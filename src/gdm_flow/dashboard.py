"""Interactive Plotly dashboard generation for GDM-Flow solver results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBus,
    DistributionCapacitor,
    DistributionRegulator,
    DistributionTransformer,
)
from gdm.systems.distribution.components.base.distribution_branch_base import (
    DistributionBranchBase,
)
from gdm.systems.distribution.enums import Phase

from ._utils import _phase_name, _phase_voltage

# ── constants ────────────────────────────────────────────────────────────

_SOLVER_COLORS = {
    "AC OPF": "#1f77b4",
    "AC PF": "#9467bd",
    "DC OPF": "#2ca02c",
    "LinDistFlow": "#ff7f0e",
}

_PHASE_COLORS = {
    "A": "#e6194b",
    "B": "#3cb44b",
    "C": "#4363d8",
    "S1": "#f58231",
    "S2": "#911eb4",
}

_PHASE_DASH = {
    "A": "solid",
    "B": "dash",
    "C": "dot",
    "S1": "dashdot",
    "S2": "longdash",
}


# ── topology helpers ─────────────────────────────────────────────────────


def _build_nominal_map(system: DistributionSystem) -> dict[tuple[str, str], float]:
    nominal: dict[tuple[str, str], float] = {}
    for bus in system.get_components(DistributionBus):
        v = _phase_voltage(bus.rated_voltage, bus.voltage_type)
        for phase in bus.phases:
            nominal[(bus.name, _phase_name(phase))] = v
    return nominal


def _build_feeder_paths(
    system: DistributionSystem,
) -> dict[str, list[tuple[str, str, str | None, float]]]:
    """BFS from source bus.  Returns {bus_name: [(bus, parent, edge_name, cumulative_distance_m)]}.

    Each path element records the bus, its parent, the connecting edge name,
    and the cumulative electrical distance in metres from the source.
    """
    digraph = system.get_directed_graph(return_radial_network=True)
    source = system.get_source_bus().name

    # Build edge length cache
    edge_length: dict[str, float] = {}
    for u, v, data in digraph.edges(data=True):
        ctype = data.get("type", "")
        cname = data.get("name")
        if not cname:
            continue
        try:
            comp = system.get_component(ctype, cname)
        except Exception:
            continue
        if hasattr(comp, "length"):
            edge_length[cname] = float(comp.length.to("m").magnitude)
        else:
            edge_length[cname] = 0.0

    # BFS
    paths: dict[str, list[tuple[str, str, str | None, float]]] = {}
    paths[source] = [(source, "", None, 0.0)]

    from collections import deque

    queue: deque[tuple[str, float]] = deque()
    queue.append((source, 0.0))
    visited = {source}

    while queue:
        node, dist = queue.popleft()
        for _, child, data in digraph.edges(node, data=True):
            if child in visited:
                continue
            visited.add(child)
            cname = data.get("name")
            child_dist = dist + edge_length.get(cname, 0.0)
            paths[child] = paths[node] + [(child, node, cname, child_dist)]
            queue.append((child, child_dist))

    return paths


def _branch_impedance(
    branch: DistributionBranchBase, phase: str
) -> tuple[float, float]:
    """Return (R, X) in ohm for a single phase of a branch."""
    phases = [_phase_name(p) for p in branch.phases]
    if phase not in phases:
        return (0.0, 0.0)
    if hasattr(branch.equipment, "r_matrix"):
        idx = phases.index(phase)
        length = float(branch.length.to("m").magnitude)
        r = float(branch.equipment.r_matrix.to("ohm/m").magnitude[idx][idx]) * length
        x = float(branch.equipment.x_matrix.to("ohm/m").magnitude[idx][idx]) * length
        return (r, x)
    if hasattr(branch.equipment, "pos_seq_resistance"):
        length = float(branch.length.to("m").magnitude)
        r = float(branch.equipment.pos_seq_resistance.to("ohm/m").magnitude) * length
        x = float(branch.equipment.pos_seq_reactance.to("ohm/m").magnitude) * length
        return (r, x)
    return (0.0, 0.0)


# ── data extraction helpers ──────────────────────────────────────────────


def _extract_ac_bus_data(result, nominal_map, label="AC OPF"):
    """Extract per-bus voltage data from an AC-type result (AC OPF or AC PF)."""
    data: dict[tuple[str, str], dict[str, float]] = {}
    for idx, (bus, phase) in enumerate(result.ybus_result.index_to_label):
        v = result.voltage[idx]
        nom = nominal_map.get((bus, phase), 1.0)
        data[(bus, phase)] = {
            "v_mag": float(abs(v)),
            "v_pu": float(abs(v)) / nom if nom > 0 else 0.0,
            "v_angle_deg": float(np.degrees(np.angle(v))),
            "p_inj_w": float(result.power_injection[idx].real),
            "q_inj_var": float(result.power_injection[idx].imag),
        }
    return data


def _extract_ac_branch_flows(system, result):
    """Extract branch P/Q flows and losses from an AC-type result."""
    v_by_label = {
        label: result.voltage[i]
        for i, label in enumerate(result.ybus_result.index_to_label)
    }
    digraph = system.get_directed_graph(return_radial_network=True)
    flows: dict[tuple[str, str], dict[str, float]] = {}

    for u, v, edata in digraph.edges(data=True):
        ctype = edata.get("type", "")
        cname = edata.get("name")
        if not cname:
            continue
        try:
            comp = system.get_component(ctype, cname)
        except Exception:
            continue
        if not isinstance(comp, DistributionBranchBase) or not comp.in_service:
            continue

        for phase in comp.phases:
            if phase == Phase.N:
                continue
            pn = _phase_name(phase)
            v_u = v_by_label.get((u, pn))
            v_v = v_by_label.get((v, pn))
            if v_u is None or v_v is None:
                continue
            r, x = _branch_impedance(comp, pn)
            z = complex(r, x)
            if abs(z) < 1e-12:
                continue
            i_br = (v_u - v_v) / z
            s_from = v_u * np.conj(i_br)
            s_to = v_v * np.conj(-i_br)
            loss_w = float(abs(i_br) ** 2 * r)
            flows[(cname, pn)] = {
                "p_from_w": float(s_from.real),
                "q_from_var": float(s_from.imag),
                "p_to_w": float(s_to.real),
                "q_to_var": float(s_to.imag),
                "loss_w": loss_w,
                "loss_var": float(abs(i_br) ** 2 * x),
                "i_mag_a": float(abs(i_br)),
            }
    return flows


def _extract_ldf_branch_flows(system, result):
    """Extract branch P/Q flows from LinDistFlow result."""
    flows: dict[tuple[str, str], dict[str, float]] = {}
    digraph = system.get_directed_graph(return_radial_network=True)

    for (branch_name, phase), p in result.p_flow_w.items():
        q = float(result.q_flow_var.get((branch_name, phase), 0.0))

        # Estimate loss from branch impedance
        parent_bus = None
        for u, _v, data in digraph.edges(data=True):
            if data.get("name") == branch_name:
                parent_bus = u
                break

        v_from = (
            float(result.voltage_v.get((parent_bus, phase), 1.0)) if parent_bus else 1.0
        )
        loss_w = 0.0
        try:
            comp = None
            for b in system.get_components(DistributionBranchBase):
                if b.name == branch_name:
                    comp = b
                    break
            if comp:
                r, _x = _branch_impedance(comp, phase)
                if r > 0 and v_from > 0:
                    loss_w = r * (float(p) ** 2 + q**2) / max(v_from**2, 1.0)
        except Exception:
            pass

        flows[(branch_name, phase)] = {
            "p_from_w": float(p),
            "q_from_var": q,
            "loss_w": max(loss_w, 0.0),
        }
    return flows


def _extract_capacitor_states(system: DistributionSystem) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cap in system.get_components(DistributionCapacitor):
        for i, (phase, pc) in enumerate(
            zip(cap.phases, cap.equipment.phase_capacitors)
        ):
            pn = _phase_name(phase)
            state_on = True
            if hasattr(cap, "state") and cap.state:
                state_on = bool(cap.state[i]) if i < len(cap.state) else True
            q_rated = float(pc.rated_reactive_power.to("kvar").magnitude)
            banks_on = int(pc.num_banks_on) if pc.num_banks_on else 0
            banks_total = int(pc.num_banks) if pc.num_banks else 1
            rows.append(
                {
                    "name": cap.name,
                    "bus": cap.bus.name,
                    "phase": pn,
                    "in_service": cap.in_service,
                    "state": "ON" if state_on else "OFF",
                    "banks_on": banks_on,
                    "banks_total": banks_total,
                    "q_rated_kvar": q_rated,
                    "q_effective_kvar": q_rated * (banks_on / banks_total)
                    if banks_total > 0 and state_on
                    else 0.0,
                }
            )
    return rows


def _extract_regulator_states(system: DistributionSystem) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reg in system.get_components(DistributionRegulator):
        for ctrl in reg.controllers:
            v_set = float((ctrl.v_setpoint * ctrl.pt_ratio).to("volt").magnitude)
            v_min = float((ctrl.min_v_limit * ctrl.pt_ratio).to("volt").magnitude)
            v_max = float((ctrl.max_v_limit * ctrl.pt_ratio).to("volt").magnitude)
            rows.append(
                {
                    "name": reg.name,
                    "controlled_bus": ctrl.controlled_bus.name,
                    "phase": _phase_name(ctrl.controlled_phase),
                    "in_service": reg.in_service,
                    "v_setpoint_v": v_set,
                    "v_min_v": v_min,
                    "v_max_v": v_max,
                    "pt_ratio": float(ctrl.pt_ratio),
                }
            )
    return rows


def _extract_transformer_info(system: DistributionSystem) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for xfmr in system.get_components(DistributionTransformer):
        windings_info = []
        for w in xfmr.equipment.windings:
            windings_info.append(f"{float(w.rated_voltage.to('kV').magnitude):.2f} kV")
        tap_str = "—"
        if xfmr.tap_positions is not None:
            tap_str = " / ".join(
                "[" + ", ".join(f"{t:.4f}" for t in winding) + "]"
                for winding in xfmr.tap_positions
            )
        rows.append(
            {
                "name": xfmr.name,
                "buses": " → ".join(b.name for b in xfmr.buses),
                "windings": " / ".join(windings_info),
                "tap_positions": tap_str,
                "in_service": xfmr.in_service,
            }
        )
    return rows


# ── plot builders ────────────────────────────────────────────────────────


def _fig_voltage_distance(
    system: DistributionSystem,
    solver_bus_data: dict[str, dict[tuple[str, str], dict]],
    nominal_map: dict[tuple[str, str], float],
) -> go.Figure:
    """Voltage-distance profile preserving feeder connectivity.

    Each edge in the radial tree is drawn as a separate line segment so
    branching points fork naturally and the radial structure is visible.
    """
    paths = _build_feeder_paths(system)
    digraph = system.get_directed_graph(return_radial_network=True)

    # Build bus → cumulative distance map
    bus_dist: dict[str, float] = {}
    for bus_name, path in paths.items():
        bus_dist[bus_name] = path[-1][3]

    # Collect edges in the modeled tree
    edges: list[tuple[str, str]] = []
    for u, v, _data in digraph.edges(data=True):
        if u in bus_dist and v in bus_dist:
            edges.append((u, v))

    phases_found: set[str] = set()
    for bus in system.get_components(DistributionBus):
        for p in bus.phases:
            if p != Phase.N:
                phases_found.add(_phase_name(p))
    phases = sorted(phases_found)

    fig = make_subplots(
        rows=len(phases),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Phase {p}" for p in phases],
        vertical_spacing=0.06,
    )

    for row_i, phase in enumerate(phases, 1):
        for solver_name, bus_data in solver_bus_data.items():
            color = _SOLVER_COLORS.get(solver_name, "#333")

            # Build segments: for each edge, draw a line from parent to child.
            # Use None separators so Plotly draws disconnected segments in one trace.
            x_seg: list[float | None] = []
            y_seg: list[float | None] = []
            hover_seg: list[str | None] = []

            for u, v in edges:
                info_u = bus_data.get((u, phase))
                info_v = bus_data.get((v, phase))
                if info_u is None or info_v is None:
                    continue
                x_seg.extend([bus_dist[u], bus_dist[v], None])
                y_seg.extend([info_u["v_pu"], info_v["v_pu"], None])
                hover_seg.extend([u, v, None])

            # Draw edge segments as a single trace
            fig.add_trace(
                go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode="lines",
                    name=solver_name if row_i == 1 else None,
                    legendgroup=solver_name,
                    showlegend=(row_i == 1),
                    line=dict(color=color, width=2),
                    customdata=hover_seg,
                    hoverinfo="skip",
                ),
                row=row_i,
                col=1,
            )

            # Draw bus markers as a separate trace for hover info
            bus_x = []
            bus_y = []
            bus_labels = []
            for bus_name in sorted(bus_dist.keys(), key=lambda b: bus_dist[b]):
                info = bus_data.get((bus_name, phase))
                if info is None:
                    continue
                bus_x.append(bus_dist[bus_name])
                bus_y.append(info["v_pu"])
                bus_labels.append(bus_name)

            fig.add_trace(
                go.Scatter(
                    x=bus_x,
                    y=bus_y,
                    mode="markers",
                    legendgroup=solver_name,
                    showlegend=False,
                    marker=dict(size=5, color=color),
                    customdata=bus_labels,
                    hovertemplate="%{customdata}<br>%{y:.4f} pu<br>%{x:.0f} m<extra>"
                    + solver_name
                    + "</extra>",
                ),
                row=row_i,
                col=1,
            )

        # ANSI limits and reference
        all_x = [bus_dist[b] for b in bus_dist]
        if all_x:
            fig.add_hline(
                y=1.05,
                line_dash="dash",
                line_color="red",
                line_width=1,
                opacity=0.7,
                row=row_i,
                col=1,
            )
            fig.add_hline(
                y=0.95,
                line_dash="dash",
                line_color="red",
                line_width=1,
                opacity=0.7,
                row=row_i,
                col=1,
            )
            fig.add_hline(
                y=1.0, line_dash="dot", line_color="gray", opacity=0.4, row=row_i, col=1
            )

        fig.update_yaxes(title_text="V (pu)", row=row_i, col=1)

    fig.update_xaxes(title_text="Distance from source (m)", row=len(phases), col=1)
    fig.update_layout(
        title="Voltage-Distance Profile",
        height=300 * len(phases) + 100,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _fig_system_power(solver_results: dict[str, dict]) -> go.Figure:
    """Source real/reactive power comparison across solvers."""
    solvers = list(solver_results.keys())
    p_vals = [solver_results[s]["source_p"] for s in solvers]
    q_vals = [solver_results[s]["source_q"] for s in solvers]
    colors = [_SOLVER_COLORS.get(s, "#333") for s in solvers]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Source Real Power (P)", "Source Reactive Power (Q)"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=solvers,
            y=p_vals,
            marker_color=colors,
            text=[f"{v / 1e3:.2f} kW" for v in p_vals],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.1f} W<extra>Source P</extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=solvers,
            y=q_vals,
            marker_color=colors,
            text=[f"{v / 1e3:.2f} kvar" for v in q_vals],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.1f} var<extra>Source Q</extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="P (W)", row=1, col=1)
    fig.update_yaxes(title_text="Q (var)", row=1, col=2)
    fig.update_layout(
        title="Total System Power — Source Injection",
        height=400,
        template="plotly_white",
    )
    return fig


def _fig_losses(
    solver_branch_flows: dict[str, dict[tuple[str, str], dict]],
) -> go.Figure:
    """Total and per-branch losses comparison."""
    # Aggregate total loss per solver
    solver_total: dict[str, float] = {}
    for solver_name, flows in solver_branch_flows.items():
        solver_total[solver_name] = sum(f.get("loss_w", 0.0) for f in flows.values())

    solvers = list(solver_total.keys())
    totals = [solver_total[s] for s in solvers]
    colors = [_SOLVER_COLORS.get(s, "#333") for s in solvers]

    # Collect all branches and build per-branch loss comparison
    all_branches = sorted(
        {k for flows in solver_branch_flows.values() for k in flows},
        key=lambda x: max(
            (
                flows.get(x, {}).get("loss_w", 0.0)
                for flows in solver_branch_flows.values()
            ),
            default=0,
        ),
        reverse=True,
    )
    top_branches = all_branches[:20]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            "Total System Losses",
            "Top Branch Losses",
        ],
        vertical_spacing=0.15,
        row_heights=[0.35, 0.65],
    )

    fig.add_trace(
        go.Bar(
            x=solvers,
            y=totals,
            marker_color=colors,
            text=[f"{v / 1e3:.3f} kW" for v in totals],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{x}<br>%{y:.1f} W<extra>Total Loss</extra>",
        ),
        row=1,
        col=1,
    )

    x_br = [f"{b}|{p}" for b, p in top_branches]
    for solver_name, flows in solver_branch_flows.items():
        y_vals = [flows.get(k, {}).get("loss_w", 0.0) for k in top_branches]
        fig.add_trace(
            go.Bar(
                x=x_br,
                y=y_vals,
                name=solver_name,
                marker_color=_SOLVER_COLORS.get(solver_name, "#333"),
                hovertemplate="%{x}<br>%{y:.2f} W<extra>" + solver_name + "</extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Loss (W)", row=1, col=1)
    fig.update_yaxes(title_text="Loss (W)", row=2, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_layout(
        title="System Losses",
        height=700,
        template="plotly_white",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _fig_branch_pq(
    solver_branch_flows: dict[str, dict[tuple[str, str], dict]],
) -> go.Figure:
    """P and Q flow at each branch/phase."""
    # Collect all branches sorted by max |P|
    all_branches = sorted(
        {k for flows in solver_branch_flows.values() for k in flows},
        key=lambda x: max(
            (
                abs(flows.get(x, {}).get("p_from_w", 0.0))
                for flows in solver_branch_flows.values()
            ),
            default=0,
        ),
        reverse=True,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Branch Real Power Flow (P)", "Branch Reactive Power Flow (Q)"],
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    x_labels = [f"{b}|{p}" for b, p in all_branches]

    for solver_name, flows in solver_branch_flows.items():
        p_vals = [flows.get(k, {}).get("p_from_w", 0.0) for k in all_branches]
        q_vals = [flows.get(k, {}).get("q_from_var", 0.0) for k in all_branches]

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=p_vals,
                mode="lines+markers",
                name=f"{solver_name}",
                legendgroup=solver_name,
                line=dict(color=_SOLVER_COLORS.get(solver_name, "#333")),
                marker=dict(size=3),
                hovertemplate="%{x}<br>%{y:.1f} W<extra>" + solver_name + " P</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=q_vals,
                mode="lines+markers",
                name=f"{solver_name}",
                legendgroup=solver_name,
                showlegend=False,
                line=dict(color=_SOLVER_COLORS.get(solver_name, "#333")),
                marker=dict(size=3),
                hovertemplate="%{x}<br>%{y:.1f} var<extra>"
                + solver_name
                + " Q</extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="P (W)", row=1, col=1)
    fig.update_yaxes(title_text="Q (var)", row=2, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_layout(
        title="Branch Power Flows",
        height=700,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _fig_equipment_state(
    cap_rows: list[dict],
    reg_rows: list[dict],
    xfmr_rows: list[dict],
    solver_bus_data: dict[str, dict[tuple[str, str], dict]],
) -> go.Figure:
    """Capacitor banks, regulator taps, transformer info, and regulator bus voltages."""
    n_sub = 3
    has_reg_voltage = bool(reg_rows and solver_bus_data)
    if has_reg_voltage:
        n_sub = 4

    titles = ["Capacitor Bank Status", "Regulator Settings", "Transformer Status"]
    specs = [[{"type": "table"}]] * 3
    heights = [0.3, 0.35, 0.35]
    if has_reg_voltage:
        titles.insert(2, "Voltage at Regulated Buses")
        specs.insert(2, [{"type": "xy"}])
        heights = [0.25, 0.25, 0.25, 0.25]

    fig = make_subplots(
        rows=n_sub,
        cols=1,
        specs=specs,
        subplot_titles=titles,
        vertical_spacing=0.08,
        row_heights=heights,
    )

    # Capacitor table
    if cap_rows:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "Name",
                        "Bus",
                        "Phase",
                        "State",
                        "Banks On/Total",
                        "Q Rated (kvar)",
                        "Q Eff. (kvar)",
                    ],
                    fill_color="#e8eaf6",
                    font=dict(size=12, color="black"),
                    align="left",
                    line_color="#c5cae9",
                ),
                cells=dict(
                    values=[
                        [r["name"] for r in cap_rows],
                        [r["bus"] for r in cap_rows],
                        [r["phase"] for r in cap_rows],
                        [
                            f'<span style="color:{"green" if r["state"] == "ON" else "red"};font-weight:bold">{r["state"]}</span>'
                            for r in cap_rows
                        ],
                        [f"{r['banks_on']}/{r['banks_total']}" for r in cap_rows],
                        [f"{r['q_rated_kvar']:.1f}" for r in cap_rows],
                        [f"{r['q_effective_kvar']:.1f}" for r in cap_rows],
                    ],
                    fill_color=[["white", "#f5f5f5"] * ((len(cap_rows) + 1) // 2)],
                    align="left",
                    font=dict(size=11),
                    line_color="#e0e0e0",
                ),
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Table(
                header=dict(values=["Capacitors"], fill_color="#e8eaf6", align="left"),
                cells=dict(values=[["No capacitors in model"]], align="left"),
            ),
            row=1,
            col=1,
        )

    # Regulator table
    if reg_rows:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "Name",
                        "Controlled Bus",
                        "Phase",
                        "V Setpoint (V)",
                        "V Min (V)",
                        "V Max (V)",
                        "PT Ratio",
                    ],
                    fill_color="#e8eaf6",
                    font=dict(size=12, color="black"),
                    align="left",
                    line_color="#c5cae9",
                ),
                cells=dict(
                    values=[
                        [r["name"] for r in reg_rows],
                        [r["controlled_bus"] for r in reg_rows],
                        [r["phase"] for r in reg_rows],
                        [f"{r['v_setpoint_v']:.1f}" for r in reg_rows],
                        [f"{r['v_min_v']:.1f}" for r in reg_rows],
                        [f"{r['v_max_v']:.1f}" for r in reg_rows],
                        [f"{r['pt_ratio']:.2f}" for r in reg_rows],
                    ],
                    fill_color=[["white", "#f5f5f5"] * ((len(reg_rows) + 1) // 2)],
                    align="left",
                    font=dict(size=11),
                    line_color="#e0e0e0",
                ),
            ),
            row=2,
            col=1,
        )
    else:
        fig.add_trace(
            go.Table(
                header=dict(values=["Regulators"], fill_color="#e8eaf6", align="left"),
                cells=dict(values=[["No regulators in model"]], align="left"),
            ),
            row=2,
            col=1,
        )

    # Regulator bus voltage bar chart
    reg_chart_row = 3
    xfmr_row = 3
    if has_reg_voltage:
        xfmr_row = 4
        reg_labels = [f"{r['controlled_bus']}|{r['phase']}" for r in reg_rows]
        reg_keys = [(r["controlled_bus"], r["phase"]) for r in reg_rows]
        v_sets = [r["v_setpoint_v"] for r in reg_rows]
        v_mins = [r["v_min_v"] for r in reg_rows]
        v_maxs = [r["v_max_v"] for r in reg_rows]

        # Setpoint band
        fig.add_trace(
            go.Scatter(
                x=reg_labels,
                y=v_maxs,
                mode="lines",
                name="V Max",
                line=dict(color="red", dash="dash", width=1),
                legendgroup="limits",
                showlegend=True,
            ),
            row=reg_chart_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=reg_labels,
                y=v_mins,
                mode="lines",
                name="V Min",
                line=dict(color="red", dash="dash", width=1),
                fill="tonexty",
                fillcolor="rgba(255,0,0,0.05)",
                legendgroup="limits",
                showlegend=True,
            ),
            row=reg_chart_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=reg_labels,
                y=v_sets,
                mode="markers+lines",
                name="Setpoint",
                marker=dict(symbol="diamond", size=8, color="black"),
                line=dict(color="black", dash="dot", width=1),
            ),
            row=reg_chart_row,
            col=1,
        )

        # Solved voltages per solver
        for solver_name, bus_data in solver_bus_data.items():
            v_solved = []
            for key in reg_keys:
                info = bus_data.get(key)
                v_solved.append(info["v_mag"] if info else None)
            fig.add_trace(
                go.Scatter(
                    x=reg_labels,
                    y=v_solved,
                    mode="markers+lines",
                    name=solver_name,
                    marker=dict(size=6),
                    line=dict(color=_SOLVER_COLORS.get(solver_name, "#333")),
                    hovertemplate="%{x}<br>%{y:.1f} V<extra>"
                    + solver_name
                    + "</extra>",
                ),
                row=reg_chart_row,
                col=1,
            )
        fig.update_yaxes(title_text="Voltage (V)", row=reg_chart_row, col=1)

    # Transformer table
    if xfmr_rows:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Name", "Buses", "Windings", "Tap Positions", "In Service"],
                    fill_color="#e8eaf6",
                    font=dict(size=12, color="black"),
                    align="left",
                    line_color="#c5cae9",
                ),
                cells=dict(
                    values=[
                        [r["name"] for r in xfmr_rows],
                        [r["buses"] for r in xfmr_rows],
                        [r["windings"] for r in xfmr_rows],
                        [r["tap_positions"] for r in xfmr_rows],
                        ["Yes" if r["in_service"] else "No" for r in xfmr_rows],
                    ],
                    fill_color=[["white", "#f5f5f5"] * ((len(xfmr_rows) + 1) // 2)],
                    align="left",
                    font=dict(size=11),
                    line_color="#e0e0e0",
                ),
            ),
            row=xfmr_row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Transformers"], fill_color="#e8eaf6", align="left"
                ),
                cells=dict(values=[["No transformers in model"]], align="left"),
            ),
            row=xfmr_row,
            col=1,
        )

    fig.update_layout(
        title="Equipment State — Capacitors, Regulators & Transformers",
        height=350 * n_sub,
        template="plotly_white",
    )
    return fig


def _fig_solver_summary(solver_results: dict[str, dict]) -> go.Figure:
    """Run summary table: success, time, iterations, objective metrics."""
    solvers = list(solver_results.keys())
    metrics = [
        "Status",
        "Source P",
        "Source Q",
        "V min (pu)",
        "V max (pu)",
        "Time (ms)",
        "Iterations",
    ]
    values_by_col: list[list[str]] = []
    for s in solvers:
        r = solver_results[s]
        nom_min = r.get("v_min_pu", "—")
        nom_max = r.get("v_max_pu", "—")
        values_by_col.append(
            [
                "✓ PASS" if r["success"] else "✗ FAIL",
                f"{r['source_p'] / 1e3:.2f} kW",
                f"{r['source_q'] / 1e3:.2f} kvar" if r["source_q"] != 0 else "—",
                f"{nom_min:.4f}" if isinstance(nom_min, float) else nom_min,
                f"{nom_max:.4f}" if isinstance(nom_max, float) else nom_max,
                f"{r['elapsed'] * 1000:.0f}",
                str(r["iterations"]),
            ]
        )

    header_vals = ["Metric"] + solvers
    cell_vals = [metrics] + values_by_col

    # Color cells based on status
    header_colors = ["#1a237e"] + [_SOLVER_COLORS.get(s, "#333") for s in solvers]

    fig = go.Figure(
        go.Table(
            header=dict(
                values=header_vals,
                fill_color=header_colors,
                font=dict(size=13, color="white"),
                align="center",
                line_color="#283593",
            ),
            cells=dict(
                values=cell_vals,
                fill_color=[["#f5f5f5", "white"] * 4]
                + [["white"] * len(metrics)] * len(solvers),
                align=["left"] + ["center"] * len(solvers),
                font=dict(size=12),
                line_color="#e0e0e0",
                height=28,
            ),
        )
    )

    fig.update_layout(
        title="Solver Run Summary",
        height=max(300, 60 * len(metrics) + 100),
        template="plotly_white",
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ── main dashboard builder ───────────────────────────────────────────────


def generate_dashboard(
    system: DistributionSystem,
    solver_results: dict[str, dict],
    output_path: Path,
    model_name: str = "",
) -> Path:
    """Generate a multi-page HTML dashboard from solver results.

    Parameters
    ----------
    system : DistributionSystem
        The loaded distribution system model.
    solver_results : dict[str, dict]
        Mapping of solver name → dict with keys from ``_run_*`` functions
        (success, source_p, source_q, elapsed, iterations, result, ...).
    output_path : Path
        Where to write the HTML file.
    model_name : str
        Display name for the model (used in title).

    Returns
    -------
    Path
        The written output file path.
    """
    nominal_map = _build_nominal_map(system)

    # ── extract per-solver bus-level data ────────────────────────────
    solver_bus_data: dict[str, dict[tuple[str, str], dict]] = {}
    solver_branch_flows: dict[str, dict[tuple[str, str], dict]] = {}

    for solver_name, sr in solver_results.items():
        result = sr.get("result")
        if result is None:
            continue

        # AC-type solvers (have .voltage and .ybus_result)
        if hasattr(result, "voltage") and hasattr(result, "ybus_result"):
            solver_bus_data[solver_name] = _extract_ac_bus_data(
                result, nominal_map, solver_name
            )
            solver_branch_flows[solver_name] = _extract_ac_branch_flows(system, result)
        # LinDistFlow
        elif hasattr(result, "voltage_v"):
            bus_data: dict[tuple[str, str], dict] = {}
            for (bus, phase), v in result.voltage_v.items():
                nom = nominal_map.get((bus, phase), 1.0)
                bus_data[(bus, phase)] = {
                    "v_mag": v,
                    "v_pu": v / nom if nom > 0 else 0.0,
                    "v_angle_deg": 0.0,
                    "p_inj_w": float(result.p_net_w.get((bus, phase), 0.0)),
                    "q_inj_var": float(result.q_net_var.get((bus, phase), 0.0)),
                }
            solver_bus_data[solver_name] = bus_data
            solver_branch_flows[solver_name] = _extract_ldf_branch_flows(system, result)
        # DC OPF (angle-only, approximate voltage)
        elif hasattr(result, "theta_rad"):
            bus_data = {}
            for (bus, phase), theta in result.theta_rad.items():
                nom = nominal_map.get((bus, phase), 1.0)
                bus_data[(bus, phase)] = {
                    "v_mag": nom,
                    "v_pu": 1.0,
                    "v_angle_deg": float(np.degrees(theta)),
                    "p_inj_w": float(result.nodal_balance_w.get((bus, phase), 0.0)),
                    "q_inj_var": 0.0,
                }
            solver_bus_data[solver_name] = bus_data

    # Compute per-unit V min/max for summary
    for solver_name, sr in solver_results.items():
        bus_data = solver_bus_data.get(solver_name, {})
        pu_vals = [d["v_pu"] for d in bus_data.values() if d.get("v_pu")]
        if pu_vals:
            sr["v_min_pu"] = min(pu_vals)
            sr["v_max_pu"] = max(pu_vals)

    # ── extract equipment data ──────────────────────────────────────
    cap_rows = _extract_capacitor_states(system)
    reg_rows = _extract_regulator_states(system)
    xfmr_rows = _extract_transformer_info(system)

    # ── build individual figures ────────────────────────────────────
    figures = {
        "summary": _fig_solver_summary(solver_results),
        "voltage_distance": _fig_voltage_distance(system, solver_bus_data, nominal_map),
        "system_power": _fig_system_power(solver_results),
        "losses": _fig_losses(solver_branch_flows),
        "branch_pq": _fig_branch_pq(solver_branch_flows),
        "equipment": _fig_equipment_state(
            cap_rows, reg_rows, xfmr_rows, solver_bus_data
        ),
    }

    # ── compose multi-section HTML dashboard ────────────────────────
    _write_dashboard_html(figures, output_path, model_name, solver_results)

    return output_path


def _write_dashboard_html(
    figures: dict[str, go.Figure],
    output_path: Path,
    model_name: str,
    solver_results: dict[str, dict],
) -> None:
    """Write a single-file HTML dashboard with tabbed navigation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tabs = [
        ("summary", "Summary"),
        ("voltage_distance", "Voltage Profile"),
        ("system_power", "System Power"),
        ("losses", "Losses"),
        ("branch_pq", "Branch P/Q"),
        ("equipment", "Equipment"),
    ]

    solver_badges = " ".join(
        f'<span class="badge" style="background:{_SOLVER_COLORS.get(s, "#333")}">'
        f"{s}: {'✓' if r['success'] else '✗'}</span>"
        for s, r in solver_results.items()
    )

    tab_buttons = "\n".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" '
        f"onclick=\"showTab('{tid}')\">{label}</button>"
        for i, (tid, label) in enumerate(tabs)
    )

    div_sections = []
    plotly_calls = []
    for i, (tid, _label) in enumerate(tabs):
        fig = figures[tid]
        fig_json = fig.to_json()
        display = "block" if i == 0 else "none"
        div_sections.append(
            f'<div id="tab-{tid}" class="tab-content" style="display:{display}">'
            f'<div id="plot-{tid}"></div></div>'
        )
        plotly_calls.append(
            f'Plotly.newPlot("plot-{tid}", {fig_json}.data, {fig_json}.layout, '
            f"{{responsive: true, displayModeBar: true, "
            f'modeBarButtonsToRemove: ["lasso2d","select2d"]}});'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GDM-Flow Dashboard — {model_name}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #f8f9fa; --surface: #ffffff; --primary: #1a237e;
    --text: #212121; --muted: #757575; --border: #e0e0e0;
    --accent: #3f51b5; --radius: 8px;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    color: white; padding: 24px 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; }}
  .header .subtitle {{ font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }}
  .badges {{ margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }}
  .badge {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; color: white;
  }}
  .tab-bar {{
    background: var(--surface); border-bottom: 2px solid var(--border);
    display: flex; gap: 0; padding: 0 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    position: sticky; top: 0; z-index: 100;
  }}
  .tab-btn {{
    padding: 12px 20px; border: none; background: none; cursor: pointer;
    font-size: 0.9rem; font-weight: 500; color: var(--muted);
    border-bottom: 3px solid transparent; transition: all 0.2s;
  }}
  .tab-btn:hover {{ color: var(--text); background: #f0f0f0; }}
  .tab-btn.active {{
    color: var(--accent); border-bottom-color: var(--accent); font-weight: 600;
  }}
  .content {{ max-width: 1400px; margin: 0 auto; padding: 16px 24px 48px; }}
  .tab-content {{ background: var(--surface); border-radius: var(--radius);
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); padding: 16px; margin-top: 12px; }}
  .footer {{ text-align: center; padding: 16px; color: var(--muted); font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="header">
  <h1>⚡ GDM-Flow Analysis Dashboard</h1>
  <div class="subtitle">{model_name}</div>
  <div class="badges">{solver_badges}</div>
</div>
<div class="tab-bar">
  {tab_buttons}
</div>
<div class="content">
  {"".join(div_sections)}
</div>
<div class="footer">Generated by GDM-Flow</div>
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).style.display = 'block';
  event.target.classList.add('active');
  Plotly.Plots.resize(document.getElementById('plot-' + id));
}}
{chr(10).join(plotly_calls)}
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# ── time series dashboard ────────────────────────────────────────────────

_TS_PHASE_COLORS = {
    "A": "#e6194b",
    "B": "#3cb44b",
    "C": "#4363d8",
    "S1": "#f58231",
    "S2": "#911eb4",
}


def _fig_ts_voltage_heatmap(conn, run_id: str) -> go.Figure | None:
    """Voltage per-unit heatmap: buses × timesteps."""
    rows = conn.execute(
        """SELECT n.bus_name, n.phase, n.timestep,
                  COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0))
           FROM ts_nodes n
           LEFT JOIN ts_bus_nominal b ON n.bus_name = b.bus_name AND n.phase = b.phase
           WHERE n.run_id = ? AND n.voltage_mag_v IS NOT NULL
           ORDER BY n.bus_name, n.phase, n.timestep""",
        (run_id,),
    ).fetchall()
    if not rows:
        return None

    # Build matrix
    from collections import defaultdict

    traces: dict[str, dict[int, float]] = defaultdict(dict)
    timesteps: set[int] = set()
    for bus, phase, t, v in rows:
        label = f"{bus}:{phase}"
        traces[label][t] = v
        timesteps.add(t)

    ts_sorted = sorted(timesteps)
    labels_sorted = sorted(traces.keys())

    z = []
    for label in labels_sorted:
        row_data = [traces[label].get(t) for t in ts_sorted]
        z.append(row_data)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=ts_sorted,
            y=labels_sorted,
            colorscale="RdYlGn",
            colorbar=dict(title="V (pu)"),
            hovertemplate="Timestep %{x}<br>%{y}<br>%{z:.4f} pu<extra></extra>",
        )
    )
    fig.update_layout(
        title="Voltage Per-Unit Heatmap",
        xaxis_title="Timestep",
        yaxis_title="Bus:Phase",
        height=max(400, 20 * len(labels_sorted) + 150),
        template="plotly_white",
    )
    return fig


def _fig_ts_voltage_envelope(
    conn, run_id: str, nominal_v: float | None = None
) -> go.Figure | None:
    """Min/mean/max voltage p.u. envelope across all buses at each timestep."""
    rows = conn.execute(
        """SELECT n.timestep,
                  MIN(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0))),
                  AVG(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0))),
                  MAX(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0)))
           FROM ts_nodes n
           LEFT JOIN ts_bus_nominal b ON n.bus_name = b.bus_name AND n.phase = b.phase
           WHERE n.run_id = ? AND n.voltage_mag_v IS NOT NULL
           GROUP BY n.timestep ORDER BY n.timestep""",
        (run_id,),
    ).fetchall()
    if not rows:
        return None

    ts = [r[0] for r in rows]
    v_min = [r[1] for r in rows]
    v_avg = [r[2] for r in rows]
    v_max = [r[3] for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=v_max,
            mode="lines",
            name="V max",
            line=dict(color="rgba(31,119,180,0.3)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=v_min,
            mode="lines",
            name="V min",
            line=dict(color="rgba(31,119,180,0.3)"),
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=v_avg,
            mode="lines",
            name="V mean",
            line=dict(color="#1f77b4", width=2),
        )
    )

    fig.update_layout(
        title="Voltage Envelope (min / mean / max across all buses)",
        xaxis_title="Timestep",
        yaxis_title="Voltage (pu)",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    # ANSI limit lines
    fig.add_hline(y=1.05, line_dash="dash", line_color="red", line_width=1, opacity=0.7)
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", line_width=1, opacity=0.7)
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.4)

    return fig


def _fig_ts_source_power(conn, run_id: str) -> go.Figure | None:
    """Source P and Q over time from ts_summary."""
    rows = conn.execute(
        """SELECT timestep, source_p_w, source_q_var, total_loss_w
           FROM ts_summary WHERE run_id = ? ORDER BY timestep""",
        (run_id,),
    ).fetchall()
    if not rows:
        return None

    ts = [r[0] for r in rows]
    p = [r[1] or 0 for r in rows]
    q = [r[2] or 0 for r in rows]
    loss = [r[3] or 0 for r in rows]

    has_q = any(v != 0 for v in q)
    has_loss = any(v != 0 for v in loss)
    n_rows = 1 + int(has_loss)

    titles = ["Source Power"]
    if has_loss:
        titles.append("Total Losses")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=[v / 1e3 for v in p],
            mode="lines",
            name="P (kW)",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1,
        col=1,
    )
    if has_q:
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=[v / 1e3 for v in q],
                mode="lines",
                name="Q (kvar)",
                line=dict(color="#ff7f0e", width=2),
            ),
            row=1,
            col=1,
        )
    fig.update_yaxes(title_text="Power (kW / kvar)", row=1, col=1)

    if has_loss:
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=[v / 1e3 for v in loss],
                mode="lines",
                name="Loss (kW)",
                line=dict(color="#d62728", width=2),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Loss (kW)", row=2, col=1)

    fig.update_xaxes(title_text="Timestep", row=n_rows, col=1)
    fig.update_layout(
        title="Source Power & Losses",
        height=350 * n_rows,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _fig_ts_battery_soc(conn, run_id: str) -> go.Figure | None:
    """Battery SOC and dispatch traces."""
    bats = conn.execute(
        "SELECT DISTINCT battery_name FROM ts_battery_soc WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    if not bats:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Battery State of Charge", "Battery Dispatch"],
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (bname,) in enumerate(bats):
        color = colors[i % len(colors)]
        rows = conn.execute(
            """SELECT timestep, soc, p_dispatch_w FROM ts_battery_soc
               WHERE run_id = ? AND battery_name = ? ORDER BY timestep""",
            (run_id, bname),
        ).fetchall()
        if not rows:
            continue
        ts = [r[0] for r in rows]
        soc = [r[1] for r in rows]
        dispatch = [(r[2] or 0) / 1e3 for r in rows]

        fig.add_trace(
            go.Scatter(
                x=ts,
                y=soc,
                mode="lines",
                name=bname,
                legendgroup=bname,
                line=dict(color=color, width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=dispatch,
                mode="lines",
                name=f"{bname} dispatch",
                legendgroup=bname,
                showlegend=False,
                line=dict(color=color, width=2),
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="SOC", row=1, col=1)
    fig.update_yaxes(title_text="Dispatch (kW)", row=2, col=1)
    fig.update_xaxes(title_text="Timestep", row=2, col=1)

    # Add SOC reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _fig_ts_convergence(conn, run_id: str) -> go.Figure | None:
    """Per-timestep convergence status and solve time."""
    rows = conn.execute(
        """SELECT timestep, success, solve_time_s FROM ts_summary
           WHERE run_id = ? ORDER BY timestep""",
        (run_id,),
    ).fetchall()
    if not rows:
        return None

    ts = [r[0] for r in rows]
    success = [r[1] for r in rows]
    solve_ms = [(r[2] or 0) * 1000 for r in rows]

    has_time = any(v > 0 for v in solve_ms)
    n_rows = 1 + int(has_time)
    titles = ["Convergence Status"]
    if has_time:
        titles.append("Solve Time (ms)")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    # Color by success: green = converged, red = failed
    colors = ["#2ca02c" if s else "#d62728" for s in success]
    fig.add_trace(
        go.Bar(
            x=ts,
            y=[1] * len(ts),
            marker_color=colors,
            name="Status",
            showlegend=False,
            hovertemplate="t=%{x}: %{customdata}<extra></extra>",
            customdata=["✓" if s else "✗" for s in success],
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(visible=False, row=1, col=1)

    if has_time:
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=solve_ms,
                mode="lines",
                name="Solve time",
                showlegend=False,
                line=dict(color="#1f77b4", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="ms", row=2, col=1)

    fig.update_xaxes(title_text="Timestep", row=n_rows, col=1)
    fig.update_layout(
        height=250 * n_rows,
        template="plotly_white",
    )
    return fig


def _fig_ts_selected_nodes(conn, run_id: str, max_nodes: int = 10) -> go.Figure | None:
    """Voltage p.u. traces for the nodes with highest voltage variation."""
    # Find nodes with most variation in p.u.
    node_stats = conn.execute(
        """SELECT n.bus_name, n.phase,
                  MAX(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0)))
                  - MIN(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0))) AS v_range,
                  MIN(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0))),
                  MAX(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0)))
           FROM ts_nodes n
           LEFT JOIN ts_bus_nominal b ON n.bus_name = b.bus_name AND n.phase = b.phase
           WHERE n.run_id = ? AND n.voltage_mag_v IS NOT NULL
           GROUP BY n.bus_name, n.phase
           ORDER BY v_range DESC
           LIMIT ?""",
        (run_id, max_nodes),
    ).fetchall()
    if not node_stats:
        return None

    fig = go.Figure()
    for bus, phase, v_range, v_min_val, v_max_val in node_stats:
        rows = conn.execute(
            """SELECT n.timestep, COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0))
               FROM ts_nodes n
               LEFT JOIN ts_bus_nominal b ON n.bus_name = b.bus_name AND n.phase = b.phase
               WHERE n.run_id = ? AND n.bus_name = ? AND n.phase = ?
               ORDER BY n.timestep""",
            (run_id, bus, phase),
        ).fetchall()
        if rows:
            ts_data = [r[0] for r in rows]
            v_data = [r[1] for r in rows]
            color = _TS_PHASE_COLORS.get(phase, "#333")
            fig.add_trace(
                go.Scatter(
                    x=ts_data,
                    y=v_data,
                    mode="lines",
                    name=f"{bus}:{phase} (Δ{v_range:.4f} pu)",
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{bus}:{phase}<br>"
                    "t=%{x}<br>V=%{y:.4f} pu<extra></extra>",
                )
            )

    # ANSI limit lines
    fig.add_hline(y=1.05, line_dash="dash", line_color="red", line_width=1, opacity=0.7)
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", line_width=1, opacity=0.7)
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.4)

    fig.update_layout(
        title=f"Top {len(node_stats)} Buses by Voltage Variation",
        xaxis_title="Timestep",
        yaxis_title="Voltage (pu)",
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _fig_ts_run_summary(conn, run_id: str) -> go.Figure:
    """Run metadata summary table."""
    row = conn.execute(
        "SELECT run_id, implementation, mode, num_timesteps, resolution_s, start_timestamp "
        "FROM ts_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()

    # Count convergence
    conv = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN success THEN 1 ELSE 0 END) "
        "FROM ts_summary WHERE run_id = ?",
        (run_id,),
    ).fetchone()

    # Voltage stats (p.u.)
    vstats = conn.execute(
        """SELECT MIN(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0))),
                  AVG(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0))),
                  MAX(COALESCE(n.voltage_pu, n.voltage_mag_v / NULLIF(b.nominal_v, 0)))
           FROM ts_nodes n
           LEFT JOIN ts_bus_nominal b ON n.bus_name = b.bus_name AND n.phase = b.phase
           WHERE n.run_id = ? AND n.voltage_mag_v IS NOT NULL""",
        (run_id,),
    ).fetchone()

    # Battery count
    bat_count = conn.execute(
        "SELECT COUNT(DISTINCT battery_name) FROM ts_battery_soc WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]

    metrics = ["Run ID", "Solver", "Mode", "Timesteps"]
    values = [row[0], row[1], row[2], str(row[3])]

    if row[4]:
        minutes = row[4] / 60
        metrics.append("Resolution")
        values.append(f"{minutes:.0f} min" if minutes >= 1 else f"{row[4]:.0f} s")

    if row[5]:
        metrics.append("Start")
        values.append(row[5])

    if conv and conv[0] > 0:
        rate = 100 * (conv[1] or 0) / conv[0]
        metrics.append("Convergence")
        values.append(f"{conv[1] or 0}/{conv[0]} ({rate:.1f}%)")

    if vstats and vstats[0] is not None:
        metrics.append("V min / max (pu)")
        values.append(f"{vstats[0]:.4f} / {vstats[2]:.4f}")

    if bat_count > 0:
        metrics.append("Batteries")
        values.append(str(bat_count))

    fig = go.Figure(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="#1a237e",
                font=dict(size=13, color="white"),
                align="left",
                line_color="#283593",
            ),
            cells=dict(
                values=[metrics, values],
                fill_color=[["#f5f5f5", "white"] * ((len(metrics) + 1) // 2)],
                align="left",
                font=dict(size=12),
                line_color="#e0e0e0",
                height=28,
            ),
        )
    )
    fig.update_layout(
        title="Time Series Run Summary",
        height=max(250, 40 * len(metrics) + 100),
        template="plotly_white",
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def generate_ts_dashboard(
    db_path: str | Path,
    output_path: str | Path,
    run_id: str | None = None,
) -> Path:
    """Generate a multi-tab HTML dashboard from QSTS or multi-period SQLite results.

    Parameters
    ----------
    db_path : str | Path
        Path to SQLite database with time series results.
    output_path : str | Path
        Where to write the HTML file.
    run_id : str | None
        Specific run to visualize. If None, uses the latest run.

    Returns
    -------
    Path
        The written output file path.
    """
    import sqlite3

    db_path = Path(db_path)
    output_path = Path(output_path)

    conn = sqlite3.connect(str(db_path))
    try:
        # Find run
        if run_id is None:
            row = conn.execute(
                "SELECT run_id FROM ts_runs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError("No time series runs found in database.")
            run_id = row[0]
        else:
            exists = conn.execute(
                "SELECT 1 FROM ts_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not exists:
                raise ValueError(f"Run ID '{run_id}' not found in database.")

        # Build all figure tabs
        tabs: list[tuple[str, str, go.Figure]] = []

        # Always include summary
        tabs.append(("summary", "Summary", _fig_ts_run_summary(conn, run_id)))

        # Source power & losses
        fig_power = _fig_ts_source_power(conn, run_id)
        if fig_power:
            tabs.append(("power", "Source Power", fig_power))

        # Voltage envelope
        fig_envelope = _fig_ts_voltage_envelope(conn, run_id)
        if fig_envelope:
            tabs.append(("voltage_envelope", "Voltage Envelope", fig_envelope))

        # Top varying nodes
        fig_nodes = _fig_ts_selected_nodes(conn, run_id)
        if fig_nodes:
            tabs.append(("voltage_traces", "Voltage Traces", fig_nodes))

        # Voltage heatmap
        fig_heatmap = _fig_ts_voltage_heatmap(conn, run_id)
        if fig_heatmap:
            tabs.append(("voltage_heatmap", "Voltage Heatmap", fig_heatmap))

        # Battery SOC & dispatch
        fig_battery = _fig_ts_battery_soc(conn, run_id)
        if fig_battery:
            tabs.append(("battery", "Battery", fig_battery))

        # Convergence
        fig_conv = _fig_ts_convergence(conn, run_id)
        if fig_conv:
            tabs.append(("convergence", "Convergence", fig_conv))

        # Get run info for title
        info = conn.execute(
            "SELECT implementation, mode FROM ts_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        model_name = f"{info[0]} / {info[1]} — {run_id}" if info else run_id

        # Build figures dict and write
        figures = {tid: fig for tid, _label, fig in tabs}
        tab_list = [(tid, label) for tid, label, _fig in tabs]

        _write_ts_dashboard_html(figures, tab_list, output_path, model_name)

    finally:
        conn.close()

    return output_path


def _write_ts_dashboard_html(
    figures: dict[str, go.Figure],
    tabs: list[tuple[str, str]],
    output_path: Path,
    model_name: str,
) -> None:
    """Write a single-file HTML time series dashboard with tabbed navigation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tab_buttons = "\n".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" '
        f"onclick=\"showTab('{tid}')\">{label}</button>"
        for i, (tid, label) in enumerate(tabs)
    )

    div_sections = []
    plotly_calls = []
    for i, (tid, _label) in enumerate(tabs):
        fig = figures[tid]
        fig_json = fig.to_json()
        display = "block" if i == 0 else "none"
        div_sections.append(
            f'<div id="tab-{tid}" class="tab-content" style="display:{display}">'
            f'<div id="plot-{tid}"></div></div>'
        )
        plotly_calls.append(
            f'Plotly.newPlot("plot-{tid}", {fig_json}.data, {fig_json}.layout, '
            f"{{responsive: true, displayModeBar: true, "
            f'modeBarButtonsToRemove: ["lasso2d","select2d"]}});'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GDM-Flow Time Series — {model_name}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #f8f9fa; --surface: #ffffff; --primary: #1a237e;
    --text: #212121; --muted: #757575; --border: #e0e0e0;
    --accent: #3f51b5; --radius: 8px;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, #004d40 0%, #00796b 100%);
    color: white; padding: 24px 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; }}
  .header .subtitle {{ font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }}
  .tab-bar {{
    background: var(--surface); border-bottom: 2px solid var(--border);
    display: flex; gap: 0; padding: 0 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    position: sticky; top: 0; z-index: 100;
  }}
  .tab-btn {{
    padding: 12px 20px; border: none; background: none; cursor: pointer;
    font-size: 0.9rem; font-weight: 500; color: var(--muted);
    border-bottom: 3px solid transparent; transition: all 0.2s;
  }}
  .tab-btn:hover {{ color: var(--text); background: #f0f0f0; }}
  .tab-btn.active {{
    color: #00796b; border-bottom-color: #00796b; font-weight: 600;
  }}
  .content {{ max-width: 1400px; margin: 0 auto; padding: 16px 24px 48px; }}
  .tab-content {{ background: var(--surface); border-radius: var(--radius);
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); padding: 16px; margin-top: 12px; }}
  .footer {{ text-align: center; padding: 16px; color: var(--muted); font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="header">
  <h1>📈 GDM-Flow Time Series Dashboard</h1>
  <div class="subtitle">{model_name}</div>
</div>
<div class="tab-bar">
  {tab_buttons}
</div>
<div class="content">
  {"".join(div_sections)}
</div>
<div class="footer">Generated by GDM-Flow</div>
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).style.display = 'block';
  event.target.classList.add('active');
  Plotly.Plots.resize(document.getElementById('plot-' + id));
}}
{chr(10).join(plotly_calls)}
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
