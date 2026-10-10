"""Compare AC OPF, AC PF, DC OPF, and LinDistFlow outputs with interactive Plotly charts."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components.base.distribution_branch_base import (
    DistributionBranchBase,
)
from gdm.systems.distribution.enums import Phase
from gdm_flow import (
    optimize_ac_power_flow_from_components,
    solve_dc_opf_from_components,
    solve_lindistflow,
)
from gdm_flow.ac_opf import _build_nominal_voltage_map
from gdm_flow.ac_pf import solve_ac_power_flow_from_components


def _label_text(label: tuple[str, str]) -> str:
    return f"{label[0]}|{label[1]}"


def _wrap_deg(values: list[float]) -> list[float]:
    return [((v + 180.0) % 360.0) - 180.0 for v in values]


def _branch_phase_resistance_ohm(branch: DistributionBranchBase, phase: str) -> float:
    if hasattr(branch, "equipment") and hasattr(branch.equipment, "r_matrix"):
        phases = [str(p.value if hasattr(p, "value") else p) for p in branch.phases]
        if phase not in phases:
            return 0.0
        pidx = phases.index(phase)
        length_m = float(branch.length.to("m").magnitude)
        return (
            float(branch.equipment.r_matrix.to("ohm/m").magnitude[pidx][pidx])
            * length_m
        )

    if hasattr(branch, "equipment") and hasattr(branch.equipment, "pos_seq_resistance"):
        phases = [str(p.value if hasattr(p, "value") else p) for p in branch.phases]
        if phase not in phases:
            return 0.0
        length_m = float(branch.length.to("m").magnitude)
        return (
            float(branch.equipment.pos_seq_resistance.to("ohm/m").magnitude) * length_m
        )

    return 0.0


def _branch_phase_impedance_ohm(
    branch: DistributionBranchBase, phase: str
) -> complex | None:
    if hasattr(branch, "equipment") and hasattr(branch.equipment, "r_matrix"):
        phases = [str(p.value if hasattr(p, "value") else p) for p in branch.phases]
        if phase not in phases:
            return None
        pidx = phases.index(phase)
        length_m = float(branch.length.to("m").magnitude)
        r = (
            float(branch.equipment.r_matrix.to("ohm/m").magnitude[pidx][pidx])
            * length_m
        )
        x = (
            float(branch.equipment.x_matrix.to("ohm/m").magnitude[pidx][pidx])
            * length_m
        )
        return complex(r, x)

    if hasattr(branch, "equipment") and hasattr(branch.equipment, "pos_seq_resistance"):
        phases = [str(p.value if hasattr(p, "value") else p) for p in branch.phases]
        if phase not in phases:
            return None
        length_m = float(branch.length.to("m").magnitude)
        r = float(branch.equipment.pos_seq_resistance.to("ohm/m").magnitude) * length_m
        x = float(branch.equipment.pos_seq_reactance.to("ohm/m").magnitude) * length_m
        return complex(r, x)

    return None


def main() -> None:
    import sys as _sys

    base_dir = Path(__file__).resolve().parent

    if len(_sys.argv) > 1:
        model_path = Path(_sys.argv[1])
    else:
        model_path = base_dir / "models" / "p5r.json"

    output_dir = base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    system = DistributionSystem.from_json(str(model_path))
    source_bus_obj = system.get_source_bus()
    source_bus = source_bus_obj.name

    ac = optimize_ac_power_flow_from_components(
        system,
        include_loads=True,
        include_solar=True,
        include_capacitor=True,
        include_regulator_targets=True,
        include_regulator_limits=True,
    )
    dc = solve_dc_opf_from_components(
        system,
        include_solar_generators=True,
        include_battery_generators=True,
        include_loads=True,
        theta_min_rad=-np.pi,
        theta_max_rad=np.pi,
        maxiter=2000,
    )
    pf = solve_ac_power_flow_from_components(
        system,
        include_loads=True,
        include_solar=True,
        include_capacitor=True,
    )
    ldf = solve_lindistflow(system)

    nominal_map = _build_nominal_voltage_map(system)

    ac_vm = {
        label: float(abs(ac.voltage[idx]))
        for idx, label in enumerate(ac.ybus_result.index_to_label)
    }
    ac_ang_deg = {
        label: float(np.degrees(np.angle(ac.voltage[idx])))
        for idx, label in enumerate(ac.ybus_result.index_to_label)
    }

    pf_vm = {
        label: float(abs(pf.voltage[idx]))
        for idx, label in enumerate(pf.ybus_result.index_to_label)
    }
    pf_ang_deg = {
        label: float(np.degrees(np.angle(pf.voltage[idx])))
        for idx, label in enumerate(pf.ybus_result.index_to_label)
    }

    common_vm = sorted(set(ac_vm).intersection(ldf.voltage_v))
    common_theta = sorted(set(ac_ang_deg).intersection(dc.theta_rad))

    x_vm = [_label_text(label) for label in common_vm]
    ac_vm_vals = [ac_vm[label] / nominal_map[label] for label in common_vm]
    pf_vm_vals = [
        pf_vm[label] / nominal_map[label] if label in pf_vm else float("nan")
        for label in common_vm
    ]
    ldf_vm_vals = [
        float(ldf.voltage_v[label]) / nominal_map[label] for label in common_vm
    ]
    vm_abs_diff = [abs(a - b) for a, b in zip(ac_vm_vals, ldf_vm_vals)]

    x_theta = [_label_text(label) for label in common_theta]
    ac_theta_vals_abs = [ac_ang_deg[label] for label in common_theta]
    dc_theta_vals_abs = [
        float(np.degrees(dc.theta_rad[label])) for label in common_theta
    ]

    if common_theta:
        ac_ref = ac_theta_vals_abs[0]
        dc_ref = dc_theta_vals_abs[0]
    else:
        ac_ref = 0.0
        dc_ref = 0.0

    ac_theta_vals = _wrap_deg([v - ac_ref for v in ac_theta_vals_abs])
    dc_theta_vals = _wrap_deg([v - dc_ref for v in dc_theta_vals_abs])

    ac_source_p = float(
        np.sum(
            [
                float(ac.power_injection[i].real)
                for i, label in enumerate(ac.ybus_result.index_to_label)
                if label[0] == source_bus
            ]
        )
    )

    pf_v = pf.voltage
    pf_ybus = pf.ybus_result.ybus
    pf_s = pf_v * np.conj(pf_ybus @ pf_v)
    pf_src_idx = [
        i for i, lbl in enumerate(pf.ybus_result.index_to_label) if lbl[0] == source_bus
    ]
    pf_source_p = float(np.sum([pf_s[i].real for i in pf_src_idx]))

    dc_source_labels = [
        label for label in dc.nodal_balance_w.keys() if label[0] == source_bus
    ]
    if dc_source_labels:
        dc_source_p = float(
            np.sum([-float(dc.nodal_balance_w[label]) for label in dc_source_labels])
        )
    else:
        dc_source_p = float(dc.slack_injection_w)

    digraph = system.get_directed_graph(return_radial_network=True)
    source_branch_names = {
        data.get("name")
        for u, _, data in digraph.edges(data=True)
        if u == source_bus and data.get("name")
    }
    ldf_source_p_first_hop = float(
        np.sum(
            [
                float(p)
                for (branch_name, _phase), p in ldf.p_flow_w.items()
                if branch_name in source_branch_names
            ]
        )
    )

    ldf_source_p = ldf_source_p_first_hop
    if abs(ldf_source_p_first_hop) < 1e-9:
        # Fallback: some feeders include source-side elements that are not fully
        # represented in this simplified LinDistFlow model. Use second-hop feeder
        # branches to produce a meaningful source-adjacent injection proxy.
        source_children = {v for u, v in digraph.edges() if u == source_bus}
        second_hop_branch_names = {
            data.get("name")
            for u, _, data in digraph.edges(data=True)
            if u in source_children and data.get("name")
        }
        ldf_source_p = float(
            np.sum(
                [
                    float(p)
                    for (branch_name, _phase), p in ldf.p_flow_w.items()
                    if branch_name in second_hop_branch_names
                ]
            )
        )
    if abs(ldf_source_p) < 1e-9:
        # Final fallback when source-adjacent branch tracking is not informative.
        # Net feeder demand is a stable source-injection proxy for this linear model.
        ldf_source_p = float(np.sum([float(v) for v in ldf.p_net_w.values()]))

    branch_loss_w = []
    for (branch_name, phase), p in ldf.p_flow_w.items():
        q = float(ldf.q_flow_var.get((branch_name, phase), 0.0))
        branch_obj = None
        for branch in system.get_components(DistributionBranchBase):
            if branch.name == branch_name:
                branch_obj = branch
                break
        if branch_obj is None:
            continue

        r_ohm = _branch_phase_resistance_ohm(branch_obj, phase)
        if r_ohm <= 0.0:
            continue

        from_bus = None
        for u, _, data in digraph.edges(data=True):
            if data.get("name") == branch_name:
                from_bus = u
                break
        v_from = float(ldf.voltage_v.get((from_bus, phase), 1.0)) if from_bus else 1.0
        loss = r_ohm * ((float(p) ** 2 + q**2) / max(v_from**2, 1.0))
        branch_loss_w.append((branch_name, phase, float(max(loss, 0.0))))

    ldf_total_loss_w = float(np.sum([x[2] for x in branch_loss_w]))

    ac_label_to_idx = ac.ybus_result.label_to_index
    ac_branch_loss_w = []
    for branch in system.get_components(DistributionBranchBase):
        if not branch.in_service:
            continue

        bus_u = branch.buses[0].name
        bus_v = branch.buses[1].name
        for i, phase in enumerate(branch.phases):
            phase_name = str(phase.value if hasattr(phase, "value") else phase)
            if phase == Phase.N:
                continue
            if (
                hasattr(branch, "is_closed")
                and i < len(branch.is_closed)
                and (not bool(branch.is_closed[i]))
            ):
                continue

            label_u = (bus_u, phase_name)
            label_v = (bus_v, phase_name)
            if label_u not in ac_label_to_idx or label_v not in ac_label_to_idx:
                continue

            z = _branch_phase_impedance_ohm(branch, phase_name)
            if z is None or abs(z) <= 0.0 or z.real <= 0.0:
                continue

            v_u = ac.voltage[ac_label_to_idx[label_u]]
            v_v = ac.voltage[ac_label_to_idx[label_v]]
            i_uv = (v_u - v_v) / z
            loss_w = float((abs(i_uv) ** 2) * z.real)
            ac_branch_loss_w.append((branch.name, phase_name, max(loss_w, 0.0)))

    ac_total_loss_w = float(np.sum([x[2] for x in ac_branch_loss_w]))
    pf_total_loss_w = float(np.sum(pf.power_injection.real))
    dc_total_loss_w = max(float(dc.slack_injection_w), 0.0)

    branch_p_abs = [
        (f"{branch}|{phase}", float(abs(p)))
        for (branch, phase), p in ldf.p_flow_w.items()
    ]
    branch_p_abs.sort(key=lambda x: x[1], reverse=True)
    top_branch_p_abs = branch_p_abs[:12]

    vm_diff_arr = np.array(vm_abs_diff, dtype=float)
    theta_diff_arr = np.array(
        [a - b for a, b in zip(ac_theta_vals, dc_theta_vals)],
        dtype=float,
    )
    # Wrap angle error to principal range for meaningful RMSE values.
    theta_diff_arr = np.array(_wrap_deg(theta_diff_arr.tolist()), dtype=float)

    vm_rmse = (
        float(np.sqrt(np.mean(vm_diff_arr**2))) if vm_diff_arr.size else float("nan")
    )
    vm_max = float(np.max(np.abs(vm_diff_arr))) if vm_diff_arr.size else float("nan")
    theta_rmse = (
        float(np.sqrt(np.mean(theta_diff_arr**2)))
        if theta_diff_arr.size
        else float("nan")
    )
    theta_max = (
        float(np.max(np.abs(theta_diff_arr))) if theta_diff_arr.size else float("nan")
    )

    summary_metrics = {
        "AC OPF success": str(bool(ac.success)),
        "AC PF success": str(bool(pf.success)),
        "DC success": str(bool(dc.success)),
        "LinDistFlow success": str(bool(ldf.success)),
        "AC OPF final objective": f"{float(ac.final_objective):.6e}",
        "AC PF max mismatch (pu)": f"{float(pf.max_mismatch_pu):.6e}",
        "DC objective": f"{float(dc.objective):.6e}",
        "Common voltage points": str(len(common_vm)),
        "Common angle points": str(len(common_theta)),
        "VM RMSE (pu)": f"{vm_rmse:.6f}",
        "VM max abs diff (pu)": f"{vm_max:.6f}",
        "Angle RMSE (deg)": f"{theta_rmse:.3f}",
        "Angle max abs diff (deg)": f"{theta_max:.3f}",
        "AC OPF source P (W)": f"{ac_source_p:.3f}",
        "AC PF source P (W)": f"{pf_source_p:.3f}",
        "DC source P (W)": f"{dc_source_p:.3f}",
        "LDF source P (W)": f"{ldf_source_p:.3f}",
        "AC OPF total loss I^2R (W)": f"{ac_total_loss_w:.3f}",
        "AC PF total loss sum(S) (W)": f"{pf_total_loss_w:.3f}",
        "DC total loss est (W)": f"{dc_total_loss_w:.3f}",
        "LDF total loss I^2R (W)": f"{ldf_total_loss_w:.3f}",
    }

    semantics_rows = [
        (
            "AC OPF source P (W)",
            "Sum of real parts of AC OPF solved nodal injections at all source-bus phases.",
        ),
        (
            "AC PF source P (W)",
            "Sum of real parts of S=V*conj(Y*V) at source-bus phases from Newton-Raphson power flow.",
        ),
        (
            "DC source P (W)",
            "Negative sum of DC nodal balances at all source-bus phases (phase-aggregated slack supply).",
        ),
        (
            "LDF source P (W)",
            "Source-adjacent branch-flow sum; fallback to second-hop branch-flow sum; final fallback to sum(p_net_w).",
        ),
        (
            "AC OPF total loss I^2R (W)",
            "Branch-wise AC OPF I^2R using solved complex voltages and per-phase branch impedances.",
        ),
        (
            "AC PF total loss sum(S) (W)",
            "Sum of real power injections from Newton-Raphson AC power flow (conservation of energy).",
        ),
        (
            "DC total loss est (W)",
            "Proxy reported as non-negative slack injection; not a physical branch I^2R loss model.",
        ),
        (
            "LDF total loss I^2R (W)",
            "Sum of per-branch R*(P^2+Q^2)/V^2 from LinDistFlow branch flows and upstream voltages.",
        ),
    ]

    fig = make_subplots(
        rows=8,
        cols=1,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
            [{"type": "table"}],
        ],
        subplot_titles=(
            "Voltage Magnitude Comparison (AC OPF / AC PF / LinDistFlow, pu)",
            "Voltage Angle Comparison (AC OPF / AC PF / DC OPF, referenced/wrapped)",
            "Absolute Voltage Magnitude Difference |AC - LinDistFlow| (pu)",
            "Power Flow: Source Injection Comparison",
            "Power Flow: Source Injection Comparison (Zoomed DC/LinDistFlow)",
            "Power Flow and Losses: Top Branch |P| and Total Loss Comparison",
            "Run Quality Summary",
            "Metric Semantics",
        ),
        vertical_spacing=0.036,
        row_heights=[0.13, 0.13, 0.10, 0.10, 0.09, 0.13, 0.16, 0.16],
    )

    fig.add_trace(
        go.Scatter(x=x_vm, y=ac_vm_vals, mode="lines+markers", name="AC OPF |V| (pu)"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vm,
            y=pf_vm_vals,
            mode="lines+markers",
            name="AC PF |V| (pu)",
            line=dict(dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vm, y=ldf_vm_vals, mode="lines+markers", name="LinDistFlow |V| (pu)"
        ),
        row=1,
        col=1,
    )

    pf_theta_vals_abs = [pf_ang_deg.get(label, float("nan")) for label in common_theta]
    if common_theta:
        pf_ref = pf_theta_vals_abs[0] if not np.isnan(pf_theta_vals_abs[0]) else 0.0
    else:
        pf_ref = 0.0
    pf_theta_vals = _wrap_deg([v - pf_ref for v in pf_theta_vals_abs])

    fig.add_trace(
        go.Scatter(
            x=x_theta, y=ac_theta_vals, mode="lines+markers", name="AC OPF angle (deg)"
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_theta,
            y=pf_theta_vals,
            mode="lines+markers",
            name="AC PF angle (deg)",
            line=dict(dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_theta, y=dc_theta_vals, mode="lines+markers", name="DC theta (deg)"
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=x_vm, y=vm_abs_diff, name="|AC-LDF| (pu)", marker_color="#d55e00"),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=["AC OPF", "AC PF", "DC", "LinDistFlow"],
            y=[ac_source_p, pf_source_p, dc_source_p, ldf_source_p],
            name="Source P injection (W)",
            marker_color=["#1f77b4", "#9467bd", "#2ca02c", "#ff7f0e"],
            text=[
                f"{ac_source_p:.2f}",
                f"{pf_source_p:.2f}",
                f"{dc_source_p:.2f}",
                f"{ldf_source_p:.2f}",
            ],
            textposition="outside",
            cliponaxis=False,
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=["AC PF", "DC", "LinDistFlow"],
            y=[pf_source_p, dc_source_p, ldf_source_p],
            name="Source P (zoom, W)",
            marker_color=["#9467bd", "#2ca02c", "#ff7f0e"],
            text=[f"{pf_source_p:.2f}", f"{dc_source_p:.2f}", f"{ldf_source_p:.2f}"],
            textposition="outside",
            cliponaxis=False,
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[x for x, _ in top_branch_p_abs],
            y=[y for _, y in top_branch_p_abs],
            name="Top branch |P| (W)",
            marker_color="#636efa",
        ),
        row=6,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=["AC OPF", "AC PF", "DC", "LinDistFlow"],
            y=[ac_total_loss_w, pf_total_loss_w, dc_total_loss_w, ldf_total_loss_w],
            mode="lines+markers",
            name="Total loss estimate (W)",
            marker_color="#d62728",
            yaxis="y5",
        ),
        row=6,
        col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color="#f0f0f0", align="left"),
            cells=dict(
                values=[list(summary_metrics.keys()), list(summary_metrics.values())],
                align="left",
            ),
        ),
        row=7,
        col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Reported metric", "Definition"],
                fill_color="#f0f0f0",
                align="left",
            ),
            cells=dict(
                values=[
                    [row[0] for row in semantics_rows],
                    [row[1] for row in semantics_rows],
                ],
                align="left",
            ),
        ),
        row=8,
        col=1,
    )

    fig.update_layout(
        title=(
            f"GDM Flow Flavor Comparison ({model_path.stem})"
            f"<br><sup>AC OPF={ac.success}, AC PF={pf.success}, DC={dc.success}, LDF={ldf.success}</sup>"
        ),
        height=4200,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(t=140, r=280, l=70, b=60),
    )
    fig.update_xaxes(title_text="Bus|Phase", row=1, col=1)
    fig.update_xaxes(title_text="Bus|Phase", row=2, col=1)
    fig.update_xaxes(title_text="Bus|Phase", row=3, col=1)
    fig.update_xaxes(title_text="Method", row=4, col=1)
    fig.update_xaxes(title_text="Method", row=5, col=1)
    fig.update_xaxes(title_text="Branch|Phase", row=6, col=1)
    fig.update_yaxes(title_text="Voltage (pu)", row=1, col=1)
    fig.update_yaxes(title_text="Angle (deg)", row=2, col=1)
    fig.update_yaxes(title_text="Abs Diff (pu)", row=3, col=1)
    fig.update_yaxes(title_text="Source P (W)", row=4, col=1)
    fig.update_yaxes(title_text="Source P (W)", row=5, col=1)
    fig.update_yaxes(title_text="Branch |P| / Losses (W)", row=6, col=1)

    output_html = output_dir / "opf_comparison_plotly.html"
    fig.write_html(str(output_html), include_plotlyjs="cdn")

    print("=== Plotly Comparison ===")
    print(f"ac_opf_success: {ac.success}")
    print(f"ac_pf_success: {pf.success}")
    print(f"dc_success: {dc.success}")
    print(f"lindistflow_success: {ldf.success}")
    print(f"saved: {output_html}")


if __name__ == "__main__":
    main()
