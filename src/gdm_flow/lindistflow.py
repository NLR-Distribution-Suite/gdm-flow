"""LinDistFlow module for radial distribution power-flow approximation."""

from __future__ import annotations

import math as _math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBattery,
    DistributionBus,
    DistributionCapacitor,
    DistributionLoad,
    DistributionReactor,
    DistributionSolar,
    DistributionTransformer,
)
from gdm.systems.distribution.components.distribution_regulator import (
    DistributionRegulator,
)
from gdm.systems.distribution.components.base.distribution_branch_base import (
    DistributionBranchBase,
)
from gdm.systems.distribution.enums import Phase

from ._utils import _phase_name, _phase_voltage
from .ybus import _reactor_impedance_ohm

BusPhaseLabel = Tuple[str, str]
BranchPhaseLabel = Tuple[str, str]


@dataclass(frozen=True)
class LinDistFlowResult:
    """Result payload from linearized DistFlow solve."""

    success: bool
    message: str
    source_bus: str
    voltage_v: Dict[BusPhaseLabel, float]
    p_flow_w: Dict[BranchPhaseLabel, float]
    q_flow_var: Dict[BranchPhaseLabel, float]
    p_net_w: Dict[BusPhaseLabel, float]
    q_net_var: Dict[BusPhaseLabel, float]


def build_lindistflow_net_injections_from_components(
    system: DistributionSystem,
    *,
    include_loads: bool = True,
    include_solar: bool = True,
    include_battery: bool = True,
    include_capacitor: bool = True,
    load_scale: float = 1.0,
    solar_scale: float = 1.0,
    battery_scale: float = 1.0,
    capacitor_scale: float = 1.0,
) -> tuple[dict[BusPhaseLabel, float], dict[BusPhaseLabel, float]]:
    """Build net nodal demand for LinDistFlow.

    Positive values indicate demand (consumption), negative values indicate net injection.
    """

    p_net: dict[BusPhaseLabel, float] = defaultdict(float)
    q_net: dict[BusPhaseLabel, float] = defaultdict(float)

    if include_loads:
        for load in system.get_components(DistributionLoad):
            if not load.in_service:
                continue
            for phase, phase_load in zip(load.phases, load.equipment.phase_loads):
                label = (load.bus.name, _phase_name(phase))
                p_net[label] += (
                    float(phase_load.real_power.to("watt").magnitude) * load_scale
                )
                q_net[label] += (
                    float(phase_load.reactive_power.to("var").magnitude) * load_scale
                )

    if include_solar:
        for solar in system.get_components(DistributionSolar):
            if not solar.in_service or not solar.phases:
                continue
            p_each = (
                float(solar.active_power.to("watt").magnitude)
                * solar_scale
                / len(solar.phases)
            )
            q_each = (
                float(solar.reactive_power.to("var").magnitude)
                * solar_scale
                / len(solar.phases)
            )
            for phase in solar.phases:
                label = (solar.bus.name, _phase_name(phase))
                p_net[label] -= p_each
                q_net[label] -= q_each

    if include_battery:
        for battery in system.get_components(DistributionBattery):
            if not battery.in_service or not battery.phases:
                continue
            p_each = (
                float(battery.active_power.to("watt").magnitude)
                * battery_scale
                / len(battery.phases)
            )
            q_each = (
                float(battery.reactive_power.to("var").magnitude)
                * battery_scale
                / len(battery.phases)
            )
            for phase in battery.phases:
                label = (battery.bus.name, _phase_name(phase))
                p_net[label] -= p_each
                q_net[label] -= q_each

    if include_capacitor:
        for capacitor in system.get_components(DistributionCapacitor):
            if not capacitor.in_service:
                continue
            if hasattr(capacitor, "state") and not any(capacitor.state):
                continue
            for phase, phase_cap in zip(
                capacitor.phases, capacitor.equipment.phase_capacitors
            ):
                label = (capacitor.bus.name, _phase_name(phase))
                banks_ratio = (
                    float(phase_cap.num_banks_on) / float(phase_cap.num_banks)
                    if phase_cap.num_banks
                    else 0.0
                )
                q_net[label] -= (
                    float(phase_cap.rated_reactive_power.to("var").magnitude)
                    * banks_ratio
                    * capacitor_scale
                )

    return dict(p_net), dict(q_net)


def _branch_phase_impedance_ohm(
    branch: DistributionBranchBase,
    phase: str,
) -> tuple[float, float]:
    if isinstance(branch, DistributionReactor):
        if phase not in [_phase_name(phase) for phase in branch.phases]:
            return 0.0, 0.0
        impedance = _reactor_impedance_ohm(branch)
        return impedance.real, impedance.imag
    # Matrix impedance branch family
    if hasattr(branch, "equipment") and hasattr(branch.equipment, "r_matrix"):
        if phase not in [_phase_name(p) for p in branch.phases]:
            return 0.0, 0.0
        pidx = [_phase_name(p) for p in branch.phases].index(phase)
        length_m = float(branch.length.to("m").magnitude)
        r = (
            float(branch.equipment.r_matrix.to("ohm/m").magnitude[pidx][pidx])
            * length_m
        )
        x = (
            float(branch.equipment.x_matrix.to("ohm/m").magnitude[pidx][pidx])
            * length_m
        )
        return r, x

    # Sequence impedance branch fallback
    if hasattr(branch, "equipment") and hasattr(branch.equipment, "pos_seq_resistance"):
        if phase not in [_phase_name(p) for p in branch.phases]:
            return 0.0, 0.0
        length_m = float(branch.length.to("m").magnitude)
        r = float(branch.equipment.pos_seq_resistance.to("ohm/m").magnitude) * length_m
        x = float(branch.equipment.pos_seq_reactance.to("ohm/m").magnitude) * length_m
        return r, x

    return 0.0, 0.0


def _branch_impedance_matrix_ohm(
    branch: DistributionBranchBase,
) -> tuple[list[str], list[list[float]], list[list[float]]] | None:
    """Return the full (phases, R_matrix, X_matrix) in ohms, or None if unavailable."""
    if not (hasattr(branch, "equipment") and hasattr(branch.equipment, "r_matrix")):
        return None
    phases = [_phase_name(p) for p in branch.phases]
    n = len(phases)
    if n < 2:
        return None
    length_m = float(branch.length.to("m").magnitude)
    r_mat = [
        [
            float(branch.equipment.r_matrix.to("ohm/m").magnitude[i][j]) * length_m
            for j in range(n)
        ]
        for i in range(n)
    ]
    x_mat = [
        [
            float(branch.equipment.x_matrix.to("ohm/m").magnitude[i][j]) * length_m
            for j in range(n)
        ]
        for i in range(n)
    ]
    return phases, r_mat, x_mat


# Nominal phase angles in radians for the angle-aware coupled formulation.

_PHASE_ANGLE_RAD: dict[str, float] = {
    "A": 0.0,
    "B": -2.0 * _math.pi / 3.0,
    "C": 2.0 * _math.pi / 3.0,
    "S1": 0.0,
    "S2": _math.pi,
}


def _transformer_phase_impedance_and_ratio(
    transformer: DistributionTransformer,
    winding_idx: int = 1,
    num_secondary: int = 1,
) -> tuple[float, float, float]:
    """Return ``(r_ohm, x_ohm, turns_ratio)`` for a transformer winding pair.

    Impedance values are in ohms referred to the *primary* (sending) side.
    Turns ratio ``a = V_primary / V_secondary``.

    For center-tapped transformers, *winding_idx* selects the secondary winding
    and *num_secondary* scales impedance so the parallel combination of all
    secondary paths equals the total transformer impedance.
    """
    if len(transformer.equipment.windings) < 2:
        return 0.0, 0.0, 1.0

    w_u = transformer.equipment.windings[0]
    w_v = transformer.equipment.windings[
        min(winding_idx, len(transformer.equipment.windings) - 1)
    ]

    v_u = _phase_voltage(w_u.rated_voltage, w_u.voltage_type)
    v_v = _phase_voltage(w_v.rated_voltage, w_v.voltage_type)

    s_phase = float(w_u.rated_power.to("va").magnitude) / max(1, int(w_u.num_phases))
    a = v_u / v_v if v_v > 0 else 1.0
    if s_phase <= 0 or v_u <= 0:
        return 0.0, 0.0, a

    r_pu = float(transformer.equipment.pct_full_load_loss) / 100.0
    x_pu = float(transformer.equipment.winding_reactances[0]) / 100.0
    z_base = (v_u * v_u) / s_phase

    return r_pu * z_base * num_secondary, x_pu * z_base * num_secondary, a


def solve_lindistflow(
    system: DistributionSystem,
    *,
    p_net_w: Dict[BusPhaseLabel, float] | None = None,
    q_net_var: Dict[BusPhaseLabel, float] | None = None,
    include_neutral: bool = False,
    include_open_switches: bool = False,
    convert_geometry_to_matrix: bool = True,
) -> LinDistFlowResult:
    """Solve radial LinDistFlow approximation for bus-phase voltages and branch flows."""

    if convert_geometry_to_matrix:
        from gdm.systems.distribution.components import GeometryBranch

        if any(True for _ in system.get_components(GeometryBranch)):
            system.convert_geometry_to_matrix_representation()

    if p_net_w is None or q_net_var is None:
        p_net_w, q_net_var = build_lindistflow_net_injections_from_components(
            system,
            include_loads=True,
            include_solar=True,
            include_battery=True,
            include_capacitor=True,
        )

    source_bus = system.get_source_bus()

    digraph = system.get_directed_graph(return_radial_network=True)

    phase_by_bus: dict[str, list[str]] = {}
    nominal_v_by_bus_phase: dict[BusPhaseLabel, float] = {}
    for bus in system.get_components(DistributionBus):
        phases = [_phase_name(p) for p in bus.phases if include_neutral or p != Phase.N]
        phase_by_bus[bus.name] = phases
        v_phase = _phase_voltage(bus.rated_voltage, bus.voltage_type)
        for p in phases:
            nominal_v_by_bus_phase[(bus.name, p)] = v_phase

    edge_branch: dict[tuple[str, str], DistributionBranchBase] = {}
    edge_transformer: dict[tuple[str, str], DistributionTransformer] = {}

    for u, v in digraph.edges():
        data = digraph.get_edge_data(u, v)
        if data is None:
            continue
        # Could be MultiDiGraph
        if isinstance(data, dict) and all(isinstance(k, int) for k in data.keys()):
            edge_items = data.values()
        else:
            edge_items = [data]

        for data in edge_items:
            branch_type = data.get("type")
            branch_name = data.get("name")
            if branch_type is None or branch_name is None:
                continue
            try:
                component = system.get_component(branch_type, branch_name)
            except Exception:
                continue
            if isinstance(component, DistributionBranchBase):
                if not component.in_service:
                    continue
                if hasattr(component, "is_closed") and (not include_open_switches):
                    if not all(bool(x) for x in component.is_closed):
                        continue
                edge_branch[(u, v)] = component
            elif isinstance(
                component, (DistributionTransformer, DistributionRegulator)
            ):
                if component.in_service:
                    edge_transformer[(u, v)] = component

    all_modeled_edges = set(edge_branch.keys()) | set(edge_transformer.keys())

    parent_of: dict[str, str] = {}
    children_of: dict[str, list[str]] = defaultdict(list)
    for u, v in digraph.edges():
        if (u, v) in all_modeled_edges:
            parent_of[v] = u
            children_of[u].append(v)

    # Map (u, v) -> component name for quick lookups in sweeps
    edge_name: dict[tuple[str, str], str] = {}
    for (u, v), br in edge_branch.items():
        edge_name[(u, v)] = br.name
    for (u, v), xf in edge_transformer.items():
        edge_name[(u, v)] = xf.name

    postorder_nodes = list(
        reversed(
            list(
                digraph.topological_sort()
                if hasattr(digraph, "topological_sort")
                else []
            )
        )
    )
    if not postorder_nodes:
        import networkx as nx

        postorder_nodes = list(reversed(list(nx.topological_sort(digraph))))

    p_flow: dict[BranchPhaseLabel, float] = defaultdict(float)
    q_flow: dict[BranchPhaseLabel, float] = defaultdict(float)

    for bus in postorder_nodes:
        if bus == source_bus.name:
            continue
        if bus not in parent_of:
            continue
        parent = parent_of[bus]
        branch = edge_branch.get((parent, bus))
        xfmr = edge_transformer.get((parent, bus))
        if branch is None and xfmr is None:
            continue

        comp_name = edge_name[(parent, bus)]

        # Determine common phases between parent, child, and component
        if branch is not None:
            branch_phases = [_phase_name(x) for x in branch.phases]
            phases = [
                p
                for p in phase_by_bus.get(bus, [])
                if p in phase_by_bus.get(parent, []) and p in branch_phases
            ]
        else:
            # Transformer: use common winding phases
            xfmr_phases = [
                _phase_name(p)
                for p in xfmr.winding_phases[0]
                if p in xfmr.winding_phases[1] and (include_neutral or p != Phase.N)
            ]
            phases = [
                p
                for p in phase_by_bus.get(bus, [])
                if p in phase_by_bus.get(parent, []) and p in xfmr_phases
            ]
            # Center-tapped (split-phase): primary/secondary phase names differ
            if not phases and len(xfmr.equipment.windings) >= 3:
                ct_primary = [
                    _phase_name(p)
                    for p in xfmr.winding_phases[0]
                    if include_neutral or p != Phase.N
                ]
                if ct_primary:
                    ct_secs: set[str] = set()
                    for w_idx in range(1, len(xfmr.winding_phases)):
                        for p in xfmr.winding_phases[w_idx]:
                            pn = _phase_name(p)
                            if pn != "N":
                                ct_secs.add(pn)
                    phases = [p for p in phase_by_bus.get(bus, []) if p in ct_secs]

        for ph in phases:
            net_p = float(p_net_w.get((bus, ph), 0.0))
            net_q = float(q_net_var.get((bus, ph), 0.0))
            child_p = sum(
                p_flow.get((edge_name[(bus, ch)], ph), 0.0)
                for ch in children_of.get(bus, [])
                if (bus, ch) in all_modeled_edges
            )
            child_q = sum(
                q_flow.get((edge_name[(bus, ch)], ph), 0.0)
                for ch in children_of.get(bus, [])
                if (bus, ch) in all_modeled_edges
            )
            p_flow[(comp_name, ph)] = net_p + child_p
            q_flow[(comp_name, ph)] = net_q + child_q

    voltage_v: dict[BusPhaseLabel, float] = {}
    source_phases = phase_by_bus.get(source_bus.name, [])
    for ph in source_phases:
        voltage_v[(source_bus.name, ph)] = nominal_v_by_bus_phase[(source_bus.name, ph)]

    import networkx as nx

    for bus in nx.topological_sort(digraph):
        if bus == source_bus.name:
            continue
        if bus not in parent_of:
            continue
        parent = parent_of[bus]
        branch = edge_branch.get((parent, bus))
        xfmr = edge_transformer.get((parent, bus))
        if branch is None and xfmr is None:
            continue

        comp_name = edge_name[(parent, bus)]

        ct_sec_map: dict[str, tuple[str, int]] = {}

        if branch is not None:
            branch_phases = [_phase_name(x) for x in branch.phases]
            phases = [
                p
                for p in phase_by_bus.get(bus, [])
                if p in phase_by_bus.get(parent, []) and p in branch_phases
            ]

            # Coupled 3-phase voltage drop using full impedance matrix
            # with angle-aware mutual coupling.
            z_mat = _branch_impedance_matrix_ohm(branch)
            if z_mat is not None:
                br_phases, r_mat, x_mat = z_mat
                for ph in phases:
                    if ph not in br_phases:
                        continue
                    i = br_phases.index(ph)
                    v_parent = float(
                        voltage_v.get(
                            (parent, ph), nominal_v_by_bus_phase[(parent, ph)]
                        )
                    )
                    alpha_phi = _PHASE_ANGLE_RAD.get(ph, 0.0)
                    drop = 0.0
                    for j, psi in enumerate(br_phases):
                        pf_j = p_flow.get((comp_name, psi), 0.0)
                        qf_j = q_flow.get((comp_name, psi), 0.0)
                        if i == j:
                            # Self-impedance (δ=0): standard R*P + X*Q
                            drop += r_mat[i][j] * pf_j + x_mat[i][j] * qf_j
                        else:
                            # Mutual: rotate by δ = α_φ − α_ψ
                            delta = alpha_phi - _PHASE_ANGLE_RAD.get(psi, 0.0)
                            cos_d = _math.cos(delta)
                            sin_d = _math.sin(delta)
                            r_ij = r_mat[i][j]
                            x_ij = x_mat[i][j]
                            drop += (r_ij * cos_d + x_ij * sin_d) * pf_j
                            drop += (x_ij * cos_d - r_ij * sin_d) * qf_j
                    voltage_v[(bus, ph)] = max(
                        v_parent - drop / max(v_parent, 1.0), 1.0
                    )
            else:
                # Fallback: diagonal-only (sequence impedance or single-phase)
                for ph in phases:
                    v_parent = float(
                        voltage_v.get(
                            (parent, ph), nominal_v_by_bus_phase[(parent, ph)]
                        )
                    )
                    pf = p_flow.get((comp_name, ph), 0.0)
                    qf = q_flow.get((comp_name, ph), 0.0)
                    r, x = _branch_phase_impedance_ohm(branch, ph)
                    dp = r * pf + x * qf
                    voltage_v[(bus, ph)] = max(v_parent - dp / max(v_parent, 1.0), 1.0)
        else:
            xfmr_phases = [
                _phase_name(p)
                for p in xfmr.winding_phases[0]
                if p in xfmr.winding_phases[1] and (include_neutral or p != Phase.N)
            ]
            phases = [
                p
                for p in phase_by_bus.get(bus, [])
                if p in phase_by_bus.get(parent, []) and p in xfmr_phases
            ]
            # Center-tapped (split-phase): primary/secondary phase names differ
            if not phases and len(xfmr.equipment.windings) >= 3:
                ct_primary = [
                    _phase_name(p)
                    for p in xfmr.winding_phases[0]
                    if include_neutral or p != Phase.N
                ]
                if ct_primary:
                    for w_idx in range(1, len(xfmr.winding_phases)):
                        for p in xfmr.winding_phases[w_idx]:
                            pn = _phase_name(p)
                            if pn != "N":
                                ct_sec_map[pn] = (ct_primary[0], w_idx)
                    phases = [p for p in phase_by_bus.get(bus, []) if p in ct_sec_map]

            for ph in phases:
                if ct_sec_map and ph in ct_sec_map:
                    pri_ph, _ = ct_sec_map[ph]
                    v_parent = float(
                        voltage_v.get(
                            (parent, pri_ph), nominal_v_by_bus_phase[(parent, pri_ph)]
                        )
                    )
                else:
                    v_parent = float(
                        voltage_v.get(
                            (parent, ph), nominal_v_by_bus_phase[(parent, ph)]
                        )
                    )
                pf = p_flow.get((comp_name, ph), 0.0)
                qf = q_flow.get((comp_name, ph), 0.0)

                if ct_sec_map and ph in ct_sec_map:
                    _, w_idx = ct_sec_map[ph]
                    num_sec = len(xfmr.equipment.windings) - 1
                    r, x, a = _transformer_phase_impedance_and_ratio(
                        xfmr, w_idx, num_sec
                    )
                else:
                    r, x, a = _transformer_phase_impedance_and_ratio(xfmr)
                # Transformer: V_sec ≈ V_pri/a - (r*P + x*Q) / (V_pri * a)
                dp = r * pf + x * qf
                v_child = v_parent / a - dp / (max(v_parent, 1.0) * a)
                voltage_v[(bus, ph)] = max(v_child, 1.0)

    return LinDistFlowResult(
        success=True,
        message="LinDistFlow solved on radial directed graph.",
        source_bus=source_bus.name,
        voltage_v=dict(voltage_v),
        p_flow_w=dict(p_flow),
        q_flow_var=dict(q_flow),
        p_net_w=dict(p_net_w),
        q_net_var=dict(q_net_var),
    )
