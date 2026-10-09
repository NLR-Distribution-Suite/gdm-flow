"""Newton-Raphson AC power flow solver for distribution systems.

Unlike the AC OPF in ``ac_opf.py`` which *optimises* voltage magnitudes
within bounds, this module solves the classical power-flow problem:
given fixed P/Q injections at PQ buses and a fixed voltage at the slack
bus, find the voltage magnitude and angle at every bus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math

import numpy as np

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionLoad,
    DistributionTransformer,
)
from gdm.systems.distribution.enums import Phase

from ._utils import _phase_name
from .ac_opf import (
    _build_nominal_voltage_map,
    _build_spec_vector,
    build_nodal_power_specs_from_components,
)
from .ybus import BusPhaseLabel, YBusResult, calculate_ybus

SPLIT_PHASE_POWER_FLOW_CONTRACT = "grounded-center-tap-pq-v1"


@dataclass(frozen=True)
class ACPowerFlowResult:
    """Result container for Newton-Raphson AC power flow."""

    success: bool
    message: str
    ybus_result: YBusResult
    voltage: np.ndarray
    """Complex bus voltages in SI volts."""
    voltage_pu: np.ndarray
    """Per-unit voltage magnitudes (|V| / V_nominal)."""
    power_injection: np.ndarray
    """Complex power injection at each bus in watts + j*var."""
    iterations: int
    max_mismatch_pu: float
    """Final maximum per-unit power mismatch (convergence metric)."""


def solve_ac_power_flow(
    system: DistributionSystem,
    *,
    p_spec_w: Dict[BusPhaseLabel, float] | None = None,
    q_spec_var: Dict[BusPhaseLabel, float] | None = None,
    slack_label: BusPhaseLabel | List[BusPhaseLabel] | None = None,
    include_neutral: bool = False,
    include_shunt: bool = False,
    convert_geometry_to_matrix: bool = True,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    v0_complex: Dict[BusPhaseLabel, complex] | None = None,
    line_to_line_power_specs: Dict[tuple[BusPhaseLabel, BusPhaseLabel], complex]
    | None = None,
) -> ACPowerFlowResult:
    """Solve AC power flow using Newton-Raphson with sparse LU factorisation.

    All non-slack buses are treated as PQ buses (fixed P and Q injections).
    The slack bus is held at nominal voltage magnitude and zero angle.

    Parameters
    ----------
    system : DistributionSystem
        Input distribution system.
    p_spec_w, q_spec_var : dict, optional
        Net active/reactive power injections in SI units. Positive = generation,
        negative = consumption. If omitted, zeros are used.
    slack_label : BusPhaseLabel or list, optional
        Bus-phase node(s) to hold as slack. Defaults to all non-neutral phases
        of the source bus.
    include_neutral, include_shunt, convert_geometry_to_matrix : bool
        Passed to Y-bus construction.
    max_iterations : int
        Maximum Newton-Raphson iterations.
    tolerance : float
        Per-unit power mismatch convergence threshold.
    v0_complex : dict[(bus_name, phase), complex], optional
        Initial complex voltages in SI volts for warm-starting. When provided,
        overrides the flat-start voltage magnitudes and angles for matching
        nodes. Useful for QSTS where consecutive timesteps have similar
        solutions.

    Returns
    -------
    ACPowerFlowResult
        Solved bus voltages, power injections, and convergence diagnostics.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    # --- Build Y-bus ---
    ybus_result = calculate_ybus(
        system,
        include_neutral=include_neutral,
        include_shunt=include_shunt,
        convert_geometry_to_matrix=convert_geometry_to_matrix,
        sparse=True,
    )
    ybus_si = ybus_result.ybus
    labels = ybus_result.index_to_label
    label_to_index = ybus_result.label_to_index
    n = len(labels)

    if n < 2:
        raise ValueError("At least two bus-phase nodes are required.")

    # --- Nominal voltages and per-unit base ---
    nominal_map = _build_nominal_voltage_map(system)
    v_base = np.array([nominal_map[label] for label in labels], dtype=float)

    s_spec = _build_spec_vector(labels, p_spec_w, q_spec_var)
    s_base = max(float(np.max(np.abs(s_spec))) if np.any(s_spec != 0) else 1e3, 1e3)

    # Per-unit Y-bus
    scale = np.outer(v_base, v_base) / s_base
    if hasattr(ybus_si, "multiply"):
        ybus_pu = ybus_si.multiply(scale).tocsr()
    else:
        ybus_pu = sp.csr_matrix(ybus_si * scale)

    s_spec_pu = s_spec / s_base
    line_to_line = []
    for (first, second), power in (line_to_line_power_specs or {}).items():
        if (
            first == second
            or first not in label_to_index
            or second not in label_to_index
            or not np.isfinite(power)
        ):
            raise ValueError(
                "Line-to-line power requires distinct known terminals and finite VA."
            )
        line_to_line.append(
            (label_to_index[first], label_to_index[second], complex(power))
        )

    def specified_power(voltage_pu):
        specified = s_spec_pu.copy()
        voltage = voltage_pu * v_base
        for first, second, power in line_to_line:
            difference = voltage[first] - voltage[second]
            if abs(difference) < 1e-12:
                raise ValueError(
                    "Line-to-line load terminal voltage collapsed to zero."
                )
            conjugate_current = power / difference / s_base
            specified[first] += voltage[first] * conjugate_current
            specified[second] -= voltage[second] * conjugate_current
        return specified

    # --- Slack bus identification ---
    if slack_label is None:
        try:
            source_bus = system.get_source_bus()
            source_phases = [_phase_name(p) for p in source_bus.phases if p != Phase.N]
            if source_phases:
                slack_label = [(source_bus.name, p) for p in source_phases]
        except Exception:
            pass

    if slack_label is None:
        slack_set = {0}
    elif isinstance(slack_label, list):
        slack_set = set()
        for sl in slack_label:
            if sl not in label_to_index:
                raise ValueError(f"Unknown slack label: {sl}")
            slack_set.add(label_to_index[sl])
    else:
        if slack_label not in label_to_index:
            raise ValueError(f"Unknown slack label: {slack_label}")
        slack_set = {label_to_index[slack_label]}

    # --- Connectivity: treat unreachable nodes as slack (fixed at flat start) ---
    from collections import deque

    y_adj = (abs(ybus_pu) + abs(ybus_pu).T).tocsr()
    reachable: set[int] = set()
    bfs_queue: deque[int] = deque()
    for seed in slack_set:
        if seed not in reachable:
            reachable.add(seed)
            bfs_queue.append(seed)
    while bfs_queue:
        node = bfs_queue.popleft()
        for neighbor in y_adj[node].indices:
            if neighbor not in reachable:
                reachable.add(neighbor)
                bfs_queue.append(neighbor)
    slack_set = slack_set | (set(range(n)) - reachable)

    non_slack = sorted(i for i in range(n) if i not in slack_set)
    m = len(non_slack)

    # --- Initial flat start: Vm=1.0 pu, θ per balanced 3-phase convention ---
    theta = np.zeros(n, dtype=float)
    _PHASE_ANGLE = {"B": -2.0 * math.pi / 3.0, "C": 2.0 * math.pi / 3.0}
    for idx, (bus, ph) in enumerate(labels):
        if ph in _PHASE_ANGLE:
            theta[idx] = _PHASE_ANGLE[ph]

    # --- Split-phase (S1/S2) angle initialization ---
    # For center-tapped transformers, S1 is in-phase with the primary and
    # S2 is anti-phase.  Build a map from each secondary-side S1/S2 bus to
    # its parent transformer's primary angle, then propagate through the
    # secondary line network so downstream customer buses also get the
    # correct angle (not just the transformer-direct bus).
    _s_bus_pri_angle: dict[str, float] = {}  # bus_name → primary angle
    for xfmr in system.get_components(DistributionTransformer):
        if not xfmr.in_service or len(xfmr.equipment.windings) < 3:
            continue
        pri_phases = [p for p in xfmr.winding_phases[0] if p != Phase.N]
        if not pri_phases:
            continue
        pri_angle = _PHASE_ANGLE.get(_phase_name(pri_phases[0]), 0.0)
        for w_idx in range(1, len(xfmr.equipment.windings)):
            bus_sec = xfmr.buses[w_idx] if w_idx < len(xfmr.buses) else xfmr.buses[-1]
            _s_bus_pri_angle[bus_sec.name] = pri_angle

    # Propagate through secondary lines: any bus reachable from a transformer
    # secondary via S1/S2 lines inherits the same primary angle.
    if _s_bus_pri_angle:
        from gdm.systems.distribution.components import DistributionBranchBase

        _sec_adj: dict[str, list[str]] = {}
        for branch in system.get_components(DistributionBranchBase):
            if not branch.in_service:
                continue
            branch_phases = {_phase_name(p) for p in branch.phases if p != Phase.N}
            if branch_phases & {"S1", "S2"}:
                b0, b1 = branch.buses[0].name, branch.buses[1].name
                _sec_adj.setdefault(b0, []).append(b1)
                _sec_adj.setdefault(b1, []).append(b0)

        bfs_queue: list[str] = list(_s_bus_pri_angle.keys())
        visited = set(bfs_queue)
        while bfs_queue:
            bus = bfs_queue.pop(0)
            angle = _s_bus_pri_angle[bus]
            for neighbor in _sec_adj.get(bus, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    _s_bus_pri_angle[neighbor] = angle
                    bfs_queue.append(neighbor)

    # Now set angles for S1/S2 nodes based on the bus→primary angle map.
    for idx, label in enumerate(labels):
        bus_name, ph = label
        if ph == "S1":
            theta[idx] = _s_bus_pri_angle.get(bus_name, 0.0)
        elif ph == "S2":
            theta[idx] = _s_bus_pri_angle.get(bus_name, 0.0) + math.pi

    vm_pu = np.ones(n, dtype=float)

    # --- External warm start from previous solve (QSTS) ---
    if v0_complex is not None:
        for label, v_cmplx in v0_complex.items():
            if label in label_to_index:
                idx = label_to_index[label]
                nom = nominal_map[label]
                if nom > 0:
                    vm_pu[idx] = abs(v_cmplx) / nom
                    theta[idx] = float(np.angle(v_cmplx))
    else:
        # --- LinDistFlow warm start ---
        # Distribution feeders with high loading can stall a flat-start NR.
        # A quick LinDistFlow solve (backward/forward sweep on the directed
        # graph) gives a much better initial voltage profile when it succeeds.
        try:
            from .lindistflow import solve_lindistflow

            ldf_result = solve_lindistflow(system)
            if ldf_result.success:
                for idx, label in enumerate(labels):
                    v_ldf = ldf_result.voltage_v.get(label)
                    nom = nominal_map[label]
                    if v_ldf is not None and nom > 0:
                        vm_pu[idx] = v_ldf / nom
        except Exception:
            pass  # Fall back to flat start

    # --- Newton-Raphson iterations ---
    max_mis = float("inf")
    converged = False

    for iteration in range(max_iterations):
        v_pu = vm_pu * np.exp(1j * theta)
        i_bus = ybus_pu @ v_pu
        s_calc = v_pu * np.conj(i_bus)

        mismatch = s_calc[non_slack] - specified_power(v_pu)[non_slack]
        max_mis = max(
            float(np.max(np.abs(mismatch.real))) if m > 0 else 0.0,
            float(np.max(np.abs(mismatch.imag))) if m > 0 else 0.0,
        )

        if max_mis < tolerance:
            converged = True
            break

        # --- Build Jacobian: dS/dθ and dS/d|V| for non-slack buses ---
        ns_arr = np.array(non_slack)
        y_ns = (
            ybus_pu[ns_arr, :][:, ns_arr]
            if sp.issparse(ybus_pu)
            else sp.csr_matrix(ybus_pu[np.ix_(ns_arr, ns_arr)])
        )

        v_ns = v_pu[non_slack]
        vm_ns = np.abs(v_ns)
        i_ns = i_bus[non_slack]
        s_diag = v_ns * np.conj(i_ns)

        v_diag = sp.diags(v_ns, format="csr")
        vc_diag = sp.diags(np.conj(v_ns), format="csr")
        m_mat = v_diag @ y_ns.conjugate() @ vc_diag

        ds_dtheta = -1j * m_mat + sp.diags(1j * s_diag, format="csr")
        vm_inv = 1.0 / vm_ns
        ds_dvm = m_mat @ sp.diags(vm_inv, format="csr") + sp.diags(
            s_diag * vm_inv, format="csr"
        )

        jac = sp.bmat(
            [
                [ds_dtheta.real, ds_dvm.real],
                [ds_dtheta.imag, ds_dvm.imag],
            ],
            format="csc",
        )
        if line_to_line:
            positions = {node: position for position, node in enumerate(non_slack)}
            jac = jac.tolil()
            voltage = v_pu * v_base
            for first, second, power in line_to_line:
                difference = voltage[first] - voltage[second]
                derivatives = (
                    power
                    / difference**2
                    / s_base
                    * np.array(
                        [
                            [-voltage[second], voltage[first]],
                            [voltage[second], -voltage[first]],
                        ]
                    )
                )
                for row_offset, row_node in enumerate((first, second)):
                    for column_offset, column_node in enumerate((first, second)):
                        if row_node not in positions or column_node not in positions:
                            continue
                        row = positions[row_node]
                        column = positions[column_node]
                        derivative = derivatives[row_offset, column_offset]
                        angle_derivative = derivative * 1j * voltage[column_node]
                        magnitude_derivative = (
                            derivative * voltage[column_node] / vm_pu[column_node]
                        )
                        jac[row, column] -= angle_derivative.real
                        jac[row + m, column] -= angle_derivative.imag
                        jac[row, column + m] -= magnitude_derivative.real
                        jac[row + m, column + m] -= magnitude_derivative.imag
            jac = jac.tocsc()

        rhs = np.concatenate([mismatch.real, mismatch.imag])
        dx = spsolve(jac, -rhs)

        # Damped update with backtracking line search
        alpha = 1.0
        for _ in range(10):
            theta_trial = theta.copy()
            vm_trial = vm_pu.copy()
            theta_trial[non_slack] += alpha * dx[:m]
            vm_trial[non_slack] += alpha * dx[m:]
            vm_trial[non_slack] = np.maximum(vm_trial[non_slack], 0.1)

            v_trial = vm_trial * np.exp(1j * theta_trial)
            s_trial = v_trial * np.conj(ybus_pu @ v_trial)
            mis_trial = s_trial[non_slack] - specified_power(v_trial)[non_slack]
            new_mis = max(
                float(np.max(np.abs(mis_trial.real))) if m > 0 else 0.0,
                float(np.max(np.abs(mis_trial.imag))) if m > 0 else 0.0,
            )
            if new_mis < max_mis:
                break
            alpha *= 0.5

        theta[non_slack] += alpha * dx[:m]
        vm_pu[non_slack] += alpha * dx[m:]
        vm_pu[non_slack] = np.maximum(vm_pu[non_slack], 0.1)

    # --- Build result ---
    v_pu_final = vm_pu * np.exp(1j * theta)
    v_si = v_pu_final * v_base
    s_si = v_si * np.conj(ybus_si @ v_si)

    if converged:
        msg = f"Converged in {iteration + 1} iterations (max mismatch {max_mis:.2e} pu)"
    else:
        msg = f"Did not converge after {max_iterations} iterations (max mismatch {max_mis:.2e} pu)"

    return ACPowerFlowResult(
        success=converged,
        message=msg,
        ybus_result=ybus_result,
        voltage=v_si,
        voltage_pu=vm_pu,
        power_injection=s_si,
        iterations=iteration + 1 if converged else max_iterations,
        max_mismatch_pu=max_mis,
    )


def solve_ac_power_flow_from_components(
    system: DistributionSystem,
    *,
    include_loads: bool = True,
    include_solar: bool = True,
    include_battery: bool = False,
    include_capacitor: bool = True,
    load_scale: float = 1.0,
    solar_scale: float = 1.0,
    battery_scale: float = 1.0,
    capacitor_scale: float = 1.0,
    slack_label: BusPhaseLabel | List[BusPhaseLabel] | None = None,
    include_neutral: bool = False,
    include_shunt: bool = False,
    convert_geometry_to_matrix: bool = True,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    v0_complex: Dict[BusPhaseLabel, complex] | None = None,
) -> ACPowerFlowResult:
    """Solve AC power flow with nodal specs auto-derived from system components.

    This is a convenience wrapper around :func:`solve_ac_power_flow` that
    builds P/Q injection specs from distribution system components (loads,
    solar, battery, capacitor).
    """
    p_spec_w, q_spec_var = build_nodal_power_specs_from_components(
        system,
        include_loads=include_loads,
        include_solar=include_solar,
        include_battery=include_battery,
        include_capacitor=include_capacitor,
        load_scale=load_scale,
        solar_scale=solar_scale,
        battery_scale=battery_scale,
        capacitor_scale=capacitor_scale,
        skip_split_phase_delta=True,
    )
    line_to_line_power_specs = {}
    if include_loads:
        for load in system.get_components(DistributionLoad):
            if not load.in_service or {_phase_name(phase) for phase in load.phases} != {
                "S1",
                "S2",
            }:
                continue
            if (
                getattr(
                    load.equipment.connection_type,
                    "value",
                    load.equipment.connection_type,
                )
                != "DELTA"
            ):
                continue
            pair = ((load.bus.name, "S1"), (load.bus.name, "S2"))
            power = -load_scale * sum(
                complex(
                    float(phase_load.real_power.to("watt").magnitude),
                    float(phase_load.reactive_power.to("var").magnitude),
                )
                for phase_load in load.equipment.phase_loads
            )
            line_to_line_power_specs[pair] = (
                line_to_line_power_specs.get(pair, 0j) + power
            )

    if slack_label is None:
        try:
            source_bus = system.get_source_bus()
            source_phases = [_phase_name(p) for p in source_bus.phases if p != Phase.N]
            if source_phases:
                slack_label = [(source_bus.name, p) for p in source_phases]
        except Exception:
            pass

    return solve_ac_power_flow(
        system,
        p_spec_w=p_spec_w,
        q_spec_var=q_spec_var,
        slack_label=slack_label,
        include_neutral=include_neutral,
        include_shunt=include_shunt,
        convert_geometry_to_matrix=convert_geometry_to_matrix,
        max_iterations=max_iterations,
        tolerance=tolerance,
        v0_complex=v0_complex,
        line_to_line_power_specs=line_to_line_power_specs,
    )
