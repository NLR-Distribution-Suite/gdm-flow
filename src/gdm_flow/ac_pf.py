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


def _slack_node_indices(system, slack_label, label_to_index, ybus_pu):
    from collections import deque

    if slack_label is None:
        try:
            source_bus = system.get_source_bus()
            source_phases = [
                _phase_name(phase) for phase in source_bus.phases if phase != Phase.N
            ]
            if source_phases:
                slack_label = [(source_bus.name, phase) for phase in source_phases]
        except Exception:
            pass
    if slack_label is None:
        slack_set = {0}
    else:
        requested = slack_label if isinstance(slack_label, list) else [slack_label]
        slack_set = set()
        for label in requested:
            if label not in label_to_index:
                raise ValueError(f"Unknown slack label: {label}")
            slack_set.add(label_to_index[label])
    adjacency = (abs(ybus_pu) + abs(ybus_pu).T).tocsr()
    reachable = set(slack_set)
    queue = deque(slack_set)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node].indices:
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    return slack_set | (set(range(len(label_to_index))) - reachable)


def _split_phase_primary_angles(system, phase_angles):
    angles = {}
    for transformer in system.get_components(DistributionTransformer):
        if not transformer.in_service or len(transformer.equipment.windings) < 3:
            continue
        primary = [phase for phase in transformer.winding_phases[0] if phase != Phase.N]
        if not primary:
            continue
        angle = phase_angles.get(_phase_name(primary[0]), 0.0)
        for index in range(1, len(transformer.equipment.windings)):
            bus = (
                transformer.buses[index]
                if index < len(transformer.buses)
                else transformer.buses[-1]
            )
            angles[bus.name] = angle
    return angles


def _propagate_secondary_angles(system, angles):
    from collections import deque
    from gdm.systems.distribution.components import DistributionBranchBase

    adjacency = {}
    for branch in system.get_components(DistributionBranchBase):
        if not branch.in_service:
            continue
        phases = {_phase_name(phase) for phase in branch.phases if phase != Phase.N}
        if phases & {"S1", "S2"}:
            first, second = branch.buses[0].name, branch.buses[1].name
            adjacency.setdefault(first, []).append(second)
            adjacency.setdefault(second, []).append(first)
    queue = deque(angles)
    visited = set(queue)
    while queue:
        bus = queue.popleft()
        for neighbor in adjacency.get(bus, []):
            if neighbor not in visited:
                visited.add(neighbor)
                angles[neighbor] = angles[bus]
                queue.append(neighbor)


def _initial_voltage_angles(system, labels):
    phase_angles = {"B": -2.0 * math.pi / 3.0, "C": 2.0 * math.pi / 3.0}
    angles = _split_phase_primary_angles(system, phase_angles)
    if angles:
        _propagate_secondary_angles(system, angles)
    theta = np.array([phase_angles.get(phase, 0.0) for _, phase in labels])
    for index, (bus, phase) in enumerate(labels):
        if phase == "S1":
            theta[index] = angles.get(bus, 0.0)
        elif phase == "S2":
            theta[index] = angles.get(bus, 0.0) + math.pi
    return theta


def _initialize_voltage(system, labels, label_to_index, nominal_map, v0_complex):
    theta = _initial_voltage_angles(system, labels)
    magnitudes = np.ones(len(labels), dtype=float)
    if v0_complex is not None:
        for label, voltage in v0_complex.items():
            if label in label_to_index and nominal_map[label] > 0:
                index = label_to_index[label]
                magnitudes[index] = abs(voltage) / nominal_map[label]
                theta[index] = float(np.angle(voltage))
    else:
        try:
            from .lindistflow import solve_lindistflow

            result = solve_lindistflow(system)
            if result.success:
                for index, label in enumerate(labels):
                    voltage = result.voltage_v.get(label)
                    if voltage is not None and nominal_map[label] > 0:
                        magnitudes[index] = voltage / nominal_map[label]
        except Exception:
            pass
    return theta, magnitudes


def _line_to_line_indices(power_specs, label_to_index):
    terminals = []
    for (first, second), power in (power_specs or {}).items():
        if (
            first == second
            or first not in label_to_index
            or second not in label_to_index
            or not np.isfinite(power)
        ):
            raise ValueError(
                "Line-to-line power requires distinct known terminals and finite VA."
            )
        terminals.append(
            (label_to_index[first], label_to_index[second], complex(power))
        )
    return terminals


def _specified_power(voltage_pu, s_spec_pu, v_base, s_base, line_to_line):
    specified = s_spec_pu.copy()
    voltage = voltage_pu * v_base
    for first, second, power in line_to_line:
        difference = voltage[first] - voltage[second]
        if abs(difference) < 1e-12:
            raise ValueError("Line-to-line load terminal voltage collapsed to zero.")
        conjugate_current = power / difference / s_base
        specified[first] += voltage[first] * conjugate_current
        specified[second] -= voltage[second] * conjugate_current
    return specified


def _nodal_power_jacobian(ybus_pu, voltage_pu, current, non_slack):
    import scipy.sparse as sp

    indices = np.array(non_slack)
    admittance = (
        ybus_pu[indices, :][:, indices]
        if sp.issparse(ybus_pu)
        else sp.csr_matrix(ybus_pu[np.ix_(indices, indices)])
    )
    voltage = voltage_pu[non_slack]
    power = voltage * np.conj(current[non_slack])
    voltage_diagonal = sp.diags(voltage, format="csr")
    conjugate_diagonal = sp.diags(np.conj(voltage), format="csr")
    product = voltage_diagonal @ admittance.conjugate() @ conjugate_diagonal
    angle_derivative = -1j * product + sp.diags(1j * power, format="csr")
    inverse_magnitude = 1.0 / np.abs(voltage)
    magnitude_derivative = product @ sp.diags(
        inverse_magnitude, format="csr"
    ) + sp.diags(power * inverse_magnitude, format="csr")
    return sp.bmat(
        [
            [angle_derivative.real, magnitude_derivative.real],
            [angle_derivative.imag, magnitude_derivative.imag],
        ],
        format="csc",
    )


def _line_to_line_jacobian(jacobian, terminals, voltage, vm_pu, non_slack, s_base):
    positions = {node: position for position, node in enumerate(non_slack)}
    count = len(non_slack)
    jacobian = jacobian.tolil()
    for first, second, power in terminals:
        difference = voltage[first] - voltage[second]
        derivatives = (
            power
            / difference**2
            / s_base
            * np.array(
                [[-voltage[second], voltage[first]], [voltage[second], -voltage[first]]]
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
                jacobian[row, column] -= angle_derivative.real
                jacobian[row + count, column] -= angle_derivative.imag
                jacobian[row, column + count] -= magnitude_derivative.real
                jacobian[row + count, column + count] -= magnitude_derivative.imag
    return jacobian.tocsc()


def _maximum_power_mismatch(mismatch):
    return max(
        float(np.max(np.abs(mismatch.real))) if mismatch.size else 0.0,
        float(np.max(np.abs(mismatch.imag))) if mismatch.size else 0.0,
    )


def _backtracking_step(
    theta, vm_pu, step, non_slack, ybus_pu, specified_power, max_mismatch
):
    count = len(non_slack)
    alpha = 1.0
    for _ in range(10):
        theta_trial = theta.copy()
        vm_trial = vm_pu.copy()
        theta_trial[non_slack] += alpha * step[:count]
        vm_trial[non_slack] += alpha * step[count:]
        vm_trial[non_slack] = np.maximum(vm_trial[non_slack], 0.1)
        voltage = vm_trial * np.exp(1j * theta_trial)
        power = voltage * np.conj(ybus_pu @ voltage)
        mismatch = power[non_slack] - specified_power(voltage)[non_slack]
        if _maximum_power_mismatch(mismatch) < max_mismatch:
            break
        alpha *= 0.5
    return alpha


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
    line_to_line = _line_to_line_indices(line_to_line_power_specs, label_to_index)

    def specified_power(voltage_pu):
        return _specified_power(voltage_pu, s_spec_pu, v_base, s_base, line_to_line)

    slack_set = _slack_node_indices(system, slack_label, label_to_index, ybus_pu)
    non_slack = sorted(i for i in range(n) if i not in slack_set)
    m = len(non_slack)
    theta, vm_pu = _initialize_voltage(
        system, labels, label_to_index, nominal_map, v0_complex
    )

    # --- Newton-Raphson iterations ---
    max_mis = float("inf")
    converged = False

    for iteration in range(max_iterations):
        v_pu = vm_pu * np.exp(1j * theta)
        i_bus = ybus_pu @ v_pu
        s_calc = v_pu * np.conj(i_bus)

        mismatch = s_calc[non_slack] - specified_power(v_pu)[non_slack]
        max_mis = _maximum_power_mismatch(mismatch)

        if max_mis < tolerance:
            converged = True
            break

        jac = _nodal_power_jacobian(ybus_pu, v_pu, i_bus, non_slack)
        if line_to_line:
            jac = _line_to_line_jacobian(
                jac, line_to_line, v_pu * v_base, vm_pu, non_slack, s_base
            )

        rhs = np.concatenate([mismatch.real, mismatch.imag])
        dx = spsolve(jac, -rhs)

        alpha = _backtracking_step(
            theta, vm_pu, dx, non_slack, ybus_pu, specified_power, max_mis
        )
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
