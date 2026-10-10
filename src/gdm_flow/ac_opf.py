"""Optimization routines that operate on the Y-bus model."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBattery,
    DistributionBus,
    DistributionCapacitor,
    DistributionLoad,
    DistributionRegulator,
    DistributionSolar,
    DistributionTransformer,
)
from gdm.systems.distribution.enums import Phase

from ._utils import _phase_name, _phase_voltage
from .ybus import YBusResult, calculate_ybus

BusPhaseLabel = tuple[str, str]


@dataclass(frozen=True)
class PowerFlowOptimizationResult:
    """Result container for AC nodal optimization using Y-bus."""

    success: bool
    message: str
    ybus_result: YBusResult
    voltage: np.ndarray
    power_injection: np.ndarray
    iterations: int
    initial_objective: float
    final_objective: float


def _build_nominal_voltage_map(
    system: DistributionSystem,
) -> dict[BusPhaseLabel, float]:
    nominal: dict[BusPhaseLabel, float] = {}
    for bus in system.get_components(DistributionBus):
        phase_voltage = _phase_voltage(
            bus.rated_voltage,
            bus.voltage_type,
            split_phase=any(_phase_name(phase) in {"S1", "S2"} for phase in bus.phases),
        )
        for phase in bus.phases:
            nominal[(bus.name, _phase_name(phase))] = phase_voltage
    return nominal


def _build_spec_vector(
    labels: list[BusPhaseLabel],
    p_spec_w: dict[BusPhaseLabel, float] | None,
    q_spec_var: dict[BusPhaseLabel, float] | None,
) -> np.ndarray:
    p_spec_w = p_spec_w or {}
    q_spec_var = q_spec_var or {}
    p = np.array([float(p_spec_w.get(label, 0.0)) for label in labels], dtype=float)
    q = np.array([float(q_spec_var.get(label, 0.0)) for label in labels], dtype=float)
    return p + 1j * q


def build_nodal_power_specs_from_components(
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
    skip_split_phase_delta: bool = False,
) -> tuple[dict[BusPhaseLabel, float], dict[BusPhaseLabel, float]]:
    """Build nodal active/reactive power specs from system components.

    Sign convention matches `optimize_ac_power_flow`:
    positive values mean net generation/injection and negative values mean load.
    """

    p_spec_w: dict[BusPhaseLabel, float] = defaultdict(float)
    q_spec_var: dict[BusPhaseLabel, float] = defaultdict(float)

    if include_loads:
        for load in system.get_components(DistributionLoad):
            if not load.in_service:
                continue
            phases = {_phase_name(phase) for phase in load.phases}
            connection = getattr(
                load.equipment.connection_type, "value", load.equipment.connection_type
            )
            if phases & {"S1", "S2"} and connection == "DELTA":
                if phases != {"S1", "S2"} or not skip_split_phase_delta:
                    raise ValueError(
                        "Split-phase DELTA loads require connection-aware AC power flow, not fixed nodal P/Q."
                    )
                continue
            for phase, phase_load in zip(load.phases, load.equipment.phase_loads):
                label = (load.bus.name, _phase_name(phase))
                p_spec_w[label] -= (
                    float(phase_load.real_power.to("watt").magnitude) * load_scale
                )
                q_spec_var[label] -= (
                    float(phase_load.reactive_power.to("var").magnitude) * load_scale
                )

    if include_solar:
        for solar in system.get_components(DistributionSolar):
            if not solar.in_service or not solar.phases:
                continue
            phase_count = len(solar.phases)
            p_each = (
                float(solar.active_power.to("watt").magnitude)
                * solar_scale
                / phase_count
            )
            q_each = (
                float(solar.reactive_power.to("var").magnitude)
                * solar_scale
                / phase_count
            )
            for phase in solar.phases:
                label = (solar.bus.name, _phase_name(phase))
                p_spec_w[label] += p_each
                q_spec_var[label] += q_each

    if include_battery:
        for battery in system.get_components(DistributionBattery):
            if not battery.in_service or not battery.phases:
                continue
            phase_count = len(battery.phases)
            p_each = (
                float(battery.active_power.to("watt").magnitude)
                * battery_scale
                / phase_count
            )
            q_each = (
                float(battery.reactive_power.to("var").magnitude)
                * battery_scale
                / phase_count
            )
            for phase in battery.phases:
                label = (battery.bus.name, _phase_name(phase))
                p_spec_w[label] += p_each
                q_spec_var[label] += q_each

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
                bank_ratio = (
                    float(phase_cap.num_banks_on) / float(phase_cap.num_banks)
                    if phase_cap.num_banks
                    else 0.0
                )
                q_spec_var[label] += (
                    float(phase_cap.rated_reactive_power.to("var").magnitude)
                    * bank_ratio
                    * capacitor_scale
                )

    return dict(p_spec_w), dict(q_spec_var)


def build_regulator_voltage_targets_from_components(
    system: DistributionSystem,
) -> dict[BusPhaseLabel, float]:
    """Build per-node voltage magnitude targets (in volts) from regulator controllers."""

    targets: dict[BusPhaseLabel, float] = {}
    for regulator in system.get_components(DistributionRegulator):
        if not regulator.in_service:
            continue
        for controller in regulator.controllers:
            label = (
                controller.controlled_bus.name,
                _phase_name(controller.controlled_phase),
            )
            # Convert controller setpoint (controller-side voltage) to controlled bus voltage.
            targets[label] = float(
                (controller.v_setpoint * controller.pt_ratio).to("volt").magnitude
            )
    return targets


def build_regulator_voltage_limits_from_components(
    system: DistributionSystem,
) -> dict[BusPhaseLabel, tuple[float, float]]:
    """Build per-node hard voltage limits (min/max in volts) from regulators."""

    limits: dict[BusPhaseLabel, tuple[float, float]] = {}
    for regulator in system.get_components(DistributionRegulator):
        if not regulator.in_service:
            continue
        for controller in regulator.controllers:
            label = (
                controller.controlled_bus.name,
                _phase_name(controller.controlled_phase),
            )
            v_min = float(
                (controller.min_v_limit * controller.pt_ratio).to("volt").magnitude
            )
            v_max = float(
                (controller.max_v_limit * controller.pt_ratio).to("volt").magnitude
            )
            low = min(v_min, v_max)
            high = max(v_min, v_max)
            if label in limits:
                prev_low, prev_high = limits[label]
                limits[label] = (max(prev_low, low), min(prev_high, high))
            else:
                limits[label] = (low, high)
    return limits


def _build_voltage_from_state(
    x: np.ndarray,
    n: int,
    slack_set: set[int],
    theta0: np.ndarray,
) -> np.ndarray:
    """Build complex per-unit voltage vector from optimisation state.

    *x* contains ``[theta_nonslack, vm_pu_nonslack]`` where ``vm_pu`` is
    the per-unit voltage magnitude (≈ 1.0).  Returns per-unit complex
    voltage: ``V_pu = vm_pu * exp(j * theta)``.
    """
    theta = theta0.copy()
    vm_pu = np.ones(n, dtype=float)

    non_slack = [i for i in range(n) if i not in slack_set]
    split = len(non_slack)

    theta[non_slack] = x[:split]
    vm_pu[non_slack] = x[split:]

    return vm_pu * np.exp(1j * theta)


def _objective_residual(
    x: np.ndarray,
    ybus_pu: np.ndarray,
    s_spec_pu: np.ndarray,
    s_scale_pu: np.ndarray,
    n: int,
    slack_set: set[int],
    theta0: np.ndarray,
    voltage_reg_weight: float,
    voltage_targets_pu: dict[BusPhaseLabel, float] | None,
    labels: list[BusPhaseLabel],
    voltage_target_weight: float,
) -> np.ndarray:
    v_pu = _build_voltage_from_state(x, n, slack_set, theta0)
    s_calc_pu = v_pu * np.conj(ybus_pu @ v_pu)

    non_slack = [i for i in range(n) if i not in slack_set]
    mismatch = (s_calc_pu[non_slack] - s_spec_pu[non_slack]) / s_scale_pu[non_slack]

    # vm_pu for non-slack nodes lives in the second half of x
    split = len(non_slack)
    vm_pu_nonslack = x[split:]
    reg = voltage_reg_weight * (vm_pu_nonslack - 1.0)

    regulator_terms: np.ndarray
    if voltage_targets_pu:
        vm_pu_all = np.abs(v_pu)
        reg_idx = [
            i
            for i in non_slack
            if labels[i] in voltage_targets_pu
            and float(voltage_targets_pu[labels[i]]) > 0
        ]
        if reg_idx:
            regulator_terms = voltage_target_weight * np.array(
                [
                    (vm_pu_all[i] - float(voltage_targets_pu[labels[i]]))
                    / float(voltage_targets_pu[labels[i]])
                    for i in reg_idx
                ],
                dtype=float,
            )
        else:
            regulator_terms = np.array([], dtype=float)
    else:
        regulator_terms = np.array([], dtype=float)

    return np.concatenate(
        [
            mismatch.real,
            mismatch.imag,
            reg,
            regulator_terms,
        ]
    )


def _objective_jacobian(
    x: np.ndarray,
    ybus_pu,
    s_spec_pu: np.ndarray,
    s_scale_pu: np.ndarray,
    n: int,
    slack_set: set[int],
    theta0: np.ndarray,
    voltage_reg_weight: float,
    voltage_targets_pu: dict[BusPhaseLabel, float] | None,
    labels: list[BusPhaseLabel],
    voltage_target_weight: float,
) -> np.ndarray:
    """Analytical Jacobian of the AC power-flow residual.

    Returns a sparse CSR matrix to enable efficient LSMR solves in
    scipy.optimize.least_squares with method='trf'.
    """
    import scipy.sparse as sp_sparse

    v_pu = _build_voltage_from_state(x, n, slack_set, theta0)
    non_slack = sorted(i for i in range(n) if i not in slack_set)
    m = len(non_slack)  # number of non-slack nodes

    # Current injection: I = Y * V
    i_bus = ybus_pu @ v_pu

    # Extract Y submatrix for non-slack rows/cols
    ns_arr = np.array(non_slack)
    if sp_sparse.issparse(ybus_pu):
        y_ns = ybus_pu[ns_arr, :][:, ns_arr]  # m × m sparse
    else:
        y_ns = sp_sparse.csr_matrix(ybus_pu[np.ix_(ns_arr, ns_arr)])

    # Diagonal matrices for non-slack nodes
    v_ns = v_pu[non_slack]  # complex voltage
    vm_ns = np.abs(v_ns)  # voltage magnitudes
    i_ns = i_bus[non_slack]  # current at non-slack

    # Include contributions from slack nodes: Y_ns_slack @ V_slack
    # This is captured in i_ns = (Y @ V)[non_slack] which already includes it.

    # Standard Newton-Raphson Jacobian subblocks:
    # dS/dtheta = j * diag(V) * (diag(conj(I)) - conj(Y) * diag(conj(V)))
    # But we need partial derivatives of P,Q w.r.t. theta and |V| for non-slack only.
    #
    # For row i (non-slack), col k (non-slack):
    #   dP_i/dtheta_k and dQ_i/dtheta_k depend on Y[i,k] and V[i], V[k]
    #   dP_i/d|V_k| and dQ_i/d|V_k| depend on Y[i,k] and V[i], V[k]

    # Efficient computation using sparse Y:
    # diag(V_ns) * conj(Y_ns) * diag(conj(V_ns))  gives element-wise product
    v_diag = sp_sparse.diags(v_ns, format="csr")
    vc_diag = sp_sparse.diags(np.conj(v_ns), format="csr")

    # M = diag(V) * conj(Y) * diag(conj(V))
    # M[i,k] = V[i] * conj(Y[i,k]) * conj(V[k])
    yc_ns = y_ns.conjugate()
    m_mat = v_diag @ yc_ns @ vc_diag  # m × m sparse

    # Diagonal correction: S_diag[i] = V[i] * conj(I[i])
    s_diag = v_ns * np.conj(i_ns)

    # dS_i / dtheta_k:
    #   k != i: j * M[i,k]
    #   k == i: j * (M[i,i] - S_diag[i])   (note: actually j*(M[i,i] - conj(S[i])))
    # Wait, the standard formulation:
    #   dS/dtheta = j * [diag(V) * conj(diag(I)) - diag(V) * conj(Y) * diag(conj(V))]
    # Hmm, let me be more careful.
    #
    # S[i] = V[i] * conj(sum_k Y[i,k] * V[k])
    # dS[i]/dtheta[k] for k != i:
    #   = V[i] * conj(Y[i,k] * dV[k]/dtheta[k])
    #   = V[i] * conj(Y[i,k] * j*V[k])
    #   = -j * V[i] * conj(Y[i,k]) * conj(V[k])
    #   = -j * M[i,k]
    # Wait, dV[k]/dtheta[k] = j * V[k], so conj(dV[k]/dtheta[k]) = -j * conj(V[k])
    # dS[i]/dtheta[k] = V[i] * (-j * conj(Y[i,k]) * conj(V[k]))
    #                  = -j * M[i,k]
    #
    # dS[i]/dtheta[i]:
    #   = dV[i]/dtheta[i] * conj(I[i]) + V[i] * conj(Y[i,i] * dV[i]/dtheta[i])
    #   = j*V[i]*conj(I[i]) + V[i]*conj(Y[i,i])*(-j)*conj(V[i])
    #   = j*S_diag[i] - j*M[i,i]
    #
    # So: dS/dtheta = -j*M + j*diag(S_diag) - (-j)*diag(M_diag)
    #              Wait, let me just write it element-wise:
    # For off-diagonal (k!=i): dS[i]/dtheta[k] = -j * M[i,k]
    # For diagonal (k==i):     dS[i]/dtheta[i] = j*(S_diag[i] - M[i,i])
    #
    # So dS_dtheta = -j*M + j*(diag(S_diag) - diag(M_diag)) + j*diag(M_diag)
    # Hmm, that simplifies to:
    # dS_dtheta[i,k] = -j*M[i,k]   for k != i
    # dS_dtheta[i,i] = j*S_diag[i] - j*M[i,i]
    #
    # Combining: dS_dtheta = -j*M + j*diag(S_diag)
    # Because for diagonal: -j*M[i,i] + j*S_diag[i] = j*(S_diag[i] - M[i,i]) ✓

    ds_dtheta = -1j * m_mat + sp_sparse.diags(1j * s_diag, format="csr")

    # dS[i]/d|V[k]| for k != i:
    #   dV[k]/d|V[k]| = V[k]/|V[k]| = exp(j*theta[k])
    #   dS[i]/d|V[k]| = V[i] * conj(Y[i,k] * exp(j*theta[k]))
    #                  = V[i] * conj(Y[i,k]) * exp(-j*theta[k])
    #                  = M[i,k] / conj(V[k]) * exp(-j*theta[k]) * conj(V[k])
    # Actually simpler:
    #   = V[i] * conj(Y[i,k]) * conj(V[k]) / |V[k]|
    #   = M[i,k] / |V[k]|
    #
    # dS[i]/d|V[i]|:
    #   = exp(j*theta[i]) * conj(I[i]) + V[i] * conj(Y[i,i]) * exp(-j*theta[i])
    #   = S_diag[i]/|V[i]| + M[i,i]/|V[i]|
    #
    # So: dS_dVm = M * diag(1/|V|) + diag((S_diag + M_diag) / |V|) - diag(M_diag / |V|)
    # Hmm, let me redo:
    # Off-diagonal: dS[i]/d|V[k]| = M[i,k] / |V[k]|
    # Diagonal:     dS[i]/d|V[i]| = (S_diag[i] + M[i,i]) / |V[i]|
    #
    # = M * diag(1/|V|)  for off-diag
    # + diag(S_diag / |V|)  for the diagonal correction
    # Because M * diag(1/|V|) gives M[i,i]/|V[i]| on diagonal already,
    # and we need (S_diag[i] + M[i,i]) / |V[i]|, so correction is +S_diag[i]/|V[i]|

    vm_inv = 1.0 / vm_ns
    ds_dvm = m_mat @ sp_sparse.diags(vm_inv, format="csr") + sp_sparse.diags(
        s_diag * vm_inv, format="csr"
    )

    # Scale by s_scale (per non-slack node)
    s_scale_ns = s_scale_pu[non_slack]
    scale_inv = sp_sparse.diags(1.0 / s_scale_ns, format="csr")

    ds_dtheta_scaled = scale_inv @ ds_dtheta
    ds_dvm_scaled = scale_inv @ ds_dvm

    # Residual = [Re(mismatch), Im(mismatch), reg, regulator_terms]
    # Decision vars = [theta_nonslack, vm_nonslack]
    #
    # dRe(S)/dtheta = Re(dS/dtheta), dRe(S)/dVm = Re(dS/dVm)
    # dIm(S)/dtheta = Im(dS/dtheta), dIm(S)/dVm = Im(dS/dVm)

    j11 = ds_dtheta_scaled.real  # dP/dtheta: m × m
    j12 = ds_dvm_scaled.real  # dP/d|V|: m × m
    j21 = ds_dtheta_scaled.imag  # dQ/dtheta: m × m
    j22 = ds_dvm_scaled.imag  # dQ/d|V|: m × m

    # Convert to sparse if dense
    if not sp_sparse.issparse(j11):
        j11 = sp_sparse.csr_matrix(j11)
        j12 = sp_sparse.csr_matrix(j12)
        j21 = sp_sparse.csr_matrix(j21)
        j22 = sp_sparse.csr_matrix(j22)

    # Voltage regulation: d(reg)/dtheta = 0, d(reg)/d|V| = w_reg * I
    reg_block = voltage_reg_weight * sp_sparse.eye(m, format="csr")
    zeros_m = sp_sparse.csr_matrix((m, m))

    blocks = [[j11, j12], [j21, j22], [zeros_m, reg_block]]

    # Voltage target terms
    if voltage_targets_pu:
        reg_idx = [
            i
            for i in non_slack
            if labels[i] in voltage_targets_pu
            and float(voltage_targets_pu[labels[i]]) > 0
        ]
        if reg_idx:
            n_reg = len(reg_idx)
            # d(target_term)/d|V_k| = voltage_target_weight / target_pu[i]  if k matches
            # Map reg_idx (global) to non_slack local index
            ns_map = {g: loc for loc, g in enumerate(non_slack)}
            reg_local = [ns_map[g] for g in reg_idx]
            reg_data = [
                voltage_target_weight / float(voltage_targets_pu[labels[g]])
                for g in reg_idx
            ]
            jt_vm = sp_sparse.csr_matrix(
                (reg_data, (list(range(n_reg)), reg_local)),
                shape=(n_reg, m),
            )
            jt_theta = sp_sparse.csr_matrix((n_reg, m))
            blocks.append([jt_theta, jt_vm])

    jac = sp_sparse.bmat(blocks, format="csr")

    # For small problems, return dense Jacobian so least_squares uses the
    # 'exact' trust-region solver (dense QR) which handles ill-conditioning
    # better than LSMR.  For large problems, keep sparse for scalability.
    if 2 * m <= 2000:
        return jac.toarray()
    return jac


def _initialize_angles(
    system: DistributionSystem,
    labels: list[BusPhaseLabel],
    label_to_index: dict[BusPhaseLabel, int],
    n: int,
) -> np.ndarray:
    """Compute initial voltage angle array with balanced phase offsets and split-phase handling."""
    theta0 = np.zeros(n, dtype=float)

    _PHASE_ANGLE = {"B": -2.0 * math.pi / 3.0, "C": 2.0 * math.pi / 3.0}
    for idx, (bus, ph) in enumerate(labels):
        if ph in _PHASE_ANGLE:
            theta0[idx] = _PHASE_ANGLE[ph]

    # Center-tapped transformers: S1 is in-phase with the primary and S2 is
    # anti-phase.  Build a map from each secondary bus to its primary angle,
    # then propagate through the secondary line network.
    _s_bus_pri_angle: dict[str, float] = {}
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

    for idx, label in enumerate(labels):
        bus_name, ph = label
        if ph == "S1":
            theta0[idx] = _s_bus_pri_angle.get(bus_name, 0.0)
        elif ph == "S2":
            theta0[idx] = _s_bus_pri_angle.get(bus_name, 0.0) + math.pi

    return theta0


def _build_per_unit_ybus(
    ybus_si: np.ndarray,
    v_base: np.ndarray,
    s_base: float,
) -> np.ndarray:
    """Convert SI Y-bus to per-unit using element-wise scaling."""
    scale = np.outer(v_base, v_base) / s_base
    if hasattr(ybus_si, "multiply"):
        ybus_pu = ybus_si.multiply(scale)
        if hasattr(ybus_pu, "tocsr"):
            ybus_pu = ybus_pu.tocsr()
    else:
        ybus_pu = ybus_si * scale
    return ybus_pu


def _identify_slack_and_filter_connectivity(
    slack_label: BusPhaseLabel | list[BusPhaseLabel] | None,
    label_to_index: dict[BusPhaseLabel, int],
    labels: list[BusPhaseLabel],
    ybus_pu: np.ndarray,
    n: int,
) -> tuple[set[int], list[int]]:
    """Determine slack set and non-slack indices with connectivity filtering."""
    if slack_label is None:
        slack_set = {0}
    elif isinstance(slack_label, list):
        for sl in slack_label:
            if sl not in label_to_index:
                raise ValueError(f"Unknown slack label: {sl}")
        slack_set = {label_to_index[sl] for sl in slack_label}
    else:
        if slack_label not in label_to_index:
            raise ValueError(f"Unknown slack label: {slack_label}")
        slack_set = {label_to_index[slack_label]}

    # BFS from slack nodes over Y-bus adjacency to find reachable nodes.
    from collections import deque

    import scipy.sparse as _sp

    if _sp.issparse(ybus_pu):
        y_adj = (abs(ybus_pu) + abs(ybus_pu).T).tocsr()
    else:
        y_adj = _sp.csr_matrix(np.abs(ybus_pu) + np.abs(ybus_pu).T)
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

    unreachable = set(range(n)) - reachable
    slack_set = slack_set | unreachable
    non_slack = sorted(i for i in range(n) if i not in slack_set)
    return slack_set, non_slack


def _run_interior_point_nr(
    system: DistributionSystem,
    x_nr: np.ndarray,
    non_slack: list[int],
    n: int,
    slack_set: set[int],
    theta0: np.ndarray,
    ybus_pu: np.ndarray,
    s_spec_pu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    m_ns: int,
    include_neutral: bool,
    include_shunt: bool,
    convert_geometry_to_matrix: bool,
    p_spec_w: dict[BusPhaseLabel, float] | None,
    q_spec_var: dict[BusPhaseLabel, float] | None,
    labels: list[BusPhaseLabel],
    label_to_index: dict[BusPhaseLabel, int],
    v_base: np.ndarray,
) -> tuple[np.ndarray, bool, int, float]:
    """Run AC PF warm-start + interior-point NR solver. Returns (x0, converged, iters, max_mis)."""
    import scipy.sparse as _sp_nr
    from scipy.sparse.linalg import spsolve

    nr_max_iter = 50
    nr_tol = 1e-6
    nr_converged = False
    max_mis = float("inf")
    nr_iter = 0

    # Use AC PF as warm-start
    try:
        from .ac_pf import solve_ac_power_flow

        pf_result = solve_ac_power_flow(
            system,
            p_spec_w=p_spec_w,
            q_spec_var=q_spec_var,
            include_neutral=include_neutral,
            include_shunt=include_shunt,
            convert_geometry_to_matrix=convert_geometry_to_matrix,
        )
        if pf_result.success:
            for idx, label in enumerate(labels):
                if label in label_to_index:
                    gidx = label_to_index[label]
                    if gidx in non_slack:
                        local_idx = non_slack.index(gidx)
                        x_nr[m_ns + local_idx] = (
                            abs(pf_result.voltage[idx]) / v_base[gidx]
                        )
                        x_nr[local_idx] = float(np.angle(pf_result.voltage[idx]))
            # PF converged — check if bounds are satisfied
            vm_pu_pf = x_nr[m_ns:]
            if np.all(vm_pu_pf >= lb[m_ns:]) and np.all(vm_pu_pf <= ub[m_ns:]):
                nr_converged = True
                max_mis = pf_result.max_mismatch_pu
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("AC PF warm-start failed: %s", exc)

    # Interior-point barrier parameters for voltage bounds
    vm_lb = lb[m_ns:]
    vm_ub = ub[m_ns:]
    mu = 1e-3
    mu_min = 1e-8

    for nr_iter in range(nr_max_iter):
        v_pu = _build_voltage_from_state(x_nr, n, slack_set, theta0)
        s_calc = v_pu * np.conj(ybus_pu @ v_pu)
        mismatch = s_calc[non_slack] - s_spec_pu[non_slack]

        max_mis = max(np.max(np.abs(mismatch.real)), np.max(np.abs(mismatch.imag)))
        if max_mis < nr_tol:
            break

        i_bus = ybus_pu @ v_pu
        ns_arr = np.array(non_slack)
        y_ns = (
            ybus_pu[ns_arr, :][:, ns_arr]
            if _sp_nr.issparse(ybus_pu)
            else _sp_nr.csr_matrix(ybus_pu[np.ix_(ns_arr, ns_arr)])
        )
        v_ns = v_pu[non_slack]
        vm_ns = np.abs(v_ns)
        i_ns = i_bus[non_slack]
        v_diag = _sp_nr.diags(v_ns, format="csr")
        vc_diag = _sp_nr.diags(np.conj(v_ns), format="csr")
        m_mat = v_diag @ y_ns.conjugate() @ vc_diag
        s_diag = v_ns * np.conj(i_ns)

        ds_dtheta = -1j * m_mat + _sp_nr.diags(1j * s_diag, format="csr")
        vm_inv = 1.0 / vm_ns
        ds_dvm = m_mat @ _sp_nr.diags(vm_inv, format="csr") + _sp_nr.diags(
            s_diag * vm_inv, format="csr"
        )

        vm_current = x_nr[m_ns:]
        dist_lb = np.maximum(vm_current - vm_lb, 1e-10)
        dist_ub = np.maximum(vm_ub - vm_current, 1e-10)
        barrier_grad = mu * (1.0 / dist_ub - 1.0 / dist_lb)
        barrier_hess_diag = mu * (1.0 / dist_ub**2 + 1.0 / dist_lb**2)

        j_nr = _sp_nr.bmat(
            [
                [ds_dtheta.real, ds_dvm.real],
                [ds_dtheta.imag, ds_dvm.imag],
            ],
            format="csc",
        )
        barrier_diag_full = np.zeros(2 * m_ns)
        barrier_diag_full[m_ns:] = barrier_hess_diag
        j_nr = j_nr + _sp_nr.diags(barrier_diag_full, format="csc")

        rhs = np.concatenate([mismatch.real, mismatch.imag])
        rhs[m_ns:] += barrier_grad

        dx = spsolve(j_nr, -rhs)

        # Fraction-to-boundary step limiting
        alpha = 1.0
        tau = 0.995
        vm_step = dx[m_ns:]
        for i in range(m_ns):
            if vm_step[i] < 0:
                max_step = tau * dist_lb[i] / (-vm_step[i])
                alpha = min(alpha, max_step)
            elif vm_step[i] > 0:
                max_step = tau * dist_ub[i] / vm_step[i]
                alpha = min(alpha, max_step)

        # Damped line-search within the feasible step
        for _bt in range(10):
            x_trial = x_nr + alpha * dx
            x_trial[m_ns:] = np.maximum(x_trial[m_ns:], vm_lb + 1e-10)
            x_trial[m_ns:] = np.minimum(x_trial[m_ns:], vm_ub - 1e-10)
            v_trial = _build_voltage_from_state(x_trial, n, slack_set, theta0)
            s_trial = v_trial * np.conj(ybus_pu @ v_trial)
            mis_trial = s_trial[non_slack] - s_spec_pu[non_slack]
            new_mis = max(
                float(np.max(np.abs(mis_trial.real))),
                float(np.max(np.abs(mis_trial.imag))),
            )
            if new_mis < max_mis:
                break
            alpha *= 0.5
        x_nr = x_nr + alpha * dx
        x_nr[m_ns:] = np.maximum(x_nr[m_ns:], vm_lb + 1e-10)
        x_nr[m_ns:] = np.minimum(x_nr[m_ns:], vm_ub - 1e-10)

        mu = max(mu * 0.5, mu_min)

    x0 = np.clip(x_nr, lb, ub)
    nr_converged = max_mis < nr_tol
    return x0, nr_converged, nr_iter, max_mis


def optimize_ac_power_flow(
    system: DistributionSystem,
    *,
    p_spec_w: dict[BusPhaseLabel, float] | None = None,
    q_spec_var: dict[BusPhaseLabel, float] | None = None,
    voltage_targets_v: dict[BusPhaseLabel, float] | None = None,
    voltage_limits_v: dict[BusPhaseLabel, tuple[float, float]] | None = None,
    slack_label: BusPhaseLabel | list[BusPhaseLabel] | None = None,
    include_neutral: bool = False,
    include_shunt: bool = False,
    convert_geometry_to_matrix: bool = True,
    vm_min_pu: float = 0.95,
    vm_max_pu: float = 1.05,
    voltage_reg_weight: float = 1e-3,
    voltage_target_weight: float = 1.0,
    mismatch_scale_floor_w: float = 1e3,
    max_nfev: int = 1000,
    v0_pu: dict[BusPhaseLabel, float] | None = None,
    theta0_rad: dict[BusPhaseLabel, float] | None = None,
) -> PowerFlowOptimizationResult:
    """Solve a Y-bus-based AC nodal optimization using nonlinear least squares.

    The objective minimizes active/reactive power mismatch at non-slack nodes:
    `S(V) - S_spec`, where `S(V) = V * conj(Ybus @ V)`.

    Parameters
    ----------
    system : DistributionSystem
        Input distribution system.
    p_spec_w, q_spec_var : dict[(bus_name, phase), float], optional
        Net active/reactive power injection targets in SI units. Positive means
        generation/injection, negative means consumption.
    slack_label : (bus_name, phase), optional
        Slack node held fixed at nominal magnitude and zero angle. If omitted,
        the first Y-bus node is used.
    include_neutral, include_shunt, convert_geometry_to_matrix : bool, optional
        Passed into Y-bus construction.
    vm_min_pu, vm_max_pu : float, optional
        Voltage magnitude bounds relative to each node nominal voltage.
    voltage_reg_weight : float, optional
        Regularization weight that nudges voltages near nominal values.
    voltage_target_weight : float, optional
        Weight for regulator-derived voltage target soft constraints.
    mismatch_scale_floor_w : float, optional
        Lower bound used to normalize complex power mismatch residuals. This
        improves solver conditioning on large SI-valued systems.
    voltage_limits_v : dict[(bus_name, phase), (vmin, vmax)], optional
        Hard node voltage limits (volts) that intersect global pu bounds.
    max_nfev : int, optional
        Maximum number of objective evaluations.
    v0_pu : dict[(bus_name, phase), float], optional
        Initial per-unit voltage magnitudes for warm-starting. When provided,
        overrides the flat start for matching nodes.
    theta0_rad : dict[(bus_name, phase), float], optional
        Initial voltage angles in radians for warm-starting. When provided,
        overrides the flat start for matching nodes.

    Returns
    -------
    PowerFlowOptimizationResult
        Optimized voltages, solved power injections, and solver diagnostics.
    """

    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "SciPy is required for optimization. Install with `pip install gdm-flow[optimization]`."
        ) from exc

    # --- Y-bus construction and nominal voltages ---
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
        raise ValueError("At least two bus-phase nodes are required for optimization.")

    nominal_map = _build_nominal_voltage_map(system)
    v_base = np.array([nominal_map[label] for label in labels], dtype=float)

    # --- Phase angle initialization ---
    theta0 = _initialize_angles(system, labels, label_to_index, n)

    # --- Warm-start override from previous solve ---
    vm0_pu = np.ones(n, dtype=float)
    if v0_pu is not None:
        for label, vm_val in v0_pu.items():
            if label in label_to_index:
                vm0_pu[label_to_index[label]] = vm_val
    if theta0_rad is not None:
        for label, th_val in theta0_rad.items():
            if label in label_to_index:
                theta0[label_to_index[label]] = th_val

    # --- Per-unit system ---
    s_spec = _build_spec_vector(labels, p_spec_w, q_spec_var)
    s_base = max(
        float(np.max(np.abs(s_spec)))
        if np.any(s_spec != 0)
        else float(mismatch_scale_floor_w),
        float(mismatch_scale_floor_w),
    )

    ybus_pu = _build_per_unit_ybus(ybus_si, v_base, s_base)
    s_spec_pu = s_spec / s_base
    s_scale_pu = np.maximum(np.abs(s_spec_pu), 1e-3)

    voltage_targets_pu = None
    if voltage_targets_v:
        voltage_targets_pu = {}
        for label, v_target in voltage_targets_v.items():
            if label in label_to_index:
                idx = label_to_index[label]
                voltage_targets_pu[label] = float(v_target) / v_base[idx]

    # --- Slack identification and connectivity BFS ---
    slack_set, non_slack = _identify_slack_and_filter_connectivity(
        slack_label, label_to_index, labels, ybus_pu, n
    )

    # --- Decision variable bounds setup ---
    x0 = np.concatenate([theta0[non_slack], vm0_pu[non_slack]])
    lb = np.concatenate(
        [
            np.full(len(non_slack), -math.pi),
            np.full(len(non_slack), vm_min_pu),
        ]
    )
    ub = np.concatenate(
        [
            np.full(len(non_slack), math.pi),
            np.full(len(non_slack), vm_max_pu),
        ]
    )

    if voltage_limits_v:
        for local_idx, global_idx in enumerate(non_slack):
            label = labels[global_idx]
            if label not in voltage_limits_v:
                continue
            v_min, v_max = voltage_limits_v[label]
            mag_pos = len(non_slack) + local_idx
            lb[mag_pos] = max(lb[mag_pos], float(v_min) / v_base[global_idx])
            ub[mag_pos] = min(ub[mag_pos], float(v_max) / v_base[global_idx])
            if lb[mag_pos] > ub[mag_pos]:
                raise ValueError(
                    f"Infeasible voltage bounds for {label}: lower={lb[mag_pos]}, upper={ub[mag_pos]}"
                )

    x0 = np.clip(x0, lb, ub)

    # --- Newton-Raphson warm start for large problems ---
    non_slack = [i for i in range(n) if i not in slack_set]
    use_nr_warmstart = len(non_slack) > 6
    nr_converged = False
    nr_max_iter = 50
    nr_iter = 0
    max_mis = float("inf")

    if use_nr_warmstart:
        m_ns = len(non_slack)
        x_nr = np.concatenate([theta0[non_slack], np.ones(m_ns, dtype=float)])
        x0, nr_converged, nr_iter, max_mis = _run_interior_point_nr(
            system,
            x_nr,
            non_slack,
            n,
            slack_set,
            theta0,
            ybus_pu,
            s_spec_pu,
            lb,
            ub,
            m_ns,
            include_neutral,
            include_shunt,
            convert_geometry_to_matrix,
            p_spec_w,
            q_spec_var,
            labels,
            label_to_index,
            v_base,
        )

    # --- Build residual args and compute initial objective ---
    args = (
        ybus_pu,
        s_spec_pu,
        s_scale_pu,
        n,
        slack_set,
        theta0,
        voltage_reg_weight,
        voltage_targets_pu,
        labels,
        voltage_target_weight,
    )

    residual0 = _objective_residual(x0, *args)
    initial_objective = float(np.dot(residual0, residual0))

    # --- Return result or fall through to least_squares ---
    if use_nr_warmstart and nr_converged:
        v_pu_opt = _build_voltage_from_state(x0, n, slack_set, theta0)
        v_si_opt = v_pu_opt * v_base
        s_si_opt = v_si_opt * np.conj(ybus_si @ v_si_opt)
        final_residual = _objective_residual(x0, *args)
        return PowerFlowOptimizationResult(
            success=True,
            message=f"Newton-Raphson converged in {nr_iter + 1} iterations",
            ybus_result=ybus_result,
            voltage=v_si_opt,
            power_injection=s_si_opt,
            iterations=nr_iter + 1,
            initial_objective=initial_objective,
            final_objective=float(np.dot(final_residual, final_residual)),
        )

    if use_nr_warmstart and not nr_converged:
        v_pu_opt = _build_voltage_from_state(x0, n, slack_set, theta0)
        v_si_opt = v_pu_opt * v_base
        s_si_opt = v_si_opt * np.conj(ybus_si @ v_si_opt)
        final_residual = _objective_residual(x0, *args)
        return PowerFlowOptimizationResult(
            success=False,
            message=f"Newton-Raphson did not converge in {nr_max_iter} iterations (max mismatch {max_mis:.4e} pu)",
            ybus_result=ybus_result,
            voltage=v_si_opt,
            power_injection=s_si_opt,
            iterations=nr_max_iter,
            initial_objective=initial_objective,
            final_objective=float(np.dot(final_residual, final_residual)),
        )

    # --- Least-squares fallback for small problems ---
    non_slack = [i for i in range(n) if i not in slack_set]
    sparse_jac = 2 * len(non_slack) > 2000
    kwargs: dict = {"method": "trf", "max_nfev": max_nfev}
    if sparse_jac:
        kwargs["x_scale"] = "jac"
        kwargs["tr_solver"] = "lsmr"
        kwargs["tr_options"] = {"atol": 1e-14, "btol": 1e-14}
        kwargs["xtol"] = 1e-12
        kwargs["ftol"] = 1e-12
        kwargs["gtol"] = 1e-12

    result = least_squares(
        _objective_residual,
        x0=x0,
        jac=_objective_jacobian,
        bounds=(lb, ub),
        args=args,
        **kwargs,
    )

    v_pu_opt = _build_voltage_from_state(result.x, n, slack_set, theta0)
    v_si_opt = v_pu_opt * v_base
    s_si_opt = v_si_opt * np.conj(ybus_si @ v_si_opt)
    final_residual = _objective_residual(result.x, *args)

    return PowerFlowOptimizationResult(
        success=bool(result.success),
        message=str(result.message),
        ybus_result=ybus_result,
        voltage=v_si_opt,
        power_injection=s_si_opt,
        iterations=int(result.nfev),
        initial_objective=initial_objective,
        final_objective=float(np.dot(final_residual, final_residual)),
    )


def optimize_ac_power_flow_from_components(
    system: DistributionSystem,
    *,
    include_loads: bool = True,
    include_solar: bool = True,
    include_battery: bool = False,
    include_capacitor: bool = True,
    include_regulator_targets: bool = True,
    include_regulator_limits: bool = True,
    load_scale: float = 1.0,
    solar_scale: float = 1.0,
    battery_scale: float = 1.0,
    capacitor_scale: float = 1.0,
    slack_label: BusPhaseLabel | list[BusPhaseLabel] | None = None,
    include_neutral: bool = False,
    include_shunt: bool = False,
    convert_geometry_to_matrix: bool = True,
    vm_min_pu: float = 0.95,
    vm_max_pu: float = 1.05,
    voltage_reg_weight: float = 1e-3,
    voltage_target_weight: float = 1.0,
    mismatch_scale_floor_w: float = 1e3,
    max_nfev: int = 1000,
) -> PowerFlowOptimizationResult:
    """Run AC optimization using nodal specs auto-derived from system components."""

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
    )

    voltage_targets_v = (
        build_regulator_voltage_targets_from_components(system)
        if include_regulator_targets
        else None
    )
    voltage_limits_v = (
        build_regulator_voltage_limits_from_components(system)
        if include_regulator_limits
        else None
    )

    # Auto-detect source bus for slack when not explicitly provided.
    # All non-neutral phases of the source bus become slack nodes.
    if slack_label is None:
        try:
            source_bus = system.get_source_bus()
            source_phases = [_phase_name(p) for p in source_bus.phases if p != Phase.N]
            if source_phases:
                slack_label = [(source_bus.name, p) for p in source_phases]
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).debug("Source bus lookup failed: %s", exc)

    return optimize_ac_power_flow(
        system,
        p_spec_w=p_spec_w,
        q_spec_var=q_spec_var,
        voltage_targets_v=voltage_targets_v,
        voltage_limits_v=voltage_limits_v,
        slack_label=slack_label,
        include_neutral=include_neutral,
        include_shunt=include_shunt,
        convert_geometry_to_matrix=convert_geometry_to_matrix,
        vm_min_pu=vm_min_pu,
        vm_max_pu=vm_max_pu,
        voltage_reg_weight=voltage_reg_weight,
        voltage_target_weight=voltage_target_weight,
        mismatch_scale_floor_w=mismatch_scale_floor_w,
        max_nfev=max_nfev,
    )
