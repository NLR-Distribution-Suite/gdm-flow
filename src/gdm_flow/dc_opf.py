"""DC OPF routines built on top of the Y-bus representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import numpy as np

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBattery,
    DistributionBus,
    DistributionLoad,
    DistributionSolar,
)
from gdm.systems.distribution.enums import Phase

from ._utils import _phase_name, _phase_voltage
from .ybus import YBusResult, calculate_ybus

BusPhaseLabel = Tuple[str, str]


@dataclass(frozen=True)
class DCGenerator:
    """Dispatchable generator model used by the DC OPF solver."""

    name: str
    node: BusPhaseLabel
    p_min_w: float
    p_max_w: float
    cost_quadratic: float = 0.0
    cost_linear: float = 1.0
    cost_constant: float = 0.0


@dataclass(frozen=True)
class DCOPFResult:
    """Result payload for DC OPF."""

    success: bool
    message: str
    objective: float
    iterations: int
    generator_dispatch_w: Dict[str, float]
    theta_rad: Dict[BusPhaseLabel, float]
    nodal_balance_w: Dict[BusPhaseLabel, float]
    slack_injection_w: float
    ybus_result: YBusResult


def build_dc_load_profile_from_components(
    system: DistributionSystem,
    *,
    include_loads: bool = True,
    include_solar_as_negative_load: bool = False,
    include_battery_as_negative_load: bool = False,
    load_scale: float = 1.0,
    solar_scale: float = 1.0,
    battery_scale: float = 1.0,
) -> dict[BusPhaseLabel, float]:
    """Build DC demand profile in watts from component models.

    Returns positive values for demand and negative values for fixed injections.
    """

    demand: dict[BusPhaseLabel, float] = {}

    if include_loads:
        for load in system.get_components(DistributionLoad):
            if not load.in_service:
                continue
            for phase, phase_load in zip(load.phases, load.equipment.phase_loads):
                label = (load.bus.name, _phase_name(phase))
                demand[label] = (
                    demand.get(label, 0.0)
                    + float(phase_load.real_power.to("watt").magnitude) * load_scale
                )

    if include_solar_as_negative_load:
        for solar in system.get_components(DistributionSolar):
            if not solar.in_service or not solar.phases:
                continue
            p_each = (
                float(solar.active_power.to("watt").magnitude)
                * solar_scale
                / len(solar.phases)
            )
            for phase in solar.phases:
                label = (solar.bus.name, _phase_name(phase))
                demand[label] = demand.get(label, 0.0) - p_each

    if include_battery_as_negative_load:
        for battery in system.get_components(DistributionBattery):
            if not battery.in_service or not battery.phases:
                continue
            p_each = (
                float(battery.active_power.to("watt").magnitude)
                * battery_scale
                / len(battery.phases)
            )
            for phase in battery.phases:
                label = (battery.bus.name, _phase_name(phase))
                demand[label] = demand.get(label, 0.0) - p_each

    return demand


def build_dc_generators_from_components(
    system: DistributionSystem,
    *,
    include_solar: bool = True,
    include_battery: bool = True,
    solar_cost_linear: float = 5.0,
    battery_cost_linear: float = 15.0,
) -> list[DCGenerator]:
    """Build dispatchable generator models from solar and battery components."""

    generators: list[DCGenerator] = []

    if include_solar:
        for solar in system.get_components(DistributionSolar):
            if not solar.in_service or not solar.phases:
                continue
            p_max_each = float(solar.active_power.to("watt").magnitude) / len(
                solar.phases
            )
            for phase in solar.phases:
                generators.append(
                    DCGenerator(
                        name=f"solar:{solar.name}:{_phase_name(phase)}",
                        node=(solar.bus.name, _phase_name(phase)),
                        p_min_w=0.0,
                        p_max_w=max(0.0, p_max_each),
                        cost_linear=solar_cost_linear,
                    )
                )

    if include_battery:
        for battery in system.get_components(DistributionBattery):
            if not battery.in_service or not battery.phases:
                continue
            p_max_each = float(battery.active_power.to("watt").magnitude) / len(
                battery.phases
            )
            for phase in battery.phases:
                generators.append(
                    DCGenerator(
                        name=f"battery:{battery.name}:{_phase_name(phase)}",
                        node=(battery.bus.name, _phase_name(phase)),
                        p_min_w=0.0,
                        p_max_w=max(0.0, p_max_each),
                        cost_linear=battery_cost_linear,
                    )
                )

    return generators


def solve_dc_opf(
    system: DistributionSystem,
    *,
    generators: List[DCGenerator],
    demand_w: Dict[BusPhaseLabel, float] | None = None,
    slack_label: BusPhaseLabel | List[BusPhaseLabel] | None = None,
    include_neutral: bool = False,
    include_shunt: bool = False,
    convert_geometry_to_matrix: bool = True,
    theta_min_rad: float = -math.pi,
    theta_max_rad: float = math.pi,
    theta_penalty: float = 1e-6,
    generation_regularization: float = 1e-9,
    maxiter: int = 500,
    theta0_rad: Dict[BusPhaseLabel, float] | None = None,
) -> DCOPFResult:
    """Solve a DC OPF with quadratic generation cost and linearized nodal constraints."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, minimize
        import scipy.sparse as sp_sparse
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "SciPy is required for DC OPF. Install with `pip install gdm-flow[optimization]`."
        ) from exc

    if not generators:
        raise ValueError("At least one generator is required for DC OPF.")

    ybus_result = calculate_ybus(
        system,
        include_neutral=include_neutral,
        include_shunt=include_shunt,
        convert_geometry_to_matrix=convert_geometry_to_matrix,
        sparse=True,
    )
    labels = ybus_result.index_to_label
    label_to_index = ybus_result.label_to_index
    n = len(labels)

    if n < 2:
        raise ValueError("At least two nodes are required for DC OPF.")

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

    ybus = ybus_result.ybus
    b_raw = -np.imag(ybus)

    # --- Exclude split-phase (S1/S2) and neutral (N) nodes from DC OPF ---
    # DC power flow is a positive-sequence single-phase approximation that
    # assumes sin(θ_ij) ≈ θ_ij.  This linearization breaks down across
    # center-tapped transformers with 60:1 turns ratios, making the B-matrix
    # extremely ill-conditioned (1B:1 ratio).  Split-phase loads are
    # aggregated to their parent primary bus so the DC OPF sees only the
    # three-phase primary network.
    _dc_phases = {"A", "B", "C"}
    dc_nodes = set(i for i, label in enumerate(labels) if label[1] in _dc_phases)
    excluded_nodes = set(range(n)) - dc_nodes

    # Aggregate excluded-node demand onto the nearest primary bus via the
    # transformer turns ratio (so the primary sees the equivalent load).
    # For simplicity, just drop S1/S2/N demand — it's a small fraction of
    # total and the DC OPF is an approximation anyway.
    demand_vec = np.array(
        [float((demand_w or {}).get(label, 0.0)) for label in labels], dtype=float
    )
    demand_vec[list(excluded_nodes)] = 0.0

    # Mask B-matrix to DC nodes only
    dc_arr = sorted(dc_nodes)

    # Scale the B matrix by nominal bus voltages so that B_eff * theta gives
    # power in watts regardless of voltage-base differences across
    # transformers: P[i] = sum_j(B_raw[i,j] * V_nom[i] * V_nom[j] * theta[j]).
    v_nom = np.ones(n, dtype=float)
    for bus in system.get_components(DistributionBus):
        v_phase = _phase_voltage(bus.rated_voltage, bus.voltage_type)
        for phase in bus.phases:
            pn = _phase_name(phase)
            label = (bus.name, pn)
            if label in label_to_index:
                v_nom[label_to_index[label]] = v_phase
    # Center-tapped transformer correction: S2 winding nodes sit at
    # θ ≈ π (antiphase to primary).  The standard DC linearization
    # sin(θ_ij) ≈ θ_ij breaks down for θ_ij ≈ π.  Linearizing around
    # the true operating point: sin(θ̃ + Δφ) ≈ sin(Δφ) + cos(Δφ)·θ̃.
    # For Δφ ∈ {0, ±π}, sin(Δφ)=0 so the constant vanishes and the
    # linear coefficient picks up cos(Δφ) = ±1.  Negating v_nom for
    # S2 nodes absorbs this sign flip into the B-matrix scaling.
    phase_offset = np.zeros(n, dtype=float)
    for i, label in enumerate(labels):
        if label[1] == "S2":
            v_nom[i] = -v_nom[i]
            phase_offset[i] = math.pi

    # Scale B by (sign-adjusted) nominal voltages:
    # B_eff = diag(v_nom) @ B_raw @ diag(v_nom).
    v_diag = sp_sparse.diags(v_nom, format="csr")
    if sp_sparse.issparse(b_raw):
        b_bus = (v_diag @ b_raw @ v_diag).tocsr()
    else:
        b_bus = v_diag @ b_raw @ v_diag

    # Restrict B-bus to DC nodes only (A/B/C phases)
    b_bus_dc = b_bus[np.ix_(dc_arr, dc_arr)].tocsr()

    gen_node_idx: list[int] = []
    for gen in generators:
        if gen.node not in label_to_index:
            raise ValueError(f"Generator {gen.name} refers to unknown node {gen.node}")
        gen_node_idx.append(label_to_index[gen.node])

    # Nodes with explicit generators are constrained in power balance even
    # if they are in the slack set.  The slack set only fixes theta = 0
    # (angle reference); power is supplied through explicit generators.
    gen_nodes = set(gen_node_idx)

    # --- Connectivity filter (on DC nodes only) ---
    from collections import deque

    dc_idx_map = {idx: local_i for local_i, idx in enumerate(dc_arr)}
    b_adj = (abs(b_bus_dc) + abs(b_bus_dc).T).tocsr()
    reachable_local = set()
    bfs_queue: deque[int] = deque()
    # Map gen/slack nodes to DC-local indices
    for seed in gen_nodes | slack_set:
        if seed in dc_idx_map:
            local = dc_idx_map[seed]
            if local not in reachable_local:
                reachable_local.add(local)
                bfs_queue.append(local)
    while bfs_queue:
        node = bfs_queue.popleft()
        for neighbor in b_adj[node].indices:
            if neighbor not in reachable_local:
                reachable_local.add(neighbor)
                bfs_queue.append(neighbor)

    # Map back to global indices
    reachable = set(dc_arr[local] for local in reachable_local)

    # Constrain: all reachable non-slack nodes PLUS any slack node with a generator.
    constraint_idx = sorted(
        i for i in reachable if i not in slack_set or i in gen_nodes
    )
    # Angle variables: all reachable non-slack nodes (slack keeps theta=0).
    theta_var_idx = sorted(i for i in reachable if i not in slack_set)

    num_gen = len(generators)
    num_theta = len(theta_var_idx)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pg = x[:num_gen]
        theta_active = x[num_gen:]
        theta = np.zeros(n, dtype=float)
        theta[theta_var_idx] = theta_active
        return pg, theta

    def objective(x: np.ndarray) -> float:
        pg, theta = unpack(x)
        gen_cost = np.sum(
            [
                (generators[k].cost_quadratic + generation_regularization)
                * pg[k]
                * pg[k]
                + generators[k].cost_linear * pg[k]
                + generators[k].cost_constant
                for k in range(num_gen)
            ]
        )
        # Tiny angle regularization removes null-space singularities in SLSQP.
        return float(gen_cost + theta_penalty * float(theta @ theta))

    def objective_jac(x: np.ndarray) -> np.ndarray:
        pg, theta = unpack(x)
        grad_pg = np.array(
            [
                2.0 * (generators[k].cost_quadratic + generation_regularization) * pg[k]
                + generators[k].cost_linear
                for k in range(num_gen)
            ],
            dtype=float,
        )
        grad_theta = 2.0 * theta_penalty * theta[theta_var_idx]
        return np.concatenate([grad_pg, grad_theta])

    _hess_diag = np.empty(num_gen + num_theta, dtype=float)
    for k in range(num_gen):
        _hess_diag[k] = 2.0 * (generators[k].cost_quadratic + generation_regularization)
    _hess_diag[num_gen:] = 2.0 * theta_penalty
    _hess_sparse = sp_sparse.diags(_hess_diag, format="csc")

    def objective_hess(_x: np.ndarray):
        return _hess_sparse

    def power_balance_non_slack(x: np.ndarray) -> np.ndarray:
        pg, theta = unpack(x)
        gen_inj = np.zeros(n, dtype=float)
        for k in range(num_gen):
            gen_inj[gen_node_idx[k]] += pg[k]

        # DC nodal equation: p_gen - p_demand - B * theta = 0.
        # Use DC-restricted B-matrix with local indexing.
        theta_full = np.zeros(len(dc_arr), dtype=float)
        for local_i, global_i in enumerate(theta_var_idx):
            theta_full[dc_idx_map[global_i]] = (
                theta[theta_var_idx.index(global_i)] if global_i in theta_var_idx else 0
            )
        # Simpler: just compute on the full vector and index
        residual = gen_inj - demand_vec
        # b_bus @ theta for constraint nodes using DC subblock
        theta_dc = np.zeros(len(dc_arr), dtype=float)
        theta_map = {
            global_i: local_i for local_i, global_i in enumerate(theta_var_idx)
        }
        for gi, li in theta_map.items():
            theta_dc[dc_idx_map[gi]] = theta[gi]
        b_flow = b_bus_dc @ theta_dc
        for idx in constraint_idx:
            residual[idx] -= b_flow[dc_idx_map[idx]]
        return residual[constraint_idx]

    x0 = np.array(
        [0.5 * (gen.p_min_w + gen.p_max_w) for gen in generators] + [0.0] * num_theta,
        dtype=float,
    )

    # --- Warm-start from previous solve (QSTS) ---
    if theta0_rad is not None:
        for local_i, global_i in enumerate(theta_var_idx):
            label = labels[global_i]
            if label in theta0_rad:
                x0[num_gen + local_i] = theta0_rad[label]

    lb = np.array(
        [gen.p_min_w for gen in generators] + [theta_min_rad] * num_theta, dtype=float
    )
    ub = np.array(
        [gen.p_max_w for gen in generators] + [theta_max_rad] * num_theta, dtype=float
    )
    # Linear equality: A_eq @ x = b_eq where x = [pg, theta_active]
    # Build as a sparse matrix to avoid O(n^2) element-by-element filling.
    constraint_idx_map = {idx: row for row, idx in enumerate(constraint_idx)}
    gen_rows, gen_cols, gen_vals = [], [], []
    for k, gen_idx in enumerate(gen_node_idx):
        if gen_idx in constraint_idx_map:
            gen_rows.append(constraint_idx_map[gen_idx])
            gen_cols.append(k)
            gen_vals.append(1.0)
    gen_block = sp_sparse.csr_matrix(
        (gen_vals, (gen_rows, gen_cols)),
        shape=(len(constraint_idx), num_gen),
    )

    # Extract the B-matrix subblock via efficient sparse row/col slicing.
    # Use the DC-restricted B-matrix with local indices.
    c_local = np.array([dc_idx_map[idx] for idx in constraint_idx])
    t_local = np.array([dc_idx_map[idx] for idx in theta_var_idx])
    if sp_sparse.issparse(b_bus_dc):
        b_block = -b_bus_dc[c_local, :][:, t_local]
    else:
        b_block = sp_sparse.csr_matrix(-b_bus_dc[np.ix_(c_local, t_local)])

    a_eq = sp_sparse.hstack([gen_block, b_block], format="csc")
    b_eq = demand_vec[constraint_idx]

    # --- Direct LP via HiGHS ---
    # With the connectivity filter above, the constraint system should be
    # feasible (all constrained nodes are reachable from generators).
    from scipy.optimize import linprog

    c_lp = np.zeros(num_gen + num_theta, dtype=float)
    for k in range(num_gen):
        c_lp[k] = generators[k].cost_linear

    bounds_seq = list(zip(lb.tolist(), ub.tolist()))

    lp_result = linprog(c_lp, A_eq=a_eq, b_eq=b_eq, bounds=bounds_seq, method="highs")

    if lp_result.success:
        pg_opt, theta_opt = unpack(lp_result.x)
        result_success = True
        result_message = "LP solve (HiGHS) converged"
        result_nit = int(lp_result.nit)
        result_obj = float(objective(lp_result.x))
    else:
        # Fallback: trust-constr for unusual failure modes.
        result = minimize(
            objective,
            x0=x0,
            method="trust-constr",
            jac=objective_jac,
            hess=objective_hess,
            bounds=Bounds(lb, ub),
            constraints=[LinearConstraint(a_eq, b_eq, b_eq)],
            options={"maxiter": maxiter, "gtol": 1e-8, "xtol": 1e-10, "verbose": 0},
        )
        pg_opt, theta_opt = unpack(result.x)
        result_success = bool(result.success)
        result_message = str(result.message)
        result_nit = int(result.nit)
        result_obj = float(objective(result.x))

    gen_inj = np.zeros(n, dtype=float)
    for k in range(num_gen):
        gen_inj[gen_node_idx[k]] += pg_opt[k]
    nodal_balance = gen_inj - demand_vec - b_bus @ theta_opt

    # Convert θ̃ (deviation from offset) back to physical angles.
    theta_physical = theta_opt + phase_offset
    theta_map = {labels[i]: float(theta_physical[i]) for i in range(n)}
    nodal_balance_map = {labels[i]: float(nodal_balance[i]) for i in range(n)}

    # Positive slack_injection_w means total source injection at slack nodes.
    slack_injection_w = float(-sum(nodal_balance[i] for i in slack_set))

    dispatch = {generators[k].name: float(pg_opt[k]) for k in range(num_gen)}

    return DCOPFResult(
        success=result_success,
        message=result_message,
        objective=result_obj,
        iterations=result_nit,
        generator_dispatch_w=dispatch,
        theta_rad=theta_map,
        nodal_balance_w=nodal_balance_map,
        slack_injection_w=slack_injection_w,
        ybus_result=ybus_result,
    )


def solve_dc_opf_from_components(
    system: DistributionSystem,
    *,
    include_solar_generators: bool = True,
    include_battery_generators: bool = True,
    include_loads: bool = True,
    include_slack_generator: bool = True,
    slack_label: BusPhaseLabel | List[BusPhaseLabel] | None = None,
    slack_cost_linear: float = 50.0,
    include_neutral: bool = False,
    include_shunt: bool = False,
    convert_geometry_to_matrix: bool = True,
    theta_min_rad: float = -math.pi / 2,
    theta_max_rad: float = math.pi / 2,
    theta_penalty: float = 1e-6,
    maxiter: int = 500,
) -> DCOPFResult:
    """Convenience wrapper that builds DC OPF inputs from DistributionSystem components.

    When *include_slack_generator* is True (default), an explicit generator is
    placed at each slack bus phase so that grid imports carry a cost in the
    objective.  Without this, the slack bus can inject unlimited free power and
    the optimizer will not dispatch DERs.
    """

    generators = build_dc_generators_from_components(
        system,
        include_solar=include_solar_generators,
        include_battery=include_battery_generators,
    )
    demand = build_dc_load_profile_from_components(system, include_loads=include_loads)

    # Auto-detect source bus if no explicit slack provided.
    if slack_label is None:
        try:
            source_bus = system.get_source_bus()
            source_phases = [_phase_name(p) for p in source_bus.phases if p != Phase.N]
            if source_phases:
                slack_label = [(source_bus.name, p) for p in source_phases]
        except Exception:
            pass

    # Add explicit grid-import generators at the slack bus so the optimizer
    # properly weighs DER dispatch against grid import cost.
    if include_slack_generator and slack_label is not None:
        slack_labels = slack_label if isinstance(slack_label, list) else [slack_label]
        total_demand = sum(demand.values()) if demand else 0.0
        p_max_slack = max(total_demand * 3.0, 1e6)  # generous upper bound
        for sl in slack_labels:
            generators.append(
                DCGenerator(
                    name=f"grid:{sl[0]}:{sl[1]}",
                    node=sl,
                    p_min_w=0.0,
                    p_max_w=p_max_slack,
                    cost_linear=slack_cost_linear,
                )
            )

    return solve_dc_opf(
        system,
        generators=generators,
        demand_w=demand,
        slack_label=slack_label,
        include_neutral=include_neutral,
        include_shunt=include_shunt,
        convert_geometry_to_matrix=convert_geometry_to_matrix,
        theta_min_rad=theta_min_rad,
        theta_max_rad=theta_max_rad,
        theta_penalty=theta_penalty,
        maxiter=maxiter,
    )
