"""Multi-period OPF formulations with inter-temporal coupling.

Provides joint optimization over a time horizon where battery SOC,
generator ramp constraints, and demand profiles are coupled across
timesteps — unlike QSTS which solves each timestep independently.

Two formulations:
- **Multi-period DC OPF**: LP via HiGHS with SOC coupling + ramp limits.
- **Multi-period LinDistFlow**: LP with voltage-drop approximation + SOC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

import numpy as np

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBattery,
    DistributionBus,
)

from ._utils import _phase_name, _phase_voltage
from .dc_opf import BusPhaseLabel, DCGenerator
from .time_series import (
    build_dc_load_profile_at_timestep,
    build_lindistflow_injections_at_timestep,
    get_time_series_resolution,
)
from .ybus import calculate_ybus


# ── Result types ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MultiPeriodResult:
    """Result from a multi-period optimization."""

    success: bool
    message: str
    solver: str
    num_timesteps: int
    objective: float
    generator_dispatch_w: dict[int, dict[str, float]]
    """timestep → {generator_name → dispatch_w}"""
    battery_soc: dict[str, list[float]]
    """battery_name → SOC values at each timestep"""
    nodal_voltage: dict[int, dict[BusPhaseLabel, float]] | None
    """timestep → {(bus,phase) → voltage} (LinDistFlow only)"""
    theta_rad: dict[int, dict[BusPhaseLabel, float]] | None
    """timestep → {(bus,phase) → angle} (DC OPF only)"""
    slack_injection_w: dict[int, float] | None
    """timestep → slack injection (DC OPF only)"""
    db_path: str | None
    run_id: str | None


# ── Battery spec ─────────────────────────────────────────────────────────


@dataclass
class BatterySpec:
    """Battery parameters for multi-period optimization."""

    name: str
    node: BusPhaseLabel
    energy_capacity_wh: float
    p_charge_max_w: float
    p_discharge_max_w: float
    soc_initial: float = 0.5
    soc_min: float = 0.1
    soc_max: float = 0.9
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    cost_linear: float = 10.0


def build_battery_specs_from_components(
    system: DistributionSystem,
    *,
    soc_initial: float = 0.5,
    soc_min: float = 0.1,
    soc_max: float = 0.9,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
    cost_linear: float = 10.0,
) -> list[BatterySpec]:
    """Extract battery specs from system components."""
    specs = []
    for bat in system.get_components(DistributionBattery):
        if not bat.in_service or not bat.phases:
            continue
        p_max = float(bat.active_power.to("watt").magnitude)
        e_cap = float(bat.equipment.rated_capacity.to("watthour").magnitude)
        for phase in bat.phases:
            label = (bat.bus.name, _phase_name(phase))
            specs.append(
                BatterySpec(
                    name=f"battery:{bat.name}:{_phase_name(phase)}",
                    node=label,
                    energy_capacity_wh=e_cap / len(bat.phases),
                    p_charge_max_w=p_max / len(bat.phases),
                    p_discharge_max_w=p_max / len(bat.phases),
                    soc_initial=soc_initial,
                    soc_min=soc_min,
                    soc_max=soc_max,
                    charge_efficiency=charge_efficiency,
                    discharge_efficiency=discharge_efficiency,
                    cost_linear=cost_linear,
                )
            )
    return specs


# ── Multi-period DC OPF ──────────────────────────────────────────────────


def solve_multiperiod_dc_opf(
    system: DistributionSystem,
    *,
    generators: list[DCGenerator],
    timestep_range: range | Sequence[int],
    battery_specs: list[BatterySpec] | None = None,
    ramp_limit_w: float | None = None,
    demand_profiles: dict[int, dict[BusPhaseLabel, float]] | None = None,
    slack_label: BusPhaseLabel | list[BusPhaseLabel] | None = None,
    include_neutral: bool = False,
    include_shunt: bool = False,
    convert_geometry_to_matrix: bool = True,
    db_path: str | None = None,
) -> MultiPeriodResult:
    """Solve multi-period DC OPF with battery SOC coupling and ramp constraints.

    Decision variables per timestep:
    - Generator dispatch ``p_g[t,k]`` for each generator k
    - Battery charge/discharge ``p_bat_ch[t,b]``, ``p_bat_dis[t,b]`` for each battery b
    - Battery SOC ``soc[t,b]``
    - Voltage angles ``theta[t,i]`` for each non-slack node

    Constraints:
    - Nodal power balance at each timestep
    - Generator capacity bounds
    - Battery charge/discharge limits
    - SOC dynamics: ``soc[t+1] = soc[t] - (p_dis/eff - p_ch*eff) * dt / E``
    - SOC bounds
    - Optional ramp limits: ``|p_g[t] - p_g[t-1]| <= ramp_limit``

    Parameters
    ----------
    system : DistributionSystem
        Input distribution system with time series data.
    generators : list[DCGenerator]
        Dispatchable generators (non-battery).
    timestep_range : range or sequence
        Timestep indices to optimize jointly.
    battery_specs : list[BatterySpec], optional
        Battery parameters. If None, extracted from system components.
    ramp_limit_w : float, optional
        Maximum inter-period ramp for generators (watts).
    demand_profiles : dict, optional
        Pre-computed demand per timestep. If None, extracted from time series.
    db_path : str, optional
        SQLite database path for streaming results.
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import scipy.sparse as sp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SciPy is required for multi-period DC OPF. "
            "Install with `pip install gdm-flow[optimization]`."
        ) from exc

    timesteps = list(timestep_range)
    T = len(timesteps)
    if T == 0:
        raise ValueError("timestep_range must be non-empty.")

    # --- System topology ---
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

    # Slack identification
    if slack_label is None:
        slack_set = {0}
    elif isinstance(slack_label, list):
        slack_set = {label_to_index[sl] for sl in slack_label}
    else:
        slack_set = {label_to_index[slack_label]}

    # B-matrix
    ybus = ybus_result.ybus
    b_raw = -np.imag(ybus)
    v_nom = np.ones(n, dtype=float)
    for bus in system.get_components(DistributionBus):
        v_phase = _phase_voltage(bus.rated_voltage, bus.voltage_type)
        for phase in bus.phases:
            pn = _phase_name(phase)
            label = (bus.name, pn)
            if label in label_to_index:
                v_nom[label_to_index[label]] = v_phase
    for i, label in enumerate(labels):
        if label[1] == "S2":
            v_nom[i] = -v_nom[i]

    v_diag = sp.diags(v_nom, format="csr")
    if sp.issparse(b_raw):
        b_bus = (v_diag @ b_raw @ v_diag).tocsr()
    else:
        b_bus = v_diag @ b_raw @ v_diag

    # Connectivity filter
    from collections import deque

    b_adj = (abs(b_bus) + abs(b_bus).T).tocsr()
    gen_nodes = set()
    for gen in generators:
        if gen.node in label_to_index:
            gen_nodes.add(label_to_index[gen.node])
    reachable = set()
    bfs_queue: deque[int] = deque()
    for seed in gen_nodes | slack_set:
        if seed not in reachable:
            reachable.add(seed)
            bfs_queue.append(seed)
    while bfs_queue:
        nd = bfs_queue.popleft()
        for neighbor in b_adj[nd].indices:
            if neighbor not in reachable:
                reachable.add(neighbor)
                bfs_queue.append(neighbor)

    constraint_idx = sorted(
        i for i in reachable if i not in slack_set or i in gen_nodes
    )
    theta_var_idx = sorted(i for i in reachable if i not in slack_set)

    # Battery specs
    if battery_specs is None:
        battery_specs = build_battery_specs_from_components(system)
    batteries = battery_specs or []

    num_gen = len(generators)
    num_theta = len(theta_var_idx)
    num_bat = len(batteries)

    # Variables per timestep: [p_g(num_gen), theta(num_theta),
    #                          p_bat_dis(num_bat), p_bat_ch(num_bat), soc(num_bat)]
    vars_per_t = num_gen + num_theta + 3 * num_bat
    total_vars = T * vars_per_t

    def var_offset(t_local: int) -> int:
        return t_local * vars_per_t

    def pg_idx(t_local: int, k: int) -> int:
        return var_offset(t_local) + k

    def theta_idx(t_local: int, j: int) -> int:
        return var_offset(t_local) + num_gen + j

    def bat_dis_idx(t_local: int, b: int) -> int:
        return var_offset(t_local) + num_gen + num_theta + b

    def bat_ch_idx(t_local: int, b: int) -> int:
        return var_offset(t_local) + num_gen + num_theta + num_bat + b

    def soc_idx(t_local: int, b: int) -> int:
        return var_offset(t_local) + num_gen + num_theta + 2 * num_bat + b

    # --- Demand profiles ---
    demand_vecs: list[np.ndarray] = []
    for t_local, t_idx in enumerate(timesteps):
        if demand_profiles is not None and t_idx in demand_profiles:
            d = demand_profiles[t_idx]
        else:
            d = build_dc_load_profile_at_timestep(system, t_idx)
        demand_vecs.append(
            np.array([float(d.get(label, 0.0)) for label in labels], dtype=float)
        )

    # --- Resolution ---
    try:
        resolution = get_time_series_resolution(system)
    except ValueError:
        resolution = timedelta(hours=1)
    dt_hours = resolution.total_seconds() / 3600.0

    # --- Build LP ---
    # Cost vector
    c = np.zeros(total_vars, dtype=float)
    for t_local in range(T):
        for k in range(num_gen):
            c[pg_idx(t_local, k)] = generators[k].cost_linear
        for b in range(num_bat):
            # Penalize both charge and discharge to avoid unnecessary cycling
            c[bat_dis_idx(t_local, b)] = batteries[b].cost_linear
            c[bat_ch_idx(t_local, b)] = batteries[b].cost_linear * 0.5

    # Bounds
    lb = np.full(total_vars, -np.inf)
    ub = np.full(total_vars, np.inf)
    for t_local in range(T):
        for k in range(num_gen):
            lb[pg_idx(t_local, k)] = generators[k].p_min_w
            ub[pg_idx(t_local, k)] = generators[k].p_max_w
        for j in range(num_theta):
            lb[theta_idx(t_local, j)] = -math.pi
            ub[theta_idx(t_local, j)] = math.pi
        for b in range(num_bat):
            lb[bat_dis_idx(t_local, b)] = 0.0
            ub[bat_dis_idx(t_local, b)] = batteries[b].p_discharge_max_w
            lb[bat_ch_idx(t_local, b)] = 0.0
            ub[bat_ch_idx(t_local, b)] = batteries[b].p_charge_max_w
            lb[soc_idx(t_local, b)] = batteries[b].soc_min
            ub[soc_idx(t_local, b)] = batteries[b].soc_max

    # --- Equality constraints ---
    # 1) Nodal power balance per timestep
    # 2) SOC dynamics

    # Pre-build constraint index maps
    gen_node_idx = []
    for gen in generators:
        gen_node_idx.append(label_to_index.get(gen.node, -1))

    bat_node_idx = []
    for bat in batteries:
        bat_node_idx.append(label_to_index.get(bat.node, -1))

    constraint_idx_map = {idx: row for row, idx in enumerate(constraint_idx)}

    num_balance = len(constraint_idx)
    num_soc_eq = num_bat
    eq_per_t = num_balance + num_soc_eq
    total_eq = T * eq_per_t

    # Build sparse A_eq
    eq_rows = []
    eq_cols = []
    eq_vals = []
    b_eq = np.zeros(total_eq, dtype=float)

    c_arr = np.array(constraint_idx)
    t_arr = np.array(theta_var_idx)

    for t_local in range(T):
        eq_base = t_local * eq_per_t

        # --- Nodal power balance: sum_gen(pg) + sum_bat(p_dis - p_ch) - B*theta = demand ---
        # Generator injection
        for k in range(num_gen):
            gi = gen_node_idx[k]
            if gi in constraint_idx_map:
                row = eq_base + constraint_idx_map[gi]
                eq_rows.append(row)
                eq_cols.append(pg_idx(t_local, k))
                eq_vals.append(1.0)

        # Battery injection (discharge - charge)
        for b_i in range(num_bat):
            bi = bat_node_idx[b_i]
            if bi in constraint_idx_map:
                row = eq_base + constraint_idx_map[bi]
                # Discharge adds power
                eq_rows.append(row)
                eq_cols.append(bat_dis_idx(t_local, b_i))
                eq_vals.append(1.0)
                # Charge removes power
                eq_rows.append(row)
                eq_cols.append(bat_ch_idx(t_local, b_i))
                eq_vals.append(-1.0)

        # -B * theta block
        if sp.issparse(b_bus):
            b_block = b_bus[c_arr, :][:, t_arr]
        else:
            b_block = sp.csr_matrix(b_bus[np.ix_(c_arr, t_arr)])
        b_coo = b_block.tocoo()
        for i, j, v in zip(b_coo.row, b_coo.col, b_coo.data):
            eq_rows.append(eq_base + i)
            eq_cols.append(theta_idx(t_local, j))
            eq_vals.append(-v)

        # RHS = demand
        b_eq[eq_base : eq_base + num_balance] = demand_vecs[t_local][constraint_idx]

        # --- SOC dynamics ---
        for b_i in range(num_bat):
            row = eq_base + num_balance + b_i
            bat = batteries[b_i]
            e_cap = bat.energy_capacity_wh

            # soc[t] = soc[t-1] - (p_dis/eff_dis - p_ch*eff_ch) * dt / E
            # Rewrite: soc[t] + (p_dis * dt)/(eff_dis * E) - (p_ch * eff_ch * dt)/E = soc[t-1]

            # soc[t] coefficient
            eq_rows.append(row)
            eq_cols.append(soc_idx(t_local, b_i))
            eq_vals.append(1.0)

            # p_dis coefficient
            dis_coeff = (
                dt_hours / (bat.discharge_efficiency * e_cap) if e_cap > 0 else 0.0
            )
            eq_rows.append(row)
            eq_cols.append(bat_dis_idx(t_local, b_i))
            eq_vals.append(dis_coeff)

            # p_ch coefficient (negative because charging increases SOC)
            ch_coeff = -dt_hours * bat.charge_efficiency / e_cap if e_cap > 0 else 0.0
            eq_rows.append(row)
            eq_cols.append(bat_ch_idx(t_local, b_i))
            eq_vals.append(ch_coeff)

            if t_local == 0:
                # soc[0] = soc_initial - delta
                b_eq[row] = bat.soc_initial
            else:
                # -soc[t-1]
                eq_rows.append(row)
                eq_cols.append(soc_idx(t_local - 1, b_i))
                eq_vals.append(-1.0)
                b_eq[row] = 0.0

    a_eq = sp.csc_matrix((eq_vals, (eq_rows, eq_cols)), shape=(total_eq, total_vars))

    # --- Ramp constraints ---
    # |p_g[t] - p_g[t-1]| <= ramp_limit
    # Encoded as LinearConstraint for milp, or A_ub rows for linprog fallback.
    ramp_constraints = None
    if ramp_limit_w is not None and ramp_limit_w > 0 and T > 1:
        num_ramp_pairs = num_gen * (T - 1)
        ub_rows = []
        ub_cols = []
        ub_vals = []

        row_offset = 0
        for t_local in range(1, T):
            for k in range(num_gen):
                # p_g[t] - p_g[t-1] <= ramp
                ub_rows.append(row_offset)
                ub_cols.append(pg_idx(t_local, k))
                ub_vals.append(1.0)
                ub_rows.append(row_offset)
                ub_cols.append(pg_idx(t_local - 1, k))
                ub_vals.append(-1.0)
                row_offset += 1

                # p_g[t-1] - p_g[t] <= ramp
                ub_rows.append(row_offset)
                ub_cols.append(pg_idx(t_local - 1, k))
                ub_vals.append(1.0)
                ub_rows.append(row_offset)
                ub_cols.append(pg_idx(t_local, k))
                ub_vals.append(-1.0)
                row_offset += 1

        ramp_constraints = sp.csc_matrix(
            (ub_vals, (ub_rows, ub_cols)), shape=(2 * num_ramp_pairs, total_vars)
        )

    # --- Solve LP via milp (better HiGHS interface than linprog) ---
    bounds_obj = Bounds(lb, ub)
    constraints_list = [LinearConstraint(a_eq, b_eq, b_eq)]
    if ramp_constraints is not None:
        ramp_ub = np.full(ramp_constraints.shape[0], ramp_limit_w)
        constraints_list.append(LinearConstraint(ramp_constraints, -np.inf, ramp_ub))

    lp_result = milp(
        c,
        constraints=constraints_list,
        bounds=bounds_obj,
    )

    if not lp_result.success:
        # HiGHS sometimes returns unrecognized status codes (e.g. Status 15)
        # even when a feasible solution was found. Check if we have a valid
        # solution vector before declaring failure.
        if lp_result.x is None or "Feasible" not in str(lp_result.message):
            return MultiPeriodResult(
                success=False,
                message=f"LP solver failed: {lp_result.message}",
                solver="dc",
                num_timesteps=T,
                objective=float("inf"),
                generator_dispatch_w={},
                battery_soc={},
                nodal_voltage=None,
                theta_rad={},
                slack_injection_w={},
                db_path=db_path,
                run_id=None,
            )

    x = lp_result.x

    # --- Extract results ---
    gen_dispatch: dict[int, dict[str, float]] = {}
    theta_results: dict[int, dict[BusPhaseLabel, float]] = {}
    slack_inj: dict[int, float] = {}
    bat_soc: dict[str, list[float]] = {bat.name: [] for bat in batteries}

    phase_offset = np.zeros(n, dtype=float)
    for i, label in enumerate(labels):
        if label[1] == "S2":
            phase_offset[i] = math.pi

    for t_local in range(T):
        t_idx = timesteps[t_local]

        # Generator dispatch
        dispatch = {}
        for k in range(num_gen):
            dispatch[generators[k].name] = float(x[pg_idx(t_local, k)])
        # Add battery net dispatch
        for b_i in range(num_bat):
            net = float(x[bat_dis_idx(t_local, b_i)] - x[bat_ch_idx(t_local, b_i)])
            dispatch[batteries[b_i].name] = net
        gen_dispatch[t_idx] = dispatch

        # SOC
        for b_i in range(num_bat):
            bat_soc[batteries[b_i].name].append(float(x[soc_idx(t_local, b_i)]))

        # Theta
        theta_full = np.zeros(n, dtype=float)
        for j in range(num_theta):
            theta_full[theta_var_idx[j]] = x[theta_idx(t_local, j)]
        theta_physical = theta_full + phase_offset
        theta_results[t_idx] = {labels[i]: float(theta_physical[i]) for i in range(n)}

        # Slack injection
        gen_inj = np.zeros(n, dtype=float)
        for k in range(num_gen):
            gi = gen_node_idx[k]
            if gi >= 0:
                gen_inj[gi] += x[pg_idx(t_local, k)]
        for b_i in range(num_bat):
            bi = bat_node_idx[b_i]
            if bi >= 0:
                gen_inj[bi] += (
                    x[bat_dis_idx(t_local, b_i)] - x[bat_ch_idx(t_local, b_i)]
                )
        nodal_balance = gen_inj - demand_vecs[t_local] - b_bus @ theta_full
        slack_inj[t_idx] = float(-sum(nodal_balance[i] for i in slack_set))

    # --- Stream to SQLite ---
    run_id = None
    if db_path is not None:
        run_id = _stream_multiperiod_to_sqlite(
            db_path,
            "dc",
            timesteps,
            gen_dispatch,
            bat_soc,
            theta_results,
            None,
            slack_inj,
            batteries,
            system=system,
        )

    return MultiPeriodResult(
        success=True,
        message=f"Multi-period DC OPF converged ({T} timesteps)",
        solver="dc",
        num_timesteps=T,
        objective=float(lp_result.fun),
        generator_dispatch_w=gen_dispatch,
        battery_soc=bat_soc,
        nodal_voltage=None,
        theta_rad=theta_results,
        slack_injection_w=slack_inj,
        db_path=db_path,
        run_id=run_id,
    )


# ── Multi-period LinDistFlow ─────────────────────────────────────────────


def solve_multiperiod_lindistflow(
    system: DistributionSystem,
    *,
    timestep_range: range | Sequence[int],
    battery_specs: list[BatterySpec] | None = None,
    injection_profiles: dict[
        int, tuple[dict[BusPhaseLabel, float], dict[BusPhaseLabel, float]]
    ]
    | None = None,
    db_path: str | None = None,
) -> MultiPeriodResult:
    """Solve multi-period LinDistFlow with battery SOC coupling.

    Uses the single-period LinDistFlow solver per timestep with battery
    dispatch optimized jointly across the horizon via LP.

    The approach:
    1. Pre-compute base demand (without batteries) at each timestep.
    2. Formulate an LP over battery charge/discharge decisions coupled by SOC.
    3. Minimize total demand deviation (or cost).
    4. Re-solve LinDistFlow per timestep with optimized battery dispatch.

    Parameters
    ----------
    system : DistributionSystem
        Input distribution system with time series data.
    timestep_range : range or sequence
        Timestep indices to optimize jointly.
    battery_specs : list[BatterySpec], optional
        Battery parameters. If None, extracted from system components.
    injection_profiles : dict, optional
        Pre-computed (p_net, q_net) per timestep.
    db_path : str, optional
        SQLite database path for streaming results.
    """
    from .lindistflow import solve_lindistflow

    try:
        from scipy.optimize import linprog
        import scipy.sparse as sp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SciPy is required for multi-period LinDistFlow. "
            "Install with `pip install gdm-flow[optimization]`."
        ) from exc

    timesteps = list(timestep_range)
    T = len(timesteps)
    if T == 0:
        raise ValueError("timestep_range must be non-empty.")

    if battery_specs is None:
        battery_specs = build_battery_specs_from_components(system)
    batteries = battery_specs or []
    num_bat = len(batteries)

    try:
        resolution = get_time_series_resolution(system)
    except ValueError:
        resolution = timedelta(hours=1)
    dt_hours = resolution.total_seconds() / 3600.0

    # Pre-compute base injections (without battery) at each timestep
    base_p: list[dict[BusPhaseLabel, float]] = []
    base_q: list[dict[BusPhaseLabel, float]] = []
    for t_idx in timesteps:
        if injection_profiles is not None and t_idx in injection_profiles:
            p, q = injection_profiles[t_idx]
        else:
            p, q = build_lindistflow_injections_at_timestep(
                system,
                t_idx,
                include_battery=False,
            )
        base_p.append(p)
        base_q.append(q)

    # If no batteries, just run LinDistFlow sequentially
    if num_bat == 0:
        voltages: dict[int, dict[BusPhaseLabel, float]] = {}
        gen_dispatch: dict[int, dict[str, float]] = {}
        inj_p: dict[int, dict[BusPhaseLabel, float]] = {}
        inj_q: dict[int, dict[BusPhaseLabel, float]] = {}
        num_converged = 0
        for t_local, t_idx in enumerate(timesteps):
            result = solve_lindistflow(
                system,
                p_net_w=base_p[t_local],
                q_net_var=base_q[t_local],
            )
            if result.success:
                num_converged += 1
            voltages[t_idx] = dict(result.voltage_v)
            gen_dispatch[t_idx] = {}
            inj_p[t_idx] = dict(result.p_net_w)
            inj_q[t_idx] = dict(result.q_net_var)

        run_id = None
        if db_path is not None:
            run_id = _stream_multiperiod_to_sqlite(
                db_path,
                "ldf",
                timesteps,
                gen_dispatch,
                {},
                None,
                voltages,
                None,
                [],
                injection_p=inj_p,
                injection_q=inj_q,
                system=system,
            )

        return MultiPeriodResult(
            success=num_converged == T,
            message=f"LinDistFlow solved {num_converged}/{T} timesteps (no batteries)",
            solver="ldf",
            num_timesteps=T,
            objective=0.0,
            generator_dispatch_w=gen_dispatch,
            battery_soc={},
            nodal_voltage=voltages,
            theta_rad=None,
            slack_injection_w=None,
            db_path=db_path,
            run_id=run_id,
        )

    # --- Battery dispatch LP ---
    # Variables: [p_dis(T*num_bat), p_ch(T*num_bat), soc(T*num_bat)]
    total_vars = 3 * T * num_bat

    def dis_idx(t_local: int, b: int) -> int:
        return t_local * num_bat + b

    def ch_idx(t_local: int, b: int) -> int:
        return T * num_bat + t_local * num_bat + b

    def s_idx(t_local: int, b: int) -> int:
        return 2 * T * num_bat + t_local * num_bat + b

    # Cost: minimize total battery cycling
    c_vec = np.zeros(total_vars, dtype=float)
    for t_local in range(T):
        for b in range(num_bat):
            c_vec[dis_idx(t_local, b)] = batteries[b].cost_linear
            c_vec[ch_idx(t_local, b)] = batteries[b].cost_linear * 0.5

    # Bounds
    lbv = np.zeros(total_vars, dtype=float)
    ubv = np.full(total_vars, np.inf)
    for t_local in range(T):
        for b in range(num_bat):
            ubv[dis_idx(t_local, b)] = batteries[b].p_discharge_max_w
            ubv[ch_idx(t_local, b)] = batteries[b].p_charge_max_w
            lbv[s_idx(t_local, b)] = batteries[b].soc_min
            ubv[s_idx(t_local, b)] = batteries[b].soc_max

    # SOC dynamics equality constraints
    eq_rows_l = []
    eq_cols_l = []
    eq_vals_l = []
    b_eq_l = np.zeros(T * num_bat, dtype=float)

    for t_local in range(T):
        for b in range(num_bat):
            row = t_local * num_bat + b
            bat = batteries[b]
            e_cap = bat.energy_capacity_wh

            # soc[t]
            eq_rows_l.append(row)
            eq_cols_l.append(s_idx(t_local, b))
            eq_vals_l.append(1.0)

            # p_dis coefficient
            dis_coeff = (
                dt_hours / (bat.discharge_efficiency * e_cap) if e_cap > 0 else 0.0
            )
            eq_rows_l.append(row)
            eq_cols_l.append(dis_idx(t_local, b))
            eq_vals_l.append(dis_coeff)

            # p_ch coefficient
            ch_coeff = -dt_hours * bat.charge_efficiency / e_cap if e_cap > 0 else 0.0
            eq_rows_l.append(row)
            eq_cols_l.append(ch_idx(t_local, b))
            eq_vals_l.append(ch_coeff)

            if t_local == 0:
                b_eq_l[row] = bat.soc_initial
            else:
                eq_rows_l.append(row)
                eq_cols_l.append(s_idx(t_local - 1, b))
                eq_vals_l.append(-1.0)
                b_eq_l[row] = 0.0

    a_eq_bat = sp.csc_matrix(
        (eq_vals_l, (eq_rows_l, eq_cols_l)), shape=(T * num_bat, total_vars)
    )

    bounds_seq = list(zip(lbv.tolist(), ubv.tolist()))
    lp_result = linprog(
        c_vec,
        A_eq=a_eq_bat,
        b_eq=b_eq_l,
        bounds=bounds_seq,
        method="highs",
    )

    if not lp_result.success:
        return MultiPeriodResult(
            success=False,
            message=f"Battery LP failed: {lp_result.message}",
            solver="ldf",
            num_timesteps=T,
            objective=float("inf"),
            generator_dispatch_w={},
            battery_soc={},
            nodal_voltage=None,
            theta_rad=None,
            slack_injection_w=None,
            db_path=db_path,
            run_id=None,
        )

    x_bat = lp_result.x

    # --- Re-solve LinDistFlow with battery dispatch ---
    voltages = {}
    gen_dispatch = {}
    inj_p_bat: dict[int, dict[BusPhaseLabel, float]] = {}
    inj_q_bat: dict[int, dict[BusPhaseLabel, float]] = {}
    bat_soc: dict[str, list[float]] = {bat.name: [] for bat in batteries}
    num_converged = 0

    for t_local, t_idx in enumerate(timesteps):
        p_net = dict(base_p[t_local])
        q_net = dict(base_q[t_local])

        # Inject battery dispatch (discharge subtracts from demand, charge adds)
        dispatch_t: dict[str, float] = {}
        for b in range(num_bat):
            p_dis = float(x_bat[dis_idx(t_local, b)])
            p_ch = float(x_bat[ch_idx(t_local, b)])
            net_bat = p_dis - p_ch
            dispatch_t[batteries[b].name] = net_bat
            bat_soc[batteries[b].name].append(float(x_bat[s_idx(t_local, b)]))

            node = batteries[b].node
            p_net[node] = p_net.get(node, 0.0) - net_bat

        result = solve_lindistflow(system, p_net_w=p_net, q_net_var=q_net)
        if result.success:
            num_converged += 1
        voltages[t_idx] = dict(result.voltage_v)
        gen_dispatch[t_idx] = dispatch_t
        inj_p_bat[t_idx] = dict(result.p_net_w)
        inj_q_bat[t_idx] = dict(result.q_net_var)

    run_id = None
    if db_path is not None:
        run_id = _stream_multiperiod_to_sqlite(
            db_path,
            "ldf",
            timesteps,
            gen_dispatch,
            bat_soc,
            None,
            voltages,
            None,
            batteries,
            injection_p=inj_p_bat,
            injection_q=inj_q_bat,
            system=system,
        )

    return MultiPeriodResult(
        success=num_converged == T,
        message=f"Multi-period LinDistFlow: {num_converged}/{T} converged",
        solver="ldf",
        num_timesteps=T,
        objective=float(lp_result.fun),
        generator_dispatch_w=gen_dispatch,
        battery_soc=bat_soc,
        nodal_voltage=voltages,
        theta_rad=None,
        slack_injection_w=None,
        db_path=db_path,
        run_id=run_id,
    )


# ── SQLite streaming ─────────────────────────────────────────────────────


def _stream_multiperiod_to_sqlite(
    db_path: str,
    solver: str,
    timesteps: list[int],
    gen_dispatch: dict[int, dict[str, float]],
    bat_soc: dict[str, list[float]],
    theta_results: dict[int, dict[BusPhaseLabel, float]] | None,
    voltage_results: dict[int, dict[BusPhaseLabel, float]] | None,
    slack_inj: dict[int, float] | None,
    batteries: list[BatterySpec],
    injection_p: dict[int, dict[BusPhaseLabel, float]] | None = None,
    injection_q: dict[int, dict[BusPhaseLabel, float]] | None = None,
    system: DistributionSystem | None = None,
) -> str:
    """Write multi-period results to SQLite and return run_id."""
    import sqlite3
    from uuid import uuid4

    from .time_series import _create_ts_schema, _populate_bus_nominal

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _create_ts_schema(conn)

    # Populate nominal voltages for p.u. computation
    nominal_map: dict[tuple[str, str], float] = {}
    if system is not None:
        nominal_map = _populate_bus_nominal(conn, system)

    run_id = f"mp_{solver}_{uuid4().hex[:12]}"
    conn.execute(
        """INSERT INTO ts_runs
           (run_id, implementation, mode, num_timesteps, resolution_s, start_timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, solver, "multiperiod", len(timesteps), None, None),
    )

    for t_local, t_idx in enumerate(timesteps):
        # Nodal data
        if theta_results and t_idx in theta_results:
            for label, theta_val in theta_results[t_idx].items():
                p_inj = None
                if injection_p and t_idx in injection_p:
                    p_inj = injection_p[t_idx].get(label)
                conn.execute(
                    """INSERT OR REPLACE INTO ts_nodes
                       (run_id, timestep, bus_name, phase,
                        voltage_mag_v, voltage_pu, voltage_angle_rad, p_injection_w, q_injection_var)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        t_idx,
                        label[0],
                        label[1],
                        None,
                        None,
                        theta_val,
                        p_inj,
                        None,
                    ),
                )
        if voltage_results and t_idx in voltage_results:
            for label, v_val in voltage_results[t_idx].items():
                p_inj = None
                q_inj = None
                if injection_p and t_idx in injection_p:
                    p_inj = injection_p[t_idx].get(label)
                if injection_q and t_idx in injection_q:
                    q_inj = injection_q[t_idx].get(label)
                nom = nominal_map.get(label, 1.0) if nominal_map else 1.0
                v_pu = v_val / nom if nom > 0 else None
                conn.execute(
                    """INSERT OR REPLACE INTO ts_nodes
                       (run_id, timestep, bus_name, phase,
                        voltage_mag_v, voltage_pu, voltage_angle_rad, p_injection_w, q_injection_var)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        t_idx,
                        label[0],
                        label[1],
                        v_val,
                        v_pu,
                        None,
                        p_inj,
                        q_inj,
                    ),
                )

        # Summary — compute source power from injections or slack
        source_p = None
        source_q = None
        if slack_inj and t_idx in slack_inj:
            source_p = slack_inj[t_idx]
        elif injection_p and t_idx in injection_p:
            source_p = sum(float(v) for v in injection_p[t_idx].values())
        if injection_q and t_idx in injection_q:
            source_q = sum(float(v) for v in injection_q[t_idx].values())
        conn.execute(
            """INSERT OR REPLACE INTO ts_summary
               (run_id, timestep, success, source_p_w, source_q_var, total_loss_w, solve_time_s)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, t_idx, 1, source_p, source_q, None, None),
        )

        # Battery SOC
        for bat in batteries:
            soc_list = bat_soc.get(bat.name, [])
            if t_local < len(soc_list):
                conn.execute(
                    """INSERT OR REPLACE INTO ts_battery_soc
                       (run_id, timestep, battery_name, soc, p_dispatch_w, energy_wh)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        t_idx,
                        bat.name,
                        soc_list[t_local],
                        gen_dispatch.get(t_idx, {}).get(bat.name, 0.0),
                        soc_list[t_local] * bat.energy_capacity_wh,
                    ),
                )

    conn.commit()
    conn.close()
    return run_id
