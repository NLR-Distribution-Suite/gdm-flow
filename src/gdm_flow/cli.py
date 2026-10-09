"""Modern CLI for GDM-Flow power flow analysis.

Usage:
    gdm-flow info    MODEL        Show system topology and component summary
    gdm-flow run     MODEL        Run one or more OPF solvers
    gdm-flow compare MODEL        Run all solvers and compare results
    gdm-flow plot    MODEL        Generate interactive analysis dashboard
    gdm-flow export  MODEL --db   Export solver results to SQLite
"""

from __future__ import annotations

import math
import time
from enum import Enum
from pathlib import Path
import sqlite3
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import DistributionBus
from gdm.systems.distribution.enums import Phase

from ._utils import _phase_name
from .ac_opf import build_regulator_voltage_limits_from_components

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="gdm-flow",
    help="[bold cyan]GDM-Flow[/] — Power flow & optimal power flow for distribution systems",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=True,
)


class Solver(str, Enum):
    ac = "ac"
    pf = "pf"
    dc = "dc"
    ldf = "ldf"


# ── helpers ──────────────────────────────────────────────────────────────


def _load_system(model: Path) -> DistributionSystem:
    """Load a DistributionSystem from a JSON file."""
    if not model.exists():
        err_console.print(f"[red]Error:[/] file not found: {model}")
        raise typer.Exit(1)
    with console.status("[cyan]Loading model…"):
        system = DistributionSystem.from_json(str(model))

    # Auto-aggregate parallel single-phase transformers/regulators
    from collections import defaultdict
    from gdm.systems.distribution.components.distribution_transformer import (
        DistributionTransformer,
    )
    from gdm.systems.distribution.components.distribution_regulator import (
        DistributionRegulator,
    )

    needs_aggregation = False
    for comp_type, label in [
        (DistributionTransformer, "transformers"),
        (DistributionRegulator, "regulators"),
    ]:
        groups: dict[tuple[str, ...], list] = defaultdict(list)
        for comp in system.get_components(comp_type):
            if all(wdg.num_phases == 1 for wdg in comp.equipment.windings):
                key = tuple(bus.name for bus in comp.buses)
                groups[key].append(comp)
        parallel = {k: v for k, v in groups.items() if 2 <= len(v) <= 3}
        if parallel:
            needs_aggregation = True
            for key, group in parallel.items():
                names = [c.name for c in group]
                config = "3-phase" if len(group) == 3 else "open-wye/open-delta"
                console.print(
                    f"[yellow]⚠ Merging {len(group)} parallel single-phase "
                    f"{label} into {config} equivalent:[/] {', '.join(names)}"
                )

    if needs_aggregation:
        from gdm.systems.distribution.utils import aggregate_single_phase_transformers

        aggregate_single_phase_transformers(system)

    return system


def _fmt_w(val: float) -> str:
    """Format watts with appropriate unit."""
    if abs(val) >= 1e6:
        return f"{val / 1e6:.2f} MW"
    if abs(val) >= 1e3:
        return f"{val / 1e3:.2f} kW"
    return f"{val:.1f} W"


def _fmt_var(val: float) -> str:
    """Format vars with appropriate unit."""
    if abs(val) >= 1e6:
        return f"{val / 1e6:.2f} Mvar"
    if abs(val) >= 1e3:
        return f"{val / 1e3:.2f} kvar"
    return f"{val:.1f} var"


def _success_badge(ok: bool) -> str:
    return "[bold green]✓ PASS[/]" if ok else "[bold red]✗ FAIL[/]"


def _to_float_quantity(value, unit: str) -> float | None:
    if value is None:
        return None
    if hasattr(value, "to"):
        try:
            return float(value.to(unit).magnitude)
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_voltage_bounds(limit_obj) -> tuple[float, float] | None:
    min_candidates = [
        "min_voltage",
        "min_v_limit",
        "lower_limit",
        "lower",
        "v_min",
        "minimum",
    ]
    max_candidates = [
        "max_voltage",
        "max_v_limit",
        "upper_limit",
        "upper",
        "v_max",
        "maximum",
    ]

    min_value = None
    max_value = None
    for name in min_candidates:
        if hasattr(limit_obj, name):
            min_value = _to_float_quantity(getattr(limit_obj, name), "volt")
            if min_value is not None:
                break
    for name in max_candidates:
        if hasattr(limit_obj, name):
            max_value = _to_float_quantity(getattr(limit_obj, name), "volt")
            if max_value is not None:
                break

    if min_value is None or max_value is None:
        return None
    return (min(min_value, max_value), max(min_value, max_value))


def _build_node_voltage_limits_v(
    system: DistributionSystem,
) -> dict[tuple[str, str], tuple[float, float]]:
    limits: dict[tuple[str, str], tuple[float, float]] = {}

    for bus in system.get_components(DistributionBus):
        bus_phases = [_phase_name(p) for p in bus.phases if p != Phase.N]
        for vlimit in getattr(bus, "voltagelimits", []) or []:
            bounds = _extract_voltage_bounds(vlimit)
            if bounds is None:
                continue

            raw_phase = getattr(vlimit, "phase", None)
            if raw_phase is None:
                phases = bus_phases
            else:
                ph_name = (
                    _phase_name(raw_phase)
                    if not isinstance(raw_phase, str)
                    else raw_phase
                )
                phases = [ph_name]

            for phase in phases:
                key = (bus.name, phase)
                if key in limits:
                    lo0, hi0 = limits[key]
                    lo1, hi1 = bounds
                    limits[key] = (max(lo0, lo1), min(hi0, hi1))
                else:
                    limits[key] = bounds

    # Regulator limits are hard constraints and should tighten any existing bus-level bounds.
    for key, bounds in build_regulator_voltage_limits_from_components(system).items():
        if key in limits:
            lo0, hi0 = limits[key]
            lo1, hi1 = bounds
            limits[key] = (max(lo0, lo1), min(hi0, hi1))
        else:
            limits[key] = bounds

    return limits


def _build_lindistflow_loading_limits_va(
    system: DistributionSystem,
) -> dict[tuple[str, str], float]:
    from gdm.systems.distribution.components.base.distribution_branch_base import (
        DistributionBranchBase,
    )

    bus_phase_voltage_v: dict[tuple[str, str], float] = {}
    for bus in system.get_components(DistributionBus):
        v_nom = _to_float_quantity(bus.rated_voltage, "volt")
        if v_nom is None:
            continue
        for phase in bus.phases:
            if phase == Phase.N:
                continue
            bus_phase_voltage_v[(bus.name, _phase_name(phase))] = v_nom

    edge_parent_bus: dict[str, str] = {}
    digraph = system.get_directed_graph(return_radial_network=True)
    for u, _v, data in digraph.edges(data=True):
        branch_name = data.get("name")
        if branch_name:
            edge_parent_bus[branch_name] = u

    limits_va: dict[tuple[str, str], float] = {}
    for branch in system.get_components(DistributionBranchBase):
        ampacity = _to_float_quantity(
            getattr(branch.equipment, "ampacity", None), "ampere"
        )
        if ampacity is None or ampacity <= 0:
            continue

        parent_bus = edge_parent_bus.get(branch.name)
        for phase in branch.phases:
            if phase == Phase.N:
                continue
            phase_name = _phase_name(phase)
            v_phase = (
                bus_phase_voltage_v.get((parent_bus, phase_name))
                if parent_bus is not None
                else None
            )
            if v_phase is None:
                continue
            limits_va[(branch.name, phase_name)] = float(v_phase * ampacity)

    return limits_va


def _branch_phase_series_impedance_ohm(branch, phase_name: str) -> tuple[float, float]:
    branch_phase_names = [_phase_name(p) for p in branch.phases]
    if phase_name not in branch_phase_names:
        return (0.0, 0.0)

    if hasattr(branch, "equipment") and hasattr(branch.equipment, "r_matrix"):
        idx = branch_phase_names.index(phase_name)
        length_m = float(branch.length.to("m").magnitude)
        r = float(branch.equipment.r_matrix.to("ohm/m").magnitude[idx][idx]) * length_m
        x = float(branch.equipment.x_matrix.to("ohm/m").magnitude[idx][idx]) * length_m
        return (r, x)

    if hasattr(branch, "equipment") and hasattr(branch.equipment, "pos_seq_resistance"):
        length_m = float(branch.length.to("m").magnitude)
        r = float(branch.equipment.pos_seq_resistance.to("ohm/m").magnitude) * length_m
        x = float(branch.equipment.pos_seq_reactance.to("ohm/m").magnitude) * length_m
        return (r, x)

    return (0.0, 0.0)


def _build_ac_branch_loading_from_result(
    system: DistributionSystem,
    ac_result,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], tuple[float, float]],
]:
    from gdm.systems.distribution.components.base.distribution_branch_base import (
        DistributionBranchBase,
    )

    v_by_label = {
        label: ac_result.voltage[i]
        for i, label in enumerate(ac_result.ybus_result.index_to_label)
    }

    digraph = system.get_directed_graph(return_radial_network=True)
    edge_component = {}
    for u, v, data in digraph.edges(data=True):
        ctype = data.get("type")
        cname = data.get("name")
        if not ctype or not cname:
            continue
        try:
            comp = system.get_component(ctype, cname)
        except Exception:
            continue
        if isinstance(comp, DistributionBranchBase) and comp.in_service:
            if hasattr(comp, "is_closed") and not all(bool(x) for x in comp.is_closed):
                continue
            edge_component[(u, v)] = comp

    loading_va: dict[tuple[str, str], float] = {}
    loading_limits_va: dict[tuple[str, str], float] = {}
    flow_w_var: dict[tuple[str, str], tuple[float, float]] = {}

    for (u, v), branch in edge_component.items():
        ampacity = _to_float_quantity(
            getattr(branch.equipment, "ampacity", None), "ampere"
        )
        for phase in branch.phases:
            if phase == Phase.N:
                continue
            phase_name = _phase_name(phase)
            v_u = v_by_label.get((u, phase_name))
            v_v = v_by_label.get((v, phase_name))
            if v_u is None or v_v is None:
                continue

            r_ohm, x_ohm = _branch_phase_series_impedance_ohm(branch, phase_name)
            z = complex(r_ohm, x_ohm)
            if abs(z) < 1e-12:
                continue

            i_branch = (v_u - v_v) / z
            s_from = v_u * np.conj(i_branch)
            key = (branch.name, phase_name)
            flow_w_var[key] = (float(s_from.real), float(s_from.imag))
            loading_va[key] = float(abs(s_from))
            if ampacity is not None and ampacity > 0:
                loading_limits_va[key] = float(abs(v_u) * ampacity)

    return loading_va, loading_limits_va, flow_w_var


def _build_dc_branch_loading_from_result(
    system: DistributionSystem,
    dc_result,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], tuple[float, float]],
]:
    """Approximate DC branch active flows from solved angle differences.

    Uses P_ij ~= (V_i * V_j / X_ij) * (theta_i - theta_j) with branch series reactance.
    Reactive flow is reported as 0.0 in this DC approximation.
    """

    from gdm.systems.distribution.components.base.distribution_branch_base import (
        DistributionBranchBase,
    )

    theta_by_label = dc_result.theta_rad

    # Match DC OPF linearization base angles used during the solve.
    phase_offset_by_name = {
        "S2": math.pi,
    }

    def _theta_effective(label: tuple[str, str]) -> float | None:
        theta = theta_by_label.get(label)
        if theta is None:
            return None
        return float(theta) - float(phase_offset_by_name.get(label[1], 0.0))

    v_nom_by_label: dict[tuple[str, str], float] = {}
    for bus in system.get_components(DistributionBus):
        v_nom = _to_float_quantity(bus.rated_voltage, "volt")
        if v_nom is None:
            continue
        for phase in bus.phases:
            if phase == Phase.N:
                continue
            v_nom_by_label[(bus.name, _phase_name(phase))] = v_nom

    digraph = system.get_directed_graph(return_radial_network=True)
    edge_component = {}
    for u, v, data in digraph.edges(data=True):
        ctype = data.get("type")
        cname = data.get("name")
        if not ctype or not cname:
            continue
        try:
            comp = system.get_component(ctype, cname)
        except Exception:
            continue
        if isinstance(comp, DistributionBranchBase) and comp.in_service:
            if hasattr(comp, "is_closed") and not all(bool(x) for x in comp.is_closed):
                continue
            edge_component[(u, v)] = comp

    loading_va: dict[tuple[str, str], float] = {}
    loading_limits_va: dict[tuple[str, str], float] = {}
    flow_w_var: dict[tuple[str, str], tuple[float, float]] = {}

    for (u, v), branch in edge_component.items():
        ampacity = _to_float_quantity(
            getattr(branch.equipment, "ampacity", None), "ampere"
        )
        for phase in branch.phases:
            if phase == Phase.N:
                continue
            phase_name = _phase_name(phase)
            theta_u = _theta_effective((u, phase_name))
            theta_v = _theta_effective((v, phase_name))
            v_u = v_nom_by_label.get((u, phase_name))
            v_v = v_nom_by_label.get((v, phase_name))
            if theta_u is None or theta_v is None or v_u is None or v_v is None:
                continue

            _r_ohm, x_ohm = _branch_phase_series_impedance_ohm(branch, phase_name)
            # Skip near-zero reactance elements (switch/fuse-like) where DC flow
            # reconstruction is numerically unstable and not physically meaningful.
            if abs(x_ohm) < 1e-3:
                continue

            p_flow_w = float((v_u * v_v / x_ohm) * (theta_u - theta_v))
            key = (branch.name, phase_name)
            flow_w_var[key] = (p_flow_w, 0.0)
            loading_va[key] = abs(p_flow_w)
            if ampacity is not None and ampacity > 0:
                loading_limits_va[key] = float(abs(v_u) * ampacity)

    return loading_va, loading_limits_va, flow_w_var


def _table_has_columns(
    conn: sqlite3.Connection, table_name: str, columns: set[str]
) -> bool:
    available = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    return columns.issubset(available)


def _resolve_latest_run_id(
    conn: sqlite3.Connection,
    implementation: str,
    requested_run_id: str | None,
) -> str | None:
    if requested_run_id:
        row = conn.execute(
            "SELECT run_id FROM runs WHERE run_id = ? AND implementation = ?",
            (requested_run_id, implementation),
        ).fetchone()
        return row[0] if row else None

    row = conn.execute(
        """
        SELECT run_id FROM runs
        WHERE implementation = ?
        ORDER BY created_at_utc DESC
        LIMIT 1
        """,
        (implementation,),
    ).fetchone()
    return row[0] if row else None


def _read_overvoltage_rows(
    db_path: str,
    implementation: str,
    run_id: str | None,
) -> tuple[str | None, list[tuple], bool]:
    table = (
        "ac_opf_nodes"
        if implementation == "ac_opf"
        else "ac_pf_nodes"
        if implementation == "ac_pf"
        else "lindistflow_nodes"
    )
    voltage_col = (
        "voltage_mag_v" if implementation in ("ac_opf", "ac_pf") else "voltage_v"
    )
    required = {"voltage_min_v", "voltage_max_v"}

    conn = sqlite3.connect(db_path)
    try:
        if not _table_has_columns(conn, table, required):
            return None, [], False

        resolved = _resolve_latest_run_id(conn, implementation, run_id)
        if resolved is None:
            return None, [], True

        rows = conn.execute(
            f"""
            SELECT bus_name, phase, {voltage_col}, voltage_min_v, voltage_max_v
            FROM {table}
            WHERE run_id = ?
              AND (
                (voltage_max_v IS NOT NULL AND {voltage_col} > voltage_max_v)
                OR
                (voltage_min_v IS NOT NULL AND {voltage_col} < voltage_min_v)
              )
            ORDER BY ({voltage_col} - COALESCE(voltage_max_v, {voltage_col})) DESC,
                     bus_name,
                     phase
            """,
            (resolved,),
        ).fetchall()
        return resolved, rows, True
    finally:
        conn.close()


def _read_overload_rows(
    db_path: str,
    implementation: str,
    run_id: str | None,
) -> tuple[str | None, list[tuple], bool]:
    conn = sqlite3.connect(db_path)
    try:
        required = {"loading_va", "loading_limit_va"}
        table_by_impl = {
            "ac_opf": "ac_opf_branches",
            "ac_pf": "ac_pf_branches",
            "dc_opf": "dc_opf_branches",
            "lindistflow": "lindistflow_branches",
        }
        table = table_by_impl.get(implementation)
        if table is None:
            return None, [], False
        if not _table_has_columns(conn, table, required):
            return None, [], False

        resolved = _resolve_latest_run_id(conn, implementation, run_id)
        if resolved is None:
            return None, [], True

        rows = conn.execute(
            f"""
            SELECT
                branch_name,
                phase,
                p_flow_w,
                q_flow_var,
                loading_va,
                loading_limit_va,
                (loading_va / loading_limit_va) AS loading_ratio
                        FROM {table}
            WHERE run_id = ?
              AND loading_limit_va IS NOT NULL
              AND loading_limit_va > 0
              AND loading_va > loading_limit_va
            ORDER BY loading_ratio DESC, branch_name, phase
                        """,
            (resolved,),
        ).fetchall()
        return resolved, rows, True
    finally:
        conn.close()


def _read_db_schema(
    db_path: str,
    *,
    include_internal: bool = False,
) -> list[tuple[str, list[str]]]:
    conn = sqlite3.connect(db_path)
    try:
        if include_internal:
            table_rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        else:
            table_rows = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()

        out: list[tuple[str, list[str]]] = []
        for (table_name,) in table_rows:
            columns = [
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            ]
            out.append((str(table_name), columns))
        return out
    finally:
        conn.close()


def _voltage_pu_stats(result_dict: dict, system: "DistributionSystem") -> dict | None:
    """Compute per-unit voltage stats from a solver result dict.

    Returns dict with v_min_pu, v_max_pu, v_mean_pu, n_under_095 or None.
    """
    from .ac_opf import _build_nominal_voltage_map

    r = result_dict.get("result")
    if r is None:
        return None

    nominal_map = _build_nominal_voltage_map(system)

    vm_pu = []
    if hasattr(r, "voltage") and hasattr(r, "ybus_result"):
        for idx, label in enumerate(r.ybus_result.index_to_label):
            if label[1] == "N":
                continue
            nom = nominal_map.get(label, 0.0)
            if nom > 0:
                vm_pu.append(float(abs(r.voltage[idx])) / nom)
    elif hasattr(r, "voltage_v"):
        for label, v in r.voltage_v.items():
            if label[1] == "N":
                continue
            nom = nominal_map.get(label, 0.0)
            if nom > 0:
                vm_pu.append(float(v) / nom)
    else:
        return None

    if not vm_pu:
        return None

    return {
        "v_min_pu": min(vm_pu),
        "v_max_pu": max(vm_pu),
        "v_mean_pu": sum(vm_pu) / len(vm_pu),
        "n_under_095": sum(1 for v in vm_pu if v < 0.95),
        "n_buses": len(vm_pu),
    }


def _run_ac(system: DistributionSystem) -> dict:
    from .ac_opf import optimize_ac_power_flow_from_components

    t0 = time.perf_counter()
    result = optimize_ac_power_flow_from_components(
        system,
        include_loads=True,
        include_solar=True,
        include_capacitor=True,
        include_regulator_targets=True,
        include_regulator_limits=True,
    )
    elapsed = time.perf_counter() - t0

    # Compute source power
    src_bus = system.get_source_bus().name
    idx_map = result.ybus_result.index_to_label
    v = result.voltage
    ybus = result.ybus_result.ybus
    s = v * np.conj(ybus @ v)
    src_idx = [i for i, lbl in enumerate(idx_map) if lbl[0] == src_bus]
    source_p = sum(s[i].real for i in src_idx)
    source_q = sum(s[i].imag for i in src_idx)

    # Voltage stats
    v_mag = np.abs(v)

    return {
        "solver": "AC OPF",
        "success": result.success,
        "message": result.message,
        "elapsed": elapsed,
        "iterations": result.iterations,
        "source_p": source_p,
        "source_q": source_q,
        "v_min": float(np.min(v_mag)),
        "v_max": float(np.max(v_mag)),
        "objective": result.final_objective,
        "result": result,
    }


def _run_dc(system: DistributionSystem) -> dict:
    from .dc_opf import solve_dc_opf_from_components

    t0 = time.perf_counter()
    result = solve_dc_opf_from_components(
        system,
        include_solar_generators=True,
        include_battery_generators=True,
        include_loads=True,
    )
    elapsed = time.perf_counter() - t0

    # Source power from grid generators
    grid_gens = {
        k: v for k, v in result.generator_dispatch_w.items() if k.startswith("grid:")
    }
    solar_gens = {
        k: v for k, v in result.generator_dispatch_w.items() if k.startswith("solar:")
    }
    battery_gens = {
        k: v for k, v in result.generator_dispatch_w.items() if k.startswith("battery:")
    }
    # Source power = grid import (what enters from the source bus)
    # Solar injects at load buses, not the source bus
    source_p = sum(grid_gens.values())

    return {
        "solver": "DC OPF",
        "success": result.success,
        "message": result.message,
        "elapsed": elapsed,
        "iterations": result.iterations,
        "source_p": source_p,
        "source_q": 0.0,
        "grid_import": sum(grid_gens.values()),
        "solar_dispatch": sum(solar_gens.values()),
        "battery_dispatch": sum(battery_gens.values()),
        "total_gen": sum(result.generator_dispatch_w.values()),
        "objective": result.objective,
        "result": result,
    }


def _run_ldf(system: DistributionSystem) -> dict:
    from .lindistflow import solve_lindistflow
    import networkx as nx

    t0 = time.perf_counter()
    try:
        result = solve_lindistflow(system)
    except (nx.NetworkXUnfeasible, nx.NetworkXError, TypeError) as exc:
        elapsed = time.perf_counter() - t0
        return {
            "solver": "LinDistFlow",
            "success": False,
            "message": f"Requires radial topology: {exc}",
            "elapsed": elapsed,
            "source_p": 0.0,
            "source_q": 0.0,
            "v_min": 0.0,
            "v_max": 0.0,
            "v_mean": 0.0,
            "result": None,
        }
    elapsed = time.perf_counter() - t0

    source_p = sum(float(v) for v in result.p_net_w.values())
    source_q = sum(float(v) for v in result.q_net_var.values())

    v_vals = list(result.voltage_v.values())

    return {
        "solver": "LinDistFlow",
        "success": result.success,
        "message": result.message,
        "elapsed": elapsed,
        "iterations": 0,
        "source_p": source_p,
        "source_q": source_q,
        "v_min": min(v_vals) if v_vals else 0.0,
        "v_max": max(v_vals) if v_vals else 0.0,
        "result": result,
    }


def _run_pf(system: DistributionSystem) -> dict:
    from .ac_pf import solve_ac_power_flow_from_components

    t0 = time.perf_counter()
    result = solve_ac_power_flow_from_components(
        system,
        include_loads=True,
        include_solar=True,
        include_capacitor=True,
    )
    elapsed = time.perf_counter() - t0

    src_bus = system.get_source_bus().name
    idx_map = result.ybus_result.index_to_label
    v = result.voltage
    ybus = result.ybus_result.ybus
    s = v * np.conj(ybus @ v)
    src_idx = [i for i, lbl in enumerate(idx_map) if lbl[0] == src_bus]
    source_p = sum(s[i].real for i in src_idx)
    source_q = sum(s[i].imag for i in src_idx)

    v_mag = np.abs(v)

    return {
        "solver": "AC PF",
        "success": result.success,
        "message": result.message,
        "elapsed": elapsed,
        "iterations": result.iterations,
        "source_p": source_p,
        "source_q": source_q,
        "v_min": float(np.min(v_mag)),
        "v_max": float(np.max(v_mag)),
        "max_mismatch_pu": result.max_mismatch_pu,
        "result": result,
    }


SOLVER_MAP = {
    Solver.ac: _run_ac,
    Solver.pf: _run_pf,
    Solver.dc: _run_dc,
    Solver.ldf: _run_ldf,
}


# ── commands ─────────────────────────────────────────────────────────────


@app.command()
def info(
    model: Path = typer.Argument(..., help="Path to GDM distribution system JSON"),
):
    """Show system topology and component summary."""
    system = _load_system(model)

    src_bus = system.get_source_bus()
    src_phases = [_phase_name(p) for p in src_bus.phases if p != Phase.N]

    # Count components
    from gdm.systems.distribution.components import (
        DistributionBus,
        DistributionLoad,
        DistributionSolar,
        DistributionTransformer,
    )

    buses = list(system.get_components(DistributionBus))
    loads = list(system.get_components(DistributionLoad))
    solars = list(system.get_components(DistributionSolar))
    transformers = list(system.get_components(DistributionTransformer))

    total_load_w = 0.0
    total_load_var = 0.0
    for ld in loads:
        for pl in ld.equipment.phase_loads:
            total_load_w += float(pl.real_power.to("watt").magnitude)
            total_load_var += float(pl.reactive_power.to("var").magnitude)

    total_solar_w = 0.0
    total_solar_rated_w = 0.0
    for s in solars:
        total_solar_w += float(s.active_power.to("watt").magnitude)
        total_solar_rated_w += float(s.equipment.rated_power.to("watt").magnitude)

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]{model.name}[/]\n[dim]{model.resolve()}[/]",
            title="[bold cyan]⚡ GDM-Flow System Info[/]",
            border_style="cyan",
        )
    )

    # Topology table
    topo = Table(title="Topology", show_header=False, border_style="dim")
    topo.add_column("Key", style="bold")
    topo.add_column("Value")
    topo.add_row("Source Bus", f"{src_bus.name}")
    topo.add_row("Source Phases", ", ".join(src_phases))
    topo.add_row("Buses", str(len(buses)))
    topo.add_row("Transformers", str(len(transformers)))
    topo.add_row("Loads", str(len(loads)))
    topo.add_row("Solar PV", str(len(solars)))
    console.print(topo)

    # Power summary
    pwr = Table(title="Power Summary", border_style="dim")
    pwr.add_column("Metric", style="bold")
    pwr.add_column("Value", justify="right")
    pwr.add_row("Total Load (P)", _fmt_w(total_load_w))
    pwr.add_row("Total Load (Q)", _fmt_var(total_load_var))
    pwr.add_row("Solar Active", _fmt_w(total_solar_w))
    pwr.add_row("Solar Rated", _fmt_w(total_solar_rated_w))
    pwr.add_row("Net Demand", _fmt_w(total_load_w - total_solar_w))
    console.print(pwr)

    # Bus details
    bus_tbl = Table(title="Bus Details", border_style="dim")
    bus_tbl.add_column("Bus", style="bold")
    bus_tbl.add_column("Phases")
    bus_tbl.add_column("Rated V", justify="right")
    bus_tbl.add_column("Type")
    for b in sorted(buses, key=lambda x: x.name):
        phases = ", ".join(_phase_name(p) for p in b.phases if p != Phase.N)
        v_str = f"{float(b.rated_voltage.to('volt').magnitude):.0f} V"
        btype = "Source" if b.name == src_bus.name else "Load"
        bus_tbl.add_row(b.name, phases, v_str, btype)
    console.print(bus_tbl)
    console.print()


@app.command()
def run(
    model: Path = typer.Argument(..., help="Path to GDM distribution system JSON"),
    solver: list[Solver] = typer.Option(
        [Solver.ac], "--solver", "-s", help="Solver(s) to run (ac, pf, dc, ldf)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed results"
    ),
):
    """Run one or more OPF solvers on a distribution system model."""
    system = _load_system(model)

    results = []
    for s in solver:
        solver_name = {
            "ac": "AC OPF",
            "pf": "AC PF",
            "dc": "DC OPF",
            "ldf": "LinDistFlow",
        }[s.value]
        with console.status(f"[cyan]Running {solver_name}…"):
            r = SOLVER_MAP[s](system)
        results.append(r)

    # Summary table
    console.print()
    tbl = Table(
        title="[bold]OPF Results[/]",
        border_style="cyan",
        show_lines=True,
    )
    tbl.add_column("Solver", style="bold")
    tbl.add_column("Status", justify="center")
    tbl.add_column("Source P", justify="right")
    tbl.add_column("Source Q", justify="right")
    tbl.add_column("Time", justify="right")
    tbl.add_column("Iterations", justify="right")

    for r in results:
        tbl.add_row(
            r["solver"],
            _success_badge(r["success"]),
            _fmt_w(r["source_p"]),
            _fmt_var(r["source_q"]),
            f"{r['elapsed'] * 1000:.0f} ms",
            str(r["iterations"]),
        )
    console.print(tbl)

    # DC dispatch details
    if verbose:
        for r in results:
            if r["solver"] == "DC OPF":
                _print_dc_dispatch(r)
            if r["solver"] == "AC OPF":
                _print_ac_voltages(r, system)

    console.print()


@app.command()
def compare(
    model: Path = typer.Argument(..., help="Path to GDM distribution system JSON"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Export comparison to HTML (requires plotly)"
    ),
):
    """Run all solvers and compare results side-by-side."""
    system = _load_system(model)

    results = {}
    for s in [Solver.ac, Solver.pf, Solver.dc, Solver.ldf]:
        solver_name = {
            "ac": "AC OPF",
            "pf": "AC PF",
            "dc": "DC OPF",
            "ldf": "LinDistFlow",
        }[s.value]
        with console.status(f"[cyan]Running {solver_name}…"):
            if s == Solver.ldf:
                results[s.value] = _run_ldf(system)
            else:
                results[s.value] = SOLVER_MAP[s](system)

    ac_r = results["ac"]
    pf_r = results["pf"]
    dc_r = results["dc"]
    ldf_r = results["ldf"]

    # Comparison table
    console.print()
    tbl = Table(
        title="[bold]⚡ Solver Comparison[/]",
        border_style="cyan",
        show_lines=True,
    )
    tbl.add_column("Metric", style="bold")
    tbl.add_column("AC OPF", justify="right", style="green")
    tbl.add_column("AC PF", justify="right", style="magenta")
    tbl.add_column("DC OPF", justify="right", style="yellow")
    tbl.add_column("LinDistFlow", justify="right", style="blue")

    tbl.add_row(
        "Status",
        _success_badge(ac_r["success"]),
        _success_badge(pf_r["success"]),
        _success_badge(dc_r["success"]),
        _success_badge(ldf_r["success"]),
    )
    tbl.add_row(
        "Source P",
        _fmt_w(ac_r["source_p"]),
        _fmt_w(pf_r["source_p"]),
        _fmt_w(dc_r["source_p"]),
        _fmt_w(ldf_r["source_p"]),
    )
    tbl.add_row(
        "Source Q",
        _fmt_var(ac_r["source_q"]),
        _fmt_var(pf_r["source_q"]),
        "—",
        _fmt_var(ldf_r["source_q"]),
    )
    tbl.add_row(
        "Time",
        f"{ac_r['elapsed'] * 1000:.0f} ms",
        f"{pf_r['elapsed'] * 1000:.0f} ms",
        f"{dc_r['elapsed'] * 1000:.0f} ms",
        f"{ldf_r['elapsed'] * 1000:.0f} ms",
    )
    tbl.add_row(
        "Iterations",
        str(ac_r["iterations"]),
        str(pf_r["iterations"]),
        str(dc_r["iterations"]),
        "—",
    )

    # Voltage comparison rows
    vstats = {
        name: _voltage_pu_stats(r, system)
        for name, r in [("ac", ac_r), ("pf", pf_r), ("dc", dc_r), ("ldf", ldf_r)]
    }

    def _vfmt(name: str, key: str) -> str:
        s = vstats.get(name)
        if s is None:
            return "—"
        return f"{s[key]:.4f}"

    def _vint(name: str, key: str) -> str:
        s = vstats.get(name)
        if s is None:
            return "—"
        return str(s[key])

    tbl.add_row(
        "V_min (pu)",
        _vfmt("ac", "v_min_pu"),
        _vfmt("pf", "v_min_pu"),
        _vfmt("dc", "v_min_pu"),
        _vfmt("ldf", "v_min_pu"),
    )
    tbl.add_row(
        "V_max (pu)",
        _vfmt("ac", "v_max_pu"),
        _vfmt("pf", "v_max_pu"),
        _vfmt("dc", "v_max_pu"),
        _vfmt("ldf", "v_max_pu"),
    )
    tbl.add_row(
        "V_mean (pu)",
        _vfmt("ac", "v_mean_pu"),
        _vfmt("pf", "v_mean_pu"),
        _vfmt("dc", "v_mean_pu"),
        _vfmt("ldf", "v_mean_pu"),
    )
    tbl.add_row(
        "V < 0.95 pu",
        _vint("ac", "n_under_095"),
        _vint("pf", "n_under_095"),
        _vint("dc", "n_under_095"),
        _vint("ldf", "n_under_095"),
    )

    console.print(tbl)

    # Per-phase loading table
    from gdm.systems.distribution.components.distribution_load import DistributionLoad

    phase_load_p: dict[str, float] = {}
    phase_load_q: dict[str, float] = {}
    for ld in system.get_components(DistributionLoad):
        for ph, pl in zip(ld.phases, ld.equipment.phase_loads):
            pname = _phase_name(ph)
            if pname == "N":
                continue
            phase_load_p[pname] = phase_load_p.get(pname, 0.0) + float(
                pl.real_power.to("watt").magnitude
            )
            phase_load_q[pname] = phase_load_q.get(pname, 0.0) + float(
                pl.reactive_power.to("var").magnitude
            )

    phase_list = sorted(phase_load_p.keys())
    if phase_list:
        # Build per-phase source injection from each solver
        def _source_per_phase(r: dict) -> dict[str, tuple[float, float]]:
            """Return {phase: (P_w, Q_var)} for source bus injection."""
            result = r.get("result")
            if result is None:
                return {}
            out: dict[str, tuple[float, float]] = {}
            if hasattr(result, "voltage") and hasattr(result, "ybus_result"):
                src = system.get_source_bus().name
                v = result.voltage
                ybus = result.ybus_result.ybus
                s = v * np.conj(ybus @ v)
                for i, lbl in enumerate(result.ybus_result.index_to_label):
                    if lbl[0] == src and lbl[1] != "N":
                        out[lbl[1]] = (float(s[i].real), float(s[i].imag))
            elif hasattr(result, "p_net_w"):
                # LDF: sum net injections by phase (source not in p_net_w)
                for (bus, ph), p in result.p_net_w.items():
                    if ph == "N":
                        continue
                    q = result.q_net_var.get((bus, ph), 0.0)
                    prev = out.get(ph, (0.0, 0.0))
                    out[ph] = (prev[0] + float(p), prev[1] + float(q))
            return out

        src_phases = {
            "AC OPF": _source_per_phase(ac_r),
            "AC PF": _source_per_phase(pf_r),
            "LinDistFlow": _source_per_phase(ldf_r),
        }

        # DC OPF per-phase from grid generator dispatch
        dc_phase: dict[str, tuple[float, float]] = {}
        if dc_r.get("result") is not None:
            for k, v in dc_r["result"].generator_dispatch_w.items():
                if not k.startswith("grid:"):
                    continue
                # key format: "grid:bus:phase" e.g. "grid:150:A"
                parts = k.split(":")
                ph = parts[-1]
                if ph != "N":
                    prev = dc_phase.get(ph, (0.0, 0.0))
                    dc_phase[ph] = (prev[0] + v, prev[1])
        src_phases["DC OPF"] = dc_phase

        ltbl = Table(
            title="[bold]⚡ Per-Phase Loading & Source Injection[/]",
            border_style="cyan",
            show_lines=True,
        )
        ltbl.add_column("Phase", style="bold")
        ltbl.add_column("Load P", justify="right")
        ltbl.add_column("Load Q", justify="right")
        ltbl.add_column("AC OPF Src P", justify="right", style="green")
        ltbl.add_column("AC PF Src P", justify="right", style="magenta")
        ltbl.add_column("DC OPF Src P", justify="right", style="yellow")
        ltbl.add_column("LDF Src P", justify="right", style="blue")

        total_lp = 0.0
        total_lq = 0.0
        solver_totals: dict[str, float] = {n: 0.0 for n in src_phases}

        for ph in phase_list:
            lp = phase_load_p.get(ph, 0.0)
            lq = phase_load_q.get(ph, 0.0)
            total_lp += lp
            total_lq += lq

            cols = []
            for sname in ["AC OPF", "AC PF", "DC OPF", "LinDistFlow"]:
                sp = src_phases.get(sname, {}).get(ph)
                if sp is not None:
                    solver_totals[sname] += sp[0]
                    cols.append(_fmt_w(sp[0]))
                else:
                    cols.append("—")

            ltbl.add_row(ph, _fmt_w(lp), _fmt_var(lq), *cols)

        ltbl.add_row(
            "[bold]Total[/]",
            f"[bold]{_fmt_w(total_lp)}[/]",
            f"[bold]{_fmt_var(total_lq)}[/]",
            *[
                f"[bold]{_fmt_w(solver_totals[n])}[/]"
                for n in ["AC OPF", "AC PF", "DC OPF", "LinDistFlow"]
            ],
        )

        console.print()
        console.print(ltbl)

    # Dispatch breakdown for DC
    _print_dc_dispatch(dc_r)

    # Agreement check
    vals = [ac_r["source_p"], pf_r["source_p"], dc_r["source_p"], ldf_r["source_p"]]
    max_diff = max(vals) - min(vals)
    console.print()
    if max_diff < 100:
        console.print(
            Panel(
                f"[green]All solvers agree within {max_diff:.1f} W[/]",
                border_style="green",
                title="[bold green]✓ Agreement[/]",
            )
        )
    else:
        console.print(
            Panel(
                f"[yellow]Max disagreement: {_fmt_w(max_diff)}[/]",
                border_style="yellow",
                title="[bold yellow]⚠ Disagreement[/]",
            )
        )

    # Optionally generate HTML
    if output is not None:
        _export_html(system, ac_r, dc_r, ldf_r, output, pf_r=pf_r)

    console.print()


@app.command()
def plot(
    model: Path = typer.Argument(..., help="Path to GDM distribution system JSON"),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HTML path (default: <model>_dashboard.html)",
    ),
    solver: list[Solver] = typer.Option(
        [Solver.ac, Solver.pf, Solver.dc, Solver.ldf],
        "--solver",
        "-s",
        help="Solver(s) to include (ac, pf, dc, ldf)",
    ),
):
    """Generate an interactive analysis dashboard with voltage profiles, power flows, losses, and equipment state."""
    system = _load_system(model)

    solver_names = {
        "ac": "AC OPF",
        "pf": "AC PF",
        "dc": "DC OPF",
        "ldf": "LinDistFlow",
    }
    results = {}
    for s in solver:
        name = solver_names[s.value]
        with console.status(f"[cyan]Running {name}…"):
            try:
                results[name] = SOLVER_MAP[s](system)
            except Exception as exc:
                results[name] = {
                    "solver": name,
                    "success": False,
                    "message": str(exc),
                    "elapsed": 0.0,
                    "iterations": 0,
                    "source_p": 0.0,
                    "source_q": 0.0,
                    "result": None,
                }
                err_console.print(f"[yellow]  {name}: failed — {exc!r}[/]")

    if output is None:
        output = model.parent / f"{model.stem}_dashboard.html"

    from .dashboard import generate_dashboard

    with console.status("[cyan]Generating dashboard…"):
        generate_dashboard(system, results, output, model_name=model.stem)

    console.print()
    console.print(
        Panel(
            f"[green]Dashboard written to [bold]{output}[/][/]",
            border_style="green",
            title="[bold green]✓ Dashboard[/]",
        )
    )
    console.print()


@app.command()
def export(
    model: Path = typer.Argument(..., help="Path to GDM distribution system JSON"),
    db: Path = typer.Option(..., "--db", help="SQLite database path to create/update"),
    solver: list[Solver] = typer.Option(
        [Solver.ac, Solver.pf, Solver.dc, Solver.ldf],
        "--solver",
        "-s",
        help="Solver(s) to export",
    ),
):
    """Run solvers and export results to a SQLite database."""
    system = _load_system(model)

    ac_result = None
    pf_result = None
    dc_result = None
    ldf_result = None

    for s in solver:
        solver_name = {
            "ac": "AC OPF",
            "pf": "AC PF",
            "dc": "DC OPF",
            "ldf": "LinDistFlow",
        }[s.value]
        with console.status(f"[cyan]Running {solver_name}…"):
            r = SOLVER_MAP[s](system)

        if s == Solver.ac:
            ac_result = r["result"]
        elif s == Solver.pf:
            pf_result = r["result"]
        elif s == Solver.dc:
            dc_result = r["result"]
        elif s == Solver.ldf:
            ldf_result = r["result"]

        status = _success_badge(r["success"])
        console.print(f"  {solver_name}: {status}  ({_fmt_w(r['source_p'])})")

    from .sqlite_export import export_all_results_to_sqlite

    node_voltage_limits_v = _build_node_voltage_limits_v(system)
    ldf_loading_limits_va = _build_lindistflow_loading_limits_va(system)
    ac_branch_loading_va = {}
    ac_branch_loading_limits_va = {}
    ac_branch_flow_w_var = {}
    pf_branch_loading_va = {}
    pf_branch_loading_limits_va = {}
    pf_branch_flow_w_var = {}
    dc_branch_loading_va = {}
    dc_branch_loading_limits_va = {}
    dc_branch_flow_w_var = {}
    if (
        ac_result is not None
        and hasattr(ac_result, "voltage")
        and hasattr(ac_result, "ybus_result")
        and hasattr(ac_result.ybus_result, "index_to_label")
    ):
        (
            ac_branch_loading_va,
            ac_branch_loading_limits_va,
            ac_branch_flow_w_var,
        ) = _build_ac_branch_loading_from_result(system, ac_result)
    if (
        pf_result is not None
        and hasattr(pf_result, "voltage")
        and hasattr(pf_result, "ybus_result")
        and hasattr(pf_result.ybus_result, "index_to_label")
    ):
        (
            pf_branch_loading_va,
            pf_branch_loading_limits_va,
            pf_branch_flow_w_var,
        ) = _build_ac_branch_loading_from_result(system, pf_result)
    if (
        dc_result is not None
        and hasattr(dc_result, "theta_rad")
        and isinstance(getattr(dc_result, "theta_rad"), dict)
    ):
        (
            dc_branch_loading_va,
            dc_branch_loading_limits_va,
            dc_branch_flow_w_var,
        ) = _build_dc_branch_loading_from_result(system, dc_result)

    with console.status("[cyan]Writing SQLite…"):
        export_all_results_to_sqlite(
            db_path=str(db),
            ac_result=ac_result,
            pf_result=pf_result,
            dc_result=dc_result,
            lindistflow_result=ldf_result,
            ac_voltage_limits_v=node_voltage_limits_v,
            ac_branch_loading_va=ac_branch_loading_va,
            ac_branch_loading_limits_va=ac_branch_loading_limits_va,
            ac_branch_flow_w_var=ac_branch_flow_w_var,
            pf_voltage_limits_v=node_voltage_limits_v,
            pf_branch_loading_va=pf_branch_loading_va,
            pf_branch_loading_limits_va=pf_branch_loading_limits_va,
            pf_branch_flow_w_var=pf_branch_flow_w_var,
            dc_branch_loading_va=dc_branch_loading_va,
            dc_branch_loading_limits_va=dc_branch_loading_limits_va,
            dc_branch_flow_w_var=dc_branch_flow_w_var,
            lindistflow_voltage_limits_v=node_voltage_limits_v,
            lindistflow_loading_limits_va=ldf_loading_limits_va,
        )

    console.print()
    console.print(
        Panel(
            f"[green]Database written to [bold]{db}[/][/]",
            border_style="green",
            title="[bold green]✓ Export Complete[/]",
        )
    )
    console.print()


@app.command("report-overvoltage")
def report_overvoltage(
    db: Path = typer.Option(..., "--db", help="SQLite database path"),
    solver: Solver = typer.Option(
        Solver.ac,
        "--solver",
        "-s",
        help="Solver result set to inspect for voltage violations (ac or ldf)",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Specific run_id to inspect. Defaults to latest run for selected solver.",
    ),
):
    """Print overvoltage/undervoltage violations from exported results."""
    implementation = {
        "ac": "ac_opf",
        "pf": "ac_pf",
        "dc": "dc_opf",
        "ldf": "lindistflow",
    }[solver.value]
    resolved_run_id, rows, has_columns = _read_overvoltage_rows(
        str(db), implementation, run_id
    )

    if not has_columns:
        err_console.print(
            "[yellow]This database does not include voltage limit columns. Re-run `gdm-flow export` to add them.[/]"
        )
        raise typer.Exit(1)

    if resolved_run_id is None:
        console.print(
            Panel(
                f"[yellow]No {implementation} run found in {db}[/]",
                border_style="yellow",
                title="No Run",
            )
        )
        return

    if not rows:
        console.print(
            Panel(
                f"[green]No voltage violations for {implementation} run [bold]{resolved_run_id}[/].[/]",
                border_style="green",
                title="No Overvoltage",
            )
        )
        return

    tbl = Table(
        title=f"Voltage Violations ({implementation}, run={resolved_run_id})",
        border_style="red",
    )
    tbl.add_column("Bus", style="bold")
    tbl.add_column("Phase")
    tbl.add_column("Voltage (V)", justify="right")
    tbl.add_column("Min (V)", justify="right")
    tbl.add_column("Max (V)", justify="right")
    tbl.add_column("Violation", justify="right")

    for bus_name, phase, voltage, v_min, v_max in rows:
        if v_max is not None and voltage > v_max:
            delta = voltage - v_max
            violation = f"+{delta:.2f} V"
        elif v_min is not None and voltage < v_min:
            delta = v_min - voltage
            violation = f"-{delta:.2f} V"
        else:
            violation = "0.00 V"
        tbl.add_row(
            str(bus_name),
            str(phase),
            f"{float(voltage):.2f}",
            "-" if v_min is None else f"{float(v_min):.2f}",
            "-" if v_max is None else f"{float(v_max):.2f}",
            violation,
        )

    console.print()
    console.print(tbl)
    console.print()


@app.command("report-overload")
def report_overload(
    db: Path = typer.Option(..., "--db", help="SQLite database path"),
    solver: Solver = typer.Option(
        Solver.ldf,
        "--solver",
        "-s",
        help="Solver result set to inspect for overloads (ac or ldf)",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Specific run_id to inspect. Defaults to latest one for selected solver.",
    ),
    dc_severity_only: bool = typer.Option(
        True,
        "--dc-severity-only/--no-dc-severity-only",
        help=(
            "For DC reports, show ranked severity instead of percentage magnitudes "
            "(recommended due to DC approximation)."
        ),
    ),
):
    """Print branch overload violations from exported AC OPF or LinDistFlow results."""
    implementation = {
        "ac": "ac_opf",
        "pf": "ac_pf",
        "dc": "dc_opf",
        "ldf": "lindistflow",
    }[solver.value]
    resolved_run_id, rows, has_columns = _read_overload_rows(
        str(db), implementation, run_id
    )

    if not has_columns:
        err_console.print(
            "[yellow]This database does not include loading limit columns. Re-run `gdm-flow export` to add them.[/]"
        )
        raise typer.Exit(1)

    if resolved_run_id is None:
        console.print(
            Panel(
                f"[yellow]No {implementation} run found in {db}[/]",
                border_style="yellow",
                title="No Run",
            )
        )
        return

    if not rows:
        console.print(
            Panel(
                f"[green]No branch overloads for {implementation} run [bold]{resolved_run_id}[/].[/]",
                border_style="green",
                title="No Overload",
            )
        )
        return

    title_suffix = " (DC Approximation)" if implementation == "dc_opf" else ""
    if implementation == "dc_opf" and dc_severity_only:
        tbl = Table(
            title=(
                f"Branch Overloads ({implementation}{title_suffix}, run={resolved_run_id}, "
                "Ranked Severity)"
            ),
            border_style="red",
        )
        tbl.add_column("Rank", justify="right")
        tbl.add_column("Branch", style="bold")
        tbl.add_column("Phase")
        tbl.add_column("Severity", justify="right")
        tbl.add_column("Band", justify="center")

        for idx, row in enumerate(rows, start=1):
            (
                branch_name,
                phase,
                _p_flow_w,
                _q_flow_var,
                _loading_va,
                _loading_limit_va,
                ratio,
            ) = row
            ratio_f = float(ratio)
            if ratio_f >= 2.0:
                band = "[red]Critical[/]"
            elif ratio_f >= 1.4:
                band = "[yellow]High[/]"
            else:
                band = "[cyan]Moderate[/]"
            tbl.add_row(
                str(idx),
                str(branch_name),
                str(phase),
                f"{ratio_f:.2f}x",
                band,
            )
    else:
        tbl = Table(
            title=f"Branch Overloads ({implementation}{title_suffix}, run={resolved_run_id})",
            border_style="red",
        )
        tbl.add_column("Branch", style="bold")
        tbl.add_column("Phase")
        tbl.add_column("P (W)", justify="right")
        tbl.add_column("Q (var)", justify="right")
        tbl.add_column("|S| (VA)", justify="right")
        tbl.add_column("Limit (VA)", justify="right")
        tbl.add_column("Loading", justify="right")

        for (
            branch_name,
            phase,
            p_flow_w,
            q_flow_var,
            loading_va,
            loading_limit_va,
            ratio,
        ) in rows:
            tbl.add_row(
                str(branch_name),
                str(phase),
                f"{float(p_flow_w):.2f}",
                f"{float(q_flow_var):.2f}",
                f"{float(loading_va):.2f}",
                f"{float(loading_limit_va):.2f}",
                f"{100.0 * float(ratio):.1f}%",
            )

    console.print()
    console.print(tbl)
    if implementation == "dc_opf":
        console.print(
            "[yellow]Note:[/] DC overload values are post-processed approximations from angle differences (P-only proxy)."
        )
    console.print()


@app.command("db-schema")
def db_schema(
    db: Path = typer.Option(..., "--db", help="SQLite database path"),
    include_internal: bool = typer.Option(
        False,
        "--include-internal",
        help="Include sqlite_* internal tables",
    ),
):
    """Print SQLite table/column schema for quick inspection."""
    if not db.exists():
        err_console.print(f"[red]Error:[/] database not found: {db}")
        raise typer.Exit(1)

    schema = _read_db_schema(str(db), include_internal=include_internal)
    if not schema:
        console.print(
            Panel(
                f"[yellow]No tables found in {db}[/]",
                border_style="yellow",
                title="Empty Schema",
            )
        )
        return

    tbl = Table(title=f"SQLite Schema ({db})", border_style="cyan", show_lines=True)
    tbl.add_column("Table", style="bold")
    tbl.add_column("Columns")

    for table_name, columns in schema:
        tbl.add_row(table_name, ", ".join(columns))

    console.print()
    console.print(tbl)
    console.print()


@app.command("ts-info")
def ts_info(
    model: Path = typer.Argument(..., help="Path to GDM distribution system JSON"),
):
    """Show time series data availability and metadata for each component type."""
    system = _load_system(model)

    from .time_series import (
        get_time_series_length,
        get_time_series_resolution,
        has_time_series_data,
        list_component_time_series,
    )

    if not has_time_series_data(system):
        console.print(
            Panel(
                "[yellow]No time series data found on any load, solar, or battery component.[/]",
                border_style="yellow",
                title="Time Series Info",
            )
        )
        return

    ts_map = list_component_time_series(system)

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Time Series Summary[/]\n"
            f"  Length: [green]{get_time_series_length(system):,}[/] timesteps\n"
            f"  Resolution: [green]{get_time_series_resolution(system)}[/]",
            border_style="cyan",
            title="Time Series Info",
        )
    )

    for comp_type, entries in ts_map.items():
        tbl = Table(
            title=f"{comp_type} Time Series",
            border_style="dim",
            show_lines=False,
        )
        tbl.add_column("Component", style="bold")
        tbl.add_column("Variable", style="cyan")
        tbl.add_column("Length", justify="right")
        tbl.add_column("Resolution", justify="right")
        tbl.add_column("Start", style="dim")
        tbl.add_column("Units", style="dim")

        for info in entries:
            res_str = str(info.resolution) if info.resolution else "—"
            start_str = str(info.initial_timestamp) if info.initial_timestamp else "—"
            tbl.add_row(
                info.component_name,
                info.variable_name,
                f"{info.length:,}",
                res_str,
                start_str,
                info.units,
            )

        console.print()
        console.print(tbl)

    console.print()


@app.command()
def qsts(
    model: Path = typer.Argument(..., help="Path to GDM distribution system JSON"),
    solver: Solver = typer.Option(
        Solver.ldf, "--solver", "-s", help="Solver to use (ac, pf, dc, ldf)"
    ),
    start: int = typer.Option(0, "--start", help="First timestep index"),
    end: int = typer.Option(
        None, "--end", help="Last timestep index (default: all available)"
    ),
    step: int = typer.Option(1, "--step", help="Timestep stride"),
    db: Optional[Path] = typer.Option(
        None, "--db", help="SQLite database path for streaming results"
    ),
):
    """Run Quasi-Static Time Series simulation over a time horizon."""
    system = _load_system(model)

    from .time_series import (
        get_time_series_length,
        get_time_series_resolution,
        has_time_series_data,
        run_qsts,
    )

    if not has_time_series_data(system):
        err_console.print(
            "[red]Error:[/] model has no time series data. Use [bold]ts-info[/] to check."
        )
        raise typer.Exit(1)

    ts_len = get_time_series_length(system)
    resolution = get_time_series_resolution(system)
    actual_end = min(end, ts_len) if end is not None else ts_len
    timestep_range = range(start, actual_end, step)
    num_steps = len(timestep_range)

    console.print()
    console.print(
        Panel(
            f"[bold cyan]QSTS Simulation[/]\n"
            f"  Solver: [green]{solver.value}[/]\n"
            f"  Timesteps: [green]{num_steps:,}[/] ({start}–{actual_end - 1}, step={step})\n"
            f"  Resolution: [green]{resolution}[/]",
            border_style="cyan",
        )
    )

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Simulating…", total=num_steps)

        def _progress_cb(done: int, total: int) -> None:
            progress.update(task, completed=done)

        import time as _time

        t0 = _time.perf_counter()
        summary = run_qsts(
            system,
            solver.value,
            timestep_range,
            db_path=str(db) if db else None,
            progress_callback=_progress_cb,
        )
        elapsed = _time.perf_counter() - t0

    console.print()
    tbl = Table(title="QSTS Summary", border_style="cyan")
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Timesteps", f"{summary.num_timesteps:,}")
    tbl.add_row("Converged", f"{summary.num_converged:,}")
    tbl.add_row(
        "Convergence Rate",
        f"{100.0 * summary.num_converged / max(summary.num_timesteps, 1):.1f}%",
    )
    tbl.add_row("Elapsed", f"{elapsed:.1f}s")
    tbl.add_row(
        "Per Timestep",
        f"{elapsed / max(summary.num_timesteps, 1) * 1000:.1f} ms",
    )
    if summary.db_path:
        tbl.add_row("Database", summary.db_path)
    if summary.run_id:
        tbl.add_row("Run ID", summary.run_id)
    console.print(tbl)
    console.print()


@app.command()
def multiperiod(
    model: Path = typer.Argument(..., help="Path to GDM distribution system JSON"),
    solver: Solver = typer.Option(
        Solver.dc, "--solver", "-s", help="Solver (dc or ldf)"
    ),
    start: int = typer.Option(0, "--start", help="First timestep index"),
    end: int = typer.Option(
        None, "--end", help="Last timestep index (default: 96 = 24h at 15min)"
    ),
    step: int = typer.Option(1, "--step", help="Timestep stride"),
    ramp: Optional[float] = typer.Option(
        None, "--ramp", help="Generator ramp limit in watts (DC OPF only)"
    ),
    db: Optional[Path] = typer.Option(
        None, "--db", help="SQLite database path for results"
    ),
):
    """Run multi-period OPF with battery SOC coupling across the time horizon."""
    if solver.value not in ("dc", "ldf"):
        err_console.print(
            "[red]Error:[/] multi-period only supports [bold]dc[/] and [bold]ldf[/] solvers."
        )
        raise typer.Exit(1)

    system = _load_system(model)

    from .time_series import (
        get_time_series_length,
        get_time_series_resolution,
        has_time_series_data,
    )

    if not has_time_series_data(system):
        err_console.print(
            "[red]Error:[/] model has no time series data. Use [bold]ts-info[/] to check."
        )
        raise typer.Exit(1)

    ts_len = get_time_series_length(system)
    resolution = get_time_series_resolution(system)
    default_end = min(96, ts_len)
    actual_end = min(end, ts_len) if end is not None else default_end
    timestep_range = range(start, actual_end, step)
    num_steps = len(timestep_range)

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Multi-Period OPF[/]\n"
            f"  Solver: [green]{solver.value.upper()}[/]\n"
            f"  Timesteps: [green]{num_steps:,}[/] ({start}–{actual_end - 1}, step={step})\n"
            f"  Resolution: [green]{resolution}[/]"
            + (f"\n  Ramp Limit: [green]{_fmt_w(ramp)}[/]" if ramp else ""),
            border_style="cyan",
        )
    )

    import time as _time

    t0 = _time.perf_counter()

    if solver.value == "dc":
        from .dc_opf import build_dc_generators_from_components
        from .multiperiod import solve_multiperiod_dc_opf

        with console.status("[cyan]Solving multi-period DC OPF…"):
            generators = build_dc_generators_from_components(system)
            result = solve_multiperiod_dc_opf(
                system,
                generators=generators,
                timestep_range=timestep_range,
                ramp_limit_w=ramp,
                db_path=str(db) if db else None,
            )
    else:
        from .multiperiod import solve_multiperiod_lindistflow

        with console.status("[cyan]Solving multi-period LinDistFlow…"):
            result = solve_multiperiod_lindistflow(
                system,
                timestep_range=timestep_range,
                db_path=str(db) if db else None,
            )

    elapsed = _time.perf_counter() - t0

    console.print()
    tbl = Table(title="Multi-Period Results", border_style="cyan")
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Status", _success_badge(result.success))
    tbl.add_row("Solver", result.solver.upper())
    tbl.add_row("Timesteps", f"{result.num_timesteps:,}")
    tbl.add_row("Objective", f"{result.objective:,.2f}")
    tbl.add_row("Elapsed", f"{elapsed:.2f}s")
    if result.battery_soc:
        tbl.add_row("Batteries", str(len(result.battery_soc)))
    if result.db_path:
        tbl.add_row("Database", result.db_path)
    if result.run_id:
        tbl.add_row("Run ID", result.run_id)
    console.print(tbl)

    # Show battery SOC summary
    if result.battery_soc:
        console.print()
        btbl = Table(title="Battery SOC Summary", border_style="dim")
        btbl.add_column("Battery", style="bold")
        btbl.add_column("Initial", justify="right")
        btbl.add_column("Final", justify="right")
        btbl.add_column("Min", justify="right")
        btbl.add_column("Max", justify="right")
        for name, soc_list in result.battery_soc.items():
            if soc_list:
                btbl.add_row(
                    name,
                    f"{soc_list[0]:.3f}",
                    f"{soc_list[-1]:.3f}",
                    f"{min(soc_list):.3f}",
                    f"{max(soc_list):.3f}",
                )
        console.print(btbl)

    if not result.success:
        console.print()
        err_console.print(f"[yellow]Warning:[/] {result.message}")

    console.print()


@app.command("plot-ts")
def plot_ts(  # pragma: no cover
    db: Path = typer.Argument(
        ..., help="SQLite database with QSTS/multi-period results"
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Specific run ID (default: latest)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output HTML path (default: <db>_ts.html)"
    ),
):
    """Generate time series plots from QSTS or multi-period results in SQLite."""
    if not db.exists():
        err_console.print(f"[red]Error:[/] database not found: {db}")
        raise typer.Exit(1)

    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError:
        err_console.print(
            "[red]Error:[/] plotly is required for time series plots. "
            "Install with [bold]pip install gdm-flow\\[plotly][/]"
        )
        raise typer.Exit(1)

    if output is None:
        output = db.parent / f"{db.stem}_ts.html"

    from .dashboard import generate_ts_dashboard

    try:
        generate_ts_dashboard(db_path=db, output_path=output, run_id=run_id)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1)

    console.print()
    console.print(
        Panel(
            f"[green]Time series dashboard written to [bold]{output}[/][/]",
            border_style="green",
            title="[bold green]✓ Dashboard[/]",
        )
    )
    console.print()


# ── display helpers ──────────────────────────────────────────────────────


def _print_dc_dispatch(dc_r: dict) -> None:
    """Print DC OPF generator dispatch table."""
    result = dc_r["result"]
    dispatch = result.generator_dispatch_w

    if not dispatch:
        return

    console.print()
    dtbl = Table(title="DC Generator Dispatch", border_style="dim")
    dtbl.add_column("Generator", style="bold")
    dtbl.add_column("Type", style="dim")
    dtbl.add_column("Dispatch", justify="right")

    for name, val in sorted(dispatch.items()):
        if name.startswith("grid:"):
            gtype = "[red]Grid[/]"
        elif name.startswith("solar:"):
            gtype = "[yellow]Solar[/]"
        elif name.startswith("battery:"):
            gtype = "[cyan]Battery[/]"
        else:
            gtype = "Other"
        # Shorten name for display
        short = name.split(":", 1)[1] if ":" in name else name
        dtbl.add_row(short, gtype, _fmt_w(val))

    dtbl.add_section()
    dtbl.add_row(
        "[bold]Total Grid[/]", "", f"[bold]{_fmt_w(dc_r.get('grid_import', 0))}[/]"
    )
    dtbl.add_row(
        "[bold]Total Solar[/]", "", f"[bold]{_fmt_w(dc_r.get('solar_dispatch', 0))}[/]"
    )
    dtbl.add_row(
        "[bold]Total Battery[/]",
        "",
        f"[bold]{_fmt_w(dc_r.get('battery_dispatch', 0))}[/]",
    )
    console.print(dtbl)


def _print_ac_voltages(ac_r: dict, system: DistributionSystem) -> None:
    """Print AC voltage magnitude table."""
    result = ac_r["result"]
    idx_map = result.ybus_result.index_to_label
    v = result.voltage

    console.print()
    vtbl = Table(title="AC Bus Voltages", border_style="dim")
    vtbl.add_column("Bus", style="bold")
    vtbl.add_column("Phase")
    vtbl.add_column("|V| (V)", justify="right")
    vtbl.add_column("∠V (°)", justify="right")

    for i, lbl in enumerate(idx_map):
        vm = abs(v[i])
        va = np.degrees(np.angle(v[i]))
        vtbl.add_row(lbl[0], lbl[1], f"{vm:.2f}", f"{va:.2f}")

    console.print(vtbl)


def _export_html(  # pragma: no cover
    system: DistributionSystem,
    ac_r: dict,
    dc_r: dict,
    ldf_r: dict,
    output: Path,
    *,
    pf_r: dict | None = None,
) -> None:
    """Generate an interactive Plotly HTML comparison from already-computed results."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        err_console.print("[yellow]plotly not installed — skipping HTML export[/]")
        return

    from .ac_opf import _build_nominal_voltage_map
    import networkx as nx

    nominal_map = _build_nominal_voltage_map(system)

    # --- Compute per-solver electrical distance (hops from source) ---
    src_bus = system.get_source_bus().name

    # Full undirected graph for Y-bus-based solvers (AC PF, AC OPF)
    full_graph = system.get_undirected_graph()
    try:
        full_hop = nx.single_source_shortest_path_length(full_graph, src_bus)
    except Exception:
        full_hop = {}

    # Radial directed graph for LinDistFlow
    try:
        radial_graph = system.get_directed_graph(return_radial_network=True)
        radial_hop = nx.single_source_shortest_path_length(
            radial_graph.to_undirected(), src_bus
        )
    except Exception:
        radial_hop = {}

    solver_hop = {
        "AC OPF": full_hop,
        "AC PF": full_hop,
        "LinDistFlow": radial_hop,
    }

    # --- Extract voltage data from solver results ---
    solvers_vm: dict[str, dict] = {}  # solver_name -> {label: vm_pu}

    for name, r in [("AC OPF", ac_r), ("AC PF", pf_r), ("LinDistFlow", ldf_r)]:
        if r is None or r.get("result") is None:
            continue
        result = r["result"]

        if hasattr(result, "voltage") and hasattr(result, "ybus_result"):
            vm = {}
            for idx, label in enumerate(result.ybus_result.index_to_label):
                nom = nominal_map.get(label, 1.0)
                if nom > 0:
                    vm[label] = float(abs(result.voltage[idx])) / nom
            solvers_vm[name] = vm
        elif hasattr(result, "voltage_v"):
            vm = {}
            for label, v in result.voltage_v.items():
                nom = nominal_map.get(label, 1.0)
                if nom > 0:
                    vm[label] = float(v) / nom
            solvers_vm[name] = vm

    # Determine which phases are present (excluding neutral)
    all_phases_present: set[str] = set()
    for vm_dict in solvers_vm.values():
        for label in vm_dict:
            if label[1] != "N":
                all_phases_present.add(label[1])
    phase_order = [p for p in ("A", "B", "C") if p in all_phases_present]
    n_phases = len(phase_order)

    # --- Build figure ---
    n_voltage_rows = max(n_phases, 1)
    total_rows = n_voltage_rows + 2  # voltage subplots + power bar + summary table
    row_heights = [0.25] * n_voltage_rows + [0.15, 0.25]
    # Normalise so they sum to 1
    rh_sum = sum(row_heights)
    row_heights = [h / rh_sum for h in row_heights]

    specs = [[{"type": "xy"}] for _ in range(n_voltage_rows)]
    specs.append([{"type": "xy"}])
    specs.append([{"type": "table"}])

    subplot_titles = [
        f"Phase {p} — Voltage vs Distance from Source" for p in phase_order
    ]
    subplot_titles += ["Source Power Injection Comparison", "Solver Summary"]

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    colors = {
        "AC OPF": "#1f77b4",
        "AC PF": "#9467bd",
        "DC OPF": "#2ca02c",
        "LinDistFlow": "#ff7f0e",
    }

    for phase_idx, phase in enumerate(phase_order):
        row = phase_idx + 1
        show_legend = phase_idx == 0  # only show legend entries once

        for solver_name, vm_dict in solvers_vm.items():
            hop_map = solver_hop.get(solver_name, {})
            # Collect (distance, voltage, bus_name) for this phase
            points = []
            for label, vm_pu in vm_dict.items():
                if label[1] != phase:
                    continue
                bus_name = label[0]
                dist = hop_map.get(bus_name)
                if dist is None:
                    continue
                points.append((dist, vm_pu, bus_name))
            points.sort()

            if not points:
                continue

            x_dist = [p[0] for p in points]
            y_vm = [p[1] for p in points]
            hover_text = [
                f"{p[2]}|{phase}<br>V={p[1]:.4f} pu<br>Hops={p[0]}" for p in points
            ]

            fig.add_trace(
                go.Scatter(
                    x=x_dist,
                    y=y_vm,
                    mode="markers",
                    name=solver_name,
                    legendgroup=solver_name,
                    showlegend=show_legend,
                    marker=dict(
                        size=6,
                        color=colors.get(solver_name),
                        opacity=0.7,
                    ),
                    hovertext=hover_text,
                    hoverinfo="text",
                ),
                row=row,
                col=1,
            )

        # ANSI limits
        fig.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="red",
            line_width=1,
            annotation_text="0.95 pu",
            row=row,
            col=1,
        )
        fig.add_hline(
            y=1.05,
            line_dash="dash",
            line_color="red",
            line_width=1,
            annotation_text="1.05 pu",
            row=row,
            col=1,
        )

        fig.update_xaxes(title_text="Hops from source", row=row, col=1)
        fig.update_yaxes(title_text="Voltage (pu)", row=row, col=1)

    power_row = n_voltage_rows + 1
    table_row = n_voltage_rows + 2

    # Source power bar chart
    solver_names = []
    source_p_vals = []
    source_q_vals = []
    bar_colors = []
    for name, r in [
        ("AC OPF", ac_r),
        ("AC PF", pf_r),
        ("DC OPF", dc_r),
        ("LinDistFlow", ldf_r),
    ]:
        if r is None:
            continue
        solver_names.append(name)
        source_p_vals.append(r["source_p"])
        source_q_vals.append(r.get("source_q", 0.0))
        bar_colors.append(colors.get(name, "#333"))

    fig.add_trace(
        go.Bar(
            x=solver_names,
            y=source_p_vals,
            name="Source P (W)",
            marker_color=bar_colors,
            text=[f"{v / 1e6:.2f} MW" for v in source_p_vals],
            textposition="outside",
        ),
        row=power_row,
        col=1,
    )

    # Summary table
    all_results = [
        ("AC OPF", ac_r),
        ("AC PF", pf_r),
        ("DC OPF", dc_r),
        ("LinDistFlow", ldf_r),
    ]
    all_results = [(n, r) for n, r in all_results if r is not None]
    header_vals = ["Metric"] + [n for n, _ in all_results]
    status_row = [
        "\u2713 PASS" if r["success"] else "\u2717 FAIL" for _, r in all_results
    ]
    p_row = [f"{r['source_p'] / 1e6:.2f} MW" for _, r in all_results]
    q_row = [
        f"{r['source_q'] / 1e3:.1f} kvar" if r.get("source_q") else "\u2014"
        for _, r in all_results
    ]
    time_row = [f"{r['elapsed'] * 1000:.0f} ms" for _, r in all_results]
    iter_row = [str(r.get("iterations", "\u2014")) for _, r in all_results]

    fig.add_trace(
        go.Table(
            header=dict(values=header_vals, fill_color="#f0f0f0", align="left"),
            cells=dict(
                values=[
                    ["Status", "Source P", "Source Q", "Time", "Iterations"],
                    *[
                        [s, p, q, t, i]
                        for s, p, q, t, i in zip(
                            status_row, p_row, q_row, time_row, iter_row
                        )
                    ],
                ],
                align="left",
            ),
        ),
        row=table_row,
        col=1,
    )

    # Transpose cells: each column is one solver
    cell_columns = []
    for s, p, q, t, i in zip(status_row, p_row, q_row, time_row, iter_row):
        cell_columns.append([s, p, q, t, i])

    fig.data[-1].cells.values = [
        ["Status", "Source P", "Source Q", "Time", "Iterations"],
        *cell_columns,
    ]

    model_name = output.stem
    fig.update_layout(
        title=f"GDM Flow Solver Comparison — {model_name}",
        height=400 * n_voltage_rows + 600,
        template="plotly_white",
        margin=dict(t=100, r=200, l=70, b=60),
    )
    fig.update_yaxes(title_text="Source P (W)", row=power_row, col=1)

    # --- Voltage vs Active Power jointplot (scatter + marginal histograms) ---
    # Extract per-bus active power injection for AC solvers
    from .ac_opf import build_nodal_power_specs_from_components

    try:
        p_spec, _ = build_nodal_power_specs_from_components(
            system,
            include_loads=True,
            include_solar=True,
            include_capacitor=False,
        )
    except Exception:
        p_spec = {}

    # Build jointplot figure using make_subplots with marginals
    joint_fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    joint_data_added = False

    for solver_name, vm_dict in solvers_vm.items():
        x_power = []
        y_voltage = []
        hover = []
        for label, vm_pu in vm_dict.items():
            if label[1] == "N":
                continue
            p_w = p_spec.get(label, 0.0)
            if p_w == 0.0:
                continue
            x_power.append(float(p_w) / 1e3)  # kW
            y_voltage.append(vm_pu)
            hover.append(
                f"{label[0]}|{label[1]}<br>P={p_w / 1e3:.1f} kW<br>V={vm_pu:.4f} pu"
            )

        if not x_power:
            continue

        joint_data_added = True
        color = colors.get(solver_name, "#999")

        # Main scatter
        joint_fig.add_trace(
            go.Scatter(
                x=x_power,
                y=y_voltage,
                mode="markers",
                name=solver_name,
                marker=dict(size=6, color=color, opacity=0.7),
                hovertext=hover,
                hoverinfo="text",
                legendgroup=solver_name,
            ),
            row=2,
            col=1,
        )

        # Top marginal histogram (P distribution)
        joint_fig.add_trace(
            go.Histogram(
                x=x_power,
                nbinsx=30,
                marker_color=color,
                opacity=0.4,
                showlegend=False,
                legendgroup=solver_name,
            ),
            row=1,
            col=1,
        )

        # Right marginal histogram (V distribution)
        joint_fig.add_trace(
            go.Histogram(
                y=y_voltage,
                nbinsy=30,
                marker_color=color,
                opacity=0.4,
                showlegend=False,
                legendgroup=solver_name,
            ),
            row=2,
            col=2,
        )

    if joint_data_added:
        # ANSI voltage limits on scatter
        joint_fig.add_hline(
            y=0.95, line_dash="dash", line_color="red", line_width=1, row=2, col=1
        )
        joint_fig.add_hline(
            y=1.05, line_dash="dash", line_color="red", line_width=1, row=2, col=1
        )

        joint_fig.update_xaxes(title_text="Active Power Load (kW)", row=2, col=1)
        joint_fig.update_yaxes(title_text="Voltage (pu)", row=2, col=1)
        joint_fig.update_xaxes(showticklabels=False, row=1, col=1)
        joint_fig.update_yaxes(showticklabels=False, row=1, col=1)
        joint_fig.update_xaxes(showticklabels=False, row=2, col=2)
        joint_fig.update_yaxes(showticklabels=False, row=2, col=2)
        joint_fig.update_layout(
            title=f"Voltage vs Active Power — {model_name}",
            height=700,
            width=900,
            template="plotly_white",
            margin=dict(t=80, r=40, l=70, b=60),
            barmode="overlay",
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    # Write main comparison and jointplot to single HTML
    main_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    if joint_data_added:
        joint_html = joint_fig.to_html(include_plotlyjs=False, full_html=False)
    else:
        joint_html = ""

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>GDM Flow — {model_name}</title></head>
<body style="font-family: sans-serif; max-width: 1400px; margin: auto; padding: 20px;">
<h1>GDM Flow Solver Comparison — {model_name}</h1>
{main_html}
{"<hr><h2>Voltage vs Active Power</h2>" + joint_html if joint_html else ""}
</body></html>"""

    output.write_text(html_content)
    console.print(
        Panel(
            f"[green]HTML report written to [bold]{output}[/][/]",
            border_style="green",
        )
    )


# ── fix command ───────────────────────────────────────────────────────────


@app.command("fix")
def fix_command(
    model: Path = typer.Argument(..., help="Path to GDM DistributionSystem JSON file"),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for fixed model JSON. Defaults to <model>_fixed.json",
    ),
    max_iter: int = typer.Option(10, "--max-iter", "-n", help="Maximum fix iterations"),
    solver: Solver = typer.Option(
        Solver.ldf, "--solver", "-s", help="Solver for violation detection"
    ),
    vm_min_pu: float = typer.Option(
        0.95, "--vm-min", help="Minimum voltage in per-unit"
    ),
    vm_max_pu: float = typer.Option(
        1.05, "--vm-max", help="Maximum voltage in per-unit"
    ),
):
    """Fix voltage and loading violations by iteratively applying remediation strategies."""
    from .fix import fix_violations

    system = _load_system(model)

    solver_name = {"ac": "ac", "pf": "ac", "dc": "ldf", "ldf": "ldf"}[solver.value]

    with console.status("[cyan]Running violation fix loop…"):
        result = fix_violations(
            system,
            max_iterations=max_iter,
            solver=solver_name,
            vm_min_pu=vm_min_pu,
            vm_max_pu=vm_max_pu,
        )

    # Print iteration summary table
    if result.iterations:
        tbl = Table(title="Fix Iterations", border_style="cyan")
        tbl.add_column("#", justify="right")
        tbl.add_column("Voltage Violations", justify="right")
        tbl.add_column("Loading Violations", justify="right")
        tbl.add_column("Actions", justify="right")
        for it in result.iterations:
            tbl.add_row(
                str(it.iteration),
                str(it.voltage_violations),
                str(it.loading_violations),
                str(len(it.actions)),
            )
        console.print()
        console.print(tbl)
        console.print()

    # Print result
    style = "green" if result.success else "yellow"
    console.print(
        Panel(
            f"[{style}]{result.message}[/]\n\n"
            f"Initial: {result.initial_voltage_violations} voltage + "
            f"{result.initial_loading_violations} loading violations\n"
            f"Final:   {result.final_voltage_violations} voltage + "
            f"{result.final_loading_violations} loading violations\n"
            f"Actions: {result.total_actions}",
            border_style=style,
            title="Fix Result",
        )
    )

    # Export fixed model
    if result.total_actions > 0:
        out_path = output or model.with_stem(model.stem + "_fixed")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        system.to_json(out_path)
        console.print(f"\n[green]Fixed model written to [bold]{out_path}[/][/]")


# ── entry point ──────────────────────────────────────────────────────────


def main() -> None:
    app()


if __name__ == "__main__":
    main()
