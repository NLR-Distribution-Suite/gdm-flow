"""Time series utilities for QSTS and multi-period simulation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence, Tuple

import numpy as np

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBattery,
    DistributionCapacitor,
    DistributionLoad,
    DistributionSolar,
)

from ._utils import _phase_name, _phase_voltage

BusPhaseLabel = Tuple[str, str]


# ── Discovery ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TimeSeriesInfo:
    """Metadata for a single time series attached to a component."""

    component_type: str
    component_name: str
    variable_name: str
    length: int
    resolution: timedelta | None
    initial_timestamp: datetime | None
    units: str


def list_component_time_series(
    system: DistributionSystem,
) -> dict[str, list[TimeSeriesInfo]]:
    """Discover all time series attached to components in the system.

    Returns a dict keyed by component type name (e.g. ``"DistributionLoad"``)
    mapping to a list of :class:`TimeSeriesInfo` entries.
    """
    component_types = [
        DistributionLoad,
        DistributionSolar,
        DistributionBattery,
    ]

    result: dict[str, list[TimeSeriesInfo]] = {}
    for comp_type in component_types:
        entries: list[TimeSeriesInfo] = []
        for comp in system.get_components(comp_type):
            if not system.has_time_series(comp):
                continue
            for md in system.list_time_series_metadata(comp):
                units_str = str(md.units.units) if md.units else ""
                entries.append(
                    TimeSeriesInfo(
                        component_type=comp_type.__name__,
                        component_name=comp.name,
                        variable_name=md.name,
                        length=md.length,
                        resolution=md.resolution,
                        initial_timestamp=md.initial_timestamp,
                        units=units_str,
                    )
                )
        if entries:
            result[comp_type.__name__] = entries
    return result


def has_time_series_data(system: DistributionSystem) -> bool:
    """Return True if any load, solar, or battery component has time series."""
    for comp_type in (DistributionLoad, DistributionSolar, DistributionBattery):
        for comp in system.get_components(comp_type):
            if system.has_time_series(comp):
                return True
    return False


def get_time_series_length(system: DistributionSystem) -> int:
    """Return the number of timesteps available.

    Checks all components with time series and returns the minimum length
    found, ensuring all components can be evaluated over the same range.

    Raises
    ------
    ValueError
        If no components have time series data.
    """
    lengths: list[int] = []
    for comp_type in (DistributionLoad, DistributionSolar, DistributionBattery):
        for comp in system.get_components(comp_type):
            for md in system.list_time_series_metadata(comp):
                lengths.append(md.length)
    if not lengths:
        raise ValueError("No time series data found on any component.")
    return min(lengths)


def get_time_series_resolution(system: DistributionSystem) -> timedelta:
    """Return the time series resolution (timestep duration).

    Raises
    ------
    ValueError
        If no time series data is found.
    """
    for comp_type in (DistributionLoad, DistributionSolar, DistributionBattery):
        for comp in system.get_components(comp_type):
            for md in system.list_time_series_metadata(comp):
                if md.resolution is not None:
                    return md.resolution
    raise ValueError("No time series data found on any component.")


def get_time_series_timestamps(system: DistributionSystem) -> np.ndarray:
    """Return array of timestamps for the time series data.

    Returns
    -------
    np.ndarray
        Array of ``datetime64`` values.
    """
    for comp_type in (DistributionLoad, DistributionSolar, DistributionBattery):
        for comp in system.get_components(comp_type):
            if not system.has_time_series(comp):
                continue
            md_list = system.list_time_series_metadata(comp)
            if md_list:
                md = md_list[0]
                ts = system.get_time_series(comp, name=md.name)
                if hasattr(ts, "make_timestamps"):
                    return ts.make_timestamps()
                # Build manually from initial_timestamp + resolution
                start = np.datetime64(md.initial_timestamp)
                step = np.timedelta64(int(md.resolution.total_seconds()), "s")
                return np.arange(start, start + step * md.length, step)
    raise ValueError("No time series data found on any component.")


# ── Per-timestep power extraction ────────────────────────────────────────


def _get_ts_value_w(
    system: DistributionSystem,
    component: Any,
    name: str,
    t_idx: int,
    target_unit: str,
) -> float | None:
    """Get a single scalar value from a component's time series at index *t_idx*.

    Returns the value converted to *target_unit* (``"watt"`` or ``"var"``),
    or ``None`` if the component has no time series for *name*.
    """
    try:
        if not system.has_time_series(component):
            return None
        ts = system.get_time_series(component, name=name)
    except Exception:
        return None

    data = ts.data
    if t_idx < 0 or t_idx >= len(data):
        return None
    val = data[t_idx]
    if hasattr(val, "to"):
        return float(val.to(target_unit).magnitude)
    return float(val)


def build_nodal_power_specs_at_timestep(
    system: DistributionSystem,
    t_idx: int,
    *,
    include_loads: bool = True,
    include_solar: bool = True,
    include_battery: bool = False,
    include_capacitor: bool = True,
    load_scale: float = 1.0,
    solar_scale: float = 1.0,
    battery_scale: float = 1.0,
    capacitor_scale: float = 1.0,
) -> tuple[dict[BusPhaseLabel, float], dict[BusPhaseLabel, float]]:
    """Build nodal P/Q specs at a specific timestep.

    For components with time series, uses the value at *t_idx*.
    For components without time series, falls back to static properties
    (same as :func:`build_nodal_power_specs_from_components`).

    Sign convention: positive = generation/injection, negative = load.
    """

    p_spec_w: dict[BusPhaseLabel, float] = defaultdict(float)
    q_spec_var: dict[BusPhaseLabel, float] = defaultdict(float)

    if include_loads:
        for load in system.get_components(DistributionLoad):
            if not load.in_service:
                continue

            # Try time series first
            p_ts = _get_ts_value_w(system, load, "active_power", t_idx, "watt")
            q_ts = _get_ts_value_w(system, load, "reactive_power", t_idx, "var")

            for phase, phase_load in zip(load.phases, load.equipment.phase_loads):
                label = (load.bus.name, _phase_name(phase))
                if p_ts is not None:
                    # Time series gives total load power — distribute equally
                    # across phases (same pattern as static per-phase loads).
                    phase_count = len(load.phases)
                    p_spec_w[label] -= p_ts * load_scale / phase_count
                else:
                    p_spec_w[label] -= (
                        float(phase_load.real_power.to("watt").magnitude) * load_scale
                    )

                if q_ts is not None:
                    phase_count = len(load.phases)
                    q_spec_var[label] -= q_ts * load_scale / phase_count
                else:
                    q_spec_var[label] -= (
                        float(phase_load.reactive_power.to("var").magnitude)
                        * load_scale
                    )

    if include_solar:
        for solar in system.get_components(DistributionSolar):
            if not solar.in_service or not solar.phases:
                continue
            phase_count = len(solar.phases)

            # Solar TS is often "irradiance" (kW/m²) used as a multiplier
            # against rated_power, or "active_power" directly.
            p_ts = _get_ts_value_w(system, solar, "active_power", t_idx, "watt")
            if p_ts is None:
                # Try irradiance: treat as per-unit multiplier on rated_power
                irr = _get_ts_value_w(
                    system, solar, "irradiance", t_idx, "kilowatt / meter ** 2"
                )
                if irr is not None:
                    rated_w = float(solar.equipment.rated_power.to("watt").magnitude)
                    p_ts = irr * rated_w  # irradiance in kW/m² as pu multiplier
                else:
                    p_ts = float(solar.active_power.to("watt").magnitude)
            q_ts = _get_ts_value_w(system, solar, "reactive_power", t_idx, "var")
            if q_ts is None:
                q_ts = float(solar.reactive_power.to("var").magnitude)

            p_each = p_ts * solar_scale / phase_count
            q_each = q_ts * solar_scale / phase_count
            for phase in solar.phases:
                label = (solar.bus.name, _phase_name(phase))
                p_spec_w[label] += p_each
                q_spec_var[label] += q_each

    if include_battery:
        for battery in system.get_components(DistributionBattery):
            if not battery.in_service or not battery.phases:
                continue
            phase_count = len(battery.phases)

            p_ts = _get_ts_value_w(system, battery, "active_power", t_idx, "watt")
            if p_ts is None:
                p_ts = float(battery.active_power.to("watt").magnitude)
            q_ts = _get_ts_value_w(system, battery, "reactive_power", t_idx, "var")
            if q_ts is None:
                q_ts = float(battery.reactive_power.to("var").magnitude)

            p_each = p_ts * battery_scale / phase_count
            q_each = q_ts * battery_scale / phase_count
            for phase in battery.phases:
                label = (battery.bus.name, _phase_name(phase))
                p_spec_w[label] += p_each
                q_spec_var[label] += q_each

    if include_capacitor:
        # Capacitors typically don't have time series — use static values
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


def build_dc_load_profile_at_timestep(
    system: DistributionSystem,
    t_idx: int,
    *,
    include_loads: bool = True,
    include_solar_as_negative_load: bool = False,
    include_battery_as_negative_load: bool = False,
    load_scale: float = 1.0,
    solar_scale: float = 1.0,
    battery_scale: float = 1.0,
) -> dict[BusPhaseLabel, float]:
    """Build DC demand profile at a specific timestep.

    Positive = demand, negative = fixed injection.
    """
    demand: dict[BusPhaseLabel, float] = {}

    if include_loads:
        for load in system.get_components(DistributionLoad):
            if not load.in_service:
                continue
            p_ts = _get_ts_value_w(system, load, "active_power", t_idx, "watt")
            for phase, phase_load in zip(load.phases, load.equipment.phase_loads):
                label = (load.bus.name, _phase_name(phase))
                if p_ts is not None:
                    phase_count = len(load.phases)
                    demand[label] = (
                        demand.get(label, 0.0) + p_ts * load_scale / phase_count
                    )
                else:
                    demand[label] = (
                        demand.get(label, 0.0)
                        + float(phase_load.real_power.to("watt").magnitude) * load_scale
                    )

    if include_solar_as_negative_load:
        for solar in system.get_components(DistributionSolar):
            if not solar.in_service or not solar.phases:
                continue
            p_ts = _get_ts_value_w(system, solar, "active_power", t_idx, "watt")
            if p_ts is None:
                irr = _get_ts_value_w(
                    system, solar, "irradiance", t_idx, "kilowatt / meter ** 2"
                )
                if irr is not None:
                    rated_w = float(solar.equipment.rated_power.to("watt").magnitude)
                    p_ts = irr * rated_w
                else:
                    p_ts = float(solar.active_power.to("watt").magnitude)
            p_each = p_ts * solar_scale / len(solar.phases)
            for phase in solar.phases:
                label = (solar.bus.name, _phase_name(phase))
                demand[label] = demand.get(label, 0.0) - p_each

    if include_battery_as_negative_load:
        for battery in system.get_components(DistributionBattery):
            if not battery.in_service or not battery.phases:
                continue
            p_ts = _get_ts_value_w(system, battery, "active_power", t_idx, "watt")
            if p_ts is None:
                p_ts = float(battery.active_power.to("watt").magnitude)
            p_each = p_ts * battery_scale / len(battery.phases)
            for phase in battery.phases:
                label = (battery.bus.name, _phase_name(phase))
                demand[label] = demand.get(label, 0.0) - p_each

    return demand


def build_lindistflow_injections_at_timestep(
    system: DistributionSystem,
    t_idx: int,
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
    """Build LinDistFlow net injections at a specific timestep.

    Positive = demand/consumption, negative = injection.
    """
    p_net: dict[BusPhaseLabel, float] = defaultdict(float)
    q_net: dict[BusPhaseLabel, float] = defaultdict(float)

    if include_loads:
        for load in system.get_components(DistributionLoad):
            if not load.in_service:
                continue
            p_ts = _get_ts_value_w(system, load, "active_power", t_idx, "watt")
            q_ts = _get_ts_value_w(system, load, "reactive_power", t_idx, "var")
            for phase, phase_load in zip(load.phases, load.equipment.phase_loads):
                label = (load.bus.name, _phase_name(phase))
                if p_ts is not None:
                    phase_count = len(load.phases)
                    p_net[label] += p_ts * load_scale / phase_count
                else:
                    p_net[label] += (
                        float(phase_load.real_power.to("watt").magnitude) * load_scale
                    )
                if q_ts is not None:
                    phase_count = len(load.phases)
                    q_net[label] += q_ts * load_scale / phase_count
                else:
                    q_net[label] += (
                        float(phase_load.reactive_power.to("var").magnitude)
                        * load_scale
                    )

    if include_solar:
        for solar in system.get_components(DistributionSolar):
            if not solar.in_service or not solar.phases:
                continue
            p_ts = _get_ts_value_w(system, solar, "active_power", t_idx, "watt")
            if p_ts is None:
                irr = _get_ts_value_w(
                    system, solar, "irradiance", t_idx, "kilowatt / meter ** 2"
                )
                if irr is not None:
                    rated_w = float(solar.equipment.rated_power.to("watt").magnitude)
                    p_ts = irr * rated_w
                else:
                    p_ts = float(solar.active_power.to("watt").magnitude)
            q_ts = _get_ts_value_w(system, solar, "reactive_power", t_idx, "var")
            if q_ts is None:
                q_ts = float(solar.reactive_power.to("var").magnitude)

            p_each = p_ts * solar_scale / len(solar.phases)
            q_each = q_ts * solar_scale / len(solar.phases)
            for phase in solar.phases:
                label = (solar.bus.name, _phase_name(phase))
                p_net[label] -= p_each
                q_net[label] -= q_each

    if include_battery:
        for battery in system.get_components(DistributionBattery):
            if not battery.in_service or not battery.phases:
                continue
            p_ts = _get_ts_value_w(system, battery, "active_power", t_idx, "watt")
            if p_ts is None:
                p_ts = float(battery.active_power.to("watt").magnitude)
            q_ts = _get_ts_value_w(system, battery, "reactive_power", t_idx, "var")
            if q_ts is None:
                q_ts = float(battery.reactive_power.to("var").magnitude)

            p_each = p_ts * battery_scale / len(battery.phases)
            q_each = q_ts * battery_scale / len(battery.phases)
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


# ── Battery SOC Tracker ──────────────────────────────────────────────────


@dataclass
class BatterySOCTracker:
    """Track battery state-of-charge across QSTS timesteps.

    Parameters
    ----------
    name : str
        Battery component name.
    energy_capacity_wh : float
        Total energy capacity in watt-hours.
    p_charge_max_w : float
        Maximum charging power in watts (positive value).
    p_discharge_max_w : float
        Maximum discharging power in watts (positive value).
    soc : float
        Current state of charge (0.0 to 1.0).
    soc_min : float
        Minimum allowable SOC.
    soc_max : float
        Maximum allowable SOC.
    charge_efficiency : float
        Charging efficiency (0.0 to 1.0).
    discharge_efficiency : float
        Discharging efficiency (0.0 to 1.0).
    """

    name: str
    energy_capacity_wh: float
    p_charge_max_w: float
    p_discharge_max_w: float
    soc: float = 0.5
    soc_min: float = 0.1
    soc_max: float = 0.9
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    soc_history: list[float] = field(default_factory=list, repr=False)

    def get_available_bounds(self, dt_hours: float) -> tuple[float, float]:
        """Return ``(p_min_w, p_max_w)`` constrained by current SOC.

        Negative = charging, positive = discharging (generation convention).
        """
        if self.energy_capacity_wh <= 0 or dt_hours <= 0:
            return (0.0, 0.0)

        # Max discharge limited by available energy above soc_min
        energy_available_wh = (
            (self.soc - self.soc_min)
            * self.energy_capacity_wh
            * self.discharge_efficiency
        )
        p_discharge_max = min(self.p_discharge_max_w, energy_available_wh / dt_hours)
        p_discharge_max = max(p_discharge_max, 0.0)

        # Max charge limited by remaining capacity below soc_max
        energy_headroom_wh = (self.soc_max - self.soc) * self.energy_capacity_wh
        if self.charge_efficiency > 0:
            p_charge_max = min(
                self.p_charge_max_w,
                energy_headroom_wh / (dt_hours * self.charge_efficiency),
            )
        else:
            p_charge_max = 0.0
        p_charge_max = max(p_charge_max, 0.0)

        return (-p_charge_max, p_discharge_max)

    def update(self, p_dispatch_w: float, dt_hours: float) -> float:
        """Update SOC given actual dispatch and return clamped dispatch.

        Parameters
        ----------
        p_dispatch_w : float
            Actual dispatch in watts. Positive = discharging, negative = charging.
        dt_hours : float
            Duration of the timestep in hours.

        Returns
        -------
        float
            The clamped dispatch value actually applied.
        """
        p_min, p_max = self.get_available_bounds(dt_hours)
        p_clamped = max(p_min, min(p_max, p_dispatch_w))

        if p_clamped >= 0:
            # Discharging
            energy_wh = p_clamped * dt_hours / self.discharge_efficiency
        else:
            # Charging
            energy_wh = p_clamped * dt_hours * self.charge_efficiency

        self.soc -= energy_wh / self.energy_capacity_wh
        self.soc = max(self.soc_min, min(self.soc_max, self.soc))
        self.soc_history.append(self.soc)
        return p_clamped


# ── QSTS Orchestrator ────────────────────────────────────────────────────


@dataclass(frozen=True)
class QSTSSummary:
    """Summary metadata for a completed QSTS simulation."""

    solver: str
    num_timesteps: int
    num_converged: int
    resolution: timedelta
    initial_timestamp: datetime | None
    db_path: str | None
    run_id: str | None
    battery_soc_traces: dict[str, list[float]]
    """Battery name → SOC history."""


def run_qsts(
    system: DistributionSystem,
    solver: str,
    timestep_range: range | Sequence[int],
    *,
    db_path: str | None = None,
    include_loads: bool = True,
    include_solar: bool = True,
    include_battery: bool = False,
    include_capacitor: bool = True,
    battery_soc_trackers: dict[str, BatterySOCTracker] | None = None,
    progress_callback: Any | None = None,
) -> QSTSSummary:
    """Run Quasi-Static Time Series simulation.

    Iterates over *timestep_range*, extracting per-timestep P/Q specs from
    component time series, solving with the chosen solver, and streaming
    results to SQLite.

    Parameters
    ----------
    system : DistributionSystem
        Input distribution system with time series data.
    solver : str
        Solver name: ``"ac"``, ``"pf"``, ``"dc"``, or ``"ldf"``.
    timestep_range : range or sequence of int
        Timestep indices to simulate.
    db_path : str, optional
        Path to SQLite database for streaming results.
    battery_soc_trackers : dict, optional
        Pre-configured SOC trackers keyed by battery name.
    progress_callback : callable, optional
        Called with ``(t_idx, total)`` after each timestep.
    """
    from .ybus import calculate_ybus

    # Precompute Y-bus (topology is static)
    ybus_result = calculate_ybus(system, sparse=True)

    # Get resolution for SOC tracking
    try:
        resolution = get_time_series_resolution(system)
    except ValueError:
        resolution = timedelta(hours=1)
    dt_hours = resolution.total_seconds() / 3600.0

    try:
        timestamps = get_time_series_timestamps(system)
    except ValueError:
        timestamps = None

    initial_timestamp = None
    if timestamps is not None and len(timestamps) > 0:
        initial_timestamp = timestamps[0].astype("datetime64[us]").astype(datetime)

    # Setup SQLite streaming
    run_id = None
    conn = None
    if db_path is not None:
        import sqlite3
        from uuid import uuid4

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        _create_ts_schema(conn)
        run_id = f"qsts_{solver}_{uuid4().hex[:12]}"
        conn.execute(
            """INSERT INTO ts_runs
               (run_id, implementation, mode, num_timesteps, resolution_s, start_timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                solver,
                "qsts",
                len(timestep_range),
                resolution.total_seconds(),
                str(initial_timestamp) if initial_timestamp else None,
            ),
        )
        conn.commit()

        # Populate nominal voltage map for p.u. computation
        nominal_map = _populate_bus_nominal(conn, system)
    else:
        nominal_map = None

    soc_trackers = battery_soc_trackers or {}
    num_converged = 0
    prev_result = None
    timestep_list = list(timestep_range)

    try:
        for step_i, t_idx in enumerate(timestep_list):
            # Build per-timestep specs
            p_spec, q_spec = build_nodal_power_specs_at_timestep(
                system,
                t_idx,
                include_loads=include_loads,
                include_solar=include_solar,
                include_battery=include_battery,
                include_capacitor=include_capacitor,
            )

            # Run solver
            result = _run_solver_snapshot(
                system, solver, p_spec, q_spec, ybus_result, prev_result
            )

            if result is not None and getattr(result, "success", False):
                num_converged += 1
                prev_result = result

            # Update SOC trackers
            for bname, tracker in soc_trackers.items():
                # Extract battery dispatch from result if available
                dispatch = 0.0
                if solver == "dc" and hasattr(result, "generator_dispatch_w"):
                    for gname, val in result.generator_dispatch_w.items():
                        if bname in gname:
                            dispatch = val
                            break
                tracker.update(dispatch, dt_hours)

            # Stream to SQLite
            if conn is not None and run_id is not None and result is not None:
                _stream_timestep_to_sqlite(
                    conn, run_id, t_idx, solver, result, nominal_map
                )
                # Stream SOC
                for bname, tracker in soc_trackers.items():
                    conn.execute(
                        """INSERT OR REPLACE INTO ts_battery_soc
                           (run_id, timestep, battery_name, soc, p_dispatch_w, energy_wh)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            run_id,
                            t_idx,
                            bname,
                            tracker.soc,
                            0.0,
                            tracker.soc * tracker.energy_capacity_wh,
                        ),
                    )
                conn.commit()

            if progress_callback is not None:
                progress_callback(step_i + 1, len(timestep_list))
    finally:
        if conn is not None:
            conn.close()

    return QSTSSummary(
        solver=solver,
        num_timesteps=len(timestep_list),
        num_converged=num_converged,
        resolution=resolution,
        initial_timestamp=initial_timestamp,
        db_path=db_path,
        run_id=run_id,
        battery_soc_traces={
            name: list(t.soc_history) for name, t in soc_trackers.items()
        },
    )


# ── Internal helpers ─────────────────────────────────────────────────────


def _run_solver_snapshot(
    system: DistributionSystem,
    solver: str,
    p_spec: dict[BusPhaseLabel, float],
    q_spec: dict[BusPhaseLabel, float],
    ybus_result: Any,
    prev_result: Any | None,
) -> Any:
    """Run a single-snapshot solve with the given specs.

    When *prev_result* is available, extracts voltage from the previous
    solution to warm-start the solver.
    """
    if solver == "ac":
        from .ac_opf import optimize_ac_power_flow

        kwargs: dict[str, Any] = {"p_spec_w": p_spec, "q_spec_var": q_spec}
        if prev_result is not None and hasattr(prev_result, "voltage"):
            labels = prev_result.ybus_result.index_to_label
            v = prev_result.voltage
            # Extract voltage magnitudes and angles
            v0_pu = {}
            theta0 = {}
            for idx, label in enumerate(labels):
                v0_pu[label] = abs(v[idx])
                theta0[label] = float(np.angle(v[idx]))
            # v0_pu needs to be in per-unit — divide by nominal
            from .ac_opf import _build_nominal_voltage_map

            nominal_map = _build_nominal_voltage_map(system)
            v0_pu_clean = {}
            for label in v0_pu:
                nom = nominal_map.get(label, 1.0)
                if nom > 0:
                    v0_pu_clean[label] = v0_pu[label] / nom
            kwargs["v0_pu"] = v0_pu_clean
            kwargs["theta0_rad"] = theta0
        return optimize_ac_power_flow(system, **kwargs)

    elif solver == "pf":
        from .ac_pf import solve_ac_power_flow

        kwargs = {"p_spec_w": p_spec, "q_spec_var": q_spec}
        if prev_result is not None and hasattr(prev_result, "voltage"):
            labels = prev_result.ybus_result.index_to_label
            v0_complex = {
                label: complex(prev_result.voltage[idx])
                for idx, label in enumerate(labels)
            }
            kwargs["v0_complex"] = v0_complex
        return solve_ac_power_flow(system, **kwargs)

    elif solver == "dc":
        from .dc_opf import (
            build_dc_generators_from_components,
            solve_dc_opf,
        )

        generators = build_dc_generators_from_components(system)
        kwargs = {"generators": generators, "demand_w": p_spec}
        if prev_result is not None and hasattr(prev_result, "theta_rad"):
            kwargs["theta0_rad"] = prev_result.theta_rad
        return solve_dc_opf(system, **kwargs)

    elif solver == "ldf":
        from .lindistflow import solve_lindistflow

        # LinDistFlow uses positive = demand convention, need to flip signs
        p_net = {k: -v for k, v in p_spec.items()}
        q_net = {k: -v for k, v in q_spec.items()}
        return solve_lindistflow(system, p_net_w=p_net, q_net_var=q_net)

    else:
        raise ValueError(f"Unknown solver: {solver!r}")


def _create_ts_schema(conn: Any) -> None:
    """Create time series tables in the SQLite database."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ts_runs (
            run_id TEXT PRIMARY KEY,
            implementation TEXT NOT NULL,
            mode TEXT NOT NULL,
            num_timesteps INTEGER NOT NULL,
            resolution_s REAL,
            start_timestamp TEXT
        );

        CREATE TABLE IF NOT EXISTS ts_bus_nominal (
            bus_name TEXT NOT NULL,
            phase TEXT NOT NULL,
            nominal_v REAL NOT NULL,
            PRIMARY KEY (bus_name, phase)
        );

        CREATE TABLE IF NOT EXISTS ts_nodes (
            run_id TEXT NOT NULL,
            timestep INTEGER NOT NULL,
            bus_name TEXT NOT NULL,
            phase TEXT NOT NULL,
            voltage_mag_v REAL,
            voltage_pu REAL,
            voltage_angle_rad REAL,
            p_injection_w REAL,
            q_injection_var REAL,
            PRIMARY KEY (run_id, timestep, bus_name, phase),
            FOREIGN KEY(run_id) REFERENCES ts_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ts_branches (
            run_id TEXT NOT NULL,
            timestep INTEGER NOT NULL,
            branch_name TEXT NOT NULL,
            phase TEXT NOT NULL,
            p_flow_w REAL,
            q_flow_var REAL,
            loading_va REAL,
            PRIMARY KEY (run_id, timestep, branch_name, phase),
            FOREIGN KEY(run_id) REFERENCES ts_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ts_battery_soc (
            run_id TEXT NOT NULL,
            timestep INTEGER NOT NULL,
            battery_name TEXT NOT NULL,
            soc REAL,
            p_dispatch_w REAL,
            energy_wh REAL,
            PRIMARY KEY (run_id, timestep, battery_name),
            FOREIGN KEY(run_id) REFERENCES ts_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ts_summary (
            run_id TEXT NOT NULL,
            timestep INTEGER NOT NULL,
            success INTEGER,
            source_p_w REAL,
            source_q_var REAL,
            total_loss_w REAL,
            solve_time_s REAL,
            PRIMARY KEY (run_id, timestep),
            FOREIGN KEY(run_id) REFERENCES ts_runs(run_id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def _populate_bus_nominal(conn: Any, system: Any) -> dict[tuple[str, str], float]:
    """Populate ts_bus_nominal table and return nominal voltage map."""
    from gdm.systems.distribution.components import DistributionBus

    nominal: dict[tuple[str, str], float] = {}
    for bus in system.get_components(DistributionBus):
        v = _phase_voltage(bus.rated_voltage, bus.voltage_type)
        for phase in bus.phases:
            pn = _phase_name(phase)
            nominal[(bus.name, pn)] = v
            conn.execute(
                "INSERT OR IGNORE INTO ts_bus_nominal (bus_name, phase, nominal_v) VALUES (?, ?, ?)",
                (bus.name, pn, v),
            )
    conn.commit()
    return nominal


def _stream_timestep_to_sqlite(
    conn: Any,
    run_id: str,
    t_idx: int,
    solver: str,
    result: Any,
    nominal_map: dict[tuple[str, str], float] | None = None,
) -> None:
    """Write a single timestep result to the SQLite database."""

    success = getattr(result, "success", False)

    # Summary row
    source_p = 0.0
    source_q = 0.0

    if solver in ("ac", "pf") and hasattr(result, "voltage"):
        ybus_res = result.ybus_result
        v = result.voltage
        for idx, label in enumerate(ybus_res.index_to_label):
            v_mag = abs(v[idx])
            v_angle = float(np.angle(v[idx]))
            p_inj = float(result.power_injection[idx].real)
            q_inj = float(result.power_injection[idx].imag)
            nom = nominal_map.get(label, 1.0) if nominal_map else 1.0
            v_pu = v_mag / nom if nom > 0 else None
            conn.execute(
                """INSERT OR REPLACE INTO ts_nodes
                   (run_id, timestep, bus_name, phase,
                    voltage_mag_v, voltage_pu, voltage_angle_rad, p_injection_w, q_injection_var)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, t_idx, label[0], label[1], v_mag, v_pu, v_angle, p_inj, q_inj),
            )
            source_p += p_inj
            source_q += q_inj

    elif solver == "ldf" and hasattr(result, "voltage_v"):
        for label, v_val in result.voltage_v.items():
            v_f = float(v_val)
            nom = nominal_map.get(label, 1.0) if nominal_map else 1.0
            v_pu = v_f / nom if nom > 0 else None
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
                    v_f,
                    v_pu,
                    None,
                    float(result.p_net_w.get(label, 0.0)),
                    float(result.q_net_var.get(label, 0.0)),
                ),
            )
        source_p = sum(float(v) for v in result.p_net_w.values())
        source_q = sum(float(v) for v in result.q_net_var.values())

    elif solver == "dc" and hasattr(result, "theta_rad"):
        for label, theta in result.theta_rad.items():
            bal = result.nodal_balance_w.get(label, 0.0)
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
                    float(theta),
                    float(bal),
                    None,
                ),
            )
        source_p = result.slack_injection_w

    conn.execute(
        """INSERT OR REPLACE INTO ts_summary
           (run_id, timestep, success, source_p_w, source_q_var, total_loss_w, solve_time_s)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (run_id, t_idx, int(success), source_p, source_q, None, None),
    )
