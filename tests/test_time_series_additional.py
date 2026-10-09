"""Additional time_series tests covering SOC tracker, QSTS, and edge cases."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gdm_flow.time_series import (
    BatterySOCTracker,
    QSTSSummary,
    _create_ts_schema,
    _run_solver_snapshot,
    _stream_timestep_to_sqlite,
    build_dc_load_profile_at_timestep,
    build_lindistflow_injections_at_timestep,
    build_nodal_power_specs_at_timestep,
    get_time_series_length,
    get_time_series_resolution,
    get_time_series_timestamps,
    has_time_series_data,
    run_qsts,
)


MODEL_PATH = Path("examples/models/p5r.json")


@pytest.fixture()
def system():
    if not MODEL_PATH.exists():
        pytest.skip("p5r.json model not found")
    from gdm.systems.distribution import DistributionSystem

    return DistributionSystem.from_json(str(MODEL_PATH))


class TestBatterySOCTracker:
    def test_initial_state(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
        )
        assert tracker.soc == 0.5
        assert tracker.soc_min == 0.1
        assert tracker.soc_max == 0.9

    def test_get_available_bounds_zero_capacity(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=0,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
        )
        assert tracker.get_available_bounds(1.0) == (0.0, 0.0)

    def test_get_available_bounds_zero_dt(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
        )
        assert tracker.get_available_bounds(0.0) == (0.0, 0.0)

    def test_get_available_bounds_normal(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
            soc=0.5,
        )
        p_min, p_max = tracker.get_available_bounds(1.0)
        assert p_min < 0  # Can charge
        assert p_max > 0  # Can discharge

    def test_get_available_bounds_at_soc_min(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
            soc=0.1,  # at min
            soc_min=0.1,
        )
        p_min, p_max = tracker.get_available_bounds(1.0)
        assert p_max == 0.0  # Cannot discharge further

    def test_get_available_bounds_at_soc_max(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
            soc=0.9,  # at max
            soc_max=0.9,
        )
        p_min, p_max = tracker.get_available_bounds(1.0)
        assert p_min == 0.0  # Cannot charge further

    def test_update_discharge(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
            soc=0.5,
        )
        clamped = tracker.update(1000.0, 1.0)  # discharge 1000W for 1h
        assert clamped == 1000.0
        assert tracker.soc < 0.5
        assert len(tracker.soc_history) == 1

    def test_update_charge(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
            soc=0.5,
        )
        clamped = tracker.update(-1000.0, 1.0)  # charge 1000W for 1h
        assert clamped == -1000.0
        assert tracker.soc > 0.5

    def test_update_clamps_to_bounds(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
            soc=0.11,  # just above min
            soc_min=0.1,
        )
        # Try to discharge more than available
        clamped = tracker.update(50000.0, 1.0)
        assert clamped < 50000.0
        assert tracker.soc >= tracker.soc_min

    def test_zero_charge_efficiency(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
            soc=0.5,
            charge_efficiency=0.0,
        )
        p_min, p_max = tracker.get_available_bounds(1.0)
        assert p_min == 0.0  # Can't charge with 0 efficiency


class TestCreateTsSchema:
    def test_creates_tables(self):
        conn = sqlite3.connect(":memory:")
        _create_ts_schema(conn)
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "ts_runs" in tables
        assert "ts_nodes" in tables
        assert "ts_branches" in tables
        assert "ts_battery_soc" in tables
        assert "ts_summary" in tables
        conn.close()

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        _create_ts_schema(conn)
        _create_ts_schema(conn)  # Should not raise
        conn.close()


class TestStreamTimestepToSqlite:
    def test_ldf_result(self):
        conn = sqlite3.connect(":memory:")
        _create_ts_schema(conn)
        conn.execute(
            "INSERT INTO ts_runs VALUES (?, ?, ?, ?, ?, ?)",
            ("run1", "ldf", "qsts", 10, 900, None),
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.voltage_v = {("bus1", "A"): 120.0, ("bus2", "B"): 119.0}
        mock_result.p_net_w = {("bus1", "A"): 1000.0, ("bus2", "B"): 500.0}
        mock_result.q_net_var = {("bus1", "A"): 200.0, ("bus2", "B"): 100.0}

        nominal = {("bus1", "A"): 120.0, ("bus2", "B"): 120.0}
        _stream_timestep_to_sqlite(conn, "run1", 0, "ldf", mock_result, nominal)

        rows = conn.execute("SELECT * FROM ts_nodes").fetchall()
        assert len(rows) == 2

        summary = conn.execute("SELECT * FROM ts_summary").fetchall()
        assert len(summary) == 1
        conn.close()

    def test_dc_result(self):
        conn = sqlite3.connect(":memory:")
        _create_ts_schema(conn)
        conn.execute(
            "INSERT INTO ts_runs VALUES (?, ?, ?, ?, ?, ?)",
            ("run1", "dc", "qsts", 10, 900, None),
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.theta_rad = {("bus1", "A"): 0.01, ("bus2", "A"): -0.02}
        mock_result.nodal_balance_w = {("bus1", "A"): 100.0, ("bus2", "A"): -50.0}
        mock_result.slack_injection_w = 100.0

        _stream_timestep_to_sqlite(conn, "run1", 0, "dc", mock_result, None)

        rows = conn.execute("SELECT * FROM ts_nodes").fetchall()
        assert len(rows) == 2
        conn.close()


class TestRunSolverSnapshot:
    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            _run_solver_snapshot(None, "invalid", {}, {}, None, None)


class TestTimestepExtraction:
    def test_build_dc_load_profile_at_timestep(self, system):
        demand = build_dc_load_profile_at_timestep(system, 0)
        assert isinstance(demand, dict)
        assert len(demand) > 0
        # All demand should be positive for loads
        assert all(v >= 0 for v in demand.values())

    def test_build_dc_load_with_solar_negative(self, system):
        demand = build_dc_load_profile_at_timestep(
            system, 0, include_solar_as_negative_load=True
        )
        assert isinstance(demand, dict)

    def test_build_lindistflow_injections_at_timestep(self, system):
        p_net, q_net = build_lindistflow_injections_at_timestep(system, 0)
        assert isinstance(p_net, dict)
        assert isinstance(q_net, dict)
        assert len(p_net) > 0

    def test_build_nodal_specs_with_battery(self, system):
        p_spec, q_spec = build_nodal_power_specs_at_timestep(
            system, 0, include_battery=True
        )
        assert isinstance(p_spec, dict)

    def test_build_nodal_specs_with_scales(self, system):
        p_base, _ = build_nodal_power_specs_at_timestep(system, 0)
        p_scaled, _ = build_nodal_power_specs_at_timestep(system, 0, load_scale=2.0)
        # Scaled loads should roughly double the demand
        if p_base:
            key = next(iter(p_base))
            assert abs(p_scaled.get(key, 0)) > abs(p_base[key]) * 1.5

    def test_get_time_series_timestamps(self, system):
        timestamps = get_time_series_timestamps(system)
        assert isinstance(timestamps, np.ndarray)
        assert len(timestamps) > 0

    def test_has_time_series_data_false(self):
        mock_sys = MagicMock()
        mock_sys.get_components.return_value = []
        assert has_time_series_data(mock_sys) is False

    def test_get_time_series_resolution_raises_on_no_ts(self):
        mock_sys = MagicMock()
        mock_sys.get_components.return_value = []
        with pytest.raises(ValueError, match="No time series"):
            get_time_series_resolution(mock_sys)


class TestRunQSTS:
    def test_qsts_ldf_small_range(self, system):
        summary = run_qsts(system, "ldf", range(3))
        assert isinstance(summary, QSTSSummary)
        assert summary.solver == "ldf"
        assert summary.num_timesteps == 3
        assert summary.num_converged > 0

    def test_qsts_with_db_output(self, system, tmp_path):
        db_file = str(tmp_path / "qsts.db")
        summary = run_qsts(system, "ldf", range(2), db_path=db_file)
        assert summary.db_path == db_file
        assert summary.run_id is not None
        # Verify DB has data
        conn = sqlite3.connect(db_file)
        rows = conn.execute("SELECT COUNT(*) FROM ts_nodes").fetchone()[0]
        assert rows > 0
        conn.close()

    def test_qsts_with_progress_callback(self, system):
        calls = []
        summary = run_qsts(
            system, "ldf", range(2), progress_callback=lambda i, t: calls.append(i)
        )
        assert len(calls) == 2

    def test_qsts_with_battery_tracker(self, system):
        tracker = BatterySOCTracker(
            name="test_bat",
            energy_capacity_wh=10000,
            p_charge_max_w=5000,
            p_discharge_max_w=5000,
            soc=0.5,
        )
        summary = run_qsts(
            system,
            "ldf",
            range(3),
            battery_soc_trackers={"test_bat": tracker},
        )
        assert "test_bat" in summary.battery_soc_traces
        assert len(summary.battery_soc_traces["test_bat"]) == 3
