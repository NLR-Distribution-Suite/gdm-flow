"""Tests for time_series module — discovery, extraction, SOC tracker, and QSTS."""

import sqlite3
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gdm_flow.time_series import (
    BatterySOCTracker,
    QSTSSummary,
    TimeSeriesInfo,
    _create_ts_schema,
    build_dc_load_profile_at_timestep,
    build_lindistflow_injections_at_timestep,
    build_nodal_power_specs_at_timestep,
    get_time_series_length,
    get_time_series_resolution,
    has_time_series_data,
    list_component_time_series,
    run_qsts,
)


MODEL_PATH = Path("examples/models/p5r.json")


@pytest.fixture()
def system():
    """Load the p5r model which has time series data."""
    if not MODEL_PATH.exists():
        pytest.skip("p5r.json model not found")
    from gdm.systems.distribution import DistributionSystem

    return DistributionSystem.from_json(str(MODEL_PATH))


# ── Discovery ────────────────────────────────────────────────────────────


class TestDiscovery:
    def test_list_component_time_series_returns_entries(self, system):
        result = list_component_time_series(system)
        assert isinstance(result, dict)
        assert len(result) > 0
        # p5r has loads with TS and solar with irradiance
        assert "DistributionLoad" in result
        for info in result["DistributionLoad"]:
            assert isinstance(info, TimeSeriesInfo)
            assert info.length > 0
            assert info.variable_name in ("active_power", "reactive_power")

    def test_has_time_series_data_true(self, system):
        assert has_time_series_data(system) is True

    def test_get_time_series_length(self, system):
        length = get_time_series_length(system)
        assert length == 35040  # 15-min resolution for a year

    def test_get_time_series_resolution(self, system):
        res = get_time_series_resolution(system)
        assert res == timedelta(minutes=15)

    def test_get_time_series_length_raises_on_no_ts(self):
        mock_sys = MagicMock()
        mock_sys.get_components.return_value = []
        with pytest.raises(ValueError, match="No time series"):
            get_time_series_length(mock_sys)


# ── Per-timestep extraction ──────────────────────────────────────────────


class TestTimestepExtraction:
    def test_build_nodal_specs_at_t0_produces_values(self, system):
        p_spec, q_spec = build_nodal_power_specs_at_timestep(system, 0)
        assert isinstance(p_spec, dict)
        assert isinstance(q_spec, dict)
        # Should have entries for load buses
        assert len(p_spec) > 0
        # All values should be negative (loads consume)
        for v in p_spec.values():
            assert v < 0, "Loads should produce negative P specs"

    def test_build_nodal_specs_with_solar_at_noon(self, system):
        # Timestep ~48 = noon on day 1 (15-min intervals, 48*15min = 12h)
        p_spec, q_spec = build_nodal_power_specs_at_timestep(
            system, 48, include_solar=True
        )
        assert len(p_spec) > 0

    def test_build_nodal_specs_differs_across_timesteps(self, system):
        p0, _ = build_nodal_power_specs_at_timestep(system, 0)
        p100, _ = build_nodal_power_specs_at_timestep(system, 100)
        # Values should differ between timesteps
        assert p0 != p100, "Different timesteps should produce different specs"

    def test_build_dc_load_profile_at_timestep(self, system):
        demand = build_dc_load_profile_at_timestep(system, 0)
        assert isinstance(demand, dict)
        assert len(demand) > 0
        # Demand should be positive for loads
        for v in demand.values():
            assert v > 0

    def test_build_lindistflow_injections_at_timestep(self, system):
        p_net, q_net = build_lindistflow_injections_at_timestep(system, 0)
        assert isinstance(p_net, dict)
        assert len(p_net) > 0
        # Demand should be positive for loads in LDF convention
        for v in p_net.values():
            assert v > 0


# ── Battery SOC Tracker ──────────────────────────────────────────────────


class TestBatterySOCTracker:
    def test_initial_state(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10_000.0,
            p_charge_max_w=5_000.0,
            p_discharge_max_w=5_000.0,
            soc=0.5,
        )
        assert tracker.soc == 0.5
        assert tracker.energy_capacity_wh == 10_000.0

    def test_discharge_reduces_soc(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10_000.0,
            p_charge_max_w=5_000.0,
            p_discharge_max_w=5_000.0,
            soc=0.5,
            discharge_efficiency=1.0,
        )
        # Discharge 1000W for 1 hour = 1000 Wh = 10% of capacity
        tracker.update(1000.0, dt_hours=1.0)
        assert tracker.soc == pytest.approx(0.4, abs=0.01)

    def test_charge_increases_soc(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10_000.0,
            p_charge_max_w=5_000.0,
            p_discharge_max_w=5_000.0,
            soc=0.5,
            charge_efficiency=1.0,
        )
        # Charge 1000W for 1 hour = 1000 Wh = 10% of capacity
        tracker.update(-1000.0, dt_hours=1.0)
        assert tracker.soc == pytest.approx(0.6, abs=0.01)

    def test_soc_clamped_at_limits(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10_000.0,
            p_charge_max_w=50_000.0,
            p_discharge_max_w=50_000.0,
            soc=0.15,
            soc_min=0.1,
            soc_max=0.9,
            discharge_efficiency=1.0,
        )
        # Try to discharge way more than available
        tracker.update(50_000.0, dt_hours=1.0)
        assert tracker.soc >= tracker.soc_min

    def test_get_available_bounds(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10_000.0,
            p_charge_max_w=5_000.0,
            p_discharge_max_w=5_000.0,
            soc=0.5,
            soc_min=0.1,
            soc_max=0.9,
        )
        p_min, p_max = tracker.get_available_bounds(dt_hours=1.0)
        assert p_min < 0  # Can charge
        assert p_max > 0  # Can discharge

    def test_soc_history_tracked(self):
        tracker = BatterySOCTracker(
            name="bat1",
            energy_capacity_wh=10_000.0,
            p_charge_max_w=5_000.0,
            p_discharge_max_w=5_000.0,
            soc=0.5,
            discharge_efficiency=1.0,
        )
        for _ in range(5):
            tracker.update(500.0, dt_hours=0.25)
        assert len(tracker.soc_history) == 5
        # SOC should be decreasing
        assert tracker.soc_history[-1] < 0.5


# ── QSTS orchestrator ───────────────────────────────────────────────────


class TestQSTS:
    def test_run_qsts_ldf_no_db(self, system):
        summary = run_qsts(system, "ldf", range(3))
        assert isinstance(summary, QSTSSummary)
        assert summary.solver == "ldf"
        assert summary.num_timesteps == 3
        assert summary.num_converged == 3

    def test_run_qsts_with_sqlite_streaming(self, system):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            summary = run_qsts(system, "ldf", range(5), db_path=db_path)
            assert summary.run_id is not None
            assert summary.db_path == db_path

            # Verify SQLite contents
            conn = sqlite3.connect(db_path)
            try:
                runs = conn.execute("SELECT * FROM ts_runs").fetchall()
                assert len(runs) == 1

                nodes = conn.execute("SELECT COUNT(*) FROM ts_nodes").fetchone()[0]
                assert nodes > 0

                summaries = conn.execute("SELECT COUNT(*) FROM ts_summary").fetchone()[
                    0
                ]
                assert summaries == 5
            finally:
                conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_run_qsts_with_progress_callback(self, system):
        calls = []
        _summary = run_qsts(
            system,
            "ldf",
            range(3),
            progress_callback=lambda done, total: calls.append((done, total)),
        )
        assert len(calls) == 3
        assert calls[-1] == (3, 3)


# ── SQLite schema ────────────────────────────────────────────────────────


class TestTSSchema:
    def test_create_ts_schema_creates_tables(self):
        conn = sqlite3.connect(":memory:")
        _create_ts_schema(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "ts_runs" in tables
        assert "ts_nodes" in tables
        assert "ts_branches" in tables
        assert "ts_battery_soc" in tables
        assert "ts_summary" in tables
        conn.close()
