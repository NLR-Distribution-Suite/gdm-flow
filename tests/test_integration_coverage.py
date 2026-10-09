"""Integration tests using real p5r model for sqlite_export, strategies, and time_series QSTS."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gdm.systems.distribution import DistributionSystem


MODEL_PATH = Path("examples/models/p5r.json")


@pytest.fixture()
def system():
    if not MODEL_PATH.exists():
        pytest.skip("p5r.json model not found")
    return DistributionSystem.from_json(str(MODEL_PATH))


class TestSQLiteExportIntegration:
    def test_export_ac_pf_result(self, system, tmp_path):
        from gdm_flow.ac_pf import solve_ac_power_flow
        from gdm_flow.sqlite_export import export_ac_pf_result_to_sqlite

        result = solve_ac_power_flow(system)
        assert result.success

        db_path = str(tmp_path / "test.db")
        run_id = export_ac_pf_result_to_sqlite(result, db_path)
        assert run_id is not None

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        assert row is not None
        nodes = conn.execute(
            "SELECT COUNT(*) FROM ac_pf_nodes WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert nodes > 0
        conn.close()

    def test_export_all_results(self, system, tmp_path):
        from gdm_flow import (
            optimize_ac_power_flow_from_components,
            solve_dc_opf_from_components,
            solve_lindistflow,
        )
        from gdm_flow.sqlite_export import export_all_results_to_sqlite

        ac_result = optimize_ac_power_flow_from_components(system)
        dc_result = solve_dc_opf_from_components(system)
        ldf_result = solve_lindistflow(system)

        db_path = str(tmp_path / "all_results.db")
        run_ids = export_all_results_to_sqlite(
            db_path,
            ac_result=ac_result,
            dc_result=dc_result,
            lindistflow_result=ldf_result,
        )
        assert isinstance(run_ids, dict)

        conn = sqlite3.connect(db_path)
        runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert runs >= 3
        conn.close()

    def test_export_lindistflow_result(self, system, tmp_path):
        from gdm_flow import solve_lindistflow
        from gdm_flow.sqlite_export import export_lindistflow_result_to_sqlite

        result = solve_lindistflow(system)
        assert result.success

        db_path = str(tmp_path / "ldf.db")
        run_id = export_lindistflow_result_to_sqlite(result, db_path)
        assert run_id is not None

        conn = sqlite3.connect(db_path)
        nodes = conn.execute(
            "SELECT COUNT(*) FROM lindistflow_nodes WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert nodes > 0
        conn.close()

    def test_export_dc_opf_result(self, system, tmp_path):
        from gdm_flow import solve_dc_opf_from_components
        from gdm_flow.sqlite_export import export_dc_opf_result_to_sqlite

        result = solve_dc_opf_from_components(system)
        assert result.success

        db_path = str(tmp_path / "dc.db")
        run_id = export_dc_opf_result_to_sqlite(result, db_path)
        assert run_id is not None

        conn = sqlite3.connect(db_path)
        nodes = conn.execute(
            "SELECT COUNT(*) FROM dc_opf_nodes WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert nodes > 0
        conn.close()


class TestFixStrategiesIntegration:
    def test_regulator_tap_apply_with_tight_limits(self, system):
        from gdm_flow.fix.detect import detect_violations
        from gdm_flow.fix.strategies import AdjustRegulatorTapStrategy

        # Use tight limits to force violations
        report = detect_violations(
            system, solver="ldf", vm_min_pu=0.999, vm_max_pu=1.001
        )
        if not report.voltage_violations:
            pytest.skip("No voltage violations with tight limits on this system")

        strategy = AdjustRegulatorTapStrategy()
        actions = strategy.apply(system, report)
        # May or may not find regulators to adjust, but shouldn't crash
        assert isinstance(actions, list)

    def test_capacitor_strategy_with_tight_limits(self, system):
        from gdm_flow.fix.detect import detect_violations
        from gdm_flow.fix.strategies import AddCapacitorStrategy

        report = detect_violations(
            system, solver="ldf", vm_min_pu=0.999, vm_max_pu=1.001
        )
        if not any(v.kind == "undervoltage" for v in report.voltage_violations):
            pytest.skip("No undervoltage violations on this system")

        strategy = AddCapacitorStrategy()
        actions = strategy.apply(system, report)
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_resize_conductor_with_loading(self, system):
        from gdm_flow.fix.detect import detect_violations
        from gdm_flow.fix.strategies import ResizeConductorStrategy

        # Very tight limits to force violations
        report = detect_violations(
            system, solver="ldf", vm_min_pu=0.999, vm_max_pu=1.001
        )
        if not report.loading_violations and not any(
            v.kind == "undervoltage" for v in report.voltage_violations
        ):
            pytest.skip("No loading/undervoltage violations for conductor resize")

        strategy = ResizeConductorStrategy()
        if strategy.can_fix(report):
            actions = strategy.apply(system, report)
            assert isinstance(actions, list)

    def test_full_fix_loop_with_tight_limits(self, system):
        from gdm_flow.fix import fix_violations

        result = fix_violations(
            system, solver="ldf", vm_min_pu=0.999, vm_max_pu=1.001, max_iterations=3
        )
        assert isinstance(result.iterations, list)
        assert result.initial_voltage_violations >= 0


class TestTimeSeriesQSTSIntegration:
    def test_qsts_dc_solver(self, system):
        from gdm_flow.time_series import has_time_series_data, run_qsts

        if not has_time_series_data(system):
            pytest.skip("No time series data")

        summary = run_qsts(system, "dc", range(2))
        assert summary.solver == "dc"
        assert summary.num_timesteps == 2

    def test_qsts_ac_solver(self, system):
        from gdm_flow.time_series import has_time_series_data, run_qsts

        if not has_time_series_data(system):
            pytest.skip("No time series data")

        summary = run_qsts(system, "ac", range(2))
        assert summary.solver == "ac"
        assert summary.num_timesteps == 2

    def test_qsts_pf_solver(self, system):
        from gdm_flow.time_series import has_time_series_data, run_qsts

        if not has_time_series_data(system):
            pytest.skip("No time series data")

        summary = run_qsts(system, "pf", range(2))
        assert summary.solver == "pf"
        assert summary.num_timesteps == 2

    def test_qsts_dc_with_db_streaming(self, system, tmp_path):
        from gdm_flow.time_series import has_time_series_data, run_qsts

        if not has_time_series_data(system):
            pytest.skip("No time series data")

        db_file = str(tmp_path / "dc_qsts.db")
        summary = run_qsts(system, "dc", range(3), db_path=db_file)
        assert summary.db_path == db_file

        conn = sqlite3.connect(db_file)
        nodes = conn.execute("SELECT COUNT(*) FROM ts_nodes").fetchone()[0]
        assert nodes > 0
        conn.close()
