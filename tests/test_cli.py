from __future__ import annotations

from pathlib import Path
import builtins
import sqlite3

import pytest
import typer
import numpy as np

import gdm_flow.cli as cli
from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.enums import Phase
from gdm.systems.distribution.components.base.distribution_branch_base import (
    DistributionBranchBase,
)
from gdm_flow import calculate_ybus


class _DummyStatus:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyConsole:
    def print(self, *args, **kwargs):
        return None

    def status(self, *args, **kwargs):
        return _DummyStatus()


class _FakeSystem:
    """Mock system that supports get_components() for aggregation logic."""

    def get_components(self, *args, **kwargs):
        return iter([])

    def get_source_bus(self):
        class _Bus:
            name = "source"
            phases = []

        return _Bus()

    def get_undirected_graph(self):
        import networkx as nx

        return nx.Graph()

    def get_directed_graph(self, **kwargs):
        import networkx as nx

        return nx.DiGraph()


class _FakeDCResult:
    def __init__(self, dispatch: dict[str, float]):
        self.success = True
        self.message = "ok"
        self.iterations = 4
        self.objective = 12.5
        self.generator_dispatch_w = dispatch


class _FakeLDFResult:
    def __init__(
        self,
        p_net_w: dict[str, float],
        q_net_var: dict[str, float],
        voltage_v: dict[str, float],
    ):
        self.success = True
        self.message = "ok"
        self.p_net_w = p_net_w
        self.q_net_var = q_net_var
        self.voltage_v = voltage_v


class _FakeACResult:
    def __init__(self):
        self.payload = "ac"


class _FakeExportedResult:
    def __init__(self, name: str):
        self.name = name


class _Q:
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def to(self, _unit):
        return self


def test_format_helpers_and_success_badge():
    assert cli._fmt_w(2_100_000.0) == "2.10 MW"
    assert cli._fmt_w(1_200.0) == "1.20 kW"
    assert cli._fmt_w(12.0) == "12.0 W"

    assert cli._fmt_var(3_000_000.0) == "3.00 Mvar"
    assert cli._fmt_var(1_500.0) == "1.50 kvar"
    assert cli._fmt_var(9.0) == "9.0 var"

    assert "PASS" in cli._success_badge(True)
    assert "FAIL" in cli._success_badge(False)


def test_load_system_missing_file_raises_exit(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(typer.Exit):
        cli._load_system(missing)


def test_load_system_reads_existing_json(tmp_path, monkeypatch):
    model = tmp_path / "model.json"
    model.write_text("{}")

    sentinel = _FakeSystem()

    def _fake_from_json(path: str):
        assert path == str(model)
        return sentinel

    monkeypatch.setattr(DistributionSystem, "from_json", staticmethod(_fake_from_json))
    assert cli._load_system(model) is sentinel


def test_run_dc_aggregates_dispatch(monkeypatch):
    def _fake_solve(*args, **kwargs):
        return _FakeDCResult(
            {
                "grid:g1": 1000.0,
                "solar:s1": 250.0,
                "battery:b1": -100.0,
                "other": 25.0,
            }
        )

    monkeypatch.setattr("gdm_flow.dc_opf.solve_dc_opf_from_components", _fake_solve)

    out = cli._run_dc(system=_FakeSystem())
    assert out["success"] is True
    assert out["source_p"] == 1000.0
    assert out["grid_import"] == 1000.0
    assert out["solar_dispatch"] == 250.0
    assert out["battery_dispatch"] == -100.0
    assert out["total_gen"] == 1175.0


def test_run_ldf_handles_empty_voltages(monkeypatch):
    def _fake_solve(*args, **kwargs):
        return _FakeLDFResult(p_net_w={"a": 1.0}, q_net_var={"a": 2.0}, voltage_v={})

    monkeypatch.setattr("gdm_flow.lindistflow.solve_lindistflow", _fake_solve)

    out = cli._run_ldf(system=_FakeSystem())
    assert out["success"] is True
    assert out["source_p"] == 1.0
    assert out["source_q"] == 2.0
    assert out["v_min"] == 0.0
    assert out["v_max"] == 0.0


def test_print_dc_dispatch_handles_empty_dispatch(monkeypatch):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    cli._print_dc_dispatch({"result": _FakeDCResult({})})


def test_run_command_verbose_triggers_detail_views(monkeypatch):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "_load_system", lambda _model: _FakeSystem())

    calls = {"dc": 0, "ac": 0}

    def _fake_print_dc(_r):
        calls["dc"] += 1

    def _fake_print_ac(_r, _s):
        calls["ac"] += 1

    monkeypatch.setattr(cli, "_print_dc_dispatch", _fake_print_dc)
    monkeypatch.setattr(cli, "_print_ac_voltages", _fake_print_ac)

    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.ac,
        lambda _system: {
            "solver": "AC OPF",
            "success": True,
            "source_p": 1000.0,
            "source_q": 50.0,
            "elapsed": 0.01,
            "iterations": 2,
            "result": _FakeACResult(),
        },
    )
    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.dc,
        lambda _system: {
            "solver": "DC OPF",
            "success": True,
            "source_p": 1000.0,
            "source_q": 0.0,
            "elapsed": 0.01,
            "iterations": 3,
            "result": _FakeDCResult({"grid:g1": 1000.0}),
        },
    )

    cli.run(
        model=Path("ignored.json"),
        solver=[cli.Solver.ac, cli.Solver.dc],
        verbose=True,
    )

    assert calls["dc"] == 1
    assert calls["ac"] == 1


def test_compare_triggers_export_and_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    fake_system = _FakeSystem()
    monkeypatch.setattr(cli, "_load_system", lambda _model: fake_system)

    dispatched = {"called": 0}
    exported = {"path": None, "system": None}

    monkeypatch.setattr(
        cli, "_print_dc_dispatch", lambda _r: dispatched.__setitem__("called", 1)
    )

    def _fake_export(system, ac_r, dc_r, ldf_r, output, *, pf_r=None):
        exported["system"] = system
        exported["path"] = output

    monkeypatch.setattr(cli, "_export_html", _fake_export)

    _ldf_result = {
        "solver": "LinDistFlow",
        "success": True,
        "source_p": 980.0,
        "source_q": 5.0,
        "elapsed": 0.01,
        "iterations": 0,
    }
    monkeypatch.setattr(cli, "_run_ldf", lambda _system: _ldf_result)

    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.ac,
        lambda _system: {
            "solver": "AC OPF",
            "success": True,
            "source_p": 1000.0,
            "source_q": 10.0,
            "elapsed": 0.01,
            "iterations": 2,
        },
    )
    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.pf,
        lambda _system: {
            "solver": "AC PF",
            "success": True,
            "source_p": 1000.0,
            "source_q": 10.0,
            "elapsed": 0.01,
            "iterations": 3,
        },
    )
    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.dc,
        lambda _system: {
            "solver": "DC OPF",
            "success": True,
            "source_p": 1020.0,
            "source_q": 0.0,
            "elapsed": 0.01,
            "iterations": 3,
            "result": _FakeDCResult({"grid:g1": 1000.0}),
        },
    )
    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.ldf,
        lambda _system: {
            "solver": "LinDistFlow",
            "success": True,
            "source_p": 980.0,
            "source_q": 5.0,
            "elapsed": 0.01,
            "iterations": 0,
        },
    )

    out = tmp_path / "comparison.html"
    cli.compare(model=Path("ignored.json"), output=out)

    assert dispatched["called"] == 1
    assert exported["system"] is fake_system
    assert exported["path"] == out


def test_compare_disagreement_branch(monkeypatch):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "_load_system", lambda _model: _FakeSystem())
    monkeypatch.setattr(cli, "_print_dc_dispatch", lambda _r: None)
    monkeypatch.setattr(cli, "_export_html", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli,
        "_run_ldf",
        lambda _system: {
            "solver": "LinDistFlow",
            "success": True,
            "source_p": 900.0,
            "source_q": 5.0,
            "elapsed": 0.01,
            "iterations": 0,
        },
    )

    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.ac,
        lambda _system: {
            "solver": "AC OPF",
            "success": True,
            "source_p": 1000.0,
            "source_q": 10.0,
            "elapsed": 0.01,
            "iterations": 2,
        },
    )
    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.pf,
        lambda _system: {
            "solver": "AC PF",
            "success": True,
            "source_p": 1000.0,
            "source_q": 10.0,
            "elapsed": 0.01,
            "iterations": 3,
        },
    )
    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.dc,
        lambda _system: {
            "solver": "DC OPF",
            "success": True,
            "source_p": 1500.0,
            "source_q": 0.0,
            "elapsed": 0.01,
            "iterations": 3,
            "result": _FakeDCResult({"grid:g1": 1500.0}),
        },
    )
    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.ldf,
        lambda _system: {
            "solver": "LinDistFlow",
            "success": True,
            "source_p": 900.0,
            "source_q": 5.0,
            "elapsed": 0.01,
            "iterations": 0,
        },
    )

    cli.compare(model=Path("ignored.json"), output=None)


def test_export_passes_selected_solver_results(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "_load_system", lambda _model: _FakeSystem())
    monkeypatch.setattr(cli, "_build_node_voltage_limits_v", lambda _system: {})
    monkeypatch.setattr(cli, "_build_lindistflow_loading_limits_va", lambda _system: {})

    captured: dict[str, object] = {}

    def _fake_export_all_results_to_sqlite(
        db_path,
        ac_result,
        pf_result,
        dc_result,
        lindistflow_result,
        ac_voltage_limits_v,
        ac_branch_loading_va,
        ac_branch_loading_limits_va,
        ac_branch_flow_w_var,
        pf_voltage_limits_v,
        pf_branch_loading_va,
        pf_branch_loading_limits_va,
        pf_branch_flow_w_var,
        dc_branch_loading_va,
        dc_branch_loading_limits_va,
        dc_branch_flow_w_var,
        lindistflow_voltage_limits_v,
        lindistflow_loading_limits_va,
    ):
        captured["db_path"] = db_path
        captured["ac_result"] = ac_result
        captured["pf_result"] = pf_result
        captured["dc_result"] = dc_result
        captured["ldf_result"] = lindistflow_result
        captured["ac_limits"] = ac_voltage_limits_v
        captured["ac_loading"] = ac_branch_loading_va
        captured["ac_loading_limits"] = ac_branch_loading_limits_va
        captured["ac_flow"] = ac_branch_flow_w_var
        captured["pf_limits"] = pf_voltage_limits_v
        captured["pf_loading"] = pf_branch_loading_va
        captured["pf_loading_limits"] = pf_branch_loading_limits_va
        captured["pf_flow"] = pf_branch_flow_w_var
        captured["dc_loading"] = dc_branch_loading_va
        captured["dc_loading_limits"] = dc_branch_loading_limits_va
        captured["dc_flow"] = dc_branch_flow_w_var
        captured["ldf_limits"] = lindistflow_voltage_limits_v
        captured["ldf_loading"] = lindistflow_loading_limits_va

    monkeypatch.setattr(
        "gdm_flow.sqlite_export.export_all_results_to_sqlite",
        _fake_export_all_results_to_sqlite,
    )

    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.ac,
        lambda _system: {
            "solver": "AC OPF",
            "success": True,
            "source_p": 1000.0,
            "result": _FakeExportedResult("ac"),
        },
    )
    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.ldf,
        lambda _system: {
            "solver": "LinDistFlow",
            "success": True,
            "source_p": 995.0,
            "result": _FakeExportedResult("ldf"),
        },
    )

    db_path = tmp_path / "results.sqlite"
    cli.export(
        model=Path("ignored.json"), db=db_path, solver=[cli.Solver.ac, cli.Solver.ldf]
    )

    assert captured["db_path"] == str(db_path)
    assert captured["ac_result"].name == "ac"
    assert captured["pf_result"] is None
    assert captured["dc_result"] is None
    assert captured["ldf_result"].name == "ldf"
    assert captured["ac_limits"] == {}
    assert captured["ac_loading"] == {}
    assert captured["ac_loading_limits"] == {}
    assert captured["ac_flow"] == {}
    assert captured["pf_limits"] == {}
    assert captured["pf_loading"] == {}
    assert captured["pf_loading_limits"] == {}
    assert captured["pf_flow"] == {}
    assert captured["dc_loading"] == {}
    assert captured["dc_loading_limits"] == {}
    assert captured["dc_flow"] == {}
    assert captured["ldf_limits"] == {}
    assert captured["ldf_loading"] == {}


def test_export_includes_dc_when_selected(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "_load_system", lambda _model: _FakeSystem())
    monkeypatch.setattr(cli, "_build_node_voltage_limits_v", lambda _system: {})
    monkeypatch.setattr(cli, "_build_lindistflow_loading_limits_va", lambda _system: {})

    captured: dict[str, object] = {}

    def _fake_export_all_results_to_sqlite(
        db_path,
        ac_result,
        pf_result,
        dc_result,
        lindistflow_result,
        ac_voltage_limits_v,
        ac_branch_loading_va,
        ac_branch_loading_limits_va,
        ac_branch_flow_w_var,
        pf_voltage_limits_v,
        pf_branch_loading_va,
        pf_branch_loading_limits_va,
        pf_branch_flow_w_var,
        dc_branch_loading_va,
        dc_branch_loading_limits_va,
        dc_branch_flow_w_var,
        lindistflow_voltage_limits_v,
        lindistflow_loading_limits_va,
    ):
        captured["ac"] = ac_result
        captured["pf"] = pf_result
        captured["dc"] = dc_result
        captured["ldf"] = lindistflow_result
        captured["ac_limits"] = ac_voltage_limits_v
        captured["ac_loading"] = ac_branch_loading_va
        captured["ac_loading_limits"] = ac_branch_loading_limits_va
        captured["ac_flow"] = ac_branch_flow_w_var
        captured["pf_limits"] = pf_voltage_limits_v
        captured["pf_loading"] = pf_branch_loading_va
        captured["pf_loading_limits"] = pf_branch_loading_limits_va
        captured["pf_flow"] = pf_branch_flow_w_var
        captured["dc_loading"] = dc_branch_loading_va
        captured["dc_loading_limits"] = dc_branch_loading_limits_va
        captured["dc_flow"] = dc_branch_flow_w_var
        captured["ldf_limits"] = lindistflow_voltage_limits_v
        captured["ldf_loading"] = lindistflow_loading_limits_va

    monkeypatch.setattr(
        "gdm_flow.sqlite_export.export_all_results_to_sqlite",
        _fake_export_all_results_to_sqlite,
    )

    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.dc,
        lambda _system: {
            "solver": "DC OPF",
            "success": True,
            "source_p": 1000.0,
            "result": _FakeDCResult({"grid:g1": 1000.0}),
        },
    )

    cli.export(
        model=Path("ignored.json"), db=tmp_path / "x.sqlite", solver=[cli.Solver.dc]
    )
    assert captured["ac"] is None
    assert captured["pf"] is None
    assert captured["dc"] is not None
    assert captured["ldf"] is None
    assert captured["ac_limits"] == {}
    assert captured["ac_loading"] == {}
    assert captured["ac_loading_limits"] == {}
    assert captured["ac_flow"] == {}
    assert captured["pf_limits"] == {}
    assert captured["pf_loading"] == {}
    assert captured["pf_loading_limits"] == {}
    assert captured["pf_flow"] == {}
    assert captured["dc_loading"] == {}
    assert captured["dc_loading_limits"] == {}
    assert captured["dc_flow"] == {}
    assert captured["ldf_limits"] == {}
    assert captured["ldf_loading"] == {}


def test_read_overvoltage_rows_filters_violations(tmp_path):
    db_path = tmp_path / "violations.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                implementation TEXT NOT NULL,
                success INTEGER NOT NULL,
                message TEXT,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE ac_opf_nodes (
                run_id TEXT NOT NULL,
                bus_name TEXT NOT NULL,
                phase TEXT NOT NULL,
                voltage_mag_v REAL NOT NULL,
                voltage_min_v REAL,
                voltage_max_v REAL,
                voltage_angle_rad REAL NOT NULL,
                p_injection_w REAL NOT NULL,
                q_injection_var REAL NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            ("ac_1", "ac_opf", 1, "ok", "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO ac_opf_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("ac_1", "b1", "A", 112.0, 110.0, 111.0, 0.0, 0.0, 0.0),
        )
        conn.execute(
            "INSERT INTO ac_opf_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("ac_1", "b2", "A", 109.5, 109.0, 111.0, 0.0, 0.0, 0.0),
        )
        conn.commit()
    finally:
        conn.close()

    run_id, rows, has_columns = cli._read_overvoltage_rows(str(db_path), "ac_opf", None)
    assert has_columns is True
    assert run_id == "ac_1"
    assert len(rows) == 1
    assert rows[0][0] == "b1"


def test_read_overload_rows_filters_violations(tmp_path):
    db_path = tmp_path / "overload.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                implementation TEXT NOT NULL,
                success INTEGER NOT NULL,
                message TEXT,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE lindistflow_branches (
                run_id TEXT NOT NULL,
                branch_name TEXT NOT NULL,
                phase TEXT NOT NULL,
                p_flow_w REAL NOT NULL,
                q_flow_var REAL NOT NULL,
                loading_va REAL,
                loading_limit_va REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            ("ldf_1", "lindistflow", 1, "ok", "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO lindistflow_branches VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ldf_1", "line_1", "A", 1000.0, 500.0, 1118.0, 1000.0),
        )
        conn.execute(
            "INSERT INTO lindistflow_branches VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ldf_1", "line_2", "A", 500.0, 100.0, 510.0, 1000.0),
        )
        conn.commit()
    finally:
        conn.close()

    run_id, rows, has_columns = cli._read_overload_rows(
        str(db_path), "lindistflow", None
    )
    assert has_columns is True
    assert run_id == "ldf_1"
    assert len(rows) == 1
    assert rows[0][0] == "line_1"


def test_read_overload_rows_filters_ac_violations(tmp_path):
    db_path = tmp_path / "ac_overload.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                implementation TEXT NOT NULL,
                success INTEGER NOT NULL,
                message TEXT,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE ac_opf_branches (
                run_id TEXT NOT NULL,
                branch_name TEXT NOT NULL,
                phase TEXT NOT NULL,
                p_flow_w REAL,
                q_flow_var REAL,
                loading_va REAL,
                loading_limit_va REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            ("ac_1", "ac_opf", 1, "ok", "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO ac_opf_branches VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ac_1", "line_1", "A", 1000.0, 100.0, 1005.0, 900.0),
        )
        conn.execute(
            "INSERT INTO ac_opf_branches VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ac_1", "line_2", "A", 500.0, 100.0, 510.0, 900.0),
        )
        conn.commit()
    finally:
        conn.close()

    run_id, rows, has_columns = cli._read_overload_rows(str(db_path), "ac_opf", None)
    assert has_columns is True
    assert run_id == "ac_1"
    assert len(rows) == 1
    assert rows[0][0] == "line_1"


def test_read_overload_rows_filters_dc_violations(tmp_path):
    db_path = tmp_path / "dc_overload.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                implementation TEXT NOT NULL,
                success INTEGER NOT NULL,
                message TEXT,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE dc_opf_branches (
                run_id TEXT NOT NULL,
                branch_name TEXT NOT NULL,
                phase TEXT NOT NULL,
                p_flow_w REAL,
                q_flow_var REAL,
                loading_va REAL,
                loading_limit_va REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            ("dc_1", "dc_opf", 1, "ok", "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO dc_opf_branches VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("dc_1", "line_1", "A", 1000.0, 0.0, 1000.0, 900.0),
        )
        conn.execute(
            "INSERT INTO dc_opf_branches VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("dc_1", "line_2", "A", 500.0, 0.0, 500.0, 900.0),
        )
        conn.commit()
    finally:
        conn.close()

    run_id, rows, has_columns = cli._read_overload_rows(str(db_path), "dc_opf", None)
    assert has_columns is True
    assert run_id == "dc_1"
    assert len(rows) == 1
    assert rows[0][0] == "line_1"


def test_info_command_smoke_with_stubbed_system(monkeypatch):
    monkeypatch.setattr(cli, "console", _DummyConsole())

    class _PhaseLoad:
        real_power = _Q(1200.0)
        reactive_power = _Q(300.0)

    class _LoadEq:
        phase_loads = [_PhaseLoad()]

    class _Load:
        def __init__(self, bus):
            self.bus = bus
            self.equipment = _LoadEq()

    class _SolarEq:
        rated_power = _Q(2000.0)

    class _Solar:
        def __init__(self, bus):
            self.bus = bus
            self.active_power = _Q(800.0)
            self.equipment = _SolarEq()

    class _Bus:
        def __init__(self, name):
            self.name = name
            self.phases = [Phase.A, Phase.N]
            self.rated_voltage = _Q(230.0)

    class _Transformer:
        pass

    bus_src = _Bus("source")
    bus_ld = _Bus("load")

    class _System:
        def get_source_bus(self):
            return bus_src

        def get_components(self, comp_type):
            name = comp_type.__name__
            if name == "DistributionBus":
                return [bus_src, bus_ld]
            if name == "DistributionLoad":
                return [_Load(bus_ld)]
            if name == "DistributionSolar":
                return [_Solar(bus_ld)]
            if name == "DistributionTransformer":
                return [_Transformer()]
            return []

    monkeypatch.setattr(cli, "_load_system", lambda _model: _System())
    cli.info(model=Path("ignored.json"))


def test_run_ac_computes_source_and_voltage_stats(monkeypatch):
    class _SourceBus:
        name = "src"

    class _System:
        def get_source_bus(self):
            return _SourceBus()

    class _Y:
        index_to_label = [("src", "A"), ("load", "A")]
        ybus = np.array([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]], dtype=np.complex128)

    class _ACRes:
        success = True
        message = "ok"
        iterations = 3
        final_objective = 1.23
        ybus_result = _Y()
        voltage = np.array([2 + 0j, 1 + 0j], dtype=np.complex128)

    monkeypatch.setattr(
        "gdm_flow.ac_opf.optimize_ac_power_flow_from_components",
        lambda *_a, **_k: _ACRes(),
    )

    out = cli._run_ac(_System())
    assert out["success"] is True
    assert out["source_p"] == 4.0
    assert out["v_min"] == 1.0
    assert out["v_max"] == 2.0


def test_print_dc_dispatch_covers_all_generator_types(monkeypatch):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    result = _FakeDCResult(
        {
            "grid:g": 10.0,
            "solar:s": 5.0,
            "battery:b": -2.0,
            "other": 1.0,
        }
    )
    cli._print_dc_dispatch(
        {
            "result": result,
            "grid_import": 10.0,
            "solar_dispatch": 5.0,
            "battery_dispatch": -2.0,
        }
    )


def test_export_html_handles_missing_plotly(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "err_console", _DummyConsole())

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "plotly.graph_objects":
            raise ImportError("plotly missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    cli._export_html(_FakeSystem(), {}, {}, {}, tmp_path / "out.html")


def test_export_html_handles_missing_compare_script(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "err_console", _DummyConsole())
    monkeypatch.setattr("gdm_flow.ac_opf._build_nominal_voltage_map", lambda _s: {})

    import pathlib

    real_exists = pathlib.Path.exists

    def _fake_exists(self):
        if str(self).endswith("compare_plotly_results.py"):
            return False
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", _fake_exists)
    _fake_r = {
        "solver": "X",
        "success": True,
        "source_p": 100.0,
        "source_q": 0.0,
        "elapsed": 0.01,
        "iterations": 1,
    }
    cli._export_html(_FakeSystem(), _fake_r, _fake_r, _fake_r, tmp_path / "out.html")


def test_print_ac_voltages_smoke(monkeypatch):
    monkeypatch.setattr(cli, "console", _DummyConsole())

    class _Y:
        index_to_label = [("b1", "A"), ("b2", "A")]

    class _Res:
        ybus_result = _Y()
        voltage = np.array([1 + 0j, 0.9 + 0.1j], dtype=np.complex128)

    cli._print_ac_voltages({"result": _Res()}, system=_FakeSystem())


def test_db_schema_prints_existing_tables(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "console", _DummyConsole())

    db_path = tmp_path / "schema.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE sample_table (id INTEGER, name TEXT)")
        conn.commit()
    finally:
        conn.close()

    cli.db_schema(db=db_path, include_internal=False)


def test_db_schema_missing_file_raises_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "err_console", _DummyConsole())

    with pytest.raises(typer.Exit):
        cli.db_schema(db=tmp_path / "missing.sqlite", include_internal=False)


def test_to_float_quantity_and_extract_voltage_bounds_paths():
    class _GoodQ:
        magnitude = 123.0

        def to(self, _unit):
            return self

    class _BadQ:
        def to(self, _unit):
            raise RuntimeError("bad conversion")

    class _Bounds:
        minimum = _GoodQ()
        maximum = _GoodQ()

    class _MissingUpper:
        minimum = _GoodQ()

    assert cli._to_float_quantity(None, "volt") is None
    assert cli._to_float_quantity(_GoodQ(), "volt") == 123.0
    assert cli._to_float_quantity(_BadQ(), "volt") is None
    assert cli._to_float_quantity("7.5", "volt") == 7.5
    assert cli._to_float_quantity(object(), "volt") is None

    assert cli._extract_voltage_bounds(_Bounds()) == (123.0, 123.0)
    assert cli._extract_voltage_bounds(_MissingUpper()) is None


def test_build_node_voltage_limits_merges_component_and_regulator_limits(monkeypatch):
    class _VLimit:
        def __init__(self, lo, hi, phase=None):
            self.min_voltage = _Q(lo)
            self.max_voltage = _Q(hi)
            self.phase = phase

    class _Bus:
        def __init__(self):
            self.name = "b1"
            self.phases = [Phase.A, Phase.B, Phase.N]
            self.voltagelimits = [
                _VLimit(110.0, 130.0),
                _VLimit(112.0, 128.0, phase=Phase.A),
            ]

    class _System:
        def get_components(self, comp_type):
            if comp_type.__name__ == "DistributionBus":
                return [_Bus()]
            return []

    monkeypatch.setattr(
        cli,
        "build_regulator_voltage_limits_from_components",
        lambda _s: {("b1", "A"): (114.0, 126.0), ("b1", "B"): (111.0, 127.0)},
    )

    out = cli._build_node_voltage_limits_v(_System())
    assert out[("b1", "A")] == (114.0, 126.0)
    assert out[("b1", "B")] == (111.0, 127.0)


def test_report_overvoltage_branching(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "err_console", _DummyConsole())

    db_path = tmp_path / "v.sqlite"
    db_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        cli, "_read_overvoltage_rows", lambda *_a, **_k: (None, [], False)
    )
    with pytest.raises(typer.Exit):
        cli.report_overvoltage(db=db_path, solver=cli.Solver.ac, run_id=None)

    monkeypatch.setattr(
        cli, "_read_overvoltage_rows", lambda *_a, **_k: (None, [], True)
    )
    cli.report_overvoltage(db=db_path, solver=cli.Solver.ac, run_id=None)

    monkeypatch.setattr(
        cli, "_read_overvoltage_rows", lambda *_a, **_k: ("ac_1", [], True)
    )
    cli.report_overvoltage(db=db_path, solver=cli.Solver.ac, run_id=None)

    rows = [
        ("b1", "A", 112.0, 110.0, 111.0),
        ("b2", "A", 108.0, 109.0, 111.0),
        ("b3", "A", 110.5, 110.0, 111.0),
    ]
    monkeypatch.setattr(
        cli, "_read_overvoltage_rows", lambda *_a, **_k: ("ac_1", rows, True)
    )
    cli.report_overvoltage(db=db_path, solver=cli.Solver.ac, run_id=None)


def test_report_overload_branching(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "err_console", _DummyConsole())

    db_path = tmp_path / "o.sqlite"
    db_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(cli, "_read_overload_rows", lambda *_a, **_k: (None, [], False))
    with pytest.raises(typer.Exit):
        cli.report_overload(
            db=db_path, solver=cli.Solver.ldf, run_id=None, dc_severity_only=True
        )

    monkeypatch.setattr(cli, "_read_overload_rows", lambda *_a, **_k: (None, [], True))
    cli.report_overload(
        db=db_path, solver=cli.Solver.ldf, run_id=None, dc_severity_only=True
    )

    monkeypatch.setattr(
        cli, "_read_overload_rows", lambda *_a, **_k: ("ldf_1", [], True)
    )
    cli.report_overload(
        db=db_path, solver=cli.Solver.ldf, run_id=None, dc_severity_only=True
    )

    dc_rows = [
        ("line1", "A", 100.0, 0.0, 100.0, 50.0, 2.1),
        ("line2", "A", 100.0, 0.0, 100.0, 70.0, 1.5),
        ("line3", "A", 100.0, 0.0, 100.0, 90.0, 1.2),
    ]
    monkeypatch.setattr(
        cli, "_read_overload_rows", lambda *_a, **_k: ("dc_1", dc_rows, True)
    )
    cli.report_overload(
        db=db_path, solver=cli.Solver.dc, run_id=None, dc_severity_only=True
    )
    cli.report_overload(
        db=db_path, solver=cli.Solver.dc, run_id=None, dc_severity_only=False
    )


def test_db_schema_empty_schema_prints_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "_read_db_schema", lambda *_a, **_k: [])

    db_path = tmp_path / "empty.sqlite"
    db_path.write_text("x", encoding="utf-8")

    cli.db_schema(db=db_path, include_internal=False)


def test_cli_main_invokes_app(monkeypatch):
    calls = {"n": 0}

    monkeypatch.setattr(cli, "app", lambda: calls.__setitem__("n", calls["n"] + 1))
    cli.main()
    assert calls["n"] == 1


def test_branch_phase_series_impedance_paths_with_example_model():
    model = Path(__file__).resolve().parents[1] / "examples" / "models" / "p5r.json"
    try:
        system = DistributionSystem.from_json(str(model))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot load example model with installed infrasys: {exc}")
    branches = list(system.get_components(DistributionBranchBase))
    assert branches

    branch = branches[0]
    phase_name = cli._phase_name(next(p for p in branch.phases if p != Phase.N))
    r_ohm, x_ohm = cli._branch_phase_series_impedance_ohm(branch, phase_name)
    assert isinstance(r_ohm, float)
    assert isinstance(x_ohm, float)

    r_bad, x_bad = cli._branch_phase_series_impedance_ohm(branch, "Z")
    assert (r_bad, x_bad) == (0.0, 0.0)


def test_branch_phase_series_impedance_paths_mocked():
    """Exercise _branch_phase_series_impedance_ohm without loading a real model."""

    class _Quantity:
        def __init__(self, value, unit):
            self._val = value
            self._unit = unit

        def to(self, _unit):
            return self

        @property
        def magnitude(self):
            return self._val

    class _Equipment:
        r_matrix = _Quantity([[0.01, 0], [0, 0.01]], "ohm/m")
        x_matrix = _Quantity([[0.005, 0], [0, 0.005]], "ohm/m")

    class _Branch:
        phases = [Phase.A, Phase.N]
        length = _Quantity(100.0, "m")
        equipment = _Equipment()

    branch = _Branch()
    r, x = cli._branch_phase_series_impedance_ohm(branch, "A")
    assert r == pytest.approx(1.0)
    assert x == pytest.approx(0.5)

    # Unknown phase
    r_bad, x_bad = cli._branch_phase_series_impedance_ohm(branch, "Z")
    assert (r_bad, x_bad) == (0.0, 0.0)

    # Sequence impedance fallback path
    class _SeqEquipment:
        pos_seq_resistance = _Quantity(0.02, "ohm/m")
        pos_seq_reactance = _Quantity(0.01, "ohm/m")

    class _SeqBranch:
        phases = [Phase.A]
        length = _Quantity(50.0, "m")
        equipment = _SeqEquipment()

    r2, x2 = cli._branch_phase_series_impedance_ohm(_SeqBranch(), "A")
    assert r2 == pytest.approx(1.0)
    assert x2 == pytest.approx(0.5)

    # No impedance info at all
    class _NoImpedanceBranch:
        phases = [Phase.A]
        equipment = object()

    r3, x3 = cli._branch_phase_series_impedance_ohm(_NoImpedanceBranch(), "A")
    assert (r3, x3) == (0.0, 0.0)


def test_build_lindistflow_loading_limits_with_example_model():
    model = Path(__file__).resolve().parents[1] / "examples" / "models" / "p5r.json"
    try:
        system = DistributionSystem.from_json(str(model))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot load example model with installed infrasys: {exc}")
    out = cli._build_lindistflow_loading_limits_va(system)
    assert isinstance(out, dict)
    assert len(out) > 0


def test_build_lindistflow_loading_limits_mocked(monkeypatch):
    """Exercise _build_lindistflow_loading_limits_va without loading a real model."""
    import networkx as nx

    class _Quantity:
        def __init__(self, value):
            self._val = value

        def to(self, _unit):
            return self

        @property
        def magnitude(self):
            return self._val

    class _Bus:
        def __init__(self, name, phases, rated_v):
            self.name = name
            self.phases = phases
            self.rated_voltage = _Quantity(rated_v)

    class _Equipment:
        ampacity = _Quantity(100.0)

    class _Branch:
        def __init__(self, name, phases):
            self.name = name
            self.phases = phases
            self.equipment = _Equipment()

    bus_a = _Bus("bus_a", [Phase.A, Phase.N], 7200.0)
    bus_b = _Bus("bus_b", [Phase.A, Phase.N], 7200.0)
    branch = _Branch("line1", [Phase.A, Phase.N])

    G = nx.DiGraph()
    G.add_edge("bus_a", "bus_b", name="line1")

    class _System:
        def get_components(self, cls):
            cls_name = getattr(cls, "__name__", str(cls))
            if "Bus" in cls_name:
                return [bus_a, bus_b]
            return [branch]

        def get_directed_graph(self, return_radial_network=True):
            return G

    out = cli._build_lindistflow_loading_limits_va(_System())
    assert isinstance(out, dict)
    assert len(out) > 0
    assert ("line1", "A") in out
    assert out[("line1", "A")] == pytest.approx(7200.0 * 100.0)


def test_build_ac_and_dc_branch_loading_helpers_with_example_model():
    model = Path(__file__).resolve().parents[1] / "examples" / "models" / "p5r.json"
    try:
        system = DistributionSystem.from_json(str(model))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot load example model with installed infrasys: {exc}")
    yres = calculate_ybus(system)

    class _ACRes:
        ybus_result = yres
        voltage = np.ones(len(yres.index_to_label), dtype=np.complex128)

    ac_loading, ac_limits, ac_flow = cli._build_ac_branch_loading_from_result(
        system, _ACRes()
    )
    assert isinstance(ac_loading, dict)
    assert isinstance(ac_limits, dict)
    assert isinstance(ac_flow, dict)

    theta = {label: 0.0 for label in yres.index_to_label}

    class _DCRes:
        theta_rad = theta

    dc_loading, dc_limits, dc_flow = cli._build_dc_branch_loading_from_result(
        system, _DCRes()
    )
    assert isinstance(dc_loading, dict)
    assert isinstance(dc_limits, dict)
    assert isinstance(dc_flow, dict)


def test_build_ac_and_dc_branch_loading_helpers_mocked(monkeypatch):
    """Exercise AC/DC branch-loading helpers without loading a real model."""
    import networkx as nx

    class _Quantity:
        def __init__(self, value):
            self._val = value

        def to(self, _unit):
            return self

        @property
        def magnitude(self):
            return self._val

    class _Equipment:
        ampacity = _Quantity(100.0)
        r_matrix = _Quantity([[0.01, 0], [0, 0.01]])
        x_matrix = _Quantity([[0.005, 0], [0, 0.005]])

    class _Branch:
        name = "line1"
        phases = [Phase.A, Phase.N]
        length = _Quantity(100.0)
        equipment = _Equipment()
        in_service = True

    branch = _Branch()

    class _Bus:
        def __init__(self, name):
            self.name = name
            self.phases = [Phase.A, Phase.N]
            self.rated_voltage = _Quantity(7200.0)

    bus_a, bus_b = _Bus("bus_a"), _Bus("bus_b")

    G = nx.DiGraph()
    G.add_edge("bus_a", "bus_b", name="line1", type=type(branch))

    class _System:
        def get_components(self, cls):
            cls_name = getattr(cls, "__name__", str(cls))
            if "Bus" in cls_name:
                return [bus_a, bus_b]
            return [branch]

        def get_directed_graph(self, return_radial_network=True):
            return G

        def get_component(self, cls, name):
            if name == "line1":
                return branch
            raise KeyError(name)

    system = _System()

    # Patch DistributionBranchBase so isinstance(_Branch(), ...) passes inside
    # the helpers which do a local import of DistributionBranchBase.
    _OrigBase = DistributionBranchBase
    _fake_base = type("DistributionBranchBase", (_Branch, _OrigBase), {})
    monkeypatch.setattr(
        "gdm.systems.distribution.components.base.distribution_branch_base.DistributionBranchBase",
        _Branch,
    )

    system = _System()

    # Fake ybus-like result (index_to_label is a list, not a dict)
    class _YbusResult:
        index_to_label = [("bus_a", "A"), ("bus_b", "A")]

    # AC path: voltage of 1+0j at both buses → small current from impedance
    class _ACRes:
        ybus_result = _YbusResult()
        voltage = np.array([7200.0 + 0j, 7100.0 + 0j])

    ac_loading, ac_limits, ac_flow = cli._build_ac_branch_loading_from_result(
        system, _ACRes()
    )
    assert isinstance(ac_loading, dict)
    assert isinstance(ac_limits, dict)
    assert isinstance(ac_flow, dict)
    assert ("line1", "A") in ac_flow

    # DC path
    theta = {("bus_a", "A"): 0.01, ("bus_b", "A"): 0.0}

    class _DCRes:
        theta_rad = theta

    dc_loading, dc_limits, dc_flow = cli._build_dc_branch_loading_from_result(
        system, _DCRes()
    )
    assert isinstance(dc_loading, dict)
    assert isinstance(dc_limits, dict)
    assert isinstance(dc_flow, dict)
    assert ("line1", "A") in dc_flow


def test_table_and_run_id_helpers_cover_edge_paths(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                implementation TEXT NOT NULL,
                success INTEGER NOT NULL,
                message TEXT,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE t1 (a INTEGER, b INTEGER);
            """
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            ("ac_1", "ac_opf", 1, "ok", "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            ("ac_2", "ac_opf", 1, "ok", "2026-01-02T00:00:00Z"),
        )
        conn.commit()

        assert cli._table_has_columns(conn, "t1", {"a"}) is True
        assert cli._table_has_columns(conn, "t1", {"a", "c"}) is False
        assert cli._resolve_latest_run_id(conn, "ac_opf", "ac_1") == "ac_1"
        assert cli._resolve_latest_run_id(conn, "ac_opf", "missing") is None
        assert cli._resolve_latest_run_id(conn, "ac_opf", None) == "ac_2"
    finally:
        conn.close()


def test_read_overload_rows_unknown_implementation(tmp_path):
    db_path = tmp_path / "unknown.sqlite"
    sqlite3.connect(str(db_path)).close()
    run_id, rows, has_columns = cli._read_overload_rows(
        str(db_path), "unknown_impl", None
    )
    assert run_id is None
    assert rows == []
    assert has_columns is False


def test_export_invokes_ac_dc_branch_helpers_when_results_include_expected_fields(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli, "console", _DummyConsole())
    monkeypatch.setattr(cli, "_load_system", lambda _model: _FakeSystem())
    monkeypatch.setattr(cli, "_build_node_voltage_limits_v", lambda _system: {})
    monkeypatch.setattr(cli, "_build_lindistflow_loading_limits_va", lambda _system: {})

    calls = {"ac": 0, "dc": 0}

    def _fake_ac_branch_builder(_system, _result):
        calls["ac"] += 1
        return ({("l1", "A"): 1.0}, {("l1", "A"): 2.0}, {("l1", "A"): (1.0, 0.0)})

    def _fake_dc_branch_builder(_system, _result):
        calls["dc"] += 1
        return ({("l2", "A"): 3.0}, {("l2", "A"): 4.0}, {("l2", "A"): (3.0, 0.0)})

    monkeypatch.setattr(
        cli, "_build_ac_branch_loading_from_result", _fake_ac_branch_builder
    )
    monkeypatch.setattr(
        cli, "_build_dc_branch_loading_from_result", _fake_dc_branch_builder
    )

    class _ACResult:
        voltage = [1 + 0j]

        class _Y:
            index_to_label = [("b", "A")]

        ybus_result = _Y()

    class _DCResult:
        theta_rad = {("b", "A"): 0.0}

    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.ac,
        lambda _system: {"success": True, "source_p": 1.0, "result": _ACResult()},
    )
    monkeypatch.setitem(
        cli.SOLVER_MAP,
        cli.Solver.dc,
        lambda _system: {"success": True, "source_p": 1.0, "result": _DCResult()},
    )

    captured = {}

    def _fake_export_all_results_to_sqlite(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "gdm_flow.sqlite_export.export_all_results_to_sqlite",
        _fake_export_all_results_to_sqlite,
    )

    cli.export(
        model=Path("ignored.json"),
        db=tmp_path / "out.sqlite",
        solver=[cli.Solver.ac, cli.Solver.dc],
    )

    assert calls["ac"] == 1
    assert calls["dc"] == 1
    assert captured["ac_branch_loading_va"] == {("l1", "A"): 1.0}
    assert captured["dc_branch_loading_va"] == {("l2", "A"): 3.0}
