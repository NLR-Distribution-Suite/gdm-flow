from __future__ import annotations

import numpy as np
import pytest

from gdm.systems.distribution.enums import Phase

from gdm_flow import ac_opf as ac

sp = pytest.importorskip("scipy.sparse")


class _FakeSystem:
    """Minimal mock system that supports get_components() returning empty."""

    def get_components(self, *args, **kwargs):
        return iter([])


def test_build_voltage_from_state_and_residual_with_targets():
    n = 2
    slack_set = {0}
    theta0 = np.zeros(n, dtype=float)

    # non-slack: theta=0, vm=1.1
    x = np.array([0.0, 1.1], dtype=float)
    v = ac._build_voltage_from_state(x, n, slack_set, theta0)
    assert np.isclose(abs(v[0]), 1.0)
    assert np.isclose(abs(v[1]), 1.1)

    residual = ac._objective_residual(
        x,
        ybus_pu=np.zeros((2, 2), dtype=np.complex128),
        s_spec_pu=np.zeros(2, dtype=np.complex128),
        s_scale_pu=np.ones(2, dtype=float),
        n=n,
        slack_set=slack_set,
        theta0=theta0,
        voltage_reg_weight=2.0,
        voltage_targets_pu={("b2", "A"): 1.0},
        labels=[("b1", "A"), ("b2", "A")],
        voltage_target_weight=3.0,
    )

    # [P mismatch, Q mismatch, vm regularization, target term]
    assert residual.shape == (4,)
    assert np.isclose(residual[2], 0.2)
    assert np.isclose(residual[3], 0.3)


def test_optimize_from_components_auto_detects_source_slack(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        ac,
        "build_nodal_power_specs_from_components",
        lambda _s, **_k: ({("n1", "A"): -100.0}, {("n1", "A"): -10.0}),
    )
    monkeypatch.setattr(
        ac,
        "build_regulator_voltage_targets_from_components",
        lambda _s: {("n1", "A"): 230.0},
    )
    monkeypatch.setattr(
        ac,
        "build_regulator_voltage_limits_from_components",
        lambda _s: {("n1", "A"): (220.0, 240.0)},
    )

    def _fake_optimize(_system, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(ac, "optimize_ac_power_flow", _fake_optimize)

    class _SourceBus:
        name = "src"
        phases = [Phase.A, Phase.B, Phase.N]

    class _System:
        def get_source_bus(self):
            return _SourceBus()

    result = ac.optimize_ac_power_flow_from_components(_System())

    assert result == "ok"
    assert captured["slack_label"] == [("src", "A"), ("src", "B")]
    assert captured["p_spec_w"][("n1", "A")] == -100.0


class _FakeYbusResult:
    def __init__(self, labels):
        self.index_to_label = labels
        self.label_to_index = {lbl: i for i, lbl in enumerate(labels)}
        n = len(labels)
        self.ybus = np.eye(n, dtype=np.complex128) * 10.0
        if n >= 2:
            self.ybus[0, 1] = -1.0
            self.ybus[1, 0] = -1.0


def test_optimize_ac_power_flow_rejects_invalid_slack_label(monkeypatch):
    monkeypatch.setattr(
        ac,
        "calculate_ybus",
        lambda *_a, **_k: _FakeYbusResult([("b1", "A"), ("b2", "A")]),
    )
    monkeypatch.setattr(
        ac,
        "_build_nominal_voltage_map",
        lambda _s: {("b1", "A"): 230.0, ("b2", "A"): 230.0},
    )

    with pytest.raises(ValueError):
        ac.optimize_ac_power_flow(
            system=_FakeSystem(),
            slack_label=("unknown", "A"),
        )


def test_optimize_ac_power_flow_rejects_infeasible_voltage_limits(monkeypatch):
    monkeypatch.setattr(
        ac,
        "calculate_ybus",
        lambda *_a, **_k: _FakeYbusResult([("b1", "A"), ("b2", "A")]),
    )
    monkeypatch.setattr(
        ac,
        "_build_nominal_voltage_map",
        lambda _s: {("b1", "A"): 100.0, ("b2", "A"): 100.0},
    )

    with pytest.raises(ValueError):
        ac.optimize_ac_power_flow(
            system=_FakeSystem(),
            voltage_limits_v={("b2", "A"): (120.0, 80.0)},
        )


def test_objective_jacobian_dense_and_target_rows():
    n = 2
    slack_set = {0}
    theta0 = np.zeros(n, dtype=float)
    x = np.array([0.0, 1.02], dtype=float)

    jac = ac._objective_jacobian(
        x,
        ybus_pu=np.array([[2.0 + 0.0j, -1.0 + 0.0j], [-1.0 + 0.0j, 2.0 + 0.0j]]),
        s_spec_pu=np.zeros(n, dtype=np.complex128),
        s_scale_pu=np.ones(n, dtype=float),
        n=n,
        slack_set=slack_set,
        theta0=theta0,
        voltage_reg_weight=1.0,
        voltage_targets_pu={("b2", "A"): 1.0},
        labels=[("b1", "A"), ("b2", "A")],
        voltage_target_weight=2.0,
    )

    # m=1 non-slack -> rows: P,Q,reg,target = 4 ; cols: theta,vm = 2.
    assert isinstance(jac, np.ndarray)
    assert jac.shape == (4, 2)


def test_objective_jacobian_large_returns_sparse():
    n = 1002
    slack_set = {0}
    theta0 = np.zeros(n, dtype=float)

    rows = np.arange(n)
    cols = np.arange(n)
    data = np.ones(n, dtype=np.complex128)
    ybus_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    x = np.concatenate([np.zeros(n - 1, dtype=float), np.ones(n - 1, dtype=float)])
    jac = ac._objective_jacobian(
        x,
        ybus_pu=ybus_sparse,
        s_spec_pu=np.zeros(n, dtype=np.complex128),
        s_scale_pu=np.ones(n, dtype=float),
        n=n,
        slack_set=slack_set,
        theta0=theta0,
        voltage_reg_weight=1.0,
        voltage_targets_pu=None,
        labels=[(f"b{i}", "A") for i in range(n)],
        voltage_target_weight=1.0,
    )

    assert sp.issparse(jac)
    assert jac.shape == (3 * (n - 1), 2 * (n - 1))


def test_optimize_rejects_unknown_slack_in_list(monkeypatch):
    monkeypatch.setattr(
        ac,
        "calculate_ybus",
        lambda *_a, **_k: _FakeYbusResult([("b1", "A"), ("b2", "A")]),
    )
    monkeypatch.setattr(
        ac,
        "_build_nominal_voltage_map",
        lambda _s: {("b1", "A"): 230.0, ("b2", "A"): 230.0},
    )

    with pytest.raises(ValueError):
        ac.optimize_ac_power_flow(
            system=_FakeSystem(),
            slack_label=[("b1", "A"), ("unknown", "A")],
        )


def test_optimize_from_components_falls_back_when_source_lookup_fails(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        ac,
        "build_nodal_power_specs_from_components",
        lambda _s, **_k: ({("n1", "A"): -10.0}, {("n1", "A"): -1.0}),
    )
    monkeypatch.setattr(
        ac,
        "build_regulator_voltage_targets_from_components",
        lambda _s: {("n1", "A"): 230.0},
    )
    monkeypatch.setattr(
        ac,
        "build_regulator_voltage_limits_from_components",
        lambda _s: {("n1", "A"): (220.0, 240.0)},
    )

    def _fake_optimize(_system, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(ac, "optimize_ac_power_flow", _fake_optimize)

    class _System:
        def get_source_bus(self):
            raise RuntimeError("no source")

    result = ac.optimize_ac_power_flow_from_components(_System())

    assert result == "ok"
    assert captured["slack_label"] is None
