from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse as sp_sparse

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBattery,
    DistributionLoad,
    DistributionSolar,
    MatrixImpedanceBranch,
)
from gdm.systems.distribution.enums import Phase

from gdm_flow import dc_opf
from gdm_flow.dc_opf import DCGenerator


pytest.importorskip("scipy")


def test_build_dc_load_profile_with_negative_injections_and_scaling():
    system = DistributionSystem(auto_add_composed_components=True, name="dc-load-extra")
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)
    bus = branch.buses[1]

    load = DistributionLoad.example()
    load.bus = bus
    system.add_component(load)

    solar = DistributionSolar.example()
    solar.bus = bus
    system.add_component(solar)

    battery = DistributionBattery.example()
    battery.bus = bus
    system.add_component(battery)

    demand = dc_opf.build_dc_load_profile_from_components(
        system,
        include_loads=True,
        include_solar_as_negative_load=True,
        include_battery_as_negative_load=True,
        load_scale=1.0,
        solar_scale=0.5,
        battery_scale=0.5,
    )

    assert (bus.name, "A") in demand
    assert np.isfinite(demand[(bus.name, "A")])


def test_build_dc_generators_with_component_filtering():
    system = DistributionSystem(auto_add_composed_components=True, name="dc-gen-extra")
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)
    bus = branch.buses[1]

    solar = DistributionSolar.example()
    solar.bus = bus
    system.add_component(solar)

    battery = DistributionBattery.example()
    battery.bus = bus
    battery.in_service = False
    system.add_component(battery)

    gens = dc_opf.build_dc_generators_from_components(
        system,
        include_solar=True,
        include_battery=True,
        solar_cost_linear=4.0,
        battery_cost_linear=11.0,
    )

    assert any(g.name.startswith("solar:") for g in gens)
    assert not any(g.name.startswith("battery:") for g in gens)


def test_build_dc_load_profile_skips_out_of_service_components():
    system = DistributionSystem(auto_add_composed_components=True, name="dc-load-skip")
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)
    bus = branch.buses[1]

    load = DistributionLoad.example()
    load.bus = bus
    load.in_service = False
    system.add_component(load)

    solar = DistributionSolar.example()
    solar.bus = bus
    solar.in_service = False
    system.add_component(solar)

    battery = DistributionBattery.example()
    battery.bus = bus
    battery.in_service = False
    system.add_component(battery)

    demand = dc_opf.build_dc_load_profile_from_components(
        system,
        include_loads=True,
        include_solar_as_negative_load=True,
        include_battery_as_negative_load=True,
    )
    assert demand == {}


class _FakeYBusResult:
    def __init__(self):
        self.index_to_label = [("slack", "A"), ("load", "A")]
        self.label_to_index = {("slack", "A"): 0, ("load", "A"): 1}
        self.ybus = sp_sparse.csr_matrix(
            np.array([[0 + 10j, 0 - 10j], [0 - 10j, 0 + 10j]], dtype=np.complex128)
        )


class _FakeSystem:
    def get_components(self, _component_type):
        return []


def test_solve_dc_opf_validations(monkeypatch):
    monkeypatch.setattr(dc_opf, "calculate_ybus", lambda *_a, **_k: _FakeYBusResult())

    with pytest.raises(ValueError):
        dc_opf.solve_dc_opf(_FakeSystem(), generators=[])

    with pytest.raises(ValueError):
        dc_opf.solve_dc_opf(
            _FakeSystem(),
            generators=[
                DCGenerator(
                    name="g",
                    node=("unknown", "A"),
                    p_min_w=0.0,
                    p_max_w=1000.0,
                )
            ],
        )

    with pytest.raises(ValueError):
        dc_opf.solve_dc_opf(
            _FakeSystem(),
            generators=[
                DCGenerator(
                    name="g",
                    node=("load", "A"),
                    p_min_w=0.0,
                    p_max_w=1000.0,
                )
            ],
            slack_label=("bad", "A"),
        )


def test_solve_dc_opf_fallback_to_trust_constr(monkeypatch):
    monkeypatch.setattr(dc_opf, "calculate_ybus", lambda *_a, **_k: _FakeYBusResult())

    import scipy.optimize as opt

    monkeypatch.setattr(
        opt,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(success=False, nit=0),
    )

    def _fake_minimize(fun, x0, method, jac, hess, bounds, constraints, options):
        # Force execution of callback paths for coverage.
        _ = fun(x0)
        _ = jac(x0)
        _ = hess(x0)
        _ = constraints[0].A @ x0
        return SimpleNamespace(
            x=np.array([500.0, 0.01]), success=True, message="fallback ok", nit=7
        )

    monkeypatch.setattr(opt, "minimize", _fake_minimize)

    result = dc_opf.solve_dc_opf(
        _FakeSystem(),
        generators=[
            DCGenerator(
                name="g-load",
                node=("load", "A"),
                p_min_w=0.0,
                p_max_w=1000.0,
                cost_linear=1.0,
            )
        ],
        demand_w={("load", "A"): 500.0},
        slack_label=("slack", "A"),
    )

    assert result.success
    assert result.iterations == 7
    assert "fallback" in result.message


def test_solve_dc_opf_from_components_auto_slack_and_grid_generators(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        dc_opf,
        "build_dc_generators_from_components",
        lambda *_a, **_k: [
            DCGenerator(
                name="solar:x:A",
                node=("n2", "A"),
                p_min_w=0.0,
                p_max_w=1000.0,
            )
        ],
    )
    monkeypatch.setattr(
        dc_opf,
        "build_dc_load_profile_from_components",
        lambda *_a, **_k: {("n2", "A"): 1200.0},
    )

    def _fake_solve(_system, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(dc_opf, "solve_dc_opf", _fake_solve)

    class _SourceBus:
        name = "src"
        phases = [Phase.A, Phase.B, Phase.N]

    class _System:
        def get_source_bus(self):
            return _SourceBus()

    out = dc_opf.solve_dc_opf_from_components(_System(), include_slack_generator=True)

    assert out == "ok"
    assert captured["slack_label"] == [("src", "A"), ("src", "B")]
    grid_names = [g.name for g in captured["generators"] if g.name.startswith("grid:")]
    assert set(grid_names) == {"grid:src:A", "grid:src:B"}
