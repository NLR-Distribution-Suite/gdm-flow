from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBattery,
    DistributionCapacitor,
    DistributionLoad,
    DistributionReactor,
    DistributionSolar,
    DistributionVoltageSource,
    MatrixImpedanceBranch,
    MatrixImpedanceSwitch,
)
from gdm.systems.distribution.enums import Phase, VoltageTypes

from gdm_flow import lindistflow as ldf


def test_lindistflow_reactor_impedance_is_lumped_and_validated():
    reactor = DistributionReactor.example()
    reactor.length = type(reactor.length)(250, "meter")
    reactor.equipment.resistance = type(reactor.equipment.resistance)(0.5, "ohm")
    assert ldf._branch_phase_impedance_ohm(reactor, "A") == (0.5, 1.0)
    assert ldf._branch_phase_impedance_ohm(reactor, "S1") == (0.0, 0.0)
    reactor.equipment.reactance = type(reactor.equipment.reactance)(-1, "ohm")
    with pytest.raises(ValueError, match="Reactor.*impedance"):
        ldf._branch_phase_impedance_ohm(reactor, "A")


class _Q:
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def to(self, _unit):
        return self


class _SeqEq:
    pos_seq_resistance = _Q(0.12)
    pos_seq_reactance = _Q(0.34)


class _SeqBranch:
    def __init__(self):
        self.equipment = _SeqEq()
        self.length = _Q(50.0)
        self.phases = [Phase.A, Phase.B]


class _Winding:
    def __init__(self, rated_voltage, rated_power, num_phases, voltage_type):
        self.rated_voltage = _Q(rated_voltage)
        self.rated_power = _Q(rated_power)
        self.num_phases = num_phases
        self.voltage_type = voltage_type


class _TransformerEq:
    def __init__(self, windings):
        self.windings = windings
        self.pct_full_load_loss = 1.5
        self.winding_reactances = [4.5]


class _Transformer:
    def __init__(self, windings):
        self.equipment = _TransformerEq(windings)


def _simple_radial_system() -> DistributionSystem:
    system = DistributionSystem(auto_add_composed_components=True, name="ldf-extra")
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)

    vsource = DistributionVoltageSource.example()
    vsource.bus = branch.buses[0]
    vsource.phases = branch.phases
    system.add_component(vsource)

    load = DistributionLoad.example()
    load.bus = branch.buses[1]
    system.add_component(load)
    return system


def test_build_lindistflow_injections_include_generation_and_capacitor():
    system = _simple_radial_system()

    p_net, q_net = ldf.build_lindistflow_net_injections_from_components(
        system,
        include_loads=True,
        include_solar=True,
        include_battery=True,
        include_capacitor=True,
        load_scale=0.5,
        solar_scale=1.0,
        battery_scale=1.0,
        capacitor_scale=1.0,
    )

    assert p_net
    assert q_net
    # Ensure at least one modeled node has finite values after mixed contributions.
    any_label = next(iter(p_net.keys()))
    assert np.isfinite(p_net[any_label])
    assert np.isfinite(q_net.get(any_label, 0.0))


def test_build_lindistflow_injections_hits_solar_battery_capacitor_paths():
    system = _simple_radial_system()
    branch = next(iter(system.get_components(MatrixImpedanceBranch)))
    bus = branch.buses[1]

    solar = DistributionSolar.example()
    solar.bus = bus
    system.add_component(solar)

    battery = DistributionBattery.example()
    battery.bus = bus
    system.add_component(battery)

    capacitor = DistributionCapacitor.example()
    capacitor.bus = bus
    system.add_component(capacitor)

    p_net, q_net = ldf.build_lindistflow_net_injections_from_components(
        system,
        include_loads=False,
        include_solar=True,
        include_battery=True,
        include_capacitor=True,
    )

    label = (bus.name, "A")
    assert label in p_net
    assert label in q_net
    # Solar and battery are modeled as negative demand contributions.
    assert p_net[label] <= 0.0


def test_build_lindistflow_injections_skip_out_of_service_and_empty_phases():
    system = _simple_radial_system()
    branch = next(iter(system.get_components(MatrixImpedanceBranch)))
    bus = branch.buses[1]

    solar = DistributionSolar.example()
    solar.bus = bus
    solar.in_service = False
    system.add_component(solar)

    battery = DistributionBattery.example()
    battery.bus = bus
    battery.phases = []
    system.add_component(battery)

    capacitor = DistributionCapacitor.example()
    capacitor.bus = bus
    capacitor.in_service = False
    system.add_component(capacitor)

    p_net, q_net = ldf.build_lindistflow_net_injections_from_components(
        system,
        include_loads=False,
        include_solar=True,
        include_battery=True,
        include_capacitor=True,
    )
    assert p_net == {}
    assert q_net == {}


def test_branch_phase_impedance_matrix_and_sequence_paths():
    branch = MatrixImpedanceBranch.example()
    r_a, x_a = ldf._branch_phase_impedance_ohm(branch, "A")
    r_missing, x_missing = ldf._branch_phase_impedance_ohm(branch, "S1")

    seq_branch = _SeqBranch()
    r_seq, x_seq = ldf._branch_phase_impedance_ohm(seq_branch, "A")

    assert r_a > 0.0 and x_a > 0.0
    assert r_missing == 0.0 and x_missing == 0.0
    assert r_seq > 0.0 and x_seq > 0.0


def test_transformer_phase_impedance_and_ratio_paths():
    primary = _Winding(7200.0, 50_000.0, 1, VoltageTypes.LINE_TO_GROUND)
    secondary = _Winding(240.0, 50_000.0, 1, VoltageTypes.LINE_TO_GROUND)

    xfmr = _Transformer([primary, secondary])
    r, x, a = ldf._transformer_phase_impedance_and_ratio(xfmr)
    assert r > 0.0 and x > 0.0 and a > 1.0

    # Degenerate path: non-positive power returns zero impedance but valid ratio.
    zero_power_primary = _Winding(7200.0, 0.0, 1, VoltageTypes.LINE_TO_GROUND)
    xfmr_zero = _Transformer([zero_power_primary, secondary])
    r0, x0, a0 = ldf._transformer_phase_impedance_and_ratio(xfmr_zero)
    assert r0 == 0.0 and x0 == 0.0 and a0 > 1.0


def test_solve_lindistflow_rebuilds_net_injections_when_inputs_missing(monkeypatch):
    system = _simple_radial_system()
    source = system.get_source_bus().name
    downstream = [
        b.name
        for b in system.get_components(type(system.get_source_bus()))
        if b.name != source
    ][0]

    monkeypatch.setattr(
        ldf,
        "build_lindistflow_net_injections_from_components",
        lambda _s, **_k: ({(downstream, "A"): 1000.0}, {(downstream, "A"): 100.0}),
    )

    result = ldf.solve_lindistflow(
        system, p_net_w=None, q_net_var={(downstream, "A"): 0.0}
    )

    assert result.success
    assert result.p_net_w[(downstream, "A")] == 1000.0


def test_solve_lindistflow_excludes_open_switch_by_default():
    system = DistributionSystem(
        auto_add_composed_components=True, name="ldf-open-switch"
    )
    switch = MatrixImpedanceSwitch.example()
    switch.is_closed = [False, False, False]
    system.add_component(switch)

    vsource = DistributionVoltageSource.example()
    vsource.bus = switch.buses[0]
    vsource.phases = switch.phases
    system.add_component(vsource)

    downstream = switch.buses[1].name
    p_net = {(downstream, "A"): 500.0}
    q_net = {(downstream, "A"): 100.0}

    result = ldf.solve_lindistflow(
        system, p_net_w=p_net, q_net_var=q_net, include_open_switches=False
    )

    # No modeled branch path through fully open switch.
    assert result.p_flow_w == {}


def test_solve_lindistflow_center_tapped_transformer_paths(monkeypatch):
    class _Bus:
        def __init__(self, name, phases):
            self.name = name
            self.phases = phases
            self.rated_voltage = _Q(120.0)
            self.voltage_type = VoltageTypes.LINE_TO_GROUND

    class _Winding:
        def __init__(self, rated_v, rated_p):
            self.rated_voltage = _Q(rated_v)
            self.rated_power = _Q(rated_p)
            self.num_phases = 1
            self.voltage_type = VoltageTypes.LINE_TO_GROUND

    class _XfmrEq:
        def __init__(self):
            self.windings = [
                _Winding(240.0, 20_000.0),
                _Winding(120.0, 10_000.0),
                _Winding(120.0, 10_000.0),
            ]
            self.pct_full_load_loss = 1.0
            self.winding_reactances = [4.0]

    class _Xfmr:
        def __init__(self):
            self.name = "xf1"
            self.in_service = True
            self.buses = [
                _Bus("src", [Phase.A]),
                _Bus("sec", ["S1", "S2"]),
                _Bus("sec", ["S1", "S2"]),
            ]
            self.equipment = _XfmrEq()
            self.winding_phases = [[Phase.A], ["S1"], [Phase.N, "S2"]]

    class _System:
        def __init__(self):
            self._source = _Bus("src", [Phase.A])
            self._sec = _Bus("sec", ["S1", "S2"])
            self._xf = _Xfmr()

        def get_source_bus(self):
            return self._source

        def get_directed_graph(self, return_radial_network=True):
            g = nx.DiGraph()
            g.add_edge("src", "sec", type="DistributionTransformer", name="xf1")
            return g

        def get_component(self, _type, _name):
            return self._xf

        def get_components(self, comp_type):
            if comp_type is ldf.DistributionBus:
                return [self._source, self._sec]
            return []

    class _DummyBranch:
        pass

    monkeypatch.setattr(ldf, "DistributionTransformer", _Xfmr)
    monkeypatch.setattr(ldf, "DistributionBranchBase", _DummyBranch)
    monkeypatch.setattr(ldf, "DistributionBus", _Bus)

    sys = _System()
    p_net = {("sec", "S1"): 400.0, ("sec", "S2"): 300.0}
    q_net = {("sec", "S1"): 40.0, ("sec", "S2"): 30.0}

    out = ldf.solve_lindistflow(sys, p_net_w=p_net, q_net_var=q_net)
    assert out.success
    assert ("sec", "S1") in out.voltage_v
    assert ("sec", "S2") in out.voltage_v
    assert ("xf1", "S1") in out.p_flow_w
    assert ("xf1", "S2") in out.p_flow_w
