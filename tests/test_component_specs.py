import pytest

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionCapacitor,
    DistributionLoad,
    DistributionRegulator,
    DistributionSolar,
    MatrixImpedanceBranch,
)
from gdm.systems.distribution.enums import Phase
from gdm.quantities import ActivePower, ReactivePower, Voltage

from gdm_flow import (
    build_nodal_power_specs_from_components,
    build_regulator_voltage_limits_from_components,
    build_regulator_voltage_targets_from_components,
    optimize_ac_power_flow,
    optimize_ac_power_flow_from_components,
)


pytest.importorskip("scipy")


def test_build_nodal_power_specs_from_components_load_and_solar():
    system = DistributionSystem(
        auto_add_composed_components=True, name="spec-build-test"
    )
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)

    bus = branch.buses[1]

    load = DistributionLoad.example()
    load.bus = bus
    load.phases = [phase for phase in bus.phases if phase.name in ["A", "B", "C"]]
    system.add_component(load)

    solar = DistributionSolar.example()
    solar.bus = bus
    solar.phases = [phase for phase in bus.phases if phase.name in ["A", "B", "C"]]
    solar.active_power = ActivePower(3000, "watt")
    solar.reactive_power = ReactivePower(0, "var")
    system.add_component(solar)

    p_spec, q_spec = build_nodal_power_specs_from_components(system)

    # Load example has 2.5 kW per phase; solar contributes +1 kW per phase.
    for phase_name in ["A", "B", "C"]:
        label = (bus.name, phase_name)
        assert label in p_spec
        assert p_spec[label] == pytest.approx(-1500.0)
        assert q_spec[label] == pytest.approx(0.0)


def test_optimize_ac_power_flow_from_components_runs():
    system = DistributionSystem(auto_add_composed_components=True, name="auto-opf-test")
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)

    bus = branch.buses[1]

    load = DistributionLoad.example()
    load.bus = bus
    system.add_component(load)

    result = optimize_ac_power_flow_from_components(system)

    assert result.success
    assert result.final_objective <= result.initial_objective


def test_build_nodal_power_specs_includes_capacitor_q_injection():
    system = DistributionSystem(auto_add_composed_components=True, name="spec-cap-test")
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)
    bus = branch.buses[1]

    capacitor = DistributionCapacitor.example()
    capacitor.bus = bus
    capacitor.phases = [Phase.A, Phase.B, Phase.C]
    system.add_component(capacitor)

    _, q_spec = build_nodal_power_specs_from_components(
        system, include_loads=False, include_solar=False
    )

    for phase_name in ["A", "B", "C"]:
        label = (bus.name, phase_name)
        assert q_spec[label] == pytest.approx(200_000.0)


def test_build_regulator_voltage_targets_from_components():
    system = DistributionSystem(
        auto_add_composed_components=True, name="reg-target-test"
    )
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)
    bus = branch.buses[1]

    regulator = DistributionRegulator.example()
    phase_order = [Phase.A, Phase.B, Phase.C]
    for i, controller in enumerate(regulator.controllers):
        controller.controlled_bus = bus
        controller.controlled_phase = phase_order[i]
        controller.v_setpoint = Voltage(120, "volt")
        controller.pt_ratio = 1.0
    system.add_component(regulator)

    targets = build_regulator_voltage_targets_from_components(system)

    for phase_name in ["A", "B", "C"]:
        label = (bus.name, phase_name)
        assert targets[label] == pytest.approx(120.0)


def test_build_regulator_voltage_limits_from_components():
    system = DistributionSystem(
        auto_add_composed_components=True, name="reg-limit-test"
    )
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)
    bus = branch.buses[1]

    regulator = DistributionRegulator.example()
    phase_order = [Phase.A, Phase.B, Phase.C]
    for i, controller in enumerate(regulator.controllers):
        controller.controlled_bus = bus
        controller.controlled_phase = phase_order[i]
        controller.min_v_limit = Voltage(110, "volt")
        controller.max_v_limit = Voltage(130, "volt")
        controller.pt_ratio = 1.0
    system.add_component(regulator)

    limits = build_regulator_voltage_limits_from_components(system)
    for phase_name in ["A", "B", "C"]:
        label = (bus.name, phase_name)
        assert limits[label][0] == pytest.approx(110.0)
        assert limits[label][1] == pytest.approx(130.0)


def test_optimize_ac_power_flow_honors_regulator_hard_limits():
    system = DistributionSystem(
        auto_add_composed_components=True, name="reg-hard-bound-opf"
    )
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)
    bus = branch.buses[1]

    limits = {(bus.name, "A"): (390.0, 395.0)}

    result = optimize_ac_power_flow(
        system,
        voltage_limits_v=limits,
        vm_min_pu=0.8,
        vm_max_pu=1.2,
    )

    idx = result.ybus_result.label_to_index[(bus.name, "A")]
    vm = abs(result.voltage[idx])
    assert 390.0 - 1e-6 <= vm <= 395.0 + 1e-6
