from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionLoad,
    DistributionVoltageSource,
    MatrixImpedanceBranch,
)
from gdm.systems.distribution.components import DistributionBus
from gdm.systems.distribution.enums import VoltageTypes
from gdm.systems.substation.components import PowerTransformer
from gdm.systems.substation.reference_designs import (
    hv_mv_substation_with_distribution_feeders,
)
from gdm.systems.substation.topology import FeederBoundary

from gdm_flow import build_lindistflow_net_injections_from_components, solve_lindistflow
from gdm_flow.lindistflow import _transformer_phase_impedance_and_ratio


def test_build_lindistflow_net_injections_from_components_load_only():
    system = DistributionSystem(
        auto_add_composed_components=True, name="lindistflow-inj-test"
    )
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)

    load = DistributionLoad.example()
    load.bus = branch.buses[1]
    system.add_component(load)

    p_net, q_net = build_lindistflow_net_injections_from_components(
        system,
        include_loads=True,
        include_solar=False,
        include_battery=False,
        include_capacitor=False,
    )

    assert (branch.buses[1].name, "A") in p_net
    assert p_net[(branch.buses[1].name, "A")] > 0.0
    assert q_net[(branch.buses[1].name, "A")] >= 0.0


def test_solve_lindistflow_voltage_drop_on_loaded_branch():
    system = DistributionSystem(
        auto_add_composed_components=True, name="lindistflow-solve-test"
    )
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)

    vsource = DistributionVoltageSource.example()
    vsource.bus = branch.buses[0]
    vsource.phases = branch.phases
    system.add_component(vsource)

    source = branch.buses[0].name
    downstream = branch.buses[1].name

    # Put active/reactive demand on the downstream bus so linear voltage drop is non-zero.
    p_net = {(downstream, "A"): 5_000.0}
    q_net = {(downstream, "A"): 1_000.0}

    result = solve_lindistflow(system, p_net_w=p_net, q_net_var=q_net)

    assert result.success
    assert (source, "A") in result.voltage_v
    assert (downstream, "A") in result.voltage_v
    assert result.voltage_v[(downstream, "A")] < result.voltage_v[(source, "A")]

    # Branch power should flow from source to downstream for the loaded phase.
    assert (branch.name, "A") in result.p_flow_w
    assert result.p_flow_w[(branch.name, "A")] > 0.0


def test_lindistflow_traverses_combined_station_electrical_equipment():
    station, feeder_collection = hv_mv_substation_with_distribution_feeders()
    feeder_systems = []
    for boundary in station.get_components(FeederBoundary):
        boundary.source_equivalent_id = f"{boundary.feeder_id}-source-bus"
        feeder_system = DistributionSystem(
            name=f"{boundary.feeder_id}-system",
            auto_add_composed_components=True,
        )
        for component in feeder_collection.iter_all_components():
            feeder = getattr(component, "feeder", None)
            if feeder is not None and feeder.name == boundary.feeder_id:
                if not feeder_system.has_component(component):
                    feeder_system.add_component(component)
        feeder_systems.append(feeder_system)

    complete = station.to_distribution_system(feeder_systems)
    result = solve_lindistflow(
        complete,
        p_net_w={("feeder-1-load-bus", "A"): 5_000.0},
        q_net_var={("feeder-1-load-bus", "A"): 1_000.0},
    )

    assert result.success, result.message
    solved_buses = {bus_name for bus_name, _phase in result.voltage_v}
    assert solved_buses == {bus.name for bus in complete.get_components(DistributionBus)}
    assert result.p_flow_w[("feeder-1-main-line", "A")] == 5_000.0
    assert result.q_flow_var[("feeder-1-main-line", "A")] == 1_000.0
    assert result.voltage_v[("feeder-1-load-bus", "A")] < result.voltage_v[("feeder-1-line-node", "A")]


def test_lindistflow_station_transformer_supports_reverse_tree_orientation():
    transformer = PowerTransformer.example()
    _r_forward, _x_forward, forward_ratio = (
        _transformer_phase_impedance_and_ratio(transformer)
    )
    _r_reverse, _x_reverse, reverse_ratio = (
        _transformer_phase_impedance_and_ratio(transformer, reverse=True)
    )

    assert abs(forward_ratio * reverse_ratio - 1.0) < 1e-12


def test_lindistflow_reports_disconnected_buses_instead_of_false_success():
    system = DistributionSystem(
        auto_add_composed_components=True,
        name="lindistflow-island-test",
    )
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)
    source = DistributionVoltageSource.example()
    source.bus = branch.buses[0]
    source.phases = branch.phases
    system.add_component(source)
    system.add_component(
        DistributionBus(
            name="island-bus",
            voltage_type=VoltageTypes.LINE_TO_LINE,
            phases=list(branch.buses[0].phases),
            voltagelimits=[],
            rated_voltage=branch.buses[0].rated_voltage,
        )
    )

    result = solve_lindistflow(system)

    assert not result.success
    assert "disconnected from the source" in result.message
