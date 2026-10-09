from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionLoad,
    DistributionVoltageSource,
    MatrixImpedanceBranch,
)

from gdm_flow import build_lindistflow_net_injections_from_components, solve_lindistflow


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
