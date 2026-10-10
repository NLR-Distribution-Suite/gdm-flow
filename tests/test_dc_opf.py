import pytest

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionLoad,
    DistributionSolar,
    MatrixImpedanceBranch,
)

from gdm_flow import (
    DCGenerator,
    build_dc_generators_from_components,
    build_dc_load_profile_from_components,
    solve_dc_opf,
)


pytest.importorskip("scipy")


def test_build_dc_helpers_from_components():
    system = DistributionSystem(
        auto_add_composed_components=True, name="dc-helper-test"
    )
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)
    bus = branch.buses[1]

    load = DistributionLoad.example()
    load.bus = bus
    system.add_component(load)

    solar = DistributionSolar.example()
    solar.bus = bus
    system.add_component(solar)

    demand = build_dc_load_profile_from_components(system)
    generators = build_dc_generators_from_components(system)

    assert (bus.name, "A") in demand
    assert len(generators) >= 1


def test_solve_dc_opf_with_custom_generator():
    system = DistributionSystem(auto_add_composed_components=True, name="dc-opf-test")
    branch = MatrixImpedanceBranch.example()
    system.add_component(branch)

    bus_slack = branch.buses[0].name
    bus_load = branch.buses[1].name

    demand = {(bus_load, "A"): 2000.0}
    generators = [
        DCGenerator(
            name="gen-a",
            node=(bus_load, "A"),
            p_min_w=0.0,
            p_max_w=5000.0,
            cost_quadratic=1e-6,
            cost_linear=1.0,
        )
    ]

    result = solve_dc_opf(
        system,
        generators=generators,
        demand_w=demand,
        slack_label=(bus_slack, "A"),
    )

    assert result.success
    assert 0.0 <= result.generator_dispatch_w["gen-a"] <= 5000.0
    # Non-slack balance should be close to zero by construction.
    assert abs(result.nodal_balance_w[(bus_load, "A")]) < 1e-4
