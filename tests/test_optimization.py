import numpy as np
import pytest

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import MatrixImpedanceBranch

from gdm_flow import optimize_ac_power_flow


pytest.importorskip("scipy")


def test_optimize_ac_power_flow_improves_objective():
    system = DistributionSystem(auto_add_composed_components=True, name="opf-test")
    branch = MatrixImpedanceBranch.example()
    bus_2 = branch.buses[1].name
    system.add_component(branch)

    p_spec = {
        (bus_2, "A"): -5_000.0,
        (bus_2, "B"): -5_000.0,
        (bus_2, "C"): -5_000.0,
    }
    q_spec = {
        (bus_2, "A"): -1_000.0,
        (bus_2, "B"): -1_000.0,
        (bus_2, "C"): -1_000.0,
    }

    result = optimize_ac_power_flow(
        system,
        p_spec_w=p_spec,
        q_spec_var=q_spec,
    )

    assert result.success
    assert result.final_objective <= result.initial_objective
    assert np.max(np.abs(result.voltage)) > 0
