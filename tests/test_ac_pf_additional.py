"""Tests for ac_pf module and _utils module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gdm_flow._utils import _phase_name, _phase_voltage
from gdm_flow.ac_pf import ACPowerFlowResult, solve_ac_power_flow


MODEL_PATH = Path("examples/models/p5r.json")


@pytest.fixture()
def system():
    if not MODEL_PATH.exists():
        pytest.skip("p5r.json model not found")
    from gdm.systems.distribution import DistributionSystem

    return DistributionSystem.from_json(str(MODEL_PATH))


class TestUtils:
    def test_split_phase_line_to_line_voltage_base(self):
        from gdm.systems.distribution.enums import VoltageTypes

        voltage = MagicMock()
        voltage.to.return_value.magnitude = 240.0
        assert (
            _phase_voltage(voltage, VoltageTypes.LINE_TO_LINE, split_phase=True)
            == 120.0
        )

    def test_phase_name_from_enum(self):
        from gdm.systems.distribution.enums import Phase

        assert _phase_name(Phase.A) == "A"
        assert _phase_name(Phase.B) == "B"
        assert _phase_name(Phase.C) == "C"
        assert _phase_name(Phase.N) == "N"

    def test_phase_name_from_string(self):
        assert _phase_name("A") == "A"
        assert _phase_name("S1") == "S1"

    def test_phase_voltage_line_to_ground(self):
        from gdm.systems.distribution.enums import VoltageTypes

        class _V:
            def to(self, unit):
                return MagicMock(magnitude=120.0)

        result = _phase_voltage(_V(), VoltageTypes.LINE_TO_GROUND)
        assert result == 120.0

    def test_phase_voltage_line_to_line(self):
        from gdm.systems.distribution.enums import VoltageTypes
        import math

        class _V:
            def to(self, unit):
                return MagicMock(magnitude=208.0)

        result = _phase_voltage(_V(), VoltageTypes.LINE_TO_LINE)
        expected = 208.0 / math.sqrt(3)
        assert abs(result - expected) < 0.01


class TestACPowerFlow:
    def test_solve_basic(self, system):
        result = solve_ac_power_flow(system)
        assert isinstance(result, ACPowerFlowResult)
        assert result.success is True
        assert result.iterations > 0
        assert len(result.voltage) > 0
        assert len(result.voltage_pu) > 0

    def test_solve_with_tolerances(self, system):
        result = solve_ac_power_flow(system, max_iterations=200, tolerance=1e-4)
        assert result.success is True

    def test_solve_with_warm_start(self, system):
        # First solve
        result1 = solve_ac_power_flow(system)
        assert result1.success

        # Build warm start from first result
        labels = result1.ybus_result.index_to_label
        v0 = {labels[i]: complex(result1.voltage[i]) for i in range(len(labels))}

        # Second solve with warm start should converge faster
        result2 = solve_ac_power_flow(system, v0_complex=v0)
        assert result2.success
        assert result2.iterations <= result1.iterations

    def test_power_injection_shape_matches_voltage(self, system):
        result = solve_ac_power_flow(system)
        assert result.voltage.shape == result.power_injection.shape

    def test_voltage_pu_reasonable(self, system):
        result = solve_ac_power_flow(system)
        # Per-unit voltages should be near 1.0 for a well-conditioned system
        assert np.all(result.voltage_pu > 0.8)
        assert np.all(result.voltage_pu < 1.2)

    def test_max_mismatch_below_tolerance(self, system):
        result = solve_ac_power_flow(system, tolerance=1e-5)
        if result.success:
            assert result.max_mismatch_pu < 1e-5


@pytest.fixture
def split_phase_line_system():
    from gdm.systems.distribution import DistributionSystem
    from gdm.systems.distribution.components import MatrixImpedanceBranch
    from gdm.systems.distribution.enums import Phase, VoltageTypes

    sample = MatrixImpedanceBranch.example()
    phases = [Phase.S1, Phase.S2]
    buses = [
        bus.model_copy(
            deep=True,
            update={
                "phases": phases,
                "rated_voltage": type(bus.rated_voltage)(240, "volt"),
                "voltage_type": VoltageTypes.LINE_TO_LINE,
            },
        )
        for bus in sample.buses
    ]
    equipment = sample.equipment.model_copy(
        deep=True,
        update={
            "r_matrix": type(sample.equipment.r_matrix)(np.eye(2) * 0.1, "ohm/m"),
            "x_matrix": type(sample.equipment.x_matrix)(np.eye(2) * 0.05, "ohm/m"),
            "c_matrix": type(sample.equipment.c_matrix)(np.zeros((2, 2)), "farad/m"),
        },
    )
    branch = MatrixImpedanceBranch(
        name="split-phase-line",
        phases=phases,
        buses=buses,
        equipment=equipment,
        length=type(sample.length)(1, "meter"),
    )
    system = DistributionSystem(
        name="split-phase-pq", auto_add_composed_components=True
    )
    system.add_component(branch)
    slack = [(buses[0].name, phase.value) for phase in phases]
    pair = tuple((buses[1].name, phase.value) for phase in phases)
    warm = {
        (bus.name, phase.value): 120 if phase == Phase.S1 else -120
        for bus in buses
        for phase in phases
    }
    return system, slack, pair, warm


@pytest.mark.parametrize("power", [-1000 - 200j, -5000 - 1000j, 1000 + 200j])
def test_split_phase_line_to_line_solver_preserves_device_power(
    split_phase_line_system, power
):
    system, slack, pair, warm = split_phase_line_system
    result = solve_ac_power_flow(
        system,
        slack_label=slack,
        v0_complex=warm,
        line_to_line_power_specs={pair: power},
        tolerance=1e-9,
    )
    assert result.success, result.message
    assert result.max_mismatch_pu < 1e-9
    indices = [result.ybus_result.label_to_index[label] for label in pair]
    currents = result.ybus_result.ybus @ result.voltage
    np.testing.assert_allclose(currents[indices[0]], -currents[indices[1]], atol=1e-8)
    differential = result.voltage[indices[0]] - result.voltage[indices[1]]
    np.testing.assert_allclose(
        differential * np.conj(currents[indices[0]]), power, rtol=1e-8, atol=1e-5
    )


@pytest.mark.parametrize("invalid", ["identical", "unknown", "nonfinite", "collapsed"])
def test_split_phase_line_to_line_solver_rejects_invalid_inputs(
    split_phase_line_system, invalid
):
    system, slack, pair, warm = split_phase_line_system
    power = -1000 - 200j
    message = "distinct known terminals and finite VA"
    if invalid == "identical":
        pair = (pair[0], pair[0])
    elif invalid == "unknown":
        pair = (pair[0], ("missing", "S2"))
    elif invalid == "nonfinite":
        power = complex(np.nan)
    else:
        warm[pair[1]] = warm[pair[0]]
        message = "collapsed to zero"
    with pytest.raises(ValueError, match=message):
        solve_ac_power_flow(
            system,
            slack_label=slack,
            v0_complex=warm,
            line_to_line_power_specs={pair: power},
        )


def test_split_phase_delta_uses_terminal_pair_not_fixed_nodal_power(monkeypatch):
    from gdm.systems.distribution import DistributionSystem
    from gdm.systems.distribution.components import DistributionLoad
    from gdm.systems.distribution.enums import ConnectionType, Phase
    from gdm_flow import ac_pf
    from gdm_flow.ac_opf import build_nodal_power_specs_from_components

    sample = DistributionLoad.example()
    sample.bus.phases = [Phase.S2, Phase.S1]
    equipment = type(sample.equipment)(
        name="ll-equipment",
        connection_type=ConnectionType.DELTA,
        phase_loads=sample.equipment.phase_loads[:2],
    )
    sample = DistributionLoad(
        name="ll-load", bus=sample.bus, phases=[Phase.S2, Phase.S1], equipment=equipment
    )
    system = DistributionSystem(name="ll-load", auto_add_composed_components=True)
    system.add_component(sample)
    with pytest.raises(ValueError, match="connection-aware"):
        build_nodal_power_specs_from_components(system)
    calls = []
    monkeypatch.setattr(
        ac_pf, "solve_ac_power_flow", lambda system, **options: calls.append(options)
    )
    ac_pf.solve_ac_power_flow_from_components(system, load_scale=0.5)
    assert calls[0]["p_spec_w"] == {}
    pair = ((sample.bus.name, "S1"), (sample.bus.name, "S2"))
    expected = -0.5 * sum(
        complex(
            float(load.real_power.to("watt").magnitude),
            float(load.reactive_power.to("var").magnitude),
        )
        for load in sample.equipment.phase_loads
    )
    assert calls[0]["line_to_line_power_specs"] == {pair: expected}
