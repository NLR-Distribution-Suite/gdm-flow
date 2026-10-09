"""Additional tests for fix module — detect, engine, and strategies."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gdm_flow.fix.detect import (
    BranchLoadingViolation,
    ViolationReport,
    VoltageViolation,
    _branch_loading_limits,
    _nominal_voltage_map,
    detect_violations,
)
from gdm_flow.fix.engine import (
    FixIteration,
    FixResult,
    _default_strategies,
    fix_violations,
)
from gdm_flow.fix.strategies import (
    AddCapacitorStrategy,
    AdjustRegulatorTapStrategy,
    FixAction,
    ResizeConductorStrategy,
    ResizeTransformerStrategy,
)


MODEL_PATH = Path("examples/models/p5r.json")


@pytest.fixture()
def system():
    if not MODEL_PATH.exists():
        pytest.skip("p5r.json model not found")
    from gdm.systems.distribution import DistributionSystem

    return DistributionSystem.from_json(str(MODEL_PATH))


class TestViolationReportProperties:
    def test_total_violations(self):
        report = ViolationReport(
            success=True,
            solver="ldf",
            voltage_violations=[
                VoltageViolation("b1", "A", 115.0, 120.0, 114.0, 126.0, "undervoltage")
            ],
            loading_violations=[
                BranchLoadingViolation("br1", "A", 150.0, 100.0, 140.0, 50.0)
            ],
        )
        assert report.total_violations == 2
        assert report.has_violations is True

    def test_empty_report(self):
        report = ViolationReport(success=True, solver="ldf")
        assert report.total_violations == 0
        assert report.has_violations is False


class TestVoltageViolationDeviation:
    def test_overvoltage_deviation(self):
        vv = VoltageViolation("b1", "A", 130.0, 120.0, 114.0, 126.0, "overvoltage")
        assert vv.deviation_v == 4.0

    def test_undervoltage_deviation(self):
        vv = VoltageViolation("b1", "A", 110.0, 120.0, 114.0, 126.0, "undervoltage")
        assert vv.deviation_v == 4.0


class TestBranchLoadingViolation:
    def test_loading_pct(self):
        blv = BranchLoadingViolation("br1", "A", 150.0, 100.0, 140.0, 50.0)
        assert blv.loading_pct == 150.0

    def test_loading_pct_zero_limit(self):
        blv = BranchLoadingViolation("br1", "A", 150.0, 0.0, 140.0, 50.0)
        assert blv.loading_pct == 0.0


class TestFixResult:
    def test_total_actions(self):
        result = FixResult(
            success=True,
            message="ok",
            iterations=[
                FixIteration(
                    iteration=1,
                    voltage_violations=2,
                    loading_violations=0,
                    actions=[FixAction("s1", "c1", "d1"), FixAction("s2", "c2", "d2")],
                ),
                FixIteration(
                    iteration=2,
                    voltage_violations=0,
                    loading_violations=0,
                    actions=[FixAction("s3", "c3", "d3")],
                ),
            ],
            initial_voltage_violations=3,
            initial_loading_violations=0,
        )
        assert result.total_actions == 3
        assert result.violations_fixed == 3

    def test_violations_fixed_with_remaining(self):
        result = FixResult(
            success=False,
            message="partial",
            initial_voltage_violations=5,
            initial_loading_violations=2,
            final_voltage_violations=2,
            final_loading_violations=1,
        )
        assert result.violations_fixed == 4


class TestDefaultStrategies:
    def test_returns_four_strategies(self):
        strategies = _default_strategies()
        assert len(strategies) == 4
        assert isinstance(strategies[0], AdjustRegulatorTapStrategy)
        assert isinstance(strategies[1], AddCapacitorStrategy)
        assert isinstance(strategies[2], ResizeConductorStrategy)
        assert isinstance(strategies[3], ResizeTransformerStrategy)


class TestResizeConductorStrategy:
    def test_can_fix_with_undervoltage(self):
        report = ViolationReport(
            success=True,
            solver="ldf",
            voltage_violations=[
                VoltageViolation("b1", "A", 110.0, 120.0, 114.0, 126.0, "undervoltage")
            ],
        )
        strategy = ResizeConductorStrategy()
        assert strategy.can_fix(report)

    def test_cannot_fix_no_violations(self):
        report = ViolationReport(success=True, solver="ldf")
        strategy = ResizeConductorStrategy()
        assert not strategy.can_fix(report)

    def test_name(self):
        strategy = ResizeConductorStrategy()
        assert strategy.name == "resize_conductor"

    def test_scale_equipment_impedance_with_matrix(self):
        strategy = ResizeConductorStrategy()
        equipment = MagicMock()
        equipment.r_matrix = 1.0
        equipment.x_matrix = 2.0
        del equipment.impedance_matrix
        del equipment.resistance
        del equipment.reactance
        equipment.conductors = []
        changed = strategy._scale_equipment_impedance(equipment)
        assert "r_matrix" in changed
        assert "x_matrix" in changed


class TestResizeTransformerStrategy:
    def test_name(self):
        strategy = ResizeTransformerStrategy()
        assert strategy.name == "resize_transformer"

    def test_can_fix_with_loading(self):
        report = ViolationReport(
            success=True,
            solver="ldf",
            loading_violations=[
                BranchLoadingViolation("br1", "A", 150.0, 100.0, 140.0, 50.0)
            ],
        )
        strategy = ResizeTransformerStrategy()
        assert strategy.can_fix(report)

    def test_cannot_fix_without_loading(self):
        report = ViolationReport(success=True, solver="ldf")
        strategy = ResizeTransformerStrategy()
        assert not strategy.can_fix(report)


class TestAdjustRegulatorTapStrategy:
    def test_name(self):
        strategy = AdjustRegulatorTapStrategy()
        assert strategy.name == "adjust_regulator_tap"

    def test_can_fix_with_voltage_violations(self):
        report = ViolationReport(
            success=True,
            solver="ldf",
            voltage_violations=[
                VoltageViolation("b1", "A", 130.0, 120.0, 114.0, 126.0, "overvoltage")
            ],
        )
        strategy = AdjustRegulatorTapStrategy()
        assert strategy.can_fix(report)


class TestAddCapacitorStrategy:
    def test_name(self):
        strategy = AddCapacitorStrategy()
        assert strategy.name == "add_capacitor"

    def test_cannot_fix_empty_report(self):
        report = ViolationReport(success=True, solver="ldf")
        strategy = AddCapacitorStrategy()
        assert not strategy.can_fix(report)


class TestFixEngineWithMocks:
    def test_initial_pf_failure(self, monkeypatch):
        mock_system = MagicMock()
        monkeypatch.setattr(
            "gdm_flow.fix.engine.detect_violations",
            lambda *a, **kw: ViolationReport(success=False, solver="ldf"),
        )
        result = fix_violations(mock_system)
        assert result.success is False
        assert "failed" in result.message

    def test_no_violations_returns_success(self, monkeypatch):
        mock_system = MagicMock()
        monkeypatch.setattr(
            "gdm_flow.fix.engine.detect_violations",
            lambda *a, **kw: ViolationReport(success=True, solver="ldf"),
        )
        result = fix_violations(mock_system)
        assert result.success is True
        assert "No violations" in result.message

    def test_deadlock_when_no_strategy_can_act(self, monkeypatch):
        mock_system = MagicMock()
        report_with_violations = ViolationReport(
            success=True,
            solver="ldf",
            loading_violations=[
                BranchLoadingViolation("br1", "A", 150.0, 100.0, 140.0, 50.0)
            ],
        )
        monkeypatch.setattr(
            "gdm_flow.fix.engine.detect_violations",
            lambda *a, **kw: report_with_violations,
        )

        class NoOpStrategy:
            name = "noop"

            def can_fix(self, report):
                return False

            def apply(self, system, report):
                return []

        result = fix_violations(mock_system, strategies=[NoOpStrategy()])
        assert result.success is False
        assert len(result.iterations) == 1


class TestResizeConductorApply:
    """Test ResizeConductorStrategy.apply() using real GDM p5r model."""

    def test_apply_on_loading_violations(self, system):
        from gdm.systems.distribution.components.base.distribution_branch_base import (
            DistributionBranchBase,
        )

        # Get a real branch name from the system
        branch = next(iter(system.get_components(DistributionBranchBase)))
        report = ViolationReport(
            success=True,
            solver="ldf",
            loading_violations=[
                BranchLoadingViolation(
                    branch_name=branch.name,
                    phase="A",
                    loading_va=50000.0,
                    limit_va=40000.0,
                    p_flow_w=48000.0,
                    q_flow_var=10000.0,
                )
            ],
        )
        strategy = ResizeConductorStrategy()
        actions = strategy.apply(system, report)
        assert len(actions) > 0
        assert actions[0].strategy == "resize_conductor"
        assert branch.name in actions[0].component_name

    def test_apply_on_undervoltage_near_branch(self, system):
        from gdm.systems.distribution.components.base.distribution_branch_base import (
            DistributionBranchBase,
        )

        # Get a bus connected to a branch
        branch = next(iter(system.get_components(DistributionBranchBase)))
        bus_name = branch.buses[1].name

        report = ViolationReport(
            success=True,
            solver="ldf",
            voltage_violations=[
                VoltageViolation(
                    bus_name=bus_name,
                    phase="A",
                    voltage_v=110.0,
                    nominal_v=120.0,
                    min_v=114.0,
                    max_v=126.0,
                    kind="undervoltage",
                )
            ],
        )
        strategy = ResizeConductorStrategy()
        actions = strategy.apply(system, report)
        assert isinstance(actions, list)
        # Should find branches connected to the violated bus
        assert len(actions) > 0


class TestResizeTransformerApply:
    """Test ResizeTransformerStrategy.apply() using real GDM p5r model."""

    def test_apply_on_transformer_loading(self, system):
        from gdm.systems.distribution.components import DistributionTransformer

        xfmr = next(iter(system.get_components(DistributionTransformer)))
        report = ViolationReport(
            success=True,
            solver="ldf",
            loading_violations=[
                BranchLoadingViolation(
                    branch_name=xfmr.name,
                    phase="A",
                    loading_va=100000.0,
                    limit_va=75000.0,
                    p_flow_w=95000.0,
                    q_flow_var=30000.0,
                )
            ],
        )
        strategy = ResizeTransformerStrategy()
        actions = strategy.apply(system, report)
        assert len(actions) == 1
        assert actions[0].strategy == "resize_transformer"
        assert xfmr.name in actions[0].description

    def test_apply_skips_non_transformer_branch(self, system):
        report = ViolationReport(
            success=True,
            solver="ldf",
            loading_violations=[
                BranchLoadingViolation(
                    branch_name="nonexistent_xfmr",
                    phase="A",
                    loading_va=100000.0,
                    limit_va=75000.0,
                    p_flow_w=95000.0,
                    q_flow_var=30000.0,
                )
            ],
        )
        strategy = ResizeTransformerStrategy()
        actions = strategy.apply(system, report)
        assert len(actions) == 0


class TestAdjustRegulatorTapApply:
    """Test regulator tap strategy with mocked regulators since p5r has none."""

    def test_apply_with_no_regulators(self, system):
        report = ViolationReport(
            success=True,
            solver="ldf",
            voltage_violations=[
                VoltageViolation(
                    bus_name="some_bus",
                    phase="A",
                    voltage_v=110.0,
                    nominal_v=120.0,
                    min_v=114.0,
                    max_v=126.0,
                    kind="undervoltage",
                )
            ],
        )
        strategy = AdjustRegulatorTapStrategy()
        # p5r has no regulators, so no actions should be taken
        actions = strategy.apply(system, report)
        assert actions == []


class TestAddCapacitorApply:
    """Test capacitor strategy with real model (no capacitors in p5r)."""

    def test_apply_with_no_existing_capacitors(self, system):
        from gdm.systems.distribution.components import DistributionBus
        from gdm_flow._utils import _phase_name

        # Get a non-source bus name
        buses = list(system.get_components(DistributionBus))
        non_source = [b for b in buses if b.name != system.get_source_bus().name]
        bus_name = non_source[0].name if non_source else buses[0].name

        report = ViolationReport(
            success=True,
            solver="ldf",
            voltage_violations=[
                VoltageViolation(
                    bus_name=bus_name,
                    phase="A",
                    voltage_v=110.0,
                    nominal_v=120.0,
                    min_v=114.0,
                    max_v=126.0,
                    kind="undervoltage",
                )
            ],
        )
        strategy = AddCapacitorStrategy()
        actions = strategy.apply(system, report)
        # No capacitor exists, so it should suggest adding one
        assert len(actions) > 0
        assert "new_cap_" in actions[0].component_name


class TestDetectViolationsIntegration:
    """Test detect_violations with the real p5r model."""

    def test_detect_ldf_normal_limits(self, system):
        report = detect_violations(system, solver="ldf")
        assert report.success is True
        assert isinstance(report.voltage_violations, list)
        assert isinstance(report.loading_violations, list)

    def test_detect_ac_solver(self, system):
        report = detect_violations(system, solver="ac", vm_min_pu=0.95, vm_max_pu=1.05)
        assert report.success is True

    def test_nominal_voltage_map(self, system):
        nominal = _nominal_voltage_map(system)
        assert isinstance(nominal, dict)
        assert len(nominal) > 0
        for key, v in nominal.items():
            assert v > 0

    def test_branch_loading_limits(self, system):
        limits = _branch_loading_limits(system)
        assert isinstance(limits, dict)
        # p5r has branches with ampacity
        assert len(limits) > 0
        for key, va in limits.items():
            assert va > 0
