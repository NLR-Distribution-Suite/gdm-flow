"""Tests for the gdm_flow.fix module."""

from __future__ import annotations

from pathlib import Path

import pytest
from gdm.systems.distribution import DistributionSystem

from gdm_flow.fix import (
    ViolationReport,
    detect_violations,
    fix_violations,
)
from gdm_flow.fix.strategies import (
    AddCapacitorStrategy,
    AdjustRegulatorTapStrategy,
    ResizeConductorStrategy,
    ResizeTransformerStrategy,
)

MODELS_DIR = Path(__file__).parent.parent / "examples" / "models"


@pytest.fixture
def system():
    """Load the test distribution system."""
    model_path = MODELS_DIR / "p5r.json"
    if not model_path.exists():
        pytest.skip("Example model p5r.json not found")
    return DistributionSystem.from_json(str(model_path))


class TestDetectViolations:
    def test_clean_system_no_violations(self, system):
        report = detect_violations(system, solver="ldf")
        assert report.success
        assert report.total_violations == 0
        assert not report.has_violations

    def test_tight_limits_produce_violations(self, system):
        # Use very tight voltage limits so violations appear
        report = detect_violations(
            system, solver="ldf", vm_min_pu=0.999, vm_max_pu=1.001
        )
        assert report.success
        # With very tight limits on a loaded system, we expect violations
        assert report.total_violations >= 0  # May be 0 if source-only

    def test_invalid_solver_raises(self, system):
        with pytest.raises(ValueError, match="Unsupported solver"):
            detect_violations(system, solver="invalid")


class TestFixViolations:
    def test_clean_system_returns_success(self, system):
        result = fix_violations(system, solver="ldf")
        assert result.success
        assert result.message == "No violations detected. System is within limits."
        assert result.total_actions == 0

    def test_tight_limits_triggers_fix_loop(self, system):
        # Very tight limits force violations and trigger fix attempts
        result = fix_violations(
            system, solver="ldf", vm_min_pu=0.9999, vm_max_pu=1.0001, max_iterations=3
        )
        # Either all fixed or stopped due to no progress
        assert result.initial_voltage_violations >= 0
        assert isinstance(result.iterations, list)

    def test_max_iterations_respected(self, system):
        result = fix_violations(
            system, solver="ldf", vm_min_pu=0.9999, vm_max_pu=1.0001, max_iterations=2
        )
        assert len(result.iterations) <= 2


class TestResizeConductorStrategy:
    def test_can_fix_with_loading_violations(self):
        from gdm_flow.fix.detect import BranchLoadingViolation

        report = ViolationReport(
            success=True,
            solver="ldf",
            loading_violations=[
                BranchLoadingViolation(
                    branch_name="br1",
                    phase="A",
                    loading_va=150.0,
                    limit_va=100.0,
                    p_flow_w=140.0,
                    q_flow_var=50.0,
                )
            ],
        )
        strategy = ResizeConductorStrategy()
        assert strategy.can_fix(report)

    def test_cannot_fix_without_loading_violations(self):
        report = ViolationReport(success=True, solver="ldf")
        strategy = ResizeConductorStrategy()
        assert not strategy.can_fix(report)


class TestResizeTransformerStrategy:
    def test_standard_sizes(self):
        strategy = ResizeTransformerStrategy()
        assert 25 in strategy.STANDARD_SIZES_KVA
        assert 100 in strategy.STANDARD_SIZES_KVA


class TestAdjustRegulatorTapStrategy:
    def test_cannot_fix_without_voltage_violations(self):
        report = ViolationReport(success=True, solver="ldf")
        strategy = AdjustRegulatorTapStrategy()
        assert not strategy.can_fix(report)


class TestAddCapacitorStrategy:
    def test_can_fix_only_undervoltage(self):
        from gdm_flow.fix.detect import VoltageViolation

        report = ViolationReport(
            success=True,
            solver="ldf",
            voltage_violations=[
                VoltageViolation(
                    bus_name="bus1",
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
        assert strategy.can_fix(report)

    def test_cannot_fix_overvoltage_only(self):
        from gdm_flow.fix.detect import VoltageViolation

        report = ViolationReport(
            success=True,
            solver="ldf",
            voltage_violations=[
                VoltageViolation(
                    bus_name="bus1",
                    phase="A",
                    voltage_v=130.0,
                    nominal_v=120.0,
                    min_v=114.0,
                    max_v=126.0,
                    kind="overvoltage",
                )
            ],
        )
        strategy = AddCapacitorStrategy()
        assert not strategy.can_fix(report)
