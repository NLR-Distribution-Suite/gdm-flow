"""Iterative violation fix engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from gdm.systems.distribution import DistributionSystem

from .detect import detect_violations
from .strategies import (
    AddCapacitorStrategy,
    AdjustRegulatorTapStrategy,
    FixAction,
    FixStrategy,
    ResizeConductorStrategy,
    ResizeTransformerStrategy,
)


@dataclass
class FixIteration:
    """Record of a single fix iteration."""

    iteration: int
    voltage_violations: int
    loading_violations: int
    actions: list[FixAction] = field(default_factory=list)


@dataclass
class FixResult:
    """Result of the iterative fix process."""

    success: bool
    message: str
    iterations: list[FixIteration] = field(default_factory=list)
    initial_voltage_violations: int = 0
    initial_loading_violations: int = 0
    final_voltage_violations: int = 0
    final_loading_violations: int = 0

    @property
    def total_actions(self) -> int:
        return sum(len(it.actions) for it in self.iterations)

    @property
    def violations_fixed(self) -> int:
        initial = self.initial_voltage_violations + self.initial_loading_violations
        final = self.final_voltage_violations + self.final_loading_violations
        return initial - final


def _default_strategies() -> list[FixStrategy]:
    """Return strategies in default priority order (cheapest first)."""
    return [
        AdjustRegulatorTapStrategy(),
        AddCapacitorStrategy(),
        ResizeConductorStrategy(),
        ResizeTransformerStrategy(),
    ]


def fix_violations(
    system: DistributionSystem,
    *,
    strategies: Sequence[FixStrategy] | None = None,
    max_iterations: int = 10,
    solver: str = "ldf",
    vm_min_pu: float = 0.95,
    vm_max_pu: float = 1.05,
) -> FixResult:
    """Iteratively fix voltage and loading violations in a distribution system.

    Modifies the system in-place. Runs power flow, applies fix strategies,
    and repeats until violations are resolved or max iterations reached.

    Parameters
    ----------
    system : DistributionSystem
        The GDM distribution system to fix (modified in-place).
    strategies : sequence of FixStrategy, optional
        Fix strategies in priority order. Defaults to:
        regulator taps → capacitors → conductors → transformers.
    max_iterations : int
        Maximum number of fix-detect cycles.
    solver : str
        Solver for violation detection ("ldf" or "ac").
    vm_min_pu : float
        Minimum acceptable voltage in per-unit.
    vm_max_pu : float
        Maximum acceptable voltage in per-unit.

    Returns
    -------
    FixResult
        Summary of the fix process including iteration history.
    """
    if strategies is None:
        strategies = _default_strategies()

    iterations: list[FixIteration] = []

    # Initial detection
    report = detect_violations(
        system, solver=solver, vm_min_pu=vm_min_pu, vm_max_pu=vm_max_pu
    )
    if not report.success:
        return FixResult(
            success=False,
            message=f"Initial power flow failed ({solver} solver).",
        )

    initial_v = len(report.voltage_violations)
    initial_l = len(report.loading_violations)

    if not report.has_violations:
        return FixResult(
            success=True,
            message="No violations detected. System is within limits.",
            initial_voltage_violations=0,
            initial_loading_violations=0,
            final_voltage_violations=0,
            final_loading_violations=0,
        )

    prev_total = report.total_violations

    for iteration_num in range(1, max_iterations + 1):
        all_actions: list[FixAction] = []

        # Apply strategies in priority order
        for strategy in strategies:
            if not strategy.can_fix(report):
                continue
            actions = strategy.apply(system, report)
            all_actions.extend(actions)

        if not all_actions:
            # No strategy could act — deadlock
            iterations.append(
                FixIteration(
                    iteration=iteration_num,
                    voltage_violations=len(report.voltage_violations),
                    loading_violations=len(report.loading_violations),
                    actions=[],
                )
            )
            break

        # Re-detect after fixes
        report = detect_violations(
            system, solver=solver, vm_min_pu=vm_min_pu, vm_max_pu=vm_max_pu
        )

        iterations.append(
            FixIteration(
                iteration=iteration_num,
                voltage_violations=len(report.voltage_violations),
                loading_violations=len(report.loading_violations),
                actions=all_actions,
            )
        )

        if not report.success:
            return FixResult(
                success=False,
                message=f"Power flow failed at iteration {iteration_num}.",
                iterations=iterations,
                initial_voltage_violations=initial_v,
                initial_loading_violations=initial_l,
                final_voltage_violations=len(report.voltage_violations),
                final_loading_violations=len(report.loading_violations),
            )

        if not report.has_violations:
            return FixResult(
                success=True,
                message=f"All violations resolved in {iteration_num} iteration(s).",
                iterations=iterations,
                initial_voltage_violations=initial_v,
                initial_loading_violations=initial_l,
                final_voltage_violations=0,
                final_loading_violations=0,
            )

        # Check for progress (deadlock detection)
        current_total = report.total_violations
        if current_total >= prev_total:
            # No improvement — stop to avoid infinite loop
            return FixResult(
                success=False,
                message=(
                    f"No improvement at iteration {iteration_num} "
                    f"({current_total} violations remaining). Stopping."
                ),
                iterations=iterations,
                initial_voltage_violations=initial_v,
                initial_loading_violations=initial_l,
                final_voltage_violations=len(report.voltage_violations),
                final_loading_violations=len(report.loading_violations),
            )
        prev_total = current_total

    # Max iterations reached
    return FixResult(
        success=False,
        message=f"Max iterations ({max_iterations}) reached. {report.total_violations} violations remain.",
        iterations=iterations,
        initial_voltage_violations=initial_v,
        initial_loading_violations=initial_l,
        final_voltage_violations=len(report.voltage_violations),
        final_loading_violations=len(report.loading_violations),
    )
