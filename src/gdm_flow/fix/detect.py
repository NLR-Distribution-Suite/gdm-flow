"""Unified violation detection wrapping existing solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import DistributionBus
from gdm.systems.distribution.components.base.distribution_branch_base import (
    DistributionBranchBase,
)
from gdm.systems.distribution.enums import Phase

from .._utils import _phase_name, _phase_voltage

BusPhaseLabel = Tuple[str, str]
BranchPhaseLabel = Tuple[str, str]


@dataclass(frozen=True)
class VoltageViolation:
    """A single bus-phase voltage violation."""

    bus_name: str
    phase: str
    voltage_v: float
    nominal_v: float
    min_v: float
    max_v: float
    kind: str  # "overvoltage" or "undervoltage"

    @property
    def deviation_v(self) -> float:
        if self.kind == "overvoltage":
            return self.voltage_v - self.max_v
        return self.min_v - self.voltage_v


@dataclass(frozen=True)
class BranchLoadingViolation:
    """A single branch-phase loading violation."""

    branch_name: str
    phase: str
    loading_va: float
    limit_va: float
    p_flow_w: float
    q_flow_var: float

    @property
    def loading_pct(self) -> float:
        if self.limit_va <= 0:
            return 0.0
        return 100.0 * self.loading_va / self.limit_va


@dataclass
class ViolationReport:
    """Aggregated violation report from a power flow run."""

    success: bool
    solver: str
    voltage_violations: list[VoltageViolation] = field(default_factory=list)
    loading_violations: list[BranchLoadingViolation] = field(default_factory=list)

    @property
    def total_violations(self) -> int:
        return len(self.voltage_violations) + len(self.loading_violations)

    @property
    def has_violations(self) -> bool:
        return self.total_violations > 0


def _nominal_voltage_map(
    system: DistributionSystem,
) -> Dict[BusPhaseLabel, float]:
    """Build per-bus-phase nominal voltage map in volts."""
    result: Dict[BusPhaseLabel, float] = {}
    for bus in system.get_components(DistributionBus):
        v_phase = _phase_voltage(bus.rated_voltage, bus.voltage_type)
        for p in bus.phases:
            if p == Phase.N:
                continue
            result[(bus.name, _phase_name(p))] = v_phase
    return result


def _branch_loading_limits(
    system: DistributionSystem,
) -> Dict[BranchPhaseLabel, float]:
    """Build per-branch-phase loading limit in VA from equipment ampacity and nominal voltage."""
    limits: Dict[BranchPhaseLabel, float] = {}
    nominal_map = _nominal_voltage_map(system)

    for branch in system.get_components(DistributionBranchBase):
        if not branch.in_service:
            continue
        ampacity = None
        if hasattr(branch.equipment, "ampacity"):
            ampacity = float(branch.equipment.ampacity.to("ampere").magnitude)
        elif hasattr(branch.equipment, "conductors"):
            amps = [
                float(c.ampacity.to("ampere").magnitude)
                for c in branch.equipment.conductors
            ]
            ampacity = min(amps) if amps else None

        if ampacity is None:
            continue

        for bus in branch.buses:
            for p in branch.phases:
                if p == Phase.N:
                    continue
                ph = _phase_name(p)
                v_nom = nominal_map.get((bus.name, ph))
                if v_nom and v_nom > 0:
                    limits[(branch.name, ph)] = ampacity * v_nom
                    break  # use first bus voltage found
    return limits


def detect_violations(
    system: DistributionSystem,
    *,
    solver: str = "ldf",
    vm_min_pu: float = 0.95,
    vm_max_pu: float = 1.05,
) -> ViolationReport:
    """Run power flow and detect voltage and loading violations.

    Parameters
    ----------
    system : DistributionSystem
        The GDM distribution system to analyze.
    solver : str
        Solver to use: "ldf" (LinDistFlow) or "ac" (AC OPF).
    vm_min_pu : float
        Minimum acceptable voltage in per-unit.
    vm_max_pu : float
        Maximum acceptable voltage in per-unit.

    Returns
    -------
    ViolationReport
        Report containing all detected violations.
    """
    nominal_map = _nominal_voltage_map(system)
    loading_limits = _branch_loading_limits(system)

    voltage_violations: list[VoltageViolation] = []
    loading_violations: list[BranchLoadingViolation] = []

    if solver == "ldf":
        from ..lindistflow import solve_lindistflow

        result = solve_lindistflow(system)
        if not result.success:
            return ViolationReport(success=False, solver=solver)

        # Check voltage violations
        for (bus_name, phase), voltage_v in result.voltage_v.items():
            nominal = nominal_map.get((bus_name, phase))
            if nominal is None or nominal <= 0:
                continue
            v_min = vm_min_pu * nominal
            v_max = vm_max_pu * nominal
            if voltage_v > v_max:
                voltage_violations.append(
                    VoltageViolation(
                        bus_name=bus_name,
                        phase=phase,
                        voltage_v=voltage_v,
                        nominal_v=nominal,
                        min_v=v_min,
                        max_v=v_max,
                        kind="overvoltage",
                    )
                )
            elif voltage_v < v_min:
                voltage_violations.append(
                    VoltageViolation(
                        bus_name=bus_name,
                        phase=phase,
                        voltage_v=voltage_v,
                        nominal_v=nominal,
                        min_v=v_min,
                        max_v=v_max,
                        kind="undervoltage",
                    )
                )

        # Check loading violations
        for (branch_name, phase), limit_va in loading_limits.items():
            p_w = result.p_flow_w.get((branch_name, phase), 0.0)
            q_var = result.q_flow_var.get((branch_name, phase), 0.0)
            loading_va = math.sqrt(p_w**2 + q_var**2)
            if loading_va > limit_va:
                loading_violations.append(
                    BranchLoadingViolation(
                        branch_name=branch_name,
                        phase=phase,
                        loading_va=loading_va,
                        limit_va=limit_va,
                        p_flow_w=p_w,
                        q_flow_var=q_var,
                    )
                )

    elif solver == "ac":
        from ..ac_opf import (
            optimize_ac_power_flow_from_components,
        )

        result = optimize_ac_power_flow_from_components(
            system, vm_min_pu=vm_min_pu, vm_max_pu=vm_max_pu
        )
        if not result.success:
            return ViolationReport(success=False, solver=solver)

        # Check voltage violations from AC OPF result
        for idx, (bus_name, phase) in enumerate(result.ybus_result.index_to_label):
            voltage_v = abs(result.voltage[idx])
            nominal = nominal_map.get((bus_name, phase))
            if nominal is None or nominal <= 0:
                continue
            v_min = vm_min_pu * nominal
            v_max = vm_max_pu * nominal
            if voltage_v > v_max:
                voltage_violations.append(
                    VoltageViolation(
                        bus_name=bus_name,
                        phase=phase,
                        voltage_v=voltage_v,
                        nominal_v=nominal,
                        min_v=v_min,
                        max_v=v_max,
                        kind="overvoltage",
                    )
                )
            elif voltage_v < v_min:
                voltage_violations.append(
                    VoltageViolation(
                        bus_name=bus_name,
                        phase=phase,
                        voltage_v=voltage_v,
                        nominal_v=nominal,
                        min_v=v_min,
                        max_v=v_max,
                        kind="undervoltage",
                    )
                )
    else:
        raise ValueError(f"Unsupported solver: {solver!r}. Use 'ldf' or 'ac'.")

    return ViolationReport(
        success=True,
        solver=solver,
        voltage_violations=voltage_violations,
        loading_violations=loading_violations,
    )
