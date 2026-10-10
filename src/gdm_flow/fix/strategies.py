"""Fix strategies for resolving voltage and loading violations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionTransformer,
)
from gdm.systems.distribution.components.base.distribution_branch_base import (
    DistributionBranchBase,
)

from .._utils import _phase_name

if TYPE_CHECKING:
    from .detect import ViolationReport


@dataclass
class FixAction:
    """Record of a single fix action applied."""

    strategy: str
    component_name: str
    description: str


class FixStrategy(ABC):
    """Base class for violation fix strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def can_fix(self, report: "ViolationReport") -> bool:
        """Return True if this strategy can address violations in the report."""

    @abstractmethod
    def apply(
        self, system: DistributionSystem, report: "ViolationReport"
    ) -> list[FixAction]:
        """Apply fixes to the system in-place. Return list of actions taken."""


class AdjustRegulatorTapStrategy(FixStrategy):
    """Adjust regulator tap setpoints to fix voltage violations at controlled buses."""

    def __init__(self, *, tap_step_pct: float = 0.625):
        self._tap_step_pct = tap_step_pct

    @property
    def name(self) -> str:
        return "adjust_regulator_tap"

    def can_fix(self, report: "ViolationReport") -> bool:
        return len(report.voltage_violations) > 0

    def apply(
        self, system: DistributionSystem, report: "ViolationReport"
    ) -> list[FixAction]:
        from gdm.systems.distribution.components.distribution_regulator import (
            DistributionRegulator,
        )

        actions: list[FixAction] = []

        # Build map of regulator-controlled buses
        controlled_buses: dict[tuple[str, str], tuple[DistributionRegulator, int]] = {}
        for regulator in system.get_components(DistributionRegulator):
            if not regulator.in_service:
                continue
            for idx, controller in enumerate(regulator.controllers):
                key = (
                    controller.controlled_bus.name,
                    _phase_name(controller.controlled_phase),
                )
                controlled_buses[key] = (regulator, idx)

        for vv in report.voltage_violations:
            key = (vv.bus_name, vv.phase)
            if key not in controlled_buses:
                continue

            regulator, ctrl_idx = controlled_buses[key]
            controller = regulator.controllers[ctrl_idx]

            # Calculate desired tap adjustment
            current_setpoint = float(
                (controller.v_setpoint * controller.pt_ratio).to("volt").magnitude
            )
            if vv.kind == "undervoltage":
                # Raise the setpoint
                step_v = current_setpoint * (self._tap_step_pct / 100.0)
                new_setpoint_v = current_setpoint + step_v
            else:
                # Lower the setpoint
                step_v = current_setpoint * (self._tap_step_pct / 100.0)
                new_setpoint_v = current_setpoint - step_v

            # Apply by adjusting v_setpoint (accounting for PT ratio)
            pt_ratio = float(controller.pt_ratio.magnitude)
            from gdm.quantities import Voltage as GDMVoltage

            new_v_setpoint = GDMVoltage(new_setpoint_v / pt_ratio, "volt")
            controller.v_setpoint = new_v_setpoint

            actions.append(
                FixAction(
                    strategy=self.name,
                    component_name=regulator.name,
                    description=(
                        f"Adjusted tap on {regulator.name} controller[{ctrl_idx}] "
                        f"for bus {vv.bus_name}/{vv.phase}: "
                        f"{current_setpoint:.1f}V → {new_setpoint_v:.1f}V"
                    ),
                )
            )

        return actions


class AddCapacitorStrategy(FixStrategy):
    """Add or resize shunt capacitors at buses with undervoltage violations."""

    def __init__(self, *, kvar_step: float = 50.0):
        self._kvar_step = kvar_step

    @property
    def name(self) -> str:
        return "add_capacitor"

    def can_fix(self, report: "ViolationReport") -> bool:
        return any(v.kind == "undervoltage" for v in report.voltage_violations)

    def apply(
        self, system: DistributionSystem, report: "ViolationReport"
    ) -> list[FixAction]:
        from gdm.systems.distribution.components import DistributionCapacitor

        actions: list[FixAction] = []

        # Get buses already controlled by regulators (skip those)
        from gdm.systems.distribution.components.distribution_regulator import (
            DistributionRegulator,
        )

        regulator_buses: set[str] = set()
        for reg in system.get_components(DistributionRegulator):
            if reg.in_service:
                for ctrl in reg.controllers:
                    regulator_buses.add(ctrl.controlled_bus.name)

        # Group undervoltage violations by bus
        undervoltage_buses: dict[str, list] = {}
        for vv in report.voltage_violations:
            if vv.kind == "undervoltage" and vv.bus_name not in regulator_buses:
                undervoltage_buses.setdefault(vv.bus_name, []).append(vv)

        for bus_name, violations in undervoltage_buses.items():
            # Check if capacitor already exists at this bus
            existing_cap = None
            for cap in system.get_components(DistributionCapacitor):
                if cap.bus.name == bus_name:
                    existing_cap = cap
                    break

            if existing_cap is not None:
                # Increase existing capacitor capacity
                for phase_cap in existing_cap.equipment.phase_capacitors:
                    old_kvar = float(
                        phase_cap.rated_reactive_power.to("kilovar").magnitude
                    )
                    from gdm.quantities import ReactivePower

                    phase_cap.rated_reactive_power = ReactivePower(
                        old_kvar + self._kvar_step, "kilovar"
                    )
                actions.append(
                    FixAction(
                        strategy=self.name,
                        component_name=existing_cap.name,
                        description=(
                            f"Increased capacitor {existing_cap.name} at bus {bus_name} "
                            f"by {self._kvar_step} kvar/phase"
                        ),
                    )
                )
            else:
                # Adding a new capacitor requires creating equipment + component.
                # This is complex with GDM's composed model. For now, log that
                # a capacitor is needed (full creation requires catalog support).
                actions.append(
                    FixAction(
                        strategy=self.name,
                        component_name=f"new_cap_{bus_name}",
                        description=(
                            f"Bus {bus_name} needs a new {self._kvar_step} kvar capacitor "
                            f"(no existing capacitor to resize)"
                        ),
                    )
                )

        return actions


class ResizeConductorStrategy(FixStrategy):
    """Resize branch conductors to fix loading and undervoltage violations."""

    def __init__(
        self, *, scale_factor: float = 1.5, impedance_reduction_factor: float = 0.9
    ):
        self._scale_factor = scale_factor
        self._impedance_reduction_factor = impedance_reduction_factor

    @property
    def name(self) -> str:
        return "resize_conductor"

    def can_fix(self, report: "ViolationReport") -> bool:
        return len(report.loading_violations) > 0 or any(
            v.kind == "undervoltage" for v in report.voltage_violations
        )

    def _scale_equipment_impedance(self, equipment) -> list[str]:
        """Reduce impedance-like fields on branch equipment when present."""
        changed: list[str] = []
        factor = self._impedance_reduction_factor

        for attr in (
            "r_matrix",
            "x_matrix",
            "impedance_matrix",
            "resistance",
            "reactance",
        ):
            if not hasattr(equipment, attr):
                continue
            value = getattr(equipment, attr)
            if value is None:
                continue
            try:
                setattr(equipment, attr, value * factor)
                changed.append(attr)
            except Exception:
                continue

        if hasattr(equipment, "conductors"):
            for cond in equipment.conductors:
                for attr in (
                    "r_matrix",
                    "x_matrix",
                    "impedance_matrix",
                    "resistance",
                    "reactance",
                ):
                    if not hasattr(cond, attr):
                        continue
                    value = getattr(cond, attr)
                    if value is None:
                        continue
                    try:
                        setattr(cond, attr, value * factor)
                        changed.append(f"conductors.{attr}")
                    except Exception:
                        continue

        return changed

    def apply(
        self, system: DistributionSystem, report: "ViolationReport"
    ) -> list[FixAction]:
        actions: list[FixAction] = []
        resized: set[str] = set()

        # Build quick lookup for all branch components.
        all_branches: dict[str, DistributionBranchBase] = {}
        for b in system.get_components(DistributionBranchBase):
            all_branches[b.name] = b

        target_branch_names: set[str] = {
            lv.branch_name for lv in report.loading_violations
        }

        # Undervoltage can often be addressed by reducing feeder impedance.
        undervoltage_buses = {
            vv.bus_name
            for vv in report.voltage_violations
            if getattr(vv, "kind", None) == "undervoltage"
        }
        if undervoltage_buses:
            for branch in all_branches.values():
                bus_names = {bus.name for bus in branch.buses}
                if bus_names & undervoltage_buses:
                    target_branch_names.add(branch.name)

        for branch_name in target_branch_names:
            if branch_name in resized:
                continue

            branch = all_branches.get(branch_name)
            if branch is None:
                continue

            equipment = branch.equipment

            changed_fields = self._scale_equipment_impedance(equipment)
            if changed_fields:
                actions.append(
                    FixAction(
                        strategy=self.name,
                        component_name=branch.name,
                        description=(
                            f"Reduced branch impedance on {branch.name}: "
                            f"fields={sorted(set(changed_fields))} scaled by {self._impedance_reduction_factor:.3f}x"
                        ),
                    )
                )
                resized.add(branch_name)
                continue

            if hasattr(equipment, "ampacity"):
                old_amp = float(equipment.ampacity.to("ampere").magnitude)
                from gdm.quantities import Current

                new_amp = old_amp * self._scale_factor
                equipment.ampacity = Current(new_amp, "ampere")
                actions.append(
                    FixAction(
                        strategy=self.name,
                        component_name=branch.name,
                        description=(
                            f"Resized conductor {branch.name}: "
                            f"ampacity {old_amp:.1f}A → {new_amp:.1f}A"
                        ),
                    )
                )
                resized.add(branch_name)
            elif hasattr(equipment, "conductors"):
                for cond in equipment.conductors:
                    if hasattr(cond, "ampacity"):
                        old_amp = float(cond.ampacity.to("ampere").magnitude)
                        from gdm.quantities import Current

                        new_amp = old_amp * self._scale_factor
                        cond.ampacity = Current(new_amp, "ampere")
                actions.append(
                    FixAction(
                        strategy=self.name,
                        component_name=branch.name,
                        description=(
                            f"Resized conductor {branch.name}: "
                            f"ampacity scaled by {self._scale_factor}x"
                        ),
                    )
                )
                resized.add(branch_name)

        return actions


class ResizeTransformerStrategy(FixStrategy):
    """Resize transformers to fix loading violations on transformer branches."""

    # Standard transformer sizes in kVA
    STANDARD_SIZES_KVA = [15, 25, 37.5, 50, 75, 100, 167, 250, 333, 500, 750, 1000]

    @property
    def name(self) -> str:
        return "resize_transformer"

    def can_fix(self, report: "ViolationReport") -> bool:
        return len(report.loading_violations) > 0

    def apply(
        self, system: DistributionSystem, report: "ViolationReport"
    ) -> list[FixAction]:
        actions: list[FixAction] = []
        resized: set[str] = set()

        # Build set of transformer names
        xfmr_names: dict[str, DistributionTransformer] = {}
        for xfmr in system.get_components(DistributionTransformer):
            xfmr_names[xfmr.name] = xfmr

        for lv in report.loading_violations:
            if lv.branch_name in resized:
                continue
            if lv.branch_name not in xfmr_names:
                continue

            xfmr = xfmr_names[lv.branch_name]
            equipment = xfmr.equipment

            # Find current minimum winding capacity
            current_kva = min(
                float(wdg.rated_power.to("kilova").magnitude)
                for wdg in equipment.windings
            )

            # Find next standard size
            next_kva = None
            for size in self.STANDARD_SIZES_KVA:
                if size > current_kva:
                    next_kva = size
                    break

            if next_kva is None:
                # Already at or above max standard size, scale by 1.5x
                next_kva = current_kva * 1.5

            # Resize all windings proportionally
            from gdm.quantities import ApparentPower

            scale = next_kva / current_kva
            for wdg in equipment.windings:
                old_kva = float(wdg.rated_power.to("kilova").magnitude)
                wdg.rated_power = ApparentPower(old_kva * scale, "kilova")

            actions.append(
                FixAction(
                    strategy=self.name,
                    component_name=xfmr.name,
                    description=(
                        f"Resized transformer {xfmr.name}: "
                        f"{current_kva:.1f} kVA → {next_kva:.1f} kVA"
                    ),
                )
            )
            resized.add(lv.branch_name)

        return actions
