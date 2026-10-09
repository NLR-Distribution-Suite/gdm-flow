"""Internal shared helper utilities for solver modules."""

from __future__ import annotations

import math

from gdm.distribution.enums import Phase, VoltageTypes


def _phase_name(phase: Phase | str) -> str:
    """Normalize phase enum/string values to string labels."""
    return phase.value if isinstance(phase, Phase) else str(phase)


def _phase_voltage(
    voltage, voltage_type: VoltageTypes, *, split_phase: bool = False
) -> float:
    """Return phase voltage magnitude in volts."""
    v_ll_or_lg = float(voltage.to("volt").magnitude)
    if voltage_type == VoltageTypes.LINE_TO_LINE:
        return v_ll_or_lg / (2 if split_phase else math.sqrt(3))
    return v_ll_or_lg
