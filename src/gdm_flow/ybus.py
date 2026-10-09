"""Build Y-bus matrices from a grid-data-models DistributionSystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import re

import numpy as np

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.components import (
    DistributionBus,
    DistributionReactor,
    DistributionTransformer,
    GeometryBranch,
    SequenceImpedanceBranch,
)
from gdm.systems.distribution.components.distribution_regulator import (
    DistributionRegulator,
)
from gdm.systems.distribution.components.base.distribution_branch_base import (
    DistributionBranchBase,
)
from gdm.systems.distribution.components.base.distribution_switch_base import (
    DistributionSwitchBase,
)
from gdm.systems.distribution.enums import Phase

from ._utils import _phase_name, _phase_voltage

BusPhaseLabel = Tuple[str, str]


@dataclass(frozen=True)
class YBusResult:
    """Container for Y-bus output and node indexing metadata."""

    ybus: np.ndarray
    index_to_label: List[BusPhaseLabel]
    label_to_index: Dict[BusPhaseLabel, int]


def _build_bus_phase_index(
    system: DistributionSystem,
    include_neutral: bool,
) -> tuple[list[BusPhaseLabel], dict[BusPhaseLabel, int]]:
    # Discover buses that need an explicit neutral node: any bus that
    # appears as a secondary winding bus of a 3-winding (split-phase)
    # transformer with N in its winding phases.  Without this, the neutral
    # is silently Kron-reduced to ground, which is wrong when the center
    # tap is floating (is_grounded=False).
    neutral_buses: set[str] = set()
    if include_neutral:
        for xfmr in system.get_components(DistributionTransformer):
            if not xfmr.in_service or len(xfmr.equipment.windings) < 3:
                continue
            for w_idx in range(1, len(xfmr.equipment.windings)):
                winding_phases = xfmr.winding_phases[w_idx]
                if Phase.N in winding_phases:
                    bus_sec = (
                        xfmr.buses[w_idx] if w_idx < len(xfmr.buses) else xfmr.buses[-1]
                    )
                    neutral_buses.add(bus_sec.name)

    labels: list[BusPhaseLabel] = []
    for bus in sorted(system.get_components(DistributionBus), key=lambda b: b.name):
        for phase in bus.phases:
            if not include_neutral and phase == Phase.N:
                continue
            labels.append((bus.name, _phase_name(phase)))
        # Add explicit N node for split-phase secondary buses that lack it
        if include_neutral and bus.name in neutral_buses and Phase.N not in bus.phases:
            labels.append((bus.name, "N"))

    label_to_index = {label: i for i, label in enumerate(labels)}
    return labels, label_to_index


def _active_branch_phase_indices(
    branch: DistributionBranchBase,
    include_neutral: bool,
    include_open_switches: bool,
) -> list[int]:
    active = []
    for i, phase in enumerate(branch.phases):
        if not include_neutral and phase == Phase.N:
            continue
        if isinstance(branch, DistributionSwitchBase) and not include_open_switches:
            if not bool(branch.is_closed[i]):
                continue
        active.append(i)
    return active


def _matrix_branch_series_admittance(
    branch: DistributionBranchBase,
    active_idx: list[int],
    skip_zero_impedance: bool = False,
) -> np.ndarray | None:
    # All matrix-impedance branch/switch/fuse/recloser models share these fields.
    r = branch.equipment.r_matrix.to("ohm/m").magnitude
    x = branch.equipment.x_matrix.to("ohm/m").magnitude
    length_m = float(branch.length.to("m").magnitude)
    z = (r + 1j * x) * length_m
    z = z[np.ix_(active_idx, active_idx)]

    _Z_MIN = 1e-4  # ohm – threshold for "near-zero" impedance

    has_near_zero = any(abs(z[i, i]) < _Z_MIN for i in range(z.shape[0]))

    if has_near_zero:
        if skip_zero_impedance and not isinstance(branch, DistributionSwitchBase):
            # These are ideal connections (load-bus stubs, regulator bypass
            # lines) that don't exist as physical branches in OpenDSS.
            # Skipping them avoids stamping spurious admittance on real bus
            # diagonals, matching OpenDSS's Y-bus formulation.
            return None

        # For solvers that need all buses connected (AC PF), clamp the
        # diagonal impedance so the branch behaves as a low-impedance
        # connection rather than a numerical singularity.
        for i in range(z.shape[0]):
            if abs(z[i, i]) < _Z_MIN:
                z[i, i] = _Z_MIN * (1 + 1j)

    try:
        return np.linalg.inv(z)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(z)


def _reactor_impedance_ohm(reactor: DistributionReactor) -> complex:
    resistance = float(reactor.equipment.resistance.to("ohm").magnitude)
    reactance = float(reactor.equipment.reactance.to("ohm").magnitude)
    if (
        not np.isfinite(resistance)
        or not np.isfinite(reactance)
        or resistance < 0
        or reactance < 0
        or resistance == reactance == 0
    ):
        raise ValueError(
            f"Reactor {reactor.name!r} requires finite, nonnegative, nonzero impedance."
        )
    return resistance + 1j * reactance


def _reactor_series_admittance(
    reactor: DistributionReactor, active_idx: list[int]
) -> np.ndarray:
    return np.eye(len(active_idx), dtype=complex) / _reactor_impedance_ohm(reactor)


def _matrix_branch_shunt_admittance(
    branch: DistributionBranchBase,
    active_idx: list[int],
    frequency_hz: float,
) -> np.ndarray:
    c = branch.equipment.c_matrix.to("farad/m").magnitude
    length_m = float(branch.length.to("m").magnitude)
    c_total = c[np.ix_(active_idx, active_idx)] * length_m
    return 1j * 2.0 * math.pi * frequency_hz * c_total


def _sequence_branch_series_admittance(
    branch: SequenceImpedanceBranch,
    active_idx: list[int],
) -> np.ndarray:
    phases = [branch.phases[i] for i in active_idx]
    n = len(phases)
    length_m = float(branch.length.to("m").magnitude)

    z1 = (
        branch.equipment.pos_seq_resistance.to("ohm/m").magnitude
        + 1j * branch.equipment.pos_seq_reactance.to("ohm/m").magnitude
    ) * length_m
    z0 = (
        branch.equipment.zero_seq_resistance.to("ohm/m").magnitude
        + 1j * branch.equipment.zero_seq_reactance.to("ohm/m").magnitude
    ) * length_m

    if n == 3 and set(phases) == {Phase.A, Phase.B, Phase.C}:
        z_self = (2.0 * z1 + z0) / 3.0
        z_mutual = (z0 - z1) / 3.0
        z = np.full((3, 3), z_mutual, dtype=np.complex128)
        np.fill_diagonal(z, z_self)
    else:
        z = np.eye(n, dtype=np.complex128) * z1

    try:
        return np.linalg.inv(z)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(z)


def _stamp_branch(
    ybus: np.ndarray,
    label_to_index: dict[BusPhaseLabel, int],
    bus_u: str,
    bus_v: str,
    phase_names: list[str],
    y_series: np.ndarray,
    y_shunt: np.ndarray | None = None,
) -> None:
    u_idx = [label_to_index[(bus_u, p)] for p in phase_names]
    v_idx = [label_to_index[(bus_v, p)] for p in phase_names]

    ybus[np.ix_(u_idx, u_idx)] += y_series
    ybus[np.ix_(v_idx, v_idx)] += y_series
    ybus[np.ix_(u_idx, v_idx)] -= y_series
    ybus[np.ix_(v_idx, u_idx)] -= y_series

    if y_shunt is not None:
        ybus[np.ix_(u_idx, u_idx)] += 0.5 * y_shunt
        ybus[np.ix_(v_idx, v_idx)] += 0.5 * y_shunt


SPLIT_PHASE_CONTRACT = "grounded-center-tap-v1"


def _center_tap_phases(
    transformer: DistributionTransformer,
    include_neutral: bool,
) -> list[list[str]]:
    windings = transformer.equipment.windings
    phases = [
        [_phase_name(phase) for phase in group] for group in transformer.winding_phases
    ]
    if include_neutral:
        raise ValueError("Explicit-neutral split-phase stamping is not supported.")
    if (
        len(windings) != 3
        or len(phases) != 3
        or len(transformer.buses) != 3
        or transformer.buses[1].name != transformer.buses[2].name
        or len(phases[0]) != 1
        or phases[0][0] not in {"A", "B", "C"}
        or {frozenset(group) for group in phases[1:]}
        != {frozenset({"N", "S1"}), frozenset({"N", "S2"})}
    ):
        raise ValueError(
            "Split-phase stamping requires a single-primary, shared-bus center tap."
        )
    if any(not getattr(winding, "is_grounded", False) for winding in windings[1:]):
        raise ValueError(
            "Split-phase secondary windings require explicit grounded metadata."
        )
    return phases


def _center_tap_winding_data(transformer: DistributionTransformer):
    windings = transformer.equipment.windings
    no_load_loss = float(getattr(transformer.equipment, "pct_no_load_loss", 0))
    if not np.isfinite(no_load_loss) or no_load_loss < 0:
        raise ValueError("Split-phase no-load loss must be finite and nonnegative.")
    if any(
        winding.num_phases != 1
        or getattr(winding.connection_type, "value", winding.connection_type) != "STAR"
        for winding in windings
    ):
        raise ValueError(
            "Split-phase stamping supports single-phase STAR windings only."
        )
    voltages = np.array(
        [float(winding.rated_voltage.to("volt").magnitude) for winding in windings]
    )
    powers = np.array(
        [float(winding.rated_power.to("va").magnitude) for winding in windings]
    )
    resistance = np.array([float(winding.resistance) for winding in windings]) / 100
    if (
        not np.all(np.isfinite(voltages))
        or np.any(voltages <= 0)
        or not np.all(np.isfinite(powers))
        or np.any(powers <= 0)
        or not np.all(np.isfinite(resistance))
        or np.any(resistance < 0)
    ):
        raise ValueError("Invalid split-phase winding ratings or resistance.")
    return voltages, powers, resistance, no_load_loss


def _center_tap_impedance(transformer, voltages, powers, resistance):
    reactances = {}
    for pair, value in zip(
        transformer.equipment.coupling_sequences,
        transformer.equipment.winding_reactances,
        strict=True,
    ):
        key = tuple(sorted((pair.from_index, pair.to_index)))
        if key in reactances or not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid split-phase pairwise reactances.")
        reactances[key] = float(value) / 100
    if set(reactances) == {(1, 2), (1, 3), (2, 3)}:
        reactances = {
            (first - 1, second - 1): value
            for (first, second), value in reactances.items()
        }
    if set(reactances) != {(0, 1), (0, 2), (1, 2)}:
        raise ValueError("All three split-phase pairwise reactances are required.")
    primary_base = voltages[0] ** 2 / powers[0]
    referred_resistance = resistance * voltages[0] ** 2 / powers
    mutual = (reactances[(0, 1)] + reactances[(0, 2)] - reactances[(1, 2)]) / 2
    impedance = np.array(
        [
            [
                referred_resistance[0]
                + referred_resistance[1]
                + 1j * primary_base * reactances[(0, 1)],
                referred_resistance[0] + 1j * primary_base * mutual,
            ],
            [
                referred_resistance[0] + 1j * primary_base * mutual,
                referred_resistance[0]
                + referred_resistance[2]
                + 1j * primary_base * reactances[(0, 2)],
            ],
        ]
    )
    if (
        np.min(np.linalg.eigvalsh(impedance.imag)) <= 0
        or np.linalg.cond(impedance) > 1e12
    ):
        raise ValueError(
            "Split-phase leakage impedance must be passive and nonsingular."
        )
    return impedance


def _center_tap_taps(transformer: DistributionTransformer, phases):
    taps = transformer.tap_positions
    tap_values = []
    for index, group in enumerate(phases):
        if taps is None:
            tap_values.append(1.0)
        elif len(taps[index]) == 1:
            tap_values.append(taps[index][0])
        elif len(taps[index]) == len(group):
            tap_values.append(
                taps[index][
                    next(
                        position for position, phase in enumerate(group) if phase != "N"
                    )
                ]
            )
        else:
            raise ValueError(
                "Split-phase tap indexing does not match winding terminals."
            )
    tap_values = np.asarray(tap_values)
    if not np.all(np.isfinite(tap_values)) or np.any(tap_values <= 0):
        raise ValueError("Split-phase taps must be finite and positive.")
    return tap_values


def _stamp_grounded_center_tap(
    ybus: np.ndarray,
    label_to_index: dict[BusPhaseLabel, int],
    transformer: DistributionTransformer,
    include_neutral: bool,
) -> None:
    phases = _center_tap_phases(transformer, include_neutral)
    voltages, powers, resistance, no_load_loss = _center_tap_winding_data(transformer)
    impedance = _center_tap_impedance(transformer, voltages, powers, resistance)
    tap_values = _center_tap_taps(transformer, phases)
    polarity = np.array(
        [1.0, *[1.0 if "S1" in group else -1.0 for group in phases[1:]]]
    )
    ratios = polarity * voltages[0] / voltages / tap_values
    incidence = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]) * ratios
    primitive = incidence.T @ np.linalg.solve(impedance, incidence)
    primitive[1, 1] += (
        no_load_loss / 100 * powers[0] / (voltages[1] * tap_values[1]) ** 2
    )
    labels = [
        (transformer.buses[index].name, next(phase for phase in group if phase != "N"))
        for index, group in enumerate(phases)
    ]
    indices = [label_to_index[label] for label in labels]
    ybus[np.ix_(indices, indices)] += primitive


def _power_transformer_layout(
    transformer: DistributionTransformer,
    include_neutral: bool,
):
    equipment = transformer.equipment
    windings = equipment.windings
    if include_neutral or len(windings) != 2 or len(transformer.buses) != 2:
        raise NotImplementedError(
            "Station transformers require two windings and implicit neutrals."
        )
    phases = [Phase.A, Phase.B, Phase.C]
    groups = [
        [_phase_name(phase) for phase in group] for group in transformer.winding_phases
    ]
    if any(winding.num_phases != 3 for winding in windings) or any(
        set(group) - {"N"} != {"A", "B", "C"} for group in groups
    ):
        raise NotImplementedError(
            "Station transformers require three ABC phases on each winding."
        )
    connections = [
        getattr(winding.connection_type, "value", winding.connection_type)
        for winding in windings
    ]
    if any(connection not in {"STAR", "DELTA"} for connection in connections):
        raise NotImplementedError(
            "Station transformer winding connections must be STAR or DELTA."
        )
    return phases, groups, connections


def _power_transformer_rotation(vector_group: str | None, connections):
    rotation = np.eye(3)
    if vector_group is not None:
        match = re.fullmatch(r"([DYdy])[Nn]?([DYdy])[Nn]?([0-9]|1[01])", vector_group)
        if match is None:
            raise NotImplementedError(f"Unsupported vector group {vector_group!r}.")
        symbols = ["D" if connection == "DELTA" else "Y" for connection in connections]
        if [match[1].upper(), match[2].upper()] != symbols:
            raise ValueError("Vector group disagrees with winding connections.")
        clock_step = (int(match[3]) + (symbols[0] == "D") - (symbols[1] == "D")) % 12
        if clock_step % 2:
            raise NotImplementedError(
                f"Unsupported clock orientation in {vector_group!r}."
            )
        rotation = np.linalg.matrix_power(
            -np.roll(np.eye(3), 1, axis=1), clock_step // 2
        )
    return rotation


def _power_transformer_data(transformer: DistributionTransformer, groups):
    equipment = transformer.equipment
    windings = equipment.windings
    phases = [Phase.A, Phase.B, Phase.C]
    voltage = np.array(
        [
            _phase_voltage(winding.rated_voltage, winding.voltage_type)
            for winding in windings
        ]
    )
    power = np.array(
        [float(winding.rated_power.to("va").magnitude) / 3 for winding in windings]
    )
    resistance = np.array([float(winding.resistance) for winding in windings]) / 100
    reactance = float(equipment.winding_reactances[0]) / 100
    core_loss = float(equipment.pct_no_load_loss) / 100
    taps = (
        np.ones((2, 3))
        if transformer.tap_positions is None
        else np.array(
            [
                [
                    transformer.tap_positions[index][groups[index].index(phase.value)]
                    for phase in phases
                ]
                for index in range(2)
            ]
        )
    )
    values = np.concatenate(
        [voltage, power, resistance, [reactance, core_loss], taps.ravel()]
    )
    if (
        not np.all(np.isfinite(values))
        or np.any(voltage <= 0)
        or np.any(power <= 0)
        or np.any(resistance < 0)
        or reactance < 0
        or core_loss < 0
        or np.any(taps <= 0)
    ):
        raise ValueError(
            f"Invalid station transformer ratings, losses, or taps on {transformer.name!r}."
        )
    impedance = voltage[0] ** 2 * (
        resistance[0] / power[0] + resistance[1] / power[1] + 1j * reactance / power[0]
    )
    if abs(impedance) == 0:
        raise ValueError("Station transformer leakage impedance must be nonzero.")
    return voltage, power, taps, impedance, core_loss


def _power_transformer_primitive(transformer, connections, rotation, electrical_data):
    voltage, power, taps, impedance, core_loss = electrical_data
    windings = transformer.equipment.windings
    delta = (np.eye(3) - np.roll(np.eye(3), 1, axis=1)) / np.sqrt(3)
    maps = [
        np.diag(1 / taps[index]) @ (delta if connection == "DELTA" else np.eye(3))
        for index, connection in enumerate(connections)
    ]
    ratio = voltage[0] / voltage[1]
    terminal_map = np.column_stack([maps[0], -ratio * rotation @ maps[1]])
    neutrals = []
    for index, winding in enumerate(windings):
        if connections[index] == "STAR" and not winding.is_grounded:
            neutral = 1 / taps[index]
            neutrals.append(-neutral if index == 0 else ratio * rotation @ neutral)
    if neutrals:
        terminal_map = np.column_stack([terminal_map, *neutrals])
    primitive = terminal_map.T @ terminal_map / impedance
    if core_loss:
        core_map = np.zeros((3, primitive.shape[0]))
        core_map[:, 3:6] = maps[1]
        if connections[1] == "STAR" and not windings[1].is_grounded:
            core_map[:, -1] = -1 / taps[1]
        primitive += core_loss * power[0] / voltage[1] ** 2 * core_map.T @ core_map
    if neutrals:
        primitive = (
            primitive[:6, :6]
            - primitive[:6, 6:] @ np.linalg.pinv(primitive[6:, 6:]) @ primitive[6:, :6]
        )
    return primitive


def _stamp_power_transformer(
    ybus: np.ndarray,
    label_to_index: dict[BusPhaseLabel, int],
    transformer: DistributionTransformer,
    include_neutral: bool,
) -> None:
    phases, groups, connections = _power_transformer_layout(
        transformer, include_neutral
    )
    rotation = _power_transformer_rotation(
        transformer.equipment.vector_group, connections
    )
    electrical_data = _power_transformer_data(transformer, groups)
    primitive = _power_transformer_primitive(
        transformer, connections, rotation, electrical_data
    )
    indices = [
        label_to_index[(bus.name, phase.value)]
        for bus in transformer.buses
        for phase in phases
    ]
    ybus[np.ix_(indices, indices)] += primitive


def _stamp_transformer(
    ybus: np.ndarray,
    label_to_index: dict[BusPhaseLabel, int],
    transformer: DistributionTransformer,
    include_neutral: bool,
) -> None:
    from gdm.systems.distribution.equipment import PowerTransformerEquipment

    if isinstance(transformer.equipment, PowerTransformerEquipment):
        _stamp_power_transformer(ybus, label_to_index, transformer, include_neutral)
        return
    if len(transformer.buses) < 2 or len(transformer.equipment.windings) < 2:
        return

    if any(
        _phase_name(phase) in {"S1", "S2"}
        for group in transformer.winding_phases
        for phase in group
    ):
        _stamp_grounded_center_tap(ybus, label_to_index, transformer, include_neutral)
        return

    bus_u = transformer.buses[0]
    bus_v = transformer.buses[1]
    w_u = transformer.equipment.windings[0]
    w_v = (
        transformer.equipment.windings[1]
        if len(transformer.equipment.windings) > 1
        else w_u
    )

    r_pu = float(transformer.equipment.pct_full_load_loss) / 100.0
    x_pu = float(transformer.equipment.winding_reactances[0]) / 100.0

    v_u_phase = _phase_voltage(w_u.rated_voltage, w_u.voltage_type)
    v_v_phase = _phase_voltage(w_v.rated_voltage, w_v.voltage_type)
    s_phase = float(w_u.rated_power.to("va").magnitude) / max(1, int(w_u.num_phases))
    if s_phase <= 0:
        return

    z_base = (v_u_phase * v_u_phase) / s_phase
    z = (r_pu + 1j * x_pu) * z_base
    if abs(z) == 0:
        return

    y = 1.0 / z
    # Base turns ratio from rated voltages
    a_base = v_u_phase / v_v_phase if v_v_phase > 0 else 1.0

    # Per-phase tap positions (defaults to 1.0 when not set)
    tap_pos = transformer.tap_positions

    common_phases = [
        p
        for p in transformer.winding_phases[0]
        if p in transformer.winding_phases[1] and (include_neutral or p != Phase.N)
    ]

    for phase in common_phases:
        p = _phase_name(phase)
        u_label = (bus_u.name, p)
        v_label = (bus_v.name, p)
        if u_label not in label_to_index or v_label not in label_to_index:
            continue
        i = label_to_index[u_label]
        j = label_to_index[v_label]
        # Effective turns ratio adjusted by per-phase tap positions
        a = a_base
        if tap_pos is not None:
            pri_idx = list(transformer.winding_phases[0]).index(phase)
            sec_idx = list(transformer.winding_phases[1]).index(phase)
            a = a_base * tap_pos[0][pri_idx] / tap_pos[1][sec_idx]
        # Proper two-winding transformer admittance model with turns ratio
        ybus[i, i] += y
        ybus[j, j] += a * a * y
        ybus[i, j] -= a * y
        ybus[j, i] -= a * y


def _as_sparse_if_requested(ybus: np.ndarray, sparse: bool):
    if not sparse:
        return ybus
    try:
        from scipy.sparse import csr_matrix
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "SciPy is required for sparse=True. Install with `pip install gdm-flow[sparse]`."
        ) from exc
    return csr_matrix(ybus)


def calculate_ybus(
    system: DistributionSystem,
    *,
    include_neutral: bool = False,
    include_shunt: bool = False,
    frequency_hz: float = 60.0,
    include_transformers: bool = True,
    include_open_switches: bool = False,
    convert_geometry_to_matrix: bool = True,
    sparse: bool = False,
) -> YBusResult:
    """Calculate a phase-domain Y-bus matrix for a DistributionSystem.

    Parameters
    ----------
    system : DistributionSystem
        Input distribution system.
    include_neutral : bool, optional
        Include neutral phase nodes in the Y-bus index.
    include_shunt : bool, optional
        Include line charging from branch capacitance matrix (pi model).
    frequency_hz : float, optional
        System frequency for shunt charging calculations.
    include_transformers : bool, optional
        Stamp series admittance for two-winding transformers.
    include_open_switches : bool, optional
        Include open switch phases. By default open phases are excluded.
    convert_geometry_to_matrix : bool, optional
        Convert `GeometryBranch` to `MatrixImpedanceBranch` on a system copy before stamping.
    sparse : bool, optional
        Return SciPy CSR matrix when True.

    Returns
    -------
    YBusResult
        Matrix and node indexing metadata.
    """

    for reactor in system.get_components(DistributionReactor):
        if reactor.in_service:
            _reactor_impedance_ohm(reactor)
    working_system = system.deepcopy() if convert_geometry_to_matrix else system
    if convert_geometry_to_matrix and list(
        working_system.get_components(GeometryBranch)
    ):
        working_system.convert_geometry_to_matrix_representation()

    index_to_label, label_to_index = _build_bus_phase_index(
        working_system, include_neutral
    )
    n = len(index_to_label)
    ybus = np.zeros((n, n), dtype=np.complex128)

    for branch in working_system.get_components(DistributionBranchBase):
        if not branch.in_service:
            continue
        if not hasattr(branch, "equipment"):
            raise ValueError(f"Active branch {branch.name!r} has no equipment.")

        active_idx = _active_branch_phase_indices(
            branch, include_neutral, include_open_switches
        )
        if not active_idx:
            continue

        phase_names = [_phase_name(branch.phases[i]) for i in active_idx]
        if any((branch.buses[0].name, p) not in label_to_index for p in phase_names):
            continue
        if any((branch.buses[1].name, p) not in label_to_index for p in phase_names):
            continue

        if isinstance(branch, DistributionReactor):
            y_series = _reactor_series_admittance(branch, active_idx)
            y_shunt = None
        elif isinstance(branch, SequenceImpedanceBranch):
            y_series = _sequence_branch_series_admittance(branch, active_idx)
            y_shunt = None
        elif hasattr(branch.equipment, "r_matrix") and hasattr(
            branch.equipment, "x_matrix"
        ):
            y_series = _matrix_branch_series_admittance(branch, active_idx)
            y_shunt = (
                _matrix_branch_shunt_admittance(branch, active_idx, frequency_hz)
                if include_shunt
                else None
            )
        else:
            raise NotImplementedError(
                f"Unsupported equipment {type(branch.equipment).__name__} "
                f"on active branch {branch.name!r}."
            )

        _stamp_branch(
            ybus,
            label_to_index,
            branch.buses[0].name,
            branch.buses[1].name,
            phase_names,
            y_series,
            y_shunt,
        )

    if include_transformers:
        for transformer in working_system.get_components(DistributionTransformer):
            if transformer.in_service:
                _stamp_transformer(ybus, label_to_index, transformer, include_neutral)
        for regulator in working_system.get_components(DistributionRegulator):
            if regulator.in_service:
                _stamp_transformer(ybus, label_to_index, regulator, include_neutral)

    return YBusResult(
        ybus=_as_sparse_if_requested(ybus, sparse),
        index_to_label=index_to_label,
        label_to_index=label_to_index,
    )
