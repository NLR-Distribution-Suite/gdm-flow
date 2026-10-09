import numpy as np
import pytest

from gdm.distribution import DistributionSystem
from gdm.distribution.enums import Phase, VoltageTypes
from gdm.distribution.components import MatrixImpedanceBranch, MatrixImpedanceSwitch

from gdm_flow import calculate_ybus
from gdm_flow import ybus as ybus_mod


def test_calculate_ybus_matrix_branch():
    system = DistributionSystem(auto_add_composed_components=True, name="ybus-test")
    system.add_component(MatrixImpedanceBranch.example())

    result = calculate_ybus(system)
    ybus = result.ybus

    assert ybus.shape == (6, 6)
    assert np.allclose(ybus, ybus.T)
    assert np.max(np.abs(ybus)) > 0


def test_open_switch_phase_is_excluded_by_default():
    system = DistributionSystem(
        auto_add_composed_components=True, name="ybus-switch-test"
    )
    switch = MatrixImpedanceSwitch.example()
    switch.is_closed = [True, False, True]
    system.add_component(switch)

    result = calculate_ybus(system)

    b_entries = [
        idx for idx, label in enumerate(result.index_to_label) if label[1] == "B"
    ]
    assert len(b_entries) == 2
    for idx in b_entries:
        assert np.allclose(result.ybus[idx, :], 0)
        assert np.allclose(result.ybus[:, idx], 0)


class _Q:
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def to(self, _unit):
        return self


class _Eq:
    def __init__(self, r, x, c=None):
        self.r_matrix = _Q(np.array(r, dtype=float))
        self.x_matrix = _Q(np.array(x, dtype=float))
        self.c_matrix = _Q(np.array(c if c is not None else r, dtype=float))


class _Branch:
    def __init__(self, r, x, phases):
        self.equipment = _Eq(r=r, x=x, c=[[1e-9, 0.0], [0.0, 1e-9]])
        self.length = _Q(100.0)
        self.phases = phases


def test_active_branch_phase_indices_include_open_switches():
    switch = MatrixImpedanceSwitch.example()
    switch.is_closed = [True, False, True]

    default_idx = ybus_mod._active_branch_phase_indices(
        switch, include_neutral=False, include_open_switches=False
    )
    include_open_idx = ybus_mod._active_branch_phase_indices(
        switch, include_neutral=False, include_open_switches=True
    )

    assert default_idx == [0, 2]
    assert include_open_idx == [0, 1, 2]


def test_matrix_branch_series_admittance_uses_pinv_for_singular():
    branch = _Branch(
        r=[[0.0, 0.0], [0.0, 0.0]],
        x=[[0.0, 0.0], [0.0, 0.0]],
        phases=[Phase.A, Phase.B],
    )
    y = ybus_mod._matrix_branch_series_admittance(branch, active_idx=[0, 1])

    # Zero-impedance branches are clamped to _Z_MIN*(1+1j) then inverted,
    # producing a finite admittance matrix (not zeros or None).
    assert y is not None
    assert y.shape == (2, 2)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) > 0


def test_matrix_branch_shunt_admittance_nonzero():
    branch = _Branch(
        r=[[0.1, 0.0], [0.0, 0.1]],
        x=[[0.2, 0.0], [0.0, 0.2]],
        phases=[Phase.A, Phase.B],
    )
    y_shunt = ybus_mod._matrix_branch_shunt_admittance(
        branch, active_idx=[0, 1], frequency_hz=60.0
    )

    assert y_shunt.shape == (2, 2)
    assert np.max(np.abs(y_shunt)) > 0


class _SeqEq:
    pos_seq_resistance = _Q(0.1)
    pos_seq_reactance = _Q(0.2)
    zero_seq_resistance = _Q(0.3)
    zero_seq_reactance = _Q(0.4)


class _SeqBranch:
    def __init__(self, phases):
        self.phases = phases
        self.length = _Q(100.0)
        self.equipment = _SeqEq()


def test_sequence_branch_series_admittance_three_phase_mutual_terms():
    branch = _SeqBranch([Phase.A, Phase.B, Phase.C])
    y = ybus_mod._sequence_branch_series_admittance(branch, active_idx=[0, 1, 2])

    assert y.shape == (3, 3)
    assert not np.allclose(y, np.diag(np.diag(y)))


def test_stamp_branch_with_shunt():
    ybus = np.zeros((4, 4), dtype=np.complex128)
    label_to_index = {
        ("u", "A"): 0,
        ("u", "B"): 1,
        ("v", "A"): 2,
        ("v", "B"): 3,
    }
    y_series = np.array([[10 + 5j, 1j], [1j, 8 + 3j]], dtype=np.complex128)
    y_shunt = np.array([[2j, 0], [0, 4j]], dtype=np.complex128)

    ybus_mod._stamp_branch(
        ybus,
        label_to_index,
        "u",
        "v",
        ["A", "B"],
        y_series,
        y_shunt,
    )

    assert np.max(np.abs(ybus)) > 0
    assert np.allclose(ybus, ybus.T)


class _Winding:
    def __init__(self, rated_voltage, rated_power, num_phases, voltage_type):
        self.rated_voltage = _Q(rated_voltage)
        self.rated_power = _Q(rated_power)
        self.num_phases = num_phases
        self.voltage_type = voltage_type


class _TransformerEq:
    def __init__(self, windings):
        self.windings = windings
        self.pct_full_load_loss = 1.0
        self.winding_reactances = [4.0]


class _Bus:
    def __init__(self, name):
        self.name = name


class _Transformer:
    def __init__(self, buses, windings, winding_phases, tap_positions=None):
        self.buses = buses
        self.equipment = _TransformerEq(windings)
        self.winding_phases = winding_phases
        self.tap_positions = tap_positions


def test_stamp_transformer_two_winding_and_center_tap_paths():
    ybus = np.zeros((4, 4), dtype=np.complex128)
    labels = {
        ("pri", "A"): 0,
        ("sec", "A"): 1,
        ("s1", "S1"): 2,
        ("s2", "S2"): 3,
    }

    w_primary = _Winding(7200.0, 50_000.0, 1, VoltageTypes.LINE_TO_GROUND)
    w_secondary = _Winding(240.0, 50_000.0, 1, VoltageTypes.LINE_TO_GROUND)

    two_winding = _Transformer(
        buses=[_Bus("pri"), _Bus("sec")],
        windings=[w_primary, w_secondary],
        winding_phases=[[Phase.A], [Phase.A]],
    )
    ybus_mod._stamp_transformer(ybus, labels, two_winding, include_neutral=False)

    center_tapped = _Transformer(
        buses=[_Bus("pri"), _Bus("s1"), _Bus("s2")],
        windings=[
            w_primary,
            _Winding(120.0, 25_000.0, 1, VoltageTypes.LINE_TO_GROUND),
            _Winding(120.0, 25_000.0, 1, VoltageTypes.LINE_TO_GROUND),
        ],
        winding_phases=[[Phase.A], ["S1"], [Phase.N, "S2"]],
    )
    with pytest.raises(ValueError, match="center tap"):
        ybus_mod._stamp_transformer(ybus, labels, center_tapped, include_neutral=False)

    assert np.max(np.abs(ybus)) > 0


def test_as_sparse_if_requested_false_and_true():
    arr = np.eye(2, dtype=np.complex128)
    dense = ybus_mod._as_sparse_if_requested(arr, sparse=False)
    sparse = ybus_mod._as_sparse_if_requested(arr, sparse=True)

    assert isinstance(dense, np.ndarray)
    assert hasattr(sparse, "toarray")


@pytest.mark.parametrize("no_load_loss", [0.0, 0.2, -1.0, np.nan])
@pytest.mark.parametrize("pair_offset", [0, 1])
def test_grounded_center_tap_matches_three_winding_primitive(no_load_loss, pair_offset):
    from types import SimpleNamespace

    windings = [
        _Winding(7200, 25000, 1, VoltageTypes.LINE_TO_GROUND),
        _Winding(120, 25000, 1, VoltageTypes.LINE_TO_GROUND),
        _Winding(120, 25000, 1, VoltageTypes.LINE_TO_GROUND),
    ]
    for winding, resistance in zip(windings, (0.2, 0.4, 0.4)):
        winding.resistance = resistance
        winding.is_grounded = True
        winding.connection_type = "STAR"
    transformer = _Transformer(
        [_Bus("primary"), _Bus("secondary"), _Bus("secondary")],
        windings,
        [[Phase.A], [Phase.N, Phase.S1], [Phase.N, Phase.S2]],
    )
    transformer.equipment.winding_reactances = [2, 2, 1]
    transformer.equipment.pct_no_load_loss = no_load_loss
    transformer.equipment.coupling_sequences = [
        SimpleNamespace(from_index=first + pair_offset, to_index=second + pair_offset)
        for first, second in ((0, 1), (0, 2), (1, 2))
    ]
    labels = {("primary", "A"): 0, ("secondary", "S1"): 1, ("secondary", "S2"): 2}
    matrix = np.zeros((3, 3), dtype=complex)
    if not np.isfinite(no_load_loss) or no_load_loss < 0:
        with pytest.raises(ValueError, match="no-load loss"):
            ybus_mod._stamp_transformer(matrix, labels, transformer, include_neutral=False)
        return
    ybus_mod._stamp_transformer(matrix, labels, transformer, include_neutral=False)
    impedance_base = 7200**2 / 25000
    leakage = impedance_base * np.array(
        [0.002 + 0.015j, 0.004 + 0.005j, 0.004 + 0.005j]
    )
    conductance = 1 / leakage
    primitive = (
        np.diag(conductance) - np.outer(conductance, conductance) / conductance.sum()
    )
    turns = np.diag([1.0, 60.0, -60.0])
    expected = turns @ primitive @ turns
    expected[1, 1] += no_load_loss / 100 * 25000 / 120**2
    np.testing.assert_allclose(
        matrix, expected, rtol=1e-12, atol=1e-12
    )


def test_as_sparse_if_requested_raises_when_scipy_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "scipy.sparse":
            raise ModuleNotFoundError("scipy unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError):
        ybus_mod._as_sparse_if_requested(np.eye(1), sparse=True)
