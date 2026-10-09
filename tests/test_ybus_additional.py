from __future__ import annotations

import numpy as np
import pytest

from gdm.systems.distribution.enums import Phase, VoltageTypes

from gdm_flow import ybus as ybus_mod


class _Q:
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def to(self, _unit):
        return self


class _Bus:
    def __init__(self, name, phases):
        self.name = name
        self.phases = phases


class _EquipMatrix:
    def __init__(self):
        self.r_matrix = _Q(np.array([[0.1]], dtype=float))
        self.x_matrix = _Q(np.array([[0.2]], dtype=float))
        self.c_matrix = _Q(np.array([[1e-9]], dtype=float))


class _BranchBase:
    def __init__(self, name, buses, phases, in_service=True, equipment=True):
        self.name = name
        self.buses = buses
        self.phases = phases
        self.in_service = in_service
        self.length = _Q(1.0)
        if equipment:
            self.equipment = _EquipMatrix()


class _SeqEquip:
    pos_seq_resistance = _Q(0.1)
    pos_seq_reactance = _Q(0.2)
    zero_seq_resistance = _Q(0.3)
    zero_seq_reactance = _Q(0.4)


class _SeqBranch(_BranchBase):
    def __init__(self, name, buses, phases):
        super().__init__(name, buses, phases)
        self.equipment = _SeqEquip()


class _FakeSystem:
    def __init__(self):
        self.convert_called = False
        self._buses = [_Bus("b1", [Phase.A, Phase.N]), _Bus("b2", [Phase.A, Phase.N])]
        self._branches = [
            _BranchBase(
                "off",
                [SimpleNamespace(name="b1"), SimpleNamespace(name="b2")],
                [Phase.A],
                in_service=False,
            ),
            _BranchBase(
                "noeq",
                [SimpleNamespace(name="b1"), SimpleNamespace(name="b2")],
                [Phase.A],
                equipment=False,
            ),
            _BranchBase(
                "onlyn",
                [SimpleNamespace(name="b1"), SimpleNamespace(name="b2")],
                [Phase.N],
            ),
            _BranchBase(
                "bad_from",
                [SimpleNamespace(name="x"), SimpleNamespace(name="b2")],
                [Phase.A],
            ),
            _BranchBase(
                "bad_to",
                [SimpleNamespace(name="b1"), SimpleNamespace(name="x")],
                [Phase.A],
            ),
            _SeqBranch(
                "seq",
                [SimpleNamespace(name="b1"), SimpleNamespace(name="b2")],
                [Phase.A],
            ),
            _UnsupportedBranch(
                "unsupported",
                [SimpleNamespace(name="b1"), SimpleNamespace(name="b2")],
                [Phase.A],
            ),
        ]

    def deepcopy(self):
        return self

    def convert_geometry_to_matrix_representation(self):
        self.convert_called = True

    def get_components(self, comp_type):
        if comp_type is ybus_mod.DistributionBus:
            return self._buses
        if comp_type is ybus_mod.GeometryBranch:
            return [object()]
        if comp_type is ybus_mod.DistributionBranchBase:
            return self._branches
        if comp_type is ybus_mod.DistributionTransformer:
            return [SimpleNamespace(in_service=False)]
        return []


class _UnsupportedBranch(_BranchBase):
    def __init__(self, name, buses, phases):
        super().__init__(name, buses, phases, equipment=True)
        self.equipment = SimpleNamespace()


class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _GeoMarker:
    pass


class _TransformerMarker:
    pass


def test_build_bus_index_and_active_phase_neutral_filters():
    def _get_components(typ):
        from gdm.systems.distribution.components import DistributionTransformer

        if typ is DistributionTransformer:
            return []
        return [_Bus("b", [Phase.A, Phase.N])]

    sys = SimpleNamespace(get_components=_get_components)
    labels_no_n, _ = ybus_mod._build_bus_phase_index(sys, include_neutral=False)
    labels_with_n, _ = ybus_mod._build_bus_phase_index(sys, include_neutral=True)
    assert labels_no_n == [("b", "A")]
    assert labels_with_n == [("b", "A"), ("b", "N")]

    br = _BranchBase(
        "x", [SimpleNamespace(name="b"), SimpleNamespace(name="c")], [Phase.A, Phase.N]
    )
    idx = ybus_mod._active_branch_phase_indices(
        br, include_neutral=False, include_open_switches=False
    )
    assert idx == [0]


def test_sequence_branch_fallback_and_pinv(monkeypatch):
    br = _SeqBranch(
        "s",
        [SimpleNamespace(name="b1"), SimpleNamespace(name="b2")],
        [Phase.A, Phase.B],
    )
    y = ybus_mod._sequence_branch_series_admittance(br, active_idx=[0, 1])
    assert y.shape == (2, 2)

    monkeypatch.setattr(
        np.linalg, "inv", lambda _z: (_ for _ in ()).throw(np.linalg.LinAlgError())
    )
    y2 = ybus_mod._sequence_branch_series_admittance(br, active_idx=[0, 1])
    assert y2.shape == (2, 2)


def test_stamp_transformer_guard_paths():
    y = np.zeros((2, 2), dtype=np.complex128)

    # Early return: less than two buses/windings
    t1 = SimpleNamespace(
        buses=[SimpleNamespace(name="b1")], equipment=SimpleNamespace(windings=[])
    )
    ybus_mod._stamp_transformer(y, {}, t1, include_neutral=False)

    # Early return: non-positive phase power
    w0 = SimpleNamespace(
        rated_voltage=_Q(120.0),
        rated_power=_Q(0.0),
        num_phases=1,
        voltage_type=VoltageTypes.LINE_TO_GROUND,
    )
    w1 = SimpleNamespace(
        rated_voltage=_Q(120.0),
        rated_power=_Q(1000.0),
        num_phases=1,
        voltage_type=VoltageTypes.LINE_TO_GROUND,
    )
    t2 = SimpleNamespace(
        buses=[SimpleNamespace(name="b1"), SimpleNamespace(name="b2")],
        equipment=SimpleNamespace(
            windings=[w0, w1], pct_full_load_loss=1.0, winding_reactances=[1.0]
        ),
        winding_phases=[[Phase.A], [Phase.A]],
    )
    ybus_mod._stamp_transformer(
        y, {("b1", "A"): 0, ("b2", "A"): 1}, t2, include_neutral=False
    )

    # Early return: zero impedance
    w = SimpleNamespace(
        rated_voltage=_Q(120.0),
        rated_power=_Q(1000.0),
        num_phases=1,
        voltage_type=VoltageTypes.LINE_TO_GROUND,
    )
    t3 = SimpleNamespace(
        buses=[SimpleNamespace(name="b1"), SimpleNamespace(name="b2")],
        equipment=SimpleNamespace(
            windings=[w, w], pct_full_load_loss=0.0, winding_reactances=[0.0]
        ),
        winding_phases=[[Phase.A], [Phase.A]],
    )
    ybus_mod._stamp_transformer(
        y, {("b1", "A"): 0, ("b2", "A"): 1}, t3, include_neutral=False
    )


def test_calculate_ybus_convert_and_skip_paths(monkeypatch):
    monkeypatch.setattr(ybus_mod, "DistributionBus", _Bus)
    monkeypatch.setattr(ybus_mod, "DistributionBranchBase", _BranchBase)
    monkeypatch.setattr(ybus_mod, "SequenceImpedanceBranch", _SeqBranch)
    monkeypatch.setattr(ybus_mod, "DistributionTransformer", _TransformerMarker)
    monkeypatch.setattr(ybus_mod, "GeometryBranch", _GeoMarker)

    sys = _FakeSystem()
    sys._branches = [
        branch for branch in sys._branches if branch.name not in {"noeq", "unsupported"}
    ]
    result = ybus_mod.calculate_ybus(
        sys, convert_geometry_to_matrix=True, include_transformers=True
    )

    assert sys.convert_called is True
    assert result.ybus.shape[0] == 2


@pytest.mark.parametrize(
    "name, error, message",
    [
        ("noeq", ValueError, "has no equipment"),
        ("unsupported", NotImplementedError, "Unsupported equipment"),
    ],
)
def test_active_unsupported_branches_raise(monkeypatch, name, error, message):
    monkeypatch.setattr(ybus_mod, "DistributionBus", _Bus)
    monkeypatch.setattr(ybus_mod, "DistributionBranchBase", _BranchBase)
    monkeypatch.setattr(ybus_mod, "DistributionTransformer", _TransformerMarker)
    sys = _FakeSystem()
    sys._branches = [branch for branch in sys._branches if branch.name == name]
    with pytest.raises(error, match=message):
        ybus_mod.calculate_ybus(sys, convert_geometry_to_matrix=False)
