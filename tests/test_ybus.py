import numpy as np
import pytest

from gdm.systems.distribution import DistributionSystem
from gdm.systems.distribution.enums import Phase, VoltageTypes
from gdm.systems.distribution.components import (
    DistributionReactor,
    MatrixImpedanceBranch,
    MatrixImpedanceSwitch,
)

from gdm_flow import calculate_ybus
from gdm_flow import ybus as ybus_mod
from gdm.systems.distribution.equipment import (
    CircuitBreakerEquipment,
    DisconnectorEquipment,
    EarthingSwitchEquipment,
    PowerTransformerEquipment,
)


def test_calculate_ybus_matrix_branch():
    system = DistributionSystem(auto_add_composed_components=True, name="ybus-test")
    system.add_component(MatrixImpedanceBranch.example())

    result = calculate_ybus(system)
    ybus = result.ybus

    assert ybus.shape == (6, 6)
    assert np.allclose(ybus, ybus.T)
    assert np.max(np.abs(ybus)) > 0


@pytest.mark.parametrize("length_m", [1, 200])
@pytest.mark.parametrize("sparse", [False, True])
def test_series_reactor_uses_lumped_impedance(length_m, sparse):
    reactor = DistributionReactor.example()
    reactor.length = type(reactor.length)(length_m, "meter")
    reactor.equipment.resistance = type(reactor.equipment.resistance)(0.25, "ohm")
    reactor.equipment.reactance = type(reactor.equipment.reactance)(0.75, "ohm")
    system = DistributionSystem(auto_add_composed_components=True, name="reactor-test")
    system.add_component(reactor)
    result = calculate_ybus(system, sparse=sparse, include_shunt=True)
    matrix = result.ybus.toarray() if sparse else result.ybus
    series = np.eye(3) / (0.25 + 0.75j)
    first = [
        result.label_to_index[(reactor.buses[0].name, phase.value)]
        for phase in reactor.phases
    ]
    second = [
        result.label_to_index[(reactor.buses[1].name, phase.value)]
        for phase in reactor.phases
    ]
    np.testing.assert_allclose(matrix[np.ix_(first, first)], series)
    np.testing.assert_allclose(matrix[np.ix_(second, second)], series)
    np.testing.assert_allclose(matrix[np.ix_(first, second)], -series)


@pytest.mark.parametrize(
    "resistance,reactance", [(0, 0), (-1, 1), (1, -1), (np.nan, 1), (1, np.inf)]
)
def test_series_reactor_rejects_invalid_impedance(resistance, reactance):
    reactor = DistributionReactor.example()
    reactor.equipment.resistance = type(reactor.equipment.resistance)(resistance, "ohm")
    reactor.equipment.reactance = type(reactor.equipment.reactance)(reactance, "ohm")
    system = DistributionSystem(
        auto_add_composed_components=True, name="invalid-reactor-test"
    )
    system.add_component(reactor)
    with pytest.raises(ValueError, match="Reactor.*impedance"):
        calculate_ybus(system)


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


@pytest.mark.parametrize(
    "equipment_type",
    [CircuitBreakerEquipment, DisconnectorEquipment, EarthingSwitchEquipment],
)
@pytest.mark.parametrize("include_open", [False, True])
def test_station_switch_equipment_roundtrip_and_phase_states(
    equipment_type, include_open, tmp_path
):
    switch = MatrixImpedanceSwitch.example()
    switch.equipment = equipment_type.example()
    switch.is_closed = [True, False, True]
    system = DistributionSystem(
        auto_add_composed_components=True, name="station-switch-test"
    )
    system.add_component(switch)
    path = tmp_path / "station.json"
    system.to_json(path)
    restored = DistributionSystem.from_json(path)
    restored_switch = next(restored.get_components(MatrixImpedanceSwitch))
    assert type(restored_switch.equipment) is equipment_type
    assert restored_switch.equipment.model_dump(
        mode="json"
    ) == switch.equipment.model_dump(mode="json")
    result = calculate_ybus(restored, include_open_switches=include_open)
    active = [0, 1, 2] if include_open else [0, 2]
    expected = ybus_mod._matrix_branch_series_admittance(restored_switch, active)
    labels = [(switch.buses[0].name, switch.phases[index].value) for index in active]
    indices = [result.label_to_index[label] for label in labels]
    np.testing.assert_allclose(result.ybus[np.ix_(indices, indices)], expected)
    phase_b = result.label_to_index[(switch.buses[0].name, "B")]
    assert bool(np.any(result.ybus[phase_b])) == include_open


@pytest.mark.parametrize(
    "connections,vector_group,leadlag",
    [
        ("delta wye", "Dyn1", "lag"),
        ("delta wye", "Dyn11", "lead"),
        ("wye delta", "YNd1", "lag"),
        ("wye delta", "YNd11", "lead"),
        ("wye wye", "YNyn0", "lag"),
        ("delta delta", "Dd0", "lag"),
    ],
)
@pytest.mark.parametrize("grounded", [False, True])
@pytest.mark.parametrize("tap,core_loss", [(1.0, 0.0), (1.03, 0.2)])
def test_station_transformer_matches_opendss(
    connections, vector_group, leadlag, grounded, tap, core_loss
):
    from gdm.systems.distribution.components import DistributionTransformer
    from gdm.systems.distribution.enums import ConnectionType

    dss = pytest.importorskip("opendssdirect").NewContext()
    transformer = DistributionTransformer.example()
    equipment = PowerTransformerEquipment.example()
    windings = [
        winding.model_copy(
            update={
                "resistance": 0.5,
                "is_grounded": grounded,
                "rated_power": type(winding.rated_power)(1_000_000, "va"),
                "connection_type": ConnectionType.STAR
                if connection == "wye"
                else ConnectionType.DELTA,
            }
        )
        for winding, connection in zip(equipment.windings, connections.split())
    ]
    equipment = equipment.model_copy(
        update={
            "windings": windings,
            "vector_group": vector_group,
            "winding_reactances": [5.0],
            "pct_no_load_loss": core_loss,
        }
    )
    transformer.equipment = equipment
    transformer.tap_positions = [[tap] * 3, [0.98] * 3]
    system = DistributionSystem(
        auto_add_composed_components=True, name="station-transformer-test"
    )
    system.add_component(transformer)
    result = calculate_ybus(system)
    indices = [
        result.label_to_index[(bus.name, phase.value)]
        for bus in transformer.buses
        for phase in (Phase.A, Phase.B, Phase.C)
    ]
    actual = result.ybus[np.ix_(indices, indices)]
    dss.Text.Command("clear")
    dss.Text.Command("new circuit.reference basekv=12.47 bus1=primary pu=1 phases=3")
    neutral_node = 0 if grounded else 4
    dss.Text.Command(
        f"new transformer.station phases=3 windings=2 buses=[primary.1.2.3.{neutral_node} secondary.1.2.3.{neutral_node}] "
        f"conns=[{connections}] kvs=[12.47 0.4] kvas=[1000 1000] %rs=[0.5 0.5] xhl=5 "
        f"%noloadloss={core_loss} leadlag={leadlag} taps=[{tap} 0.98]"
    )
    dss.Text.Command("solve")
    dss.Circuit.SetActiveElement("transformer.station")
    primitive = np.asarray(dss.CktElement.YPrim()).reshape(8, 8, 2)
    primitive = primitive[:, :, 0] + 1j * primitive[:, :, 1]
    terminals = [0, 1, 2, 4, 5, 6]
    expected = primitive[np.ix_(terminals, terminals)]
    if not grounded:
        neutrals = [
            index
            for index, connection in zip((3, 7), connections.split())
            if connection == "wye"
        ]
        if neutrals:
            expected -= (
                primitive[np.ix_(terminals, neutrals)]
                @ np.linalg.pinv(primitive[np.ix_(neutrals, neutrals)])
                @ primitive[np.ix_(neutrals, terminals)]
            )
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_loaded_station_transformer_power_flow_matches_opendss():
    from gdm.systems.distribution.components import (
        DistributionLoad,
        DistributionTransformer,
        DistributionVoltageSource,
    )
    from gdm_flow.ac_pf import solve_ac_power_flow_from_components

    dss = pytest.importorskip("opendssdirect").NewContext()
    transformer = DistributionTransformer.example()
    equipment = PowerTransformerEquipment.example()
    windings = [
        winding.model_copy(
            update={
                "resistance": 0.5,
                "is_grounded": True,
                "rated_power": type(winding.rated_power)(1_000_000, "va"),
            }
        )
        for winding in equipment.windings
    ]
    transformer.equipment = equipment.model_copy(
        update={
            "windings": windings,
            "vector_group": "Dyn11",
            "winding_reactances": [5.0],
            "pct_no_load_loss": 0.2,
        }
    )
    transformer.tap_positions = [[1.03] * 3, [0.98] * 3]
    source = DistributionVoltageSource.example()
    source.bus = transformer.buses[0]
    load = DistributionLoad.example()
    load.bus = transformer.buses[1]
    load.equipment.phase_loads = [
        phase_load.model_copy(
            deep=True,
            update={
                "real_power": type(phase_load.real_power)((index + 1) * 10_000, "watt"),
                "reactive_power": type(phase_load.reactive_power)(
                    (index + 3) * 1_000, "var"
                ),
            },
        )
        for index, phase_load in enumerate(load.equipment.phase_loads)
    ]
    assert len({id(item) for item in load.equipment.phase_loads}) == 3
    assert [
        float(item.real_power.to("watt").magnitude)
        for item in load.equipment.phase_loads
    ] == [10_000, 20_000, 30_000]
    system = DistributionSystem(
        auto_add_composed_components=True, name="loaded-station-test"
    )
    for component in (transformer, source, load):
        system.add_component(component)
    solved = solve_ac_power_flow_from_components(system, v0_complex={}, tolerance=1e-9)
    assert solved.success
    source_voltage = solved.voltage[
        solved.ybus_result.label_to_index[(source.bus.name, "A")]
    ]
    dss.Text.Command("clear")
    dss.Text.Command(
        f"new circuit.reference basekv={abs(source_voltage) * np.sqrt(3) / 1000} bus1=primary angle={np.angle(source_voltage, deg=True)} pu=1 phases=3"
    )
    dss.Text.Command("edit vsource.source r1=1e-9 x1=1e-9 r0=1e-9 x0=1e-9")
    dss.Text.Command(
        "new transformer.station phases=3 windings=2 buses=[primary.1.2.3.0 secondary.1.2.3.0] conns=[delta wye] kvs=[12.47 0.4] kvas=[1000 1000] %rs=[0.5 0.5] xhl=5 %noloadloss=0.2 leadlag=lead taps=[1.03 0.98]"
    )
    for index in range(3):
        dss.Text.Command(
            f"new load.phase{index} phases=1 bus1=secondary.{index + 1}.0 conn=wye model=1 kv={0.4 / np.sqrt(3)} kw={(index + 1) * 10} kvar={index + 3} vminpu=0.01 vmaxpu=2"
        )
    dss.Text.Command("set tolerance=1e-10 maxiterations=100")
    dss.Text.Command("solve")
    assert dss.Solution.Converged()
    dss.Circuit.SetActiveBus("secondary")
    expected = np.asarray(dss.Bus.Voltages()).reshape(-1, 2)
    expected = expected[:, 0] + 1j * expected[:, 1]
    actual = [
        solved.voltage[solved.ybus_result.label_to_index[(load.bus.name, phase.value)]]
        for phase in load.phases
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-5)


@pytest.mark.parametrize(
    "invalid,message",
    [
        ("loss", "Invalid station transformer"),
        ("tap", "Invalid station transformer"),
        ("impedance", "leakage impedance must be nonzero"),
        ("vector", "disagrees with winding connections"),
    ],
)
def test_station_transformer_rejects_invalid_electrical_data(invalid, message):
    from gdm.systems.distribution.components import DistributionTransformer

    transformer = DistributionTransformer.example().model_copy(
        deep=True, update={"equipment": PowerTransformerEquipment.example()}
    )
    if invalid == "loss":
        transformer.equipment = transformer.equipment.model_copy(
            update={"pct_no_load_loss": -1.0}
        )
    elif invalid == "tap":
        transformer = transformer.model_copy(
            update={"tap_positions": [[0.0] * 3, [1.0] * 3]}
        )
    elif invalid == "impedance":
        windings = [
            winding.model_copy(update={"resistance": 0.0})
            for winding in transformer.equipment.windings
        ]
        transformer.equipment = transformer.equipment.model_copy(
            update={"windings": windings, "winding_reactances": [0.0]}
        )
    else:
        transformer.equipment = transformer.equipment.model_copy(
            update={"vector_group": "YNyn0"}
        )
    with pytest.raises(ValueError, match=message):
        ybus_mod._stamp_power_transformer(
            np.zeros((6, 6), dtype=complex), {}, transformer, False
        )


def test_split_phase_neutral_index_is_added_once():
    from types import SimpleNamespace

    primary = SimpleNamespace(name="primary", phases=[Phase.A])
    secondary = SimpleNamespace(name="secondary", phases=[Phase.S1, Phase.S2])
    transformer = SimpleNamespace(
        in_service=True,
        equipment=SimpleNamespace(windings=[object(), object(), object()]),
        buses=[primary, secondary, secondary],
        winding_phases=[[Phase.A], [Phase.S1, Phase.N], [Phase.N, Phase.S2]],
    )
    system = SimpleNamespace(
        get_components=lambda kind: [primary, secondary]
        if kind is ybus_mod.DistributionBus
        else [transformer]
    )
    labels, indices = ybus_mod._build_bus_phase_index(system, include_neutral=True)
    assert labels == [
        ("primary", "A"),
        ("secondary", "S1"),
        ("secondary", "S2"),
        ("secondary", "N"),
    ]
    assert len(labels) == len(indices) == 4
    labels, _ = ybus_mod._build_bus_phase_index(system, include_neutral=False)
    assert ("secondary", "N") not in labels


@pytest.mark.parametrize("vector_group", ["Dyn0", "Dyn2"])
def test_station_transformer_rejects_incompatible_clock(vector_group):
    from gdm.systems.distribution.components import DistributionTransformer

    transformer = DistributionTransformer.example()
    transformer.equipment = PowerTransformerEquipment.example().model_copy(
        update={"vector_group": vector_group}
    )
    system = DistributionSystem(
        auto_add_composed_components=True, name="unsupported-clock-test"
    )
    system.add_component(transformer)
    with pytest.raises(NotImplementedError, match="clock orientation"):
        calculate_ybus(system)


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
            ybus_mod._stamp_transformer(
                matrix, labels, transformer, include_neutral=False
            )
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
    np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)


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
