# API — Y-Bus

## `YBusResult`

Dataclass returned by `calculate_ybus`.

| Field | Type | Description |
|-------|------|-------------|
| `ybus` | `np.ndarray` or `scipy.sparse.csr_matrix` | Complex admittance matrix |
| `index_to_label` | `list[tuple[str, str]]` | Maps integer index → `(bus_name, phase)` |
| `label_to_index` | `dict[tuple[str, str], int]` | Maps `(bus_name, phase)` → integer index |

## `calculate_ybus`

```python
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
```

Build the bus admittance matrix from all branches and transformers in the system.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `system` | — | A `DistributionSystem` instance |
| `include_neutral` | `False` | Include neutral (N) phase nodes |
| `include_shunt` | `False` | Include line charging admittance (pi model) |
| `include_transformers` | `True` | Include transformer and regulator admittance stamps |
| `include_open_switches` | `False` | Include open switch phases as connected |
| `convert_geometry_to_matrix` | `True` | Convert geometry branches on a system copy |
| `sparse` | `False` | Return `scipy.sparse.csr_matrix` instead of dense `np.ndarray` |
| `frequency_hz` | `60.0` | System frequency for computing shunt susceptance |

**Returns:** `YBusResult`

Active branches without equipment raise `ValueError`; unsupported active branch
equipment raises `NotImplementedError` rather than being silently omitted.
Station transformer layouts and vector groups outside the supported scope also
raise `NotImplementedError`. See [Y-Bus Construction](../solvers/ybus.md) for
reactor, station switch, transformer connection, and grounding semantics.

**Example:**

```python
from gdm.systems.distribution import DistributionSystem
from gdm_flow import calculate_ybus

system = DistributionSystem.from_json("model.json")
result = calculate_ybus(system, sparse=True)

print(f"Matrix shape: {result.ybus.shape}")
print(f"Number of nodes: {len(result.index_to_label)}")
```
