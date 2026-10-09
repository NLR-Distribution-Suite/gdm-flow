# Y-Bus Construction

The Y-bus (admittance matrix) is the shared foundation for all GDM-Flow solvers. It encodes the network topology, branch impedances, and transformer models in a single complex-valued matrix.

## Theory

For a system with $n$ bus-phase nodes, the Y-bus is an $n \times n$ complex matrix where:

$$Y_{bus} \cdot V = I$$

Each branch between nodes $i$ and $j$ with series admittance $y = z^{-1}$ is stamped as:

$$Y_{ii} \mathrel{+}= y, \quad Y_{jj} \mathrel{+}= y, \quad Y_{ij} \mathrel{-}= y, \quad Y_{ji} \mathrel{-}= y$$

### Transformer Model

Two-winding transformers with turns ratio $a = V_{primary} / V_{secondary}$ use the standard model:

$$\begin{bmatrix} I_p \\ I_s \end{bmatrix} = \begin{bmatrix} y & -ay \\ -ay & a^2 y \end{bmatrix} \begin{bmatrix} V_p \\ V_s \end{bmatrix}$$

This ensures zero current injection at nominal voltages (no phantom power) and properly handles voltage transformation across winding ratios.

### Center-Tapped (Split-Phase) Transformer Model

Center-tapped transformers have three windings: a primary (e.g. phase A at 7200 V) and two 120 V secondaries sharing a neutral center tap. In the GDM data model these appear as:

- **Winding 0** (primary): phases `[A]` (or `B`, `C`)
- **Winding 1**: phases `[S1, N]` — positive polarity
- **Winding 2**: phases `[N, S2]` — reversed polarity

The grounded center-tap handler constructs a coupled three-terminal primitive
for the primary, S1 and S2. It uses all three pairwise leakage reactances, winding
resistances, voltage/power ratings and tap positions; secondary paths are not
treated as independent transformers with equally divided admittance.

S1 has positive polarity and S2 negative polarity relative to the primary,
regardless of secondary winding or bus-phase ordering. This gives the expected
180-degree leg separation. Nonnegative no-load core loss is a conductance on
the second winding, using its tap-adjusted voltage base. Independent OpenDSS
fixtures check the primitive with asymmetric taps and nonzero no-load losses.

#### Supported Scope

- Exactly three single-phase STAR windings: one A/B/C primary and two
  secondaries sharing a bus, with terminal sets `{N, S1}` and `{N, S2}`.
- Both secondary windings must declare `is_grounded=True`. Ground is an
  implicit zero-voltage reference, not a solved neutral terminal.
- `include_neutral=True`, floating center taps, unsupported winding layouts,
  incomplete pairwise reactances and invalid ratings are rejected.
- Complete zero-based or one-based winding-pair indices are accepted.
- Split-phase line-to-line bus ratings use half the rating as the per-leg base.

The Y-bus does not silently infer grounded metadata. An external adapter such
as gdm-reduce may explicitly assume grounding on a copy and record that policy;
this does not establish physical grounding or modify the original model.

## Node Indexing

The Y-bus assigns a unique integer index to each `(bus_name, phase)` pair. The `YBusResult` contains the mapping:

```python
result = calculate_ybus(system)

# Index → label
label = result.index_to_label[0]  # e.g., ("bus_1", "A")

# Label → index
idx = result.label_to_index[("bus_1", "A")]  # e.g., 0
```

## Supported Branch Types

| Branch Type | Description |
|-------------|-------------|
| `MatrixImpedanceBranch` | Full phase impedance/admittance matrix |
| `SequenceImpedanceBranch` | Positive/zero sequence impedance → phase domain via symmetrical components |
| `GeometryBranch` | Wire geometry → auto-converted to matrix representation |
| `DistributionTransformer` | Two-winding or center-tapped (3-winding) transformer with per-unit leakage impedance |

## Usage

### Basic Y-Bus

```python
from gdm_flow import calculate_ybus

result = calculate_ybus(system)
print(f"Shape: {result.ybus.shape}")
print(f"Nodes: {len(result.index_to_label)}")
```

### With Options

```python
result = calculate_ybus(
    system,
    include_neutral=False,       # Required for grounded S1/S2 services
    include_shunt=True,          # Include line charging (pi model)
    include_transformers=True,   # Include transformer admittance
    sparse=True,                 # Return scipy CSR matrix
    frequency_hz=60.0,           # System frequency for shunt
)
```

### Inspecting the Matrix

```python
import numpy as np

Y = result.ybus
print(f"Non-zero entries: {np.count_nonzero(Y)}")
print(f"Symmetric: {np.allclose(Y, Y.T)}")
print(f"Condition number: {np.linalg.cond(Y):.2e}")
```
