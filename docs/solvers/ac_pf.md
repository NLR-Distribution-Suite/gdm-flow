# AC Power Flow (Newton-Raphson)

The AC PF solver uses Newton-Raphson with a sparse Jacobian and backtracking.
Unlike [AC OPF](ac_opf.md), which optimises voltage magnitudes within bounds,
AC PF solves fixed P/Q injections at PQ buses and fixed slack-bus voltage.

## Formulation

Inputs and returned voltages and powers use SI units. Internally, the solver
scales the Y-bus and power specifications to per-unit using nominal
phase-voltage bases and a common power base.

### Newton-Raphson Iteration

At each iteration, calculate complex nodal power from the current voltage:

$$
S^{calc} = V \odot \overline{YV}.
$$

The stacked active/reactive mismatch at non-slack nodes defines a sparse Jacobian system for voltage-angle and magnitude corrections:

$$
J\Delta x = -\begin{bmatrix}\operatorname{Re}(S^{calc}-S^{spec}) \\
\operatorname{Im}(S^{calc}-S^{spec})\end{bmatrix}.
$$

The Jacobian is rebuilt each iteration and solved with SciPy's sparse solver.
Backtracking starts at a full step and halves it when mismatch does not improve.

### Convergence Criterion

The largest absolute active or reactive power mismatch in per-unit must be
below `tolerance`:

$$
\max_{i \notin slack}\left\{|\Delta P_i|,|\Delta Q_i|\right\} < tolerance.
$$

### Initial Voltage Estimate

Initialization uses nominal magnitudes, balanced A/B/C angles and S1/S2 angles
propagated from each service transformer. Without an external warm start, the
solver attempts a LinDistFlow magnitude estimate and falls back to nominal
values if it fails. `v0_complex` overrides matching initial terminal voltages;
an empty mapping skips automatic LinDistFlow initialization.

## Supported Scope

- Sparse Newton steps with per-unit internal scaling and SI-unit results.
- Grounded split-phase services with opposite S1/S2 polarity and unequal leg loads.
- Line-to-line loads using actual differential voltage and corresponding Jacobian terms.
- External warm starts accepting complex terminal voltages from a prior solve.

Explicit-neutral and floating-neutral split-phase stamping is unsupported.
Component-based load evaluation uses constant P/Q, not ZIP coefficients.
Solver convergence alone does not certify grounding metadata, thermal limits
or independent agreement with an external feeder model.

## Usage

### Low-level interface

```python
from gdm.distribution import DistributionSystem
from gdm_flow import solve_ac_power_flow

system = DistributionSystem.from_json("model.json")

result = solve_ac_power_flow(
    system,
    p_spec_w={("bus_5", "A"): -20_000.0},
    q_spec_var={("bus_5", "A"): -5_000.0},
    max_iterations=100,
    tolerance=1e-6,
)

print(result.success, result.iterations)
print(result.max_mismatch_pu)
```

### Component-based interface

```python
from gdm.distribution import DistributionSystem
from gdm_flow import solve_ac_power_flow_from_components

system = DistributionSystem.from_json("model.json")

result = solve_ac_power_flow_from_components(
    system,
    include_loads=True,
    include_solar=True,
    include_capacitor=True,
    load_scale=1.0,
    solar_scale=1.0,
)

print(f"Converged: {result.success}")
print(f"Iterations: {result.iterations}")
print(f"Max power mismatch: {result.max_mismatch_pu:.2e} pu")
```

## Result Object

`ACPowerFlowResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether solver converged within tolerance |
| `message` | `str` | Convergence status message |
| `ybus_result` | `YBusResult` | Y-bus matrix and node mapping |
| `voltage` | `np.ndarray` | Complex bus voltages in SI volts |
| `voltage_pu` | `np.ndarray` | Per-unit voltage magnitudes |
| `power_injection` | `np.ndarray` | Complex power injection at each bus (W + j·var) |
| `iterations` | `int` | Number of Newton-Raphson iterations |
| `max_mismatch_pu` | `float` | Final maximum active/reactive power mismatch in per-unit |

## AC PF vs AC OPF

| Aspect | AC PF | AC OPF |
|--------|-------|--------|
| **Method** | Newton-Raphson with backtracking | Nonlinear least-squares (optimisation) |
| **Units** | SI inputs/results; per-unit internal solve | Per-unit |
| **Slack bus** | Fixed at nominal voltage | Adjusted within bounds |
| **Voltage bounds** | None — reports actual voltages | Enforced via `vm_min_pu` / `vm_max_pu` |
| **Regulator targets** | Not modeled | Soft voltage targets via penalty |
| **Use case** | Classical power flow baseline | Voltage regulation studies |

Both solvers share the same Y-bus construction and produce compatible result objects, making them easy to compare side-by-side.
