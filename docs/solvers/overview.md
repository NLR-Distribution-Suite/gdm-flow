# Solver Overview

GDM-Flow provides four power flow solvers, each suited to different analysis needs. All operate on `DistributionSystem` objects from grid-data-models. AC PF, AC OPF, and DC OPF share Y-bus construction; LinDistFlow extracts branch impedances and transformer ratios directly.

## Solver Comparison

| Feature | AC OPF | AC PF | DC OPF | LinDistFlow |
|---------|--------|-------|--------|-------------|
| **Formulation** | Nonlinear least-squares | Sparse Newton-Raphson | Quadratic program | Backward/forward sweep |
| **Variables** | $V_m$, $\theta$ (per-unit) | $V_m$, $\theta$ (per-unit internally) | $P_g$, $\theta$ | $V^2$, $P$, $Q$ |
| **Losses** | Full $I^2R$ losses | Full $I^2R$ losses | Neglected | Neglected |
| **Reactive Power** | Full Q modeling | Full Q modeling | Neglected | Modeled |
| **Network Topology** | Meshed or radial | Meshed or radial | Meshed or radial | Radial only |
| **Economic Dispatch** | No | No | Yes (generation costs) | No |
| **Speed** | Moderate (~300 ms) | Moderate (~200 ms) | Moderate (~400 ms) | Fast (~2 ms) |
| **Accuracy** | Highest | Highest | Approximate | Approximate |
| **Center-Tapped Transformers** | Grounded services (polarity-aware) | Grounded services (polarity-aware) | Excluded (small-angle violation) | Directed-graph approximation |

## When to Use Each Solver

### AC OPF
Use when you need **accurate voltages and losses**. The AC solver finds complex voltages that satisfy power balance at every node, including reactive power and $I^2R$ line losses. Best for:
- Voltage regulation studies
- Loss analysis
- Detailed power quality assessment

### AC PF (Newton-Raphson)
Use when you need **classical power flow** with fixed P/Q injections and a slack bus. Unlike the AC OPF which optimises voltage magnitudes within bounds, the AC PF solves the standard power-flow equations directly. Best for:
- Baseline power flow studies
- Validating AC OPF results against a traditional solver
- Steady-state analysis with known load/generation profiles
- Cases where you want solved bus voltages without optimization bounds

Uses SI-unit inputs/results, per-unit internal scaling, sparse Jacobian solves
and backtracking. Initialization accepts external complex voltages or attempts
a LinDistFlow magnitude estimate. Grounded center-tapped services are supported;
explicit/floating split-phase neutrals are not.

### DC OPF
Use when you need **economic dispatch with generation costs**. The DC solver minimizes total generation cost subject to linearized power balance constraints. Best for:
- DER dispatch optimization (solar, battery, grid import)
- Market-clearing simulations
- Generation scheduling

> **Note:** DC OPF uses the small-angle approximation ($\sin\Delta\theta \approx \Delta\theta$), which breaks down across center-tapped transformers where the S2 winding operates at 180° from the primary. On systems with significant split-phase residential load, DC OPF will underestimate total source power. Use AC OPF or LinDistFlow for accurate results on such systems.

### LinDistFlow
Use when you need **fast voltage drop estimates** on radial feeders. LinDistFlow performs a single backward/forward sweep without iteration. Best for:
- Screening studies and quick assessments
- Large-scale parametric sweeps
- Hosting capacity analysis

## GDM 2.4 Equipment Scope

Reactors use lumped series impedance, including in LinDistFlow. Station breaker,
disconnector, and earthing-switch equipment use the supplied matrix-switch model
and static phase states; protection trips and fault interruption are not simulated.
AC solvers use the connection-aware two-winding station-transformer primitive,
including vector-group orientation and internal floating STAR neutral elimination.
DC OPF remains a small-angle approximation, not an independently validated
station-transformer dispatch model. LinDistFlow remains a magnitude/ratio
approximation and does not solve transformer phase shifts or floating neutrals.
See [Y-Bus Construction](ybus.md) for supported equipment and rejected layouts.

## Common Workflow

All solvers follow the same pattern:

```python
from gdm.systems.distribution import DistributionSystem

# 1. Load the system
system = DistributionSystem.from_json("model.json")

# 2. Run a solver (each has a *_from_components convenience wrapper)
result = solver_from_components(system, ...)

# 3. Inspect results
print(result.success)
```

The `*_from_components` wrapper functions automatically extract loads, solar, batteries, and other components from the system. For fine-grained control, use the lower-level functions that accept explicit parameter dictionaries.

## Time Series Simulation

For models with time-varying load and solar profiles, GDM-Flow offers two simulation modes:

| Mode | Approach | Battery SOC | Ramp Limits | Solvers |
|------|----------|-------------|-------------|---------|
| **QSTS** | Sequential snapshots with warm-start | Tracked (not optimized) | No | All four |
| **Multi-Period OPF** | Joint optimization over horizon | Optimized across time | Yes | DC OPF, LinDistFlow |

- **QSTS** is ideal for impact studies and monitoring — run any solver across thousands of timesteps
- **Multi-Period OPF** is ideal for dispatch scheduling and storage optimization — jointly optimizes battery SOC and generator dispatch

See the [Time Series guide](../guide/time_series.md) for details.
