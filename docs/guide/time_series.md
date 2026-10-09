# Time Series Simulation

GDM-Flow supports two modes of time series simulation for distribution systems with time-varying load and solar profiles:

1. **QSTS (Quasi-Static Time Series)** — Sequential snapshot solves at each timestep
2. **Multi-Period OPF** — Joint optimization across a time horizon with inter-temporal coupling

## Prerequisites

Time series data must be attached to components in your GDM distribution system model. Use the `ts-info` CLI command to check availability:

```bash
gdm-flow ts-info examples/models/p5r.json
```

## Discovery API

Before running simulations, you can inspect what time series data is available:

```python
from gdm.systems.distribution import DistributionSystem
from gdm_flow import list_component_time_series, has_time_series_data

system = DistributionSystem.from_json("model.json")

# Check if any components have time series
if has_time_series_data(system):
    ts_map = list_component_time_series(system)
    for comp_type, entries in ts_map.items():
        for info in entries:
            print(f"{comp_type}: {info.component_name} — "
                  f"{info.variable_name} ({info.length} steps, {info.resolution})")
```

## QSTS Simulation

QSTS solves each timestep independently using any of the four solvers, with optional warm-starting from the previous solution for faster convergence.

```python
from gdm_flow import run_qsts

summary = run_qsts(
    system,
    solver="ldf",           # "ac", "pf", "dc", or "ldf"
    timestep_range=range(96),  # First 24 hours at 15-min resolution
    db_path="qsts_results.db",  # Optional: stream to SQLite
)

print(f"Converged: {summary.num_converged}/{summary.num_timesteps}")
print(f"Resolution: {summary.resolution}")
```

### Supported Solvers

| Solver | Code | Per-Timestep Speed | Notes |
|--------|------|--------------------|-------|
| LinDistFlow | `ldf` | ~5 ms | Fastest; radial networks only |
| AC PF | `pf` | ~200 ms | Full NR power flow with warm-start |
| AC OPF | `ac` | ~300 ms | Optimization with voltage bounds |
| DC OPF | `dc` | ~400 ms | Economic dispatch per timestep |

### Warm-Starting

When using `ac`, `pf`, or `dc` solvers, QSTS automatically passes the previous timestep's voltage solution as a warm-start to the next solve. This significantly improves convergence speed for consecutive timesteps with similar loading.

## Multi-Period OPF

Multi-period optimization jointly solves across all timesteps, coupling battery state-of-charge (SOC) and generator ramp constraints across the time horizon.

### Multi-Period DC OPF

```python
from gdm_flow import solve_multiperiod_dc_opf, BatterySpec
from gdm_flow.dc_opf import build_dc_generators_from_components

generators = build_dc_generators_from_components(system)

# Optional: define battery parameters
battery = BatterySpec(
    name="ess_1",
    node=("bus_5", "A"),
    energy_capacity_wh=50_000,
    p_charge_max_w=10_000,
    p_discharge_max_w=10_000,
    soc_initial=0.5,
    soc_min=0.1,
    soc_max=0.9,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
)

result = solve_multiperiod_dc_opf(
    system,
    generators=generators,
    timestep_range=range(96),
    battery_specs=[battery],
    ramp_limit_w=5000.0,     # Optional generator ramp limit
    db_path="mp_dc.db",      # Optional SQLite output
)

print(f"Objective: {result.objective:.2f}")
for name, soc_trace in result.battery_soc.items():
    print(f"  {name}: SOC {soc_trace[0]:.2f} → {soc_trace[-1]:.2f}")
```

### Multi-Period LinDistFlow

```python
from gdm_flow import solve_multiperiod_lindistflow

result = solve_multiperiod_lindistflow(
    system,
    timestep_range=range(96),
    battery_specs=[battery],  # Optional
    db_path="mp_ldf.db",
)

# Access per-timestep voltages
for t_idx, voltages in result.nodal_voltage.items():
    min_v = min(voltages.values())
    print(f"  t={t_idx}: min voltage = {min_v:.1f} V")
```

## QSTS vs Multi-Period

| Feature | QSTS | Multi-Period |
|---------|------|-------------|
| **Approach** | Sequential snapshots | Joint optimization |
| **Battery SOC** | Tracked but not optimized | Optimized across horizon |
| **Ramp Constraints** | Not supported | Generator ramp limits |
| **Solvers** | All four (ac, pf, dc, ldf) | DC OPF and LinDistFlow |
| **Speed** | ~5–400 ms per timestep | Single LP solve |
| **Use Case** | Impact studies, monitoring | Dispatch scheduling, storage optimization |

## SQLite Output Schema

Both QSTS and multi-period results can stream to SQLite using the `db_path` parameter. The database uses these tables:

| Table | Contents |
|-------|----------|
| `ts_runs` | Run metadata (solver, mode, timestep count, resolution) |
| `ts_nodes` | Per-timestep bus voltages and power injections |
| `ts_branches` | Per-timestep branch flows and loading |
| `ts_battery_soc` | Battery SOC, dispatch, and energy at each timestep |
| `ts_summary` | Per-timestep convergence status and source power |

## CLI Commands

```bash
# QSTS simulation
gdm-flow qsts model.json --solver ldf --end 96 --db results.db

# Multi-period optimization
gdm-flow multiperiod model.json --solver dc --end 96 --db results.db

# Plot results from SQLite
gdm-flow plot-ts results.db --output timeseries.html
```

See the [CLI guide](../guide/cli.md) for full option details.
