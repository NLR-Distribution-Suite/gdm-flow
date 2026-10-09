# SQLite Export

GDM-Flow can export solver results to a SQLite database for downstream analysis, archival, or integration with other tools.

## Quick Start

### Via CLI

```bash
gdm-flow export examples/models/p5r.json --db results.db
```

### Via Python

```python
from gdm_flow import (
    optimize_ac_power_flow_from_components,
    solve_dc_opf_from_components,
    solve_lindistflow,
    export_all_results_to_sqlite,
)
from gdm.systems.distribution import DistributionSystem

system = DistributionSystem.from_json("model.json")
ac = optimize_ac_power_flow_from_components(system)
dc = solve_dc_opf_from_components(system)
ldf = solve_lindistflow(system)

export_all_results_to_sqlite(
    db_path="results.db",
    ac_result=ac,
    dc_result=dc,
    lindistflow_result=ldf,
)
```

## Database Schema

### `runs`

Metadata for every solver execution.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT PK | Unique run identifier (`ac_<hex>`, `dc_<hex>`, `ldf_<hex>`) |
| `implementation` | TEXT | Solver type: `ac_opf`, `dc_opf`, `lindistflow` |
| `success` | INTEGER | 1 = converged, 0 = failed |
| `message` | TEXT | Solver status message |
| `created_at_utc` | TEXT | ISO 8601 timestamp |

### `ac_opf_summary`

AC OPF solver-level results.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT PK | Foreign key → `runs` |
| `iterations` | INTEGER | Number of solver iterations |
| `initial_objective` | REAL | Starting objective value |
| `final_objective` | REAL | Converged objective value |

### `ac_opf_nodes`

Per-node AC OPF results (one row per bus-phase).

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Foreign key → `runs` |
| `bus_name` | TEXT | Bus name |
| `phase` | TEXT | Phase label (A, B, C) |
| `voltage_mag_v` | REAL | Voltage magnitude (V) |
| `voltage_min_v` | REAL | Minimum voltage limit (V), when available |
| `voltage_max_v` | REAL | Maximum voltage limit (V), when available |
| `voltage_angle_rad` | REAL | Voltage angle (radians) |
| `p_injection_w` | REAL | Active power injection (W) |
| `q_injection_var` | REAL | Reactive power injection (var) |

### `ac_opf_branches`

Per-branch AC OPF loading and flow results (when branch loading data is available during export).

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Foreign key → `runs` |
| `branch_name` | TEXT | Branch identifier |
| `phase` | TEXT | Phase label |
| `p_flow_w` | REAL | Active power flow from sending side (W) |
| `q_flow_var` | REAL | Reactive power flow from sending side (var) |
| `loading_va` | REAL | Apparent loading magnitude (VA) |
| `loading_limit_va` | REAL | Branch loading limit (VA), when available |

### `dc_opf_summary`

DC OPF solver-level results.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT PK | Foreign key → `runs` |
| `objective` | REAL | Minimized total cost |
| `iterations` | INTEGER | Solver iterations |
| `slack_injection_w` | REAL | Total slack bus injection (W) |

### `dc_opf_generators`

Per-generator DC OPF dispatch.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Foreign key → `runs` |
| `generator_name` | TEXT | Generator identifier |
| `dispatch_w` | REAL | Optimal dispatch (W) |

### `dc_opf_nodes`

Per-node DC OPF results.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Foreign key → `runs` |
| `bus_name` | TEXT | Bus name |
| `phase` | TEXT | Phase label |
| `theta_rad` | REAL | Voltage angle (radians) |
| `nodal_balance_w` | REAL | Net power balance (W) |

### `dc_opf_branches`

Per-branch DC OPF loading and flow results (post-processed from solved angles).

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Foreign key → `runs` |
| `branch_name` | TEXT | Branch identifier |
| `phase` | TEXT | Phase label |
| `p_flow_w` | REAL | Approximate active power flow (W) |
| `q_flow_var` | REAL | Reactive flow placeholder (`0.0` in DC model) |
| `loading_va` | REAL | Apparent loading proxy (`abs(p_flow_w)`) |
| `loading_limit_va` | REAL | Branch loading limit (VA), when available |

### `lindistflow_summary`

LinDistFlow solver-level results.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT PK | Foreign key → `runs` |
| `source_bus` | TEXT | Root bus name |

### `lindistflow_nodes`

Per-node LinDistFlow results.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Foreign key → `runs` |
| `bus_name` | TEXT | Bus name |
| `phase` | TEXT | Phase label |
| `voltage_v` | REAL | Voltage magnitude (V) |
| `voltage_min_v` | REAL | Minimum voltage limit (V), when available |
| `voltage_max_v` | REAL | Maximum voltage limit (V), when available |
| `p_net_w` | REAL | Net active power (W) |
| `q_net_var` | REAL | Net reactive power (var) |

### `lindistflow_branches`

Per-branch LinDistFlow power flows.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Foreign key → `runs` |
| `branch_name` | TEXT | Branch identifier |
| `phase` | TEXT | Phase label |
| `p_flow_w` | REAL | Active power flow (W) |
| `q_flow_var` | REAL | Reactive power flow (var) |
| `loading_va` | REAL | Apparent branch loading magnitude (VA) |
| `loading_limit_va` | REAL | Branch loading limit (VA), when available |

### `voltage_violations`

Persisted voltage violations generated during export for AC OPF and LinDistFlow runs.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Foreign key → `runs` |
| `implementation` | TEXT | `ac_opf` or `lindistflow` |
| `bus_name` | TEXT | Bus name |
| `phase` | TEXT | Phase label |
| `voltage_v` | REAL | Actual voltage magnitude (V) |
| `voltage_min_v` | REAL | Configured minimum voltage limit (V) |
| `voltage_max_v` | REAL | Configured maximum voltage limit (V) |
| `violation_v` | REAL | Positive violation magnitude (V) |
| `violation_kind` | TEXT | `overvoltage` or `undervoltage` |

### `loading_violations`

Persisted branch loading violations generated during export for AC OPF, DC OPF, and LinDistFlow runs.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Foreign key → `runs` |
| `implementation` | TEXT | `ac_opf`, `dc_opf`, or `lindistflow` |
| `branch_name` | TEXT | Branch identifier |
| `phase` | TEXT | Phase label |
| `p_flow_w` | REAL | Active power flow (W) |
| `q_flow_var` | REAL | Reactive power flow (var) |
| `loading_va` | REAL | Apparent loading magnitude (VA) |
| `loading_limit_va` | REAL | Loading limit (VA) |
| `loading_pct` | REAL | Loading percent (`100 * loading_va / loading_limit_va`) |

### `losses`

Per-run system loss summary persisted during export.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT PK | Foreign key → `runs` |
| `implementation` | TEXT | `ac_opf`, `dc_opf`, or `lindistflow` |
| `p_loss_w` | REAL | Total active loss estimate (W) |
| `q_loss_var` | REAL | Total reactive loss estimate (var) |
| `method` | TEXT | Loss computation method/assumption |

## Querying Results

```sql
-- Compare source power across all runs
SELECT r.implementation, r.success,
       CASE r.implementation
           WHEN 'ac_opf' THEN (SELECT SUM(p_injection_w) FROM ac_opf_nodes n
                                WHERE n.run_id = r.run_id AND n.bus_name = 'source_bus')
           WHEN 'dc_opf' THEN (SELECT slack_injection_w FROM dc_opf_summary s
                                WHERE s.run_id = r.run_id)
       END AS source_p_w
FROM runs r;

-- Get all bus voltages from the latest AC run
SELECT bus_name, phase, voltage_mag_v
FROM ac_opf_nodes
WHERE run_id = (SELECT run_id FROM runs
                WHERE implementation = 'ac_opf'
                ORDER BY created_at_utc DESC LIMIT 1)
ORDER BY bus_name, phase;
```

## Export Functions

| Function | Description |
|----------|-------------|
| `export_ac_opf_result_to_sqlite(result, db_path)` | Export a single AC OPF result |
| `export_dc_opf_result_to_sqlite(result, db_path)` | Export a single DC OPF result |
| `export_lindistflow_result_to_sqlite(result, db_path)` | Export a single LinDistFlow result |
| `export_all_results_to_sqlite(db_path, ac_result, dc_result, lindistflow_result)` | Export all results in one call |

All functions accept an optional `run_id` parameter; if omitted, a unique ID is auto-generated.
