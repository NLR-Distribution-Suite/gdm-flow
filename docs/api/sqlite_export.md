# API — SQLite Export

## `export_all_results_to_sqlite`

Export AC, DC, and LinDistFlow results in a single call.

```python
def export_all_results_to_sqlite(
    db_path: str,
    *,
    ac_result: PowerFlowOptimizationResult | None = None,
    dc_result: DCOPFResult | None = None,
    lindistflow_result: LinDistFlowResult | None = None,
) -> dict[str, str]:
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `db_path` | Path to SQLite database file (created if needed) |
| `ac_result` | AC OPF result to export (optional) |
| `dc_result` | DC OPF result to export (optional) |
| `lindistflow_result` | LinDistFlow result to export (optional) |

**Returns:** Dictionary mapping solver name → `run_id`.

## `export_ac_opf_result_to_sqlite`

```python
def export_ac_opf_result_to_sqlite(
    result: PowerFlowOptimizationResult,
    db_path: str,
    *,
    run_id: str | None = None,
) -> str:
```

Export a single AC OPF result. Creates the database schema if needed. Returns the `run_id`.

## `export_dc_opf_result_to_sqlite`

```python
def export_dc_opf_result_to_sqlite(
    result: DCOPFResult,
    db_path: str,
    *,
    run_id: str | None = None,
) -> str:
```

Export a single DC OPF result. Returns the `run_id`.

## `export_lindistflow_result_to_sqlite`

```python
def export_lindistflow_result_to_sqlite(
    result: LinDistFlowResult,
    db_path: str,
    *,
    run_id: str | None = None,
) -> str:
```

Export a single LinDistFlow result. Returns the `run_id`.

## Example

```python
from gdm_flow import (
    optimize_ac_power_flow_from_components,
    solve_dc_opf_from_components,
    solve_lindistflow,
    export_all_results_to_sqlite,
    export_ac_opf_result_to_sqlite,
)
from gdm.systems.distribution import DistributionSystem

system = DistributionSystem.from_json("model.json")

# Run solvers
ac = optimize_ac_power_flow_from_components(system)
dc = solve_dc_opf_from_components(system)
ldf = solve_lindistflow(system)

# Export all at once
ids = export_all_results_to_sqlite(
    "results.db",
    ac_result=ac,
    dc_result=dc,
    lindistflow_result=ldf,
)
print(ids)  # {"ac_opf": "ac_a1b2c3...", "dc_opf": "dc_d4e5f6...", ...}

# Or export individually
run_id = export_ac_opf_result_to_sqlite(ac, "results.db")
```
