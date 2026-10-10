# Interactive Dashboards

GDM-Flow can generate rich interactive HTML dashboards using [Plotly](https://plotly.com/python/). Dashboards visualise solver results across multiple dimensions and are useful for presentations, reports, and exploratory analysis.

## Quick Start

```bash
# Generate a dashboard with all four solvers
gdm-flow plot examples/models/p5r.json

# Select specific solvers
gdm-flow plot examples/models/p5r.json -s ac -s pf

# Custom output path
gdm-flow plot examples/models/p5r.json -o my_dashboard.html
```

The dashboard is saved as a self-contained HTML file that can be opened in any browser.

## Installation

Dashboards require the `plotting` extra:

```bash
pip install -e ".[plotting]"
```

This installs `plotly>=5.0`.

## Dashboard Panels

### Voltage-Distance Profile

Shows per-unit voltage magnitude along feeder distance from the source bus. Each phase gets its own subplot. The radial tree topology is preserved — branching points fork naturally so the feeder structure is visible.

Each solver's results are overlaid with distinct colors:
- **AC OPF** — blue
- **AC PF** — purple
- **DC OPF** — green (voltage shown as nominal since DC OPF solves for angles, not magnitudes)
- **LinDistFlow** — orange

### Branch Power Flow

Active and reactive power flow per branch, broken down by phase.

### Branch Loading

Apparent power loading compared against equipment ampacity limits. Branches exceeding their thermal limit are highlighted.

### Line Losses

Per-branch $I^2R$ loss breakdown, useful for identifying the costliest segments.

### Equipment State

Tables summarising the state of:
- **Capacitors** — on/off state, banks on/total, rated and effective kvar
- **Regulators** — voltage setpoint, min/max limits, PT ratio
- **Transformers** — winding voltages, tap positions, service status

## Programmatic Usage

You can also generate dashboards from Python:

```python
from gdm.systems.distribution import DistributionSystem
from gdm_flow import optimize_ac_power_flow_from_components
from gdm_flow.ac_pf import solve_ac_power_flow_from_components
from gdm_flow.dashboard import generate_dashboard

system = DistributionSystem.from_json("model.json")

# Run solvers (results dict maps solver name to result dict)
results = {}

ac_result = optimize_ac_power_flow_from_components(system)
results["AC OPF"] = {
    "solver": "AC OPF",
    "success": ac_result.success,
    "result": ac_result,
    # ... additional fields
}

pf_result = solve_ac_power_flow_from_components(system)
results["AC PF"] = {
    "solver": "AC PF",
    "success": pf_result.success,
    "result": pf_result,
    # ... additional fields
}

generate_dashboard(system, results, "dashboard.html", model_name="my_model")
```

## Comparison HTML (via `gdm-flow compare`)

The `gdm-flow compare` command can also export a simpler comparison HTML file:

```bash
gdm-flow compare examples/models/p5r.json -o comparison.html
```

This generates a multi-panel Plotly figure focused on voltage comparison across solvers, rather than the full dashboard produced by `gdm-flow plot`.
