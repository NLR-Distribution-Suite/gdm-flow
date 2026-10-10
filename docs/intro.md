# GDM-Flow

**Power Flow & Optimal Power Flow for Distribution Systems**

GDM-Flow provides four complementary solvers for analyzing distribution power systems built on [grid-data-models](https://github.com/NLR-Distribution-Suite/grid-data-models):

| Solver | Method | Strengths |
|--------|--------|-----------|
| **AC OPF** | Nonlinear least-squares on Y-bus | Full voltage & power accuracy including losses |
| **AC PF** | Fixed-point iteration (OpenDSS-style) | Classical power flow with exact bus voltages |
| **DC OPF** | Quadratic programming with linearized constraints | Economic dispatch with generation cost optimization |
| **LinDistFlow** | Backward/forward sweep on radial tree | Fast, lightweight voltage drop analysis |

## Key Features

- **Y-Bus Construction** — Phase-domain admittance matrices from GDM components (branches, transformers, switches) with matrix and sequence impedance support
- **Four Solvers** — AC OPF nonlinear, AC PF fixed-point iteration, DC linearized, and LinDistFlow radial approximation
- **Multi-Phase Support** — Full three-phase modeling with per-phase power injection and voltage tracking
- **Component Integration** — Direct integration with GDM loads, solar PV, batteries, capacitors, and regulators
- **Interactive Dashboards** — Plotly-based HTML dashboards with voltage profiles, power flow, branch loading, losses, and equipment state
- **Modern CLI** — Rich terminal interface with formatted tables, progress indicators, solver comparison, and dashboard generation
- **SQLite Export** — Structured database output for post-processing, violation reporting, and archival
- **SI-Unit Internals** — AC PF solves directly in SI units (OpenDSS-style) avoiding per-unit ill-conditioning on multi-voltage systems; AC OPF uses per-unit for numerical robustness

## Architecture

```
DistributionSystem (GDM JSON)
        │
        ▼
   ┌─────────┐
   │  Y-Bus  │  ← Phase-domain admittance matrix
   └────┬────┘
        │
   ┌────┴───────────────────────┐
   │            │               │
   ▼            ▼               ▼
┌───────┐  ┌────────┐  ┌─────────────┐
│AC OPF │  │DC OPF  │  │ LinDistFlow │
└───┬───┘  └───┬────┘  └──────┬──────┘
    │          │               │
    ▼          ▼               ▼
 Results    Results         Results
    │          │               │
    └──────────┴───────────────┘
               │
         ┌─────┴─────┐
         │  CLI / DB  │
         └────────────┘
```

## Quick Example

```python
from gdm.systems.distribution import DistributionSystem
from gdm_flow import optimize_ac_power_flow_from_components

system = DistributionSystem.from_json("model.json")
result = optimize_ac_power_flow_from_components(system)

print(f"Success: {result.success}")
print(f"Iterations: {result.iterations}")
```

## Navigation

Use the sidebar to explore:

- **Getting Started** — Installation and a hands-on quickstart notebook
- **Solvers** — Theory and implementation details for each solver
- **User Guide** — CLI usage, interactive dashboards, SQLite export, testing workflows, and result comparison
- **API Reference** — Complete function and class documentation
