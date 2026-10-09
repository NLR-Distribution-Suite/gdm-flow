# Installation

## Requirements

- Python ≥ 3.11
- [grid-data-models](https://github.com/NLR-Distribution-Suite/grid-data-models) >= 2.4.0

## Install from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/NLR-Distribution-Suite/gdm-flow.git
cd gdm-flow
pip install -e .
```

## Optional Extras

SciPy, optimization solvers and sparse matrix support are core dependencies.
Optional dependency groups cover plotting, MCP, independent OpenDSS checks,
and development:

```bash
# For interactive Plotly dashboards
pip install -e ".[plotting]"

# For development and testing
pip install -e ".[dev]"

# For MCP server runtime and MCP tests
pip install -e ".[mcp]"

# For independent OpenDSS comparisons
pip install -e ".[opendss]"

# Install everything
pip install -e ".[plotting,dev,mcp,opendss]"
```

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `numpy` | Array operations, Y-bus matrices | Yes |
| `grid-data-models` | Distribution system data model | Yes |
| `typer` | CLI framework | Yes |
| `rich` | Terminal formatting | Yes |
| `scipy` | AC/DC optimization solvers, AC PF | Yes |
| `plotly` | Interactive HTML dashboards | Optional |
| `mcp` | MCP server runtime and MCP tests | Optional |

## Testing Notes

For a consolidated local/CI testing reference, see `docs/guide/testing.md`.

Run all tests:

```bash
pytest -v --tb=short
```

Run MCP server tests directly:

```bash
pip install -e ".[mcp,dev]"
pytest -v --tb=short tests/test_mcp_server.py
```

In GitHub Actions, MCP tests are enforced by the dedicated `mcp-test` job in `.github/workflows/ci.yml`.

## Verify Installation

After installation, verify the CLI is available:

```bash
gdm-flow --help
```

You should see:

```
Usage: gdm-flow [OPTIONS] COMMAND [ARGS]...

 GDM-Flow — Power flow & optimal power flow for distribution systems

╭─ Commands ──────────────────────────────────────────────╮
│ info                Show system topology and component summary.   │
│ run                 Run one or more OPF solvers.                  │
│ compare             Run all solvers and compare results.          │
│ plot                Generate interactive analysis dashboard.      │
│ export              Run solvers and export results to SQLite.     │
│ report-overvoltage  Print voltage violations from results.        │
│ report-overload     Print branch loading violations.              │
│ db-schema           Print SQLite table/column schema.             │
╰─────────────────────────────────────────────────────────╯
```

Or verify in Python:

```python
import gdm_flow
print(dir(gdm_flow))
```
