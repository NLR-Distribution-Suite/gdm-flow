"""MCP server for GDM-Flow solver operations."""

from __future__ import annotations

import json
import logging
import inspect
import os
from pathlib import Path
import sqlite3
from typing import Annotated, Any

import numpy as np
import typer
from gdm.systems.distribution import DistributionSystem
from mcp.server import Server
from mcp import stdio_server
from mcp_types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

from gdm_flow import (
    build_lindistflow_net_injections_from_components,
    calculate_ybus,
    export_all_results_to_sqlite,
    optimize_ac_power_flow_from_components,
    solve_ac_power_flow_from_components,
    solve_dc_opf_from_components,
    solve_lindistflow,
)
from gdm_flow.sqlite_export import (
    export_ac_opf_result_to_sqlite,
    export_ac_pf_result_to_sqlite,
    export_dc_opf_result_to_sqlite,
    export_lindistflow_result_to_sqlite,
)
import gdm_flow as gdm_flow_api
from gdm_flow.mcp import __version__

try:  # Optional dist-stack provenance integration
    from dist_stack.runstore import attach_artifact, create_run
    from dist_stack.manifest import write_manifest

    _RUNSTORE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _RUNSTORE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gdm_flow_mcp")

app = Server("gdm-flow-mcp")

DOCS_ROOT = Path(__file__).resolve().parents[3] / "docs"
_DOC_SUFFIXES = {".md", ".ipynb"}


def _iter_doc_files() -> list[Path]:
    if not DOCS_ROOT.exists():
        return []
    files: list[Path] = []
    for path in DOCS_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _DOC_SUFFIXES:
            continue
        if "_build" in path.parts:
            continue
        files.append(path)
    return sorted(files)


def _extract_snippet(text: str, query: str, radius: int = 140) -> str:
    haystack = text.lower()
    needle = query.lower()
    idx = haystack.find(needle)
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(text), idx + len(query) + radius)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(text):
        snippet = snippet + " ..."
    return snippet


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _list_public_api_symbols() -> list[str]:
    symbols = getattr(gdm_flow_api, "__all__", [])
    return sorted(str(name) for name in symbols)


def _api_reference_for_symbol(symbol_name: str) -> dict[str, Any]:
    if not hasattr(gdm_flow_api, symbol_name):
        raise ValueError(f"Unknown public API symbol: {symbol_name}")
    symbol = getattr(gdm_flow_api, symbol_name)
    signature = None
    if callable(symbol):
        try:
            signature = str(inspect.signature(symbol))
        except (TypeError, ValueError):
            signature = None
    doc = inspect.getdoc(symbol) or ""
    return {
        "symbol": symbol_name,
        "module": getattr(symbol, "__module__", ""),
        "signature": signature,
        "doc": doc,
    }


def _load_system(system_path: str) -> DistributionSystem:
    path = Path(system_path)
    if not path.exists():
        raise FileNotFoundError(f"System JSON file not found: {system_path}")
    return DistributionSystem.from_json(str(path))


def _resolve_model_ref_to_path(model_ref: dict[str, Any]) -> str:
    """Resolve model_ref payload into a concrete system JSON path."""
    for key in ("stored_path", "path", "source_path"):
        value = model_ref.get(key)
        if isinstance(value, str) and value.strip():
            return value

    model_id = model_ref.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_ref must include a path or model_id")

    version = model_ref.get("version")
    db_path = model_ref.get("registry_db") or os.getenv("DIST_STACK_MODEL_REGISTRY_DB")
    if not db_path:
        raise ValueError(
            "model_ref requires DIST_STACK_MODEL_REGISTRY_DB (or model_ref.registry_db) "
            "when path fields are not provided"
        )

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        if version is None:
            row = conn.execute(
                """
                SELECT stored_path FROM models
                WHERE model_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (model_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT stored_path FROM models
                WHERE model_id = ? AND version = ?
                LIMIT 1
                """,
                (model_id, int(version)),
            ).fetchone()

    if row is None:
        suffix = "latest" if version is None else f"version={version}"
        raise ValueError(f"model_ref not found for model_id={model_id}, {suffix}")

    return str(row["stored_path"])


def _get_system_path_arg(args: dict[str, Any]) -> str:
    """Extract system path from either legacy system_path or model_ref."""
    system_path = args.get("system_path")
    if isinstance(system_path, str) and system_path.strip():
        return system_path

    model_ref = args.get("model_ref")
    if isinstance(model_ref, dict):
        return _resolve_model_ref_to_path(model_ref)

    raise ValueError("Expected either 'system_path' or 'model_ref'")


def _serialize_complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _serialize_ybus_result(
    result: Any,
    *,
    include_matrix: bool,
    matrix_preview_limit: int,
) -> dict[str, Any]:
    ybus = result.ybus
    is_sparse = hasattr(ybus, "toarray")
    ybus_dense = ybus.toarray() if is_sparse else ybus
    n_nodes = len(result.index_to_label)

    payload: dict[str, Any] = {
        "n_nodes": n_nodes,
        "n_nonzero": int(np.count_nonzero(ybus_dense)),
        "is_sparse": bool(is_sparse),
        "index_to_label": [
            {"bus": str(bus), "phase": str(phase)}
            for bus, phase in result.index_to_label
        ],
    }

    if include_matrix:
        preview_n = max(1, min(int(matrix_preview_limit), n_nodes))
        payload["matrix_preview"] = {
            "rows": preview_n,
            "cols": preview_n,
            "values": [
                [_serialize_complex(v) for v in row[:preview_n]]
                for row in ybus_dense[:preview_n]
            ],
        }

    return payload


def _source_bus_totals(
    labels: list[tuple[str, str]], s_injection: np.ndarray
) -> dict[str, float]:
    if not labels:
        return {"source_bus": "", "p_w": 0.0, "q_var": 0.0}
    source_bus = labels[0][0]
    p = 0.0
    q = 0.0
    for idx, label in enumerate(labels):
        if label[0] == source_bus:
            p += float(s_injection[idx].real)
            q += float(s_injection[idx].imag)
    return {"source_bus": source_bus, "p_w": p, "q_var": q}


def _serialize_ac_result(result: Any, include_details: bool) -> dict[str, Any]:
    voltage_mag = np.abs(result.voltage)
    source_totals = _source_bus_totals(
        result.ybus_result.index_to_label, result.power_injection
    )
    payload: dict[str, Any] = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.iterations),
        "initial_objective": float(result.initial_objective),
        "final_objective": float(result.final_objective),
        "voltage_min_v": float(np.min(voltage_mag)) if voltage_mag.size else 0.0,
        "voltage_max_v": float(np.max(voltage_mag)) if voltage_mag.size else 0.0,
        "source_injection": source_totals,
    }
    if include_details:
        payload["nodes"] = [
            {
                "bus": bus,
                "phase": phase,
                "voltage": _serialize_complex(result.voltage[idx]),
                "power_injection": _serialize_complex(result.power_injection[idx]),
            }
            for idx, (bus, phase) in enumerate(result.ybus_result.index_to_label)
        ]
    return payload


def _serialize_dc_result(result: Any, include_details: bool) -> dict[str, Any]:
    dispatch_total = float(sum(result.generator_dispatch_w.values()))
    payload: dict[str, Any] = {
        "success": bool(result.success),
        "message": str(result.message),
        "objective": float(result.objective),
        "iterations": int(result.iterations),
        "slack_injection_w": float(result.slack_injection_w),
        "total_dispatch_w": dispatch_total,
        "generator_count": int(len(result.generator_dispatch_w)),
    }
    if include_details:
        payload["generator_dispatch_w"] = {
            name: float(value) for name, value in result.generator_dispatch_w.items()
        }
        payload["theta_rad"] = [
            {"bus": bus, "phase": phase, "theta_rad": float(theta)}
            for (bus, phase), theta in sorted(result.theta_rad.items())
        ]
        payload["nodal_balance_w"] = [
            {"bus": bus, "phase": phase, "balance_w": float(balance)}
            for (bus, phase), balance in sorted(result.nodal_balance_w.items())
        ]
    return payload


def _serialize_lindistflow_result(result: Any, include_details: bool) -> dict[str, Any]:
    voltage_values = list(result.voltage_v.values())
    payload: dict[str, Any] = {
        "success": bool(result.success),
        "message": str(result.message),
        "source_bus": str(result.source_bus),
        "voltage_min_v": float(min(voltage_values)) if voltage_values else 0.0,
        "voltage_max_v": float(max(voltage_values)) if voltage_values else 0.0,
        "modeled_nodes": int(len(result.voltage_v)),
        "modeled_branches": int(len(result.p_flow_w)),
    }
    if include_details:
        payload["voltage_v"] = [
            {"bus": bus, "phase": phase, "voltage_v": float(v)}
            for (bus, phase), v in sorted(result.voltage_v.items())
        ]
        payload["p_flow_w"] = [
            {"branch": branch, "phase": phase, "p_flow_w": float(v)}
            for (branch, phase), v in sorted(result.p_flow_w.items())
        ]
        payload["q_flow_var"] = [
            {"branch": branch, "phase": phase, "q_flow_var": float(v)}
            for (branch, phase), v in sorted(result.q_flow_var.items())
        ]
    return payload


def _serialize_ac_pf_result(result: Any, *, include_details: bool) -> dict[str, Any]:
    voltage_mag = np.abs(result.voltage)
    source_totals = _source_bus_totals(
        result.ybus_result.index_to_label, result.power_injection
    )
    payload: dict[str, Any] = {
        "success": True,
        "message": str(result.message),
        "converged": bool(result.success),
        "iterations": int(result.iterations),
        "max_mismatch_pu": float(result.max_mismatch_pu),
        "voltage_min_v": float(np.min(voltage_mag)) if voltage_mag.size else 0.0,
        "voltage_max_v": float(np.max(voltage_mag)) if voltage_mag.size else 0.0,
        "source_injection": source_totals,
        "slack_bus": str(source_totals.get("source_bus", "")),
    }
    if include_details:
        payload["nodes"] = [
            {
                "bus": bus,
                "phase": phase,
                "voltage_mag_v": float(np.abs(result.voltage[idx])),
                "voltage_pu": float(result.voltage_pu[idx]),
                "voltage_angle_rad": float(np.angle(result.voltage[idx])),
                "p_injection_w": float(result.power_injection[idx].real),
                "q_injection_var": float(result.power_injection[idx].imag),
            }
            for idx, (bus, phase) in enumerate(result.ybus_result.index_to_label)
        ]
    return payload


def _record_gdm_flow_run(
    *,
    tool: str,
    model_id: str | None,
    tool_version: str | None,
    export_path: str | None,
) -> str | None:
    if not _RUNSTORE_AVAILABLE:
        return None

    runstore_db = os.getenv("DIST_STACK_RUNSTORE_DB")
    if not runstore_db:
        return None

    try:
        rec = create_run(
            tool,
            run_type="gdm_flow_run",
            status="succeeded",
            model_id=model_id,
            tool_version=tool_version,
            runstore_db=runstore_db,
        )
        run_id = rec.run_id

        if isinstance(export_path, str) and export_path.strip():
            export_abs = os.path.abspath(export_path)
            if os.path.exists(export_abs):
                write_manifest(
                    export_abs,
                    artifact_type="artifact",
                    tool=tool,
                    tool_version=tool_version,
                    derived_from=[run_id],
                    config={"run_id": run_id},
                )
                attach_artifact(run_id, export_abs, runstore_db=runstore_db)

        return run_id
    except Exception:
        return None


async def _handle_list_tools(ctx, params: ListToolsRequest) -> ListToolsResult:
    """List available GDM-Flow MCP tools."""
    return ListToolsResult(
        tools=[
            # Solver and matrix tools
            Tool(
                name="opf_calculate_ybus",
                description="Build phase-domain Y-bus matrix metadata for a DistributionSystem JSON model.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_path": {
                            "type": "string",
                            "description": "Path to system JSON file",
                        },
                        "model_ref": {
                            "type": "object",
                            "description": "Optional model reference ({model_id/version} or direct stored_path/path)",
                        },
                        "include_neutral": {"type": "boolean", "default": False},
                        "include_shunt": {"type": "boolean", "default": False},
                        "include_transformers": {"type": "boolean", "default": True},
                        "include_open_switches": {"type": "boolean", "default": False},
                        "convert_geometry_to_matrix": {
                            "type": "boolean",
                            "default": True,
                        },
                        "sparse": {"type": "boolean", "default": True},
                        "include_matrix": {
                            "type": "boolean",
                            "description": "Include a top-left matrix preview in the result",
                            "default": False,
                        },
                        "matrix_preview_limit": {
                            "type": "integer",
                            "description": "Preview matrix side length when include_matrix=true",
                            "default": 10,
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="opf_run_ac",
                description="Run AC OPF directly from system components.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_path": {
                            "type": "string",
                            "description": "Path to system JSON file",
                        },
                        "model_ref": {
                            "type": "object",
                            "description": "Optional model reference ({model_id/version} or direct stored_path/path)",
                        },
                        "include_loads": {"type": "boolean", "default": True},
                        "include_solar": {"type": "boolean", "default": True},
                        "include_battery": {"type": "boolean", "default": False},
                        "include_capacitor": {"type": "boolean", "default": True},
                        "include_regulator_targets": {
                            "type": "boolean",
                            "default": True,
                        },
                        "include_regulator_limits": {
                            "type": "boolean",
                            "default": True,
                        },
                        "include_neutral": {"type": "boolean", "default": False},
                        "include_shunt": {"type": "boolean", "default": False},
                        "convert_geometry_to_matrix": {
                            "type": "boolean",
                            "default": True,
                        },
                        "vm_min_pu": {"type": "number", "default": 0.95},
                        "vm_max_pu": {"type": "number", "default": 1.05},
                        "max_nfev": {"type": "integer", "default": 300},
                        "include_details": {
                            "type": "boolean",
                            "description": "Include per-node solved values",
                            "default": False,
                        },
                        "db_path": {
                            "type": "string",
                            "description": "Optional SQLite DB path to export this AC OPF result",
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="opf_run_dc",
                description="Run DC OPF directly from system components.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_path": {
                            "type": "string",
                            "description": "Path to system JSON file",
                        },
                        "model_ref": {
                            "type": "object",
                            "description": "Optional model reference ({model_id/version} or direct stored_path/path)",
                        },
                        "include_solar_generators": {
                            "type": "boolean",
                            "default": True,
                        },
                        "include_battery_generators": {
                            "type": "boolean",
                            "default": True,
                        },
                        "include_loads": {"type": "boolean", "default": True},
                        "include_slack_generator": {"type": "boolean", "default": True},
                        "slack_cost_linear": {"type": "number", "default": 50.0},
                        "include_neutral": {"type": "boolean", "default": False},
                        "include_shunt": {"type": "boolean", "default": False},
                        "convert_geometry_to_matrix": {
                            "type": "boolean",
                            "default": True,
                        },
                        "theta_min_rad": {
                            "type": "number",
                            "default": -1.5707963267948966,
                        },
                        "theta_max_rad": {
                            "type": "number",
                            "default": 1.5707963267948966,
                        },
                        "theta_penalty": {"type": "number", "default": 1e-6},
                        "maxiter": {"type": "integer", "default": 500},
                        "include_details": {
                            "type": "boolean",
                            "description": "Include generator dispatch and nodal details",
                            "default": False,
                        },
                        "db_path": {
                            "type": "string",
                            "description": "Optional SQLite DB path to export this DC OPF result",
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="opf_run_powerflow",
                description="Run Newton-Raphson AC power flow (fixed P/Q injections), distinct from AC OPF optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_path": {
                            "type": "string",
                            "description": "Path to system JSON file",
                        },
                        "model_ref": {
                            "type": "object",
                            "description": "Optional model reference ({model_id/version} or direct stored_path/path)",
                        },
                        "include_loads": {"type": "boolean", "default": True},
                        "include_solar": {"type": "boolean", "default": True},
                        "include_battery": {"type": "boolean", "default": False},
                        "include_capacitor": {"type": "boolean", "default": True},
                        "load_scale": {"type": "number", "default": 1.0},
                        "solar_scale": {"type": "number", "default": 1.0},
                        "battery_scale": {"type": "number", "default": 1.0},
                        "capacitor_scale": {"type": "number", "default": 1.0},
                        "slack_label": {"type": "string"},
                        "include_neutral": {"type": "boolean", "default": False},
                        "include_shunt": {"type": "boolean", "default": False},
                        "convert_geometry_to_matrix": {
                            "type": "boolean",
                            "default": True,
                        },
                        "max_iterations": {"type": "integer", "default": 100},
                        "tolerance": {"type": "number", "default": 1e-6},
                        "include_details": {
                            "type": "boolean",
                            "description": "Include per-node solved values",
                            "default": False,
                        },
                        "db_path": {
                            "type": "string",
                            "description": "If set, also export this power-flow result to a SQLite DB",
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="opf_run_lindistflow",
                description="Run LinDistFlow from component-derived net injections.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_path": {
                            "type": "string",
                            "description": "Path to system JSON file",
                        },
                        "model_ref": {
                            "type": "object",
                            "description": "Optional model reference ({model_id/version} or direct stored_path/path)",
                        },
                        "include_loads": {"type": "boolean", "default": True},
                        "include_solar": {"type": "boolean", "default": True},
                        "include_battery": {"type": "boolean", "default": True},
                        "include_capacitor": {"type": "boolean", "default": True},
                        "include_neutral": {"type": "boolean", "default": False},
                        "include_open_switches": {"type": "boolean", "default": False},
                        "include_details": {
                            "type": "boolean",
                            "description": "Include per-node and per-branch outputs",
                            "default": False,
                        },
                        "db_path": {
                            "type": "string",
                            "description": "Optional SQLite DB path to export this LinDistFlow result",
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="opf_compare_solvers",
                description="Run AC OPF, DC OPF, and LinDistFlow and return a side-by-side summary.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_path": {
                            "type": "string",
                            "description": "Path to system JSON file",
                        },
                        "model_ref": {
                            "type": "object",
                            "description": "Optional model reference ({model_id/version} or direct stored_path/path)",
                        },
                        "include_details": {
                            "type": "boolean",
                            "description": "Include full per-solver details in addition to summary",
                            "default": False,
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="opf_export_sqlite",
                description="Run selected OPF solvers and export results to a SQLite database.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_path": {
                            "type": "string",
                            "description": "Path to system JSON file",
                        },
                        "model_ref": {
                            "type": "object",
                            "description": "Optional model reference ({model_id/version} or direct stored_path/path)",
                        },
                        "db_path": {
                            "type": "string",
                            "description": "Output SQLite database path",
                        },
                        "run_ac": {"type": "boolean", "default": True},
                        "run_pf": {"type": "boolean", "default": False},
                        "run_dc": {"type": "boolean", "default": True},
                        "run_lindistflow": {"type": "boolean", "default": True},
                    },
                    "required": ["db_path"],
                },
            ),
            # Documentation and knowledge tools
            Tool(
                name="list_opf_documentation",
                description="List available GDM-Flow documentation files (docs/*.md, docs/*.ipynb).",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="search_opf_documentation",
                description="Search GDM-Flow documentation and return relevant snippets.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term or phrase",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matches to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_opf_documentation_page",
                description="Read a specific documentation page by relative path from docs/.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_path": {
                            "type": "string",
                            "description": "Path relative to docs/ (e.g., solvers/ac_opf.md)",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based start line",
                            "default": 1,
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum number of lines to return",
                            "default": 160,
                        },
                    },
                    "required": ["doc_path"],
                },
            ),
            Tool(
                name="list_opf_api_symbols",
                description="List public API symbols exposed by gdm_flow.__all__.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="get_opf_api_reference",
                description="Get module, signature, and docstring for a public GDM-Flow API symbol.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol_name": {
                            "type": "string",
                            "description": "Public symbol name (e.g., solve_dc_opf_from_components)",
                        }
                    },
                    "required": ["symbol_name"],
                },
            ),
            Tool(
                name="opf_fix_violations",
                description="Detect and fix voltage/loading violations in a GDM distribution system model iteratively.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_path": {
                            "type": "string",
                            "description": "Path to system JSON file",
                        },
                        "model_ref": {
                            "type": "object",
                            "description": "Optional model reference ({model_id/version} or direct stored_path/path)",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Output path for the fixed model JSON. If empty, appends '_fixed' to input filename.",
                        },
                        "max_iterations": {"type": "integer", "default": 10},
                        "solver": {
                            "type": "string",
                            "enum": ["ldf", "ac"],
                            "default": "ldf",
                            "description": "Solver for violation detection (ldf=LinDistFlow, ac=AC OPF)",
                        },
                        "vm_min_pu": {"type": "number", "default": 0.95},
                        "vm_max_pu": {"type": "number", "default": 1.05},
                    },
                    "required": [],
                },
            ),
            Tool(
                name="scale_loads",
                description=(
                    "Scale every load's real/reactive power by a factor and write the "
                    "updated system JSON (use before a power-flow run to model higher demand)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system_path": {
                            "type": "string",
                            "description": "Path to system JSON file",
                        },
                        "load_scale": {
                            "type": "number",
                            "description": "Multiplier applied to every load's power",
                            "default": 1.0,
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Where to write the scaled system JSON (defaults to system_path)",
                        },
                    },
                    "required": ["load_scale"],
                },
            ),
        ]
    )


_TOOL_HANDLERS: dict[str, Any] = {
    "scale_loads": lambda args: _handle_scale_loads(args),
    "opf_calculate_ybus": lambda args: _handle_calculate_ybus(args),
    "opf_run_ac": lambda args: _handle_run_ac(args),
    "opf_run_dc": lambda args: _handle_run_dc(args),
    "opf_run_powerflow": lambda args: _handle_run_powerflow(args),
    "opf_run_lindistflow": lambda args: _handle_run_lindistflow(args),
    "opf_compare_solvers": lambda args: _handle_compare_solvers(args),
    "opf_export_sqlite": lambda args: _handle_export_sqlite(args),
    "list_opf_documentation": lambda args: _handle_list_opf_documentation(args),
    "search_opf_documentation": lambda args: _handle_search_opf_documentation(args),
    "get_opf_documentation_page": lambda args: _handle_get_opf_documentation_page(args),
    "list_opf_api_symbols": lambda args: _handle_list_opf_api_symbols(args),
    "get_opf_api_reference": lambda args: _handle_get_opf_api_reference(args),
    "opf_fix_violations": lambda args: _handle_fix_violations(args),
}


async def _handle_call_tool(ctx, params: CallToolRequestParams) -> CallToolResult:
    """Handle MCP tool calls."""
    name = params.name
    arguments = params.arguments
    try:
        logger.info("Tool called: %s", name)
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            result = {"error": f"Unknown tool: {name}"}
        else:
            result = await handler(arguments or {})
        return CallToolResult(
            content=[
                TextContent(type="text", text=json.dumps(result, indent=2, default=str))
            ]
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool %s failed", name)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}"}, indent=2
                    ),
                )
            ]
        )


# Register handlers with the v2 API
app.add_request_handler("tools/list", ListToolsRequest, _handle_list_tools)
app.add_request_handler("tools/call", CallToolRequestParams, _handle_call_tool)


async def _handle_calculate_ybus(args: dict[str, Any]) -> dict[str, Any]:
    system = _load_system(_get_system_path_arg(args))
    result = calculate_ybus(
        system,
        include_neutral=bool(args.get("include_neutral", False)),
        include_shunt=bool(args.get("include_shunt", False)),
        include_transformers=bool(args.get("include_transformers", True)),
        include_open_switches=bool(args.get("include_open_switches", False)),
        convert_geometry_to_matrix=bool(args.get("convert_geometry_to_matrix", True)),
        sparse=bool(args.get("sparse", True)),
    )
    return _serialize_ybus_result(
        result,
        include_matrix=bool(args.get("include_matrix", False)),
        matrix_preview_limit=int(args.get("matrix_preview_limit", 10)),
    )


async def _handle_run_ac(args: dict[str, Any]) -> dict[str, Any]:
    system = _load_system(_get_system_path_arg(args))
    result = optimize_ac_power_flow_from_components(
        system,
        include_loads=bool(args.get("include_loads", True)),
        include_solar=bool(args.get("include_solar", True)),
        include_battery=bool(args.get("include_battery", False)),
        include_capacitor=bool(args.get("include_capacitor", True)),
        include_regulator_targets=bool(args.get("include_regulator_targets", True)),
        include_regulator_limits=bool(args.get("include_regulator_limits", True)),
        include_neutral=bool(args.get("include_neutral", False)),
        include_shunt=bool(args.get("include_shunt", False)),
        convert_geometry_to_matrix=bool(args.get("convert_geometry_to_matrix", True)),
        vm_min_pu=float(args.get("vm_min_pu", 0.95)),
        vm_max_pu=float(args.get("vm_max_pu", 1.05)),
        max_nfev=int(args.get("max_nfev", 300)),
    )
    response = _serialize_ac_result(
        result, include_details=bool(args.get("include_details", False))
    )
    db_path = args.get("db_path")
    if isinstance(db_path, str) and db_path.strip():
        run_id = export_ac_opf_result_to_sqlite(result, str(Path(db_path)))
        response["db_path"] = str(Path(db_path))
        response["run_id"] = run_id
    model_ref = args.get("model_ref")
    model_id = model_ref.get("model_id") if isinstance(model_ref, dict) else None
    response["runstore_run_id"] = _record_gdm_flow_run(
        tool="opf_run_ac",
        model_id=model_id,
        tool_version=__version__,
        export_path=args.get("db_path"),
    )
    return response


async def _handle_run_dc(args: dict[str, Any]) -> dict[str, Any]:
    system = _load_system(_get_system_path_arg(args))
    result = solve_dc_opf_from_components(
        system,
        include_solar_generators=bool(args.get("include_solar_generators", True)),
        include_battery_generators=bool(args.get("include_battery_generators", True)),
        include_loads=bool(args.get("include_loads", True)),
        include_slack_generator=bool(args.get("include_slack_generator", True)),
        slack_cost_linear=float(args.get("slack_cost_linear", 50.0)),
        include_neutral=bool(args.get("include_neutral", False)),
        include_shunt=bool(args.get("include_shunt", False)),
        convert_geometry_to_matrix=bool(args.get("convert_geometry_to_matrix", True)),
        theta_min_rad=float(args.get("theta_min_rad", -np.pi / 2)),
        theta_max_rad=float(args.get("theta_max_rad", np.pi / 2)),
        theta_penalty=float(args.get("theta_penalty", 1e-6)),
        maxiter=int(args.get("maxiter", 500)),
    )
    response = _serialize_dc_result(
        result, include_details=bool(args.get("include_details", False))
    )
    db_path = args.get("db_path")
    if isinstance(db_path, str) and db_path.strip():
        run_id = export_dc_opf_result_to_sqlite(result, str(Path(db_path)))
        response["db_path"] = str(Path(db_path))
        response["run_id"] = run_id
    model_ref = args.get("model_ref")
    model_id = model_ref.get("model_id") if isinstance(model_ref, dict) else None
    response["runstore_run_id"] = _record_gdm_flow_run(
        tool="opf_run_dc",
        model_id=model_id,
        tool_version=__version__,
        export_path=args.get("db_path"),
    )
    return response


async def _handle_run_powerflow(args: dict[str, Any]) -> dict[str, Any]:
    try:
        system = _load_system(_get_system_path_arg(args))
        result = solve_ac_power_flow_from_components(
            system,
            include_loads=bool(args.get("include_loads", True)),
            include_solar=bool(args.get("include_solar", True)),
            include_battery=bool(args.get("include_battery", False)),
            include_capacitor=bool(args.get("include_capacitor", True)),
            load_scale=float(args.get("load_scale", 1.0)),
            solar_scale=float(args.get("solar_scale", 1.0)),
            battery_scale=float(args.get("battery_scale", 1.0)),
            capacitor_scale=float(args.get("capacitor_scale", 1.0)),
            slack_label=args.get("slack_label"),
            include_neutral=bool(args.get("include_neutral", False)),
            include_shunt=bool(args.get("include_shunt", False)),
            convert_geometry_to_matrix=bool(
                args.get("convert_geometry_to_matrix", True)
            ),
            max_iterations=int(args.get("max_iterations", 100)),
            tolerance=float(args.get("tolerance", 1e-6)),
        )
        response = _serialize_ac_pf_result(
            result,
            include_details=bool(args.get("include_details", False)),
        )
        db_path = args.get("db_path")
        if isinstance(db_path, str) and db_path.strip():
            run_id = export_ac_pf_result_to_sqlite(result, str(Path(db_path)))
            response["db_path"] = str(Path(db_path))
            response["run_id"] = run_id
        model_ref = args.get("model_ref")
        model_id = model_ref.get("model_id") if isinstance(model_ref, dict) else None
        response["runstore_run_id"] = _record_gdm_flow_run(
            tool="opf_run_powerflow",
            model_id=model_id,
            tool_version=__version__,
            export_path=args.get("db_path"),
        )
        return response
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def _handle_run_lindistflow(args: dict[str, Any]) -> dict[str, Any]:
    system = _load_system(_get_system_path_arg(args))
    p_net_w, q_net_var = build_lindistflow_net_injections_from_components(
        system,
        include_loads=bool(args.get("include_loads", True)),
        include_solar=bool(args.get("include_solar", True)),
        include_battery=bool(args.get("include_battery", True)),
        include_capacitor=bool(args.get("include_capacitor", True)),
    )
    result = solve_lindistflow(
        system,
        p_net_w=p_net_w,
        q_net_var=q_net_var,
        include_neutral=bool(args.get("include_neutral", False)),
        include_open_switches=bool(args.get("include_open_switches", False)),
    )
    response = _serialize_lindistflow_result(
        result,
        include_details=bool(args.get("include_details", False)),
    )
    db_path = args.get("db_path")
    if isinstance(db_path, str) and db_path.strip():
        run_id = export_lindistflow_result_to_sqlite(result, str(Path(db_path)))
        response["db_path"] = str(Path(db_path))
        response["run_id"] = run_id
    model_ref = args.get("model_ref")
    model_id = model_ref.get("model_id") if isinstance(model_ref, dict) else None
    response["runstore_run_id"] = _record_gdm_flow_run(
        tool="opf_run_lindistflow",
        model_id=model_id,
        tool_version=__version__,
        export_path=args.get("db_path"),
    )
    return response


async def _handle_compare_solvers(args: dict[str, Any]) -> dict[str, Any]:
    system = _load_system(_get_system_path_arg(args))
    include_details = bool(args.get("include_details", False))

    ac = optimize_ac_power_flow_from_components(system)
    dc = solve_dc_opf_from_components(system)
    ldf = solve_lindistflow(system)

    ac_summary = _serialize_ac_result(ac, include_details=include_details)
    dc_summary = _serialize_dc_result(dc, include_details=include_details)
    ldf_summary = _serialize_lindistflow_result(ldf, include_details=include_details)

    return {
        "ac": ac_summary,
        "dc": dc_summary,
        "lindistflow": ldf_summary,
        "summary": {
            "ac_success": ac_summary["success"],
            "dc_success": dc_summary["success"],
            "lindistflow_success": ldf_summary["success"],
            "ac_source_p_w": ac_summary["source_injection"]["p_w"],
            "dc_slack_injection_w": dc_summary["slack_injection_w"],
            "ldf_source_bus": ldf_summary["source_bus"],
        },
    }


async def _handle_export_sqlite(args: dict[str, Any]) -> dict[str, Any]:
    system = _load_system(_get_system_path_arg(args))
    db_path = str(Path(args["db_path"]))

    run_ac = bool(args.get("run_ac", True))
    run_pf = bool(args.get("run_pf", False))
    run_dc = bool(args.get("run_dc", True))
    run_lindistflow = bool(args.get("run_lindistflow", True))

    if not any([run_ac, run_pf, run_dc, run_lindistflow]):
        raise ValueError(
            "At least one of run_ac, run_pf, run_dc, run_lindistflow must be true"
        )

    ac_result = optimize_ac_power_flow_from_components(system) if run_ac else None
    pf_result = solve_ac_power_flow_from_components(system) if run_pf else None
    dc_result = solve_dc_opf_from_components(system) if run_dc else None
    ldf_result = solve_lindistflow(system) if run_lindistflow else None

    run_ids = export_all_results_to_sqlite(
        db_path,
        ac_result=ac_result,
        pf_result=pf_result,
        dc_result=dc_result,
        lindistflow_result=ldf_result,
    )
    return {
        "db_path": db_path,
        "run_ids": run_ids,
        "exported": {
            "ac": run_ac,
            "pf": run_pf,
            "dc": run_dc,
            "lindistflow": run_lindistflow,
        },
    }


async def _handle_list_opf_documentation(args: dict[str, Any]) -> dict[str, Any]:
    del args
    files = _iter_doc_files()
    rel_paths = [str(path.relative_to(DOCS_ROOT)) for path in files]
    return {
        "docs_root": str(DOCS_ROOT),
        "count": len(rel_paths),
        "files": rel_paths,
    }


async def _handle_search_opf_documentation(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args["query"]).strip()
    max_results = max(1, int(args.get("max_results", 5)))

    matches: list[dict[str, Any]] = []
    for doc_path in _iter_doc_files():
        text = _read_text_file(doc_path)
        snippet = _extract_snippet(text, query)
        if not snippet:
            continue
        matches.append(
            {
                "path": str(doc_path.relative_to(DOCS_ROOT)),
                "snippet": snippet,
            }
        )
        if len(matches) >= max_results:
            break

    return {
        "query": query,
        "count": len(matches),
        "results": matches,
    }


async def _handle_get_opf_documentation_page(args: dict[str, Any]) -> dict[str, Any]:
    doc_path = str(args["doc_path"]).strip()
    start_line = max(1, int(args.get("start_line", 1)))
    max_lines = max(1, int(args.get("max_lines", 160)))

    full_path = (DOCS_ROOT / doc_path).resolve()
    docs_root_resolved = DOCS_ROOT.resolve()
    if not str(full_path).startswith(str(docs_root_resolved)):
        raise ValueError("doc_path must stay within docs/ directory")
    if not full_path.exists() or not full_path.is_file():
        raise FileNotFoundError(f"Documentation file not found: {doc_path}")

    lines = _read_text_file(full_path).splitlines()
    start_idx = start_line - 1
    end_idx = min(len(lines), start_idx + max_lines)

    return {
        "path": str(full_path.relative_to(docs_root_resolved)),
        "start_line": start_line,
        "end_line": end_idx,
        "content": "\n".join(lines[start_idx:end_idx]),
    }


async def _handle_list_opf_api_symbols(args: dict[str, Any]) -> dict[str, Any]:
    del args
    symbols = _list_public_api_symbols()
    return {
        "count": len(symbols),
        "symbols": symbols,
    }


async def _handle_get_opf_api_reference(args: dict[str, Any]) -> dict[str, Any]:
    symbol_name = str(args["symbol_name"]).strip()
    return _api_reference_for_symbol(symbol_name)


async def _handle_scale_loads(args: dict[str, Any]) -> dict[str, Any]:
    """Scale every load's power by ``load_scale`` and write the updated system."""
    from pathlib import Path

    from gdm.systems.distribution.components import DistributionLoad

    system_path = _get_system_path_arg(args)
    load_scale = float(args.get("load_scale", 1.0))
    output_path = args.get("output_path") or system_path

    system = _load_system(system_path)

    def _total_kw(sys: Any) -> float:
        total = 0.0
        for load in sys.get_components(DistributionLoad):
            for phase_load in load.equipment.phase_loads:
                total += phase_load.real_power.to("kilowatt").magnitude
        return total

    before_kw = _total_kw(system)

    scaled_equipment: set[Any] = set()
    n_loads = 0
    for load in system.get_components(DistributionLoad):
        n_loads += 1
        equipment = load.equipment
        # Equipment can be shared across loads; scale each unique one once.
        if equipment.uuid in scaled_equipment:
            continue
        scaled_equipment.add(equipment.uuid)
        for phase_load in equipment.phase_loads:
            phase_load.real_power = phase_load.real_power * load_scale
            phase_load.reactive_power = phase_load.reactive_power * load_scale

    after_kw = _total_kw(system)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    system.to_json(out)

    return {
        "success": True,
        "load_scale": load_scale,
        "num_loads": n_loads,
        "total_load_kw_before": round(before_kw, 3),
        "total_load_kw_after": round(after_kw, 3),
        "output_path": str(out),
    }


async def _handle_fix_violations(args: dict[str, Any]) -> dict[str, Any]:
    system_path = _get_system_path_arg(args)
    system = _load_system(system_path)

    from ..fix import fix_violations

    solver = args.get("solver", "ldf")
    max_iterations = int(args.get("max_iterations", 10))
    vm_min_pu = float(args.get("vm_min_pu", 0.95))
    vm_max_pu = float(args.get("vm_max_pu", 1.05))

    result = fix_violations(
        system,
        max_iterations=max_iterations,
        solver=solver,
        vm_min_pu=vm_min_pu,
        vm_max_pu=vm_max_pu,
    )

    # Export fixed model if any actions were taken
    output_path = args.get("output_path", "")
    exported_path = None
    if result.total_actions > 0:
        from pathlib import Path

        if not output_path:
            p = Path(system_path)
            output_path = str(p.with_stem(p.stem + "_fixed"))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        system.to_json(out)
        exported_path = str(out)

    return {
        "success": result.success,
        "message": result.message,
        "initial_voltage_violations": result.initial_voltage_violations,
        "initial_loading_violations": result.initial_loading_violations,
        "final_voltage_violations": result.final_voltage_violations,
        "final_loading_violations": result.final_loading_violations,
        "total_actions": result.total_actions,
        "iterations": len(result.iterations),
        "output_path": exported_path,
    }


def _run_server(
    log_level: Annotated[str, typer.Option(help="Logging level")] = "INFO",
) -> None:
    """Start the GDM-Flow MCP server over stdio."""
    logging.getLogger("gdm_flow_mcp").setLevel(log_level.upper())
    logger.info("Starting GDM-Flow MCP Server v%s", __version__)

    import asyncio

    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream, write_stream, app.create_initialization_options()
            )

    asyncio.run(run())


def main() -> None:
    typer.run(_run_server)


if __name__ == "__main__":
    main()
