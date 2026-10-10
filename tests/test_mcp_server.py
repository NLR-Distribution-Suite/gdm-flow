import asyncio
import os
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("mcp")

try:
    from gdm_flow.mcp import server as mcp_server
except (AttributeError, ImportError) as exc:
    pytest.skip(f"MCP server module incompatible: {exc}", allow_module_level=True)


def test_mcp_list_tools_includes_documentation_tools():
    from mcp_types import ListToolsRequest

    result = asyncio.run(mcp_server._handle_list_tools(None, ListToolsRequest(method="tools/list")))
    tool_names = {tool.name for tool in result.tools}

    assert "opf_calculate_ybus" in tool_names
    assert "opf_run_ac" in tool_names
    assert "list_opf_documentation" in tool_names
    assert "search_opf_documentation" in tool_names
    assert "get_opf_documentation_page" in tool_names
    assert "list_opf_api_symbols" in tool_names
    assert "get_opf_api_reference" in tool_names


def test_mcp_documentation_tools_smoke(monkeypatch):
    docs_root = Path(__file__).resolve().parents[1] / "docs"
    assert docs_root.is_dir()
    monkeypatch.setattr(mcp_server, "DOCS_ROOT", docs_root)
    listing = asyncio.run(mcp_server._handle_list_opf_documentation({}))
    assert listing["count"] > 0
    assert "intro.md" in listing["files"]

    search = asyncio.run(
        mcp_server._handle_search_opf_documentation(
            {
                "query": "AC OPF",
                "max_results": 3,
            }
        )
    )
    assert search["count"] >= 1

    page = asyncio.run(
        mcp_server._handle_get_opf_documentation_page(
            {
                "doc_path": "intro.md",
                "start_line": 1,
                "max_lines": 20,
            }
        )
    )
    assert page["path"] == "intro.md"
    assert "GDM-Flow" in page["content"]


def test_mcp_api_reference_tools_smoke():
    symbol_listing = asyncio.run(mcp_server._handle_list_opf_api_symbols({}))
    assert symbol_listing["count"] > 0
    assert "calculate_ybus" in symbol_listing["symbols"]

    api_ref = asyncio.run(
        mcp_server._handle_get_opf_api_reference(
            {
                "symbol_name": "calculate_ybus",
            }
        )
    )
    assert api_ref["symbol"] == "calculate_ybus"
    assert api_ref["module"].startswith("gdm_flow")
    assert api_ref["signature"] is not None


def test_get_system_path_arg_accepts_direct_model_ref_path():
    path = mcp_server._get_system_path_arg({"model_ref": {"path": "/data/model.json"}})
    assert path == "/data/model.json"


def test_get_system_path_arg_resolves_registry_model_ref(tmp_path):
    db_path = tmp_path / "registry.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE models (
                model_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                stored_path TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO models (model_id, version, stored_path) VALUES (?, ?, ?)",
            ("opf123", 2, "/data/opf_v2.json"),
        )

    os.environ["DIST_STACK_MODEL_REGISTRY_DB"] = str(db_path)
    try:
        path = mcp_server._get_system_path_arg(
            {"model_ref": {"model_id": "opf123", "version": 2}}
        )
    finally:
        os.environ.pop("DIST_STACK_MODEL_REGISTRY_DB", None)

    assert path == "/data/opf_v2.json"
