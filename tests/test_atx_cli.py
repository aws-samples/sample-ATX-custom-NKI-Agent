"""Tests for the AWS Transform (ATX) CLI surface.

The `atx` CLI is one client of the `kernelforge-nki-mcp` MCP server. Unlike the
IDE plugin (installed via ``/plugin install``), it has its own packaging model:

    curl -fsSL https://transform-cli.awsstatic.com/install.sh | bash
                                  # installs the `atx` binary
    ~/.aws/atx/mcp.json           # registers the MCP server for atx
    atx custom def exec ...       # runs a transformation definition

so the artifacts it needs live under ``atx/``:

    atx/transformation-definition/transformation_definition.md   (the workflow)
    atx/mcp.json                                                 (the MCP wiring)

This file has two tiers, matching ``test_integration_chain.py``:

  * **Hardware-free (always runs):** the ATX artifacts exist and are wired
    correctly — the transformation definition names the MCP tools in workflow
    order, and ``atx/mcp.json`` registers the same server the IDE ``.mcp.json``
    does. These pin the ``atx -> transformation-definition -> MCP`` links
    without needing the CLI installed.

  * **Live CLI (runs when ``ATX_CLI=1`` and the ``atx`` binary is present):**
    shells out to the real ``atx`` CLI and asserts it can enumerate the seven
    ``kernelforge-nki-mcp`` tools through ``~/.aws/atx/mcp.json``. This pins the
    ``atx -> MCP`` link against the actual binary.

Run hardware-free:
    pytest tests/test_atx_cli.py

Run including the live CLI leg (needs `atx` installed + ~/.aws/atx/mcp.json):
    ATX_CLI=1 pytest tests/test_atx_cli.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_ATX = _REPO / "atx"
_TD = _ATX / "transformation-definition" / "transformation_definition.md"
_ATX_MCP = _ATX / "mcp.json"

# The seven tools, in the order the workflow calls them.
_TOOLS = (
    "nki_discover_kernels",
    "nki_skill_lookup",
    "nki_generate_kernel",
    "nki_compile",
    "nki_verify",
    "nki_profile",
    "nki_emit_diff",
)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 (hardware-free): the ATX artifacts exist and are wired correctly.
# ─────────────────────────────────────────────────────────────────────────────

def test_transformation_definition_exists() -> None:
    # ATX executes a transformation definition file, not a SKILL.md. It must be
    # present, or `atx custom def publish/exec` has nothing to run.
    assert _TD.is_file(), "atx/transformation-definition/transformation_definition.md is missing"


def test_transformation_definition_documents_the_tool_workflow() -> None:
    td = _TD.read_text()
    # The definition must name every MCP tool the agent is supposed to call...
    for tool in _TOOLS:
        assert tool in td, f"transformation definition does not document {tool}"
    # ...and it must state the two hard gates that make a kernel rewrite safe.
    assert "requirements" in td.lower(), "definition omits the requirements gate"
    assert "verif" in td.lower(), "definition omits the verification requirement"


def test_transformation_definition_calls_tools_in_workflow_order() -> None:
    td = _TD.read_text()
    # discovery precedes generation, generation precedes verification, and a
    # diff is only emitted last — the invariant the workflow depends on.
    positions = [td.find(t) for t in _TOOLS]
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions), (
        "tools are not documented in workflow order "
        "(discover -> skill_lookup -> generate -> compile -> verify -> profile -> emit_diff)"
    )


def test_atx_mcp_config_registers_the_same_server() -> None:
    # atx reads ~/.aws/atx/mcp.json, not the repo's .mcp.json — but both must
    # register the SAME server id, or the two surfaces drive different tools.
    atx_cfg = json.loads(_ATX_MCP.read_text())
    ide_cfg = json.loads((_REPO / ".mcp.json").read_text())
    assert "kernelforge-nki-mcp" in atx_cfg["mcpServers"]
    assert set(atx_cfg["mcpServers"]) == set(ide_cfg["mcpServers"]), (
        "atx/mcp.json and .mcp.json register different MCP servers"
    )
    entry = atx_cfg["mcpServers"]["kernelforge-nki-mcp"]
    assert entry["command"] == "uvx", "atx mcp entry should launch the server via uvx"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 (live CLI): shell out to the real `atx` binary.
# ─────────────────────────────────────────────────────────────────────────────

_ATX_BIN = shutil.which("atx")
_LIVE_SKIP = pytest.mark.skipif(
    not (os.environ.get("ATX_CLI") == "1" and _ATX_BIN),
    reason="live atx CLI leg: set ATX_CLI=1 and install the `atx` CLI to run",
)


@_LIVE_SKIP
def test_atx_cli_enumerates_the_seven_mcp_tools() -> None:
    """The real `atx` CLI must see all seven tools through ~/.aws/atx/mcp.json.

    This is the concrete `atx -> MCP` link: if `atx mcp tools` can list the
    kernelforge tools, the CLI can drive the migration the same way the IDE and
    Kiro surfaces do.
    """
    out = subprocess.run(
        [_ATX_BIN, "mcp", "tools", "-s", "kernelforge-nki-mcp"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, f"`atx mcp tools` failed: {out.stderr}"
    for tool in _TOOLS:
        assert tool in out.stdout, f"atx CLI did not list {tool} (check ~/.aws/atx/mcp.json)"
