"""End-to-end integration test for the full ATX migration chain.

Chain under test (the surfaces a real migration traverses):

    atx CLI  ->  skill (SKILL.md)  ->  MCP tools  ->  AgentCore runtime  ->  Trn1 reward server

The chain has two tiers, and this file exercises both:

  * **Local / hardware-free (always runs):** the plugin wiring (`.mcp.json`,
    plugin manifest, SKILL.md), the MCP tools that don't touch AWS
    (`nki_discover_kernels`, `nki_skill_lookup`, `nki_emit_diff`), and the
    contracts between them. These pin the atx-CLI -> skill -> MCP links.

  * **Live / on-device (runs when ``NKI_LIVE=1`` and ``AGENTCORE_ARN`` set):**
    `nki_generate_kernel` -> AgentCore -> reward server, which compiles,
    verifies, and profiles a kernel on a real Trainium device. This pins the
    MCP -> AgentCore -> reward-server links against real silicon. It drives the
    runtime in ``multi_turn`` mode — the compile-verify-fix loop the documented
    migration path uses — so a first-try compiler error is repaired rather than
    terminal.

Run everything hardware-free:
    pytest tests/test_integration_chain.py

Run including the live on-device leg (needs AWS creds + a deployed runtime, plus
boto3 — `uv run --with boto3 --with pytest python -m pytest ...` if your venv
lacks it):
    NKI_LIVE=1 \
    AGENTCORE_ARN=arn:aws:bedrock-agentcore:us-east-1:...:runtime/... \
    AWS_REGION=us-east-1 \
    pytest tests/test_integration_chain.py -v
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

# conftest.py at the repo root wires mcp/src onto sys.path.
from kernelforge_nki_mcp import discover as discover_mod
from kernelforge_nki_mcp.diff import emit_diff
from kernelforge_nki_mcp.skill_db import search as skill_search

_REPO = Path(__file__).resolve().parents[1]
_EXAMPLE = _REPO / "examples" / "cuda_rmsnorm_migration"
_CUDA_INPUT = _EXAMPLE / "rmsnorm_cuda_input.py"

_LIVE = os.environ.get("NKI_LIVE") == "1"
_ARN = os.environ.get("AGENTCORE_ARN", "")
_REGION = os.environ.get("AWS_REGION", "us-east-1")


# ─────────────────────────────────────────────────────────────────────────────
# Link 1: atx CLI -> skill.  The CLI is a thin client; what it loads is the
# plugin manifest + SKILL.md + .mcp.json. If those don't wire the MCP server in,
# no surface (atx, IDE, Kiro) can drive the chain. Pin that wiring.
# ─────────────────────────────────────────────────────────────────────────────

def test_plugin_manifest_wires_skill_and_mcp() -> None:
    manifest = json.loads((_REPO / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["mcp"] == ".mcp.json"
    assert any("nki-kernel" in s for s in manifest["skills"])


def test_mcp_descriptor_registers_server() -> None:
    desc = json.loads((_REPO / ".mcp.json").read_text())
    assert "kernelforge-nki-mcp" in desc["mcpServers"]


def test_skill_documents_the_mcp_tool_workflow() -> None:
    skill = (_REPO / "skills" / "nki-kernel" / "SKILL.md").read_text()
    # The skill must name the MCP tools the agent is supposed to call, in order.
    for tool in (
        "nki_discover_kernels",
        "nki_skill_lookup",
        "nki_generate_kernel",
        "nki_compile",
        "nki_verify",
    ):
        assert tool in skill, f"SKILL.md does not document {tool}"


# ─────────────────────────────────────────────────────────────────────────────
# Link 2: skill -> MCP (hardware-free tools). The skill's first step is
# discovery; a CUDA custom kernel must be found, or the migration never starts.
# ─────────────────────────────────────────────────────────────────────────────

def test_discover_finds_the_cuda_custom_kernel() -> None:
    result = discover_mod.discover(str(_EXAMPLE))
    cuda = [c for c in result.candidates if c.kind == "cuda"]
    assert cuda, "the CUDA RMSNorm custom kernel was not discovered"
    assert any(c.function == "forward" for c in cuda)


def test_skill_lookup_returns_an_rmsnorm_pattern() -> None:
    matches = skill_search("rmsnorm root mean square layer normalization", top_k=5)
    assert matches, "skill DB returned no RMSNorm pattern"


def test_emit_diff_applies_and_is_traversal_safe(tmp_path: Path) -> None:
    # Positive: a diff against a file inside the repo applies.
    target = tmp_path / "kernel.py"
    target.write_text("# original\n")
    out = emit_diff(
        repo_root=str(tmp_path),
        target_file="kernel.py",
        new_source="# migrated to NKI\n",
        open_pr=False,
    )
    assert out.get("diff") and "migrated to NKI" in out["diff"]

    # Negative: path traversal is refused (the diff tool writes files — this is
    # the security boundary for the last link in the chain).
    bad = emit_diff(
        repo_root=str(tmp_path),
        target_file="../escape.py",
        new_source="x",
        open_pr=False,
    )
    assert bad["ok"] is False and "escape" in bad["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Link 3: MCP -> AgentCore -> reward server (live, on-device). Gated so the
# suite stays hardware-free by default; runs for real when creds are present.
# ─────────────────────────────────────────────────────────────────────────────

_LIVE_SKIP = pytest.mark.skipif(
    not (_LIVE and _ARN),
    reason="live on-device leg: set NKI_LIVE=1 and AGENTCORE_ARN to run",
)


def _reference_source() -> str:
    """The PyTorch reference the device compares against.

    Must define a module-level ``reference(*args)`` function: the reward
    server's verify op imports this source and exits (code 6) if no such
    function exists, so a bare ``nn.Module`` subclass never gets verified.
    Weight is omitted deliberately — it is all-ones in the CUDA original, so
    the identity multiply adds nothing to compare against.
    """
    return (
        "import torch\n"
        "def reference(x):\n"
        "    v = x.pow(2).mean(dim=-1, keepdim=True)\n"
        "    return x * torch.rsqrt(v + 1e-6)\n"
    )


def _invoke_runtime(payload: dict) -> dict:
    """Drive the deployed AgentCore runtime the way the eval/smoke path does."""
    import boto3
    from botocore.config import Config as BotoConfig

    client = boto3.Session().client(
        "bedrock-agentcore",
        region_name=_REGION,
        config=BotoConfig(read_timeout=400, connect_timeout=30, retries={"max_attempts": 1}),
    )
    session_id = f"itest-{int(time.time())}".ljust(33, "x")[:33]
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=_ARN,
        qualifier="DEFAULT",
        runtimeSessionId=session_id,
        payload=json.dumps(payload).encode("utf-8"),
    )
    body = (
        b"".join(resp["response"].iter_chunks())
        if hasattr(resp["response"], "iter_chunks")
        else resp["response"].read()
    )
    return json.loads(body)


@_LIVE_SKIP
def test_live_migration_compiles_and_verifies_on_device() -> None:
    """The headline chain: generate a kernel and prove it on real Trainium.

    Asserts the reward server actually compiled and numerically verified the
    migrated RMSNorm kernel — the ground-truth link the whole design rests on.

    Runs in ``multi_turn`` mode, the same mode the documented migration path
    uses (``examples/cuda_rmsnorm_migration/run_migration.py``). The mode choice
    is load-bearing, not incidental: only multi-turn has the compile-verify-fix
    loop, where a rejected kernel comes back with the ``neuronx-cc`` error and
    gets repaired. ``single_shot`` emits one candidate and stops, so a first-try
    compile error is terminal — it cannot reach a verified kernel, and asserting
    verification against it would test a path the agent does not ship.
    """
    payload = {
        "task_spec": {
            "description": (
                "Convert this RMSNorm (originally a hand-written CUDA custom "
                "kernel) to an @nki.jit kernel named `nki_kernel` for AWS "
                "Trainium. Match the PyTorch reference numerically."
            ),
            "reference_source": _reference_source(),
            "input_specs": [{"shape": [4096, 4096], "dtype": "float32"}],
            "constraints": {"atol": 1e-3, "rtol": 1e-3},
            "model_pin": None,
        },
        "mode": "multi_turn",
        "max_turns": 10,
    }
    # Bedrock intermittently returns InternalServerException / ServiceUnavailable
    # after exhausting its own retries. The runtime's converse loop treats that as
    # terminal (`break`) and logs the reason only to CloudWatch, so the response
    # is status="failed" with NO compile block, NO verify block and error=None.
    #
    # That combination is the signature to retry on: a genuine generation failure
    # always reaches the reward server at least once, so it carries a compile or
    # verify block. Deliberately not keyed on turns_used — the capacity error can
    # land on any turn (turn 2 is common, after a token-refresh `continue`).
    for attempt in range(3):
        result = _invoke_runtime(payload)
        transient = (
            result.get("status") == "failed"
            and not result.get("compile")
            and not result.get("verify")
        )
        if not transient:
            break
        if attempt < 2:
            time.sleep(20 * (attempt + 1))
    else:
        pytest.skip(
            "Bedrock returned a capacity error (InternalServerException / "
            "ServiceUnavailableException) on 3 consecutive attempts — see the "
            "runtime's CloudWatch logs. Not a kernel or reward-server failure."
        )

    assert result.get("status") == "success", (
        f"status={result.get('status')} after {result.get('turns_used')} turns; "
        f"compile={result.get('compile')} verify={result.get('verify')} "
        f"error={result.get('error')}"
    )
    assert result.get("kernel_source"), "no kernel_source returned"

    # Assert on the top-level booleans, the way run_migration.py does. The
    # nested `compile` block is whichever tool result last reported a
    # compile — `compile_nki_kernel` (which returns `success`) or
    # `verify_nki_kernel` (which returns `compiled` and no `success` key), so
    # its shape depends on which tool the model chose to call.
    assert result.get("compiled") is True, f"compile failed: {result.get('compile')}"

    verify_r = result.get("verify") or {}
    assert result.get("correct") is True, (
        f"numerical verification failed on device: max_abs_error="
        f"{verify_r.get('max_abs_error')} mismatched="
        f"{verify_r.get('mismatched_elements')}/{verify_r.get('total_elements')}"
    )
