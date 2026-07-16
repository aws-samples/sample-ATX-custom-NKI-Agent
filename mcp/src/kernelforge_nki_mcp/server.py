"""MCP server entry point.

Exposes seven tools that drive the kernel-forge agent end-to-end:

  nki_discover_kernels
  nki_generate_kernel
  nki_compile
  nki_verify
  nki_profile
  nki_skill_lookup
  nki_emit_diff

The server is stateless. State (sessions, retries, model history) is owned by
AgentCore on the server side and by `.nki-agent/` on the customer side.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .clients import AgentCoreClient, RewardServerClient
from .config import Config
from .diff import emit_diff
from .discover import discover
from .skill_db import search as skill_search

_cfg = Config.from_env()
_agentcore = AgentCoreClient(_cfg)
_reward = RewardServerClient(_cfg)

mcp = FastMCP("kernelforge-nki-mcp")


@mcp.tool()
def nki_discover_kernels(repo_root: str) -> dict[str, Any]:
    """Walk a repo and return candidate kernels for NKI conversion.

    Args:
        repo_root: Absolute path to the repository to scan.

    Returns:
        repo_root, candidates[]: each with file, line, function, signature,
        kind ('triton' | 'module_forward'), hot, already_nki, snippet.
    """
    result = discover(repo_root)
    return {
        "repo_root": result.repo_root,
        "candidates": [c.__dict__ for c in result.candidates],
    }


@mcp.tool()
def nki_generate_kernel(
    description: str,
    reference_source: str,
    input_specs: list[dict[str, Any]],
    model: str = "auto",
    max_turns: int = 10,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate an NKI kernel via the AgentCore multi-turn loop.

    Args:
        description: Natural-language description of the kernel.
        reference_source: PyTorch / Triton reference implementation.
        input_specs: List of {shape, dtype} for each input tensor.
        model: Backend to use. One of 'auto', 'opus-4-8', 'qwen3-sft-v4', 'vllm'.
        max_turns: Max compile/verify retry budget.
        constraints: Optional perf / accuracy budget.

    Returns:
        kernel_source, verification, profiling, turns_used, trajectory.
    """
    payload = {
        "task_spec": {
            "description": description,
            "reference_source": reference_source,
            "input_specs": input_specs,
            "constraints": constraints or {},
            "model_pin": None if model == "auto" else model,
        },
        "mode": "multi_turn",
        "max_turns": max_turns,
    }
    return _agentcore.invoke(payload)


@mcp.tool()
def nki_compile(
    kernel_source: str,
    kernel_name: str = "nki_kernel",
    optimization_level: int = 2,
) -> dict[str, Any]:
    """Compile an NKI kernel via the Trn1 reward server (neuronx-cc).

    Returns: success, neff_path, errors[], warnings[], compile_time_s.
    """
    return _reward.reward(
        {
            "op": "compile",
            "kernel_source": kernel_source,
            "kernel_name": kernel_name,
            "optimization_level": optimization_level,
        }
    )


@mcp.tool()
def nki_verify(
    kernel_source: str,
    reference_source: str,
    input_specs: list[dict[str, Any]],
    kernel_name: str = "nki_kernel",
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Verify an NKI kernel against a PyTorch reference on real Trainium silicon.

    Returns: correct, max_abs_error, max_rel_error, mismatched_elements, total_elements, error.
    """
    return _reward.reward(
        {
            "op": "verify",
            "kernel_source": kernel_source,
            "reference_source": reference_source,
            "input_specs": input_specs,
            "kernel_name": kernel_name,
            "atol": atol,
            "rtol": rtol,
        }
    )


@mcp.tool()
def nki_profile(
    kernel_source: str,
    input_specs: list[dict[str, Any]],
    kernel_name: str = "nki_kernel",
    warmup_runs: int = 5,
    measure_runs: int = 20,
) -> dict[str, Any]:
    """Profile a compiled NKI kernel on Trainium.

    Returns: success, latency_mean_us, latency_median_us, latency_p99_us,
    latency_std_us, throughput_ops_per_s, flops_utilization, error.
    """
    return _reward.reward(
        {
            "op": "profile",
            "kernel_source": kernel_source,
            "input_specs": input_specs,
            "kernel_name": kernel_name,
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
        }
    )


@mcp.tool()
def nki_skill_lookup(
    query: str,
    top_k: int = 5,
    skill_category: str | None = None,
) -> dict[str, Any]:
    """Search the NKI skill DB for relevant patterns and recipes.

    Returns: results[], total_matches.
    """
    matches = skill_search(query, top_k=top_k, category=skill_category)
    return {
        "results": [m.__dict__ for m in matches],
        "total_matches": len(matches),
    }


@mcp.tool()
def nki_emit_diff(
    repo_root: str,
    target_file: str,
    new_source: str,
    open_pr: bool = False,
    pr_title: str = "Convert kernel to NKI",
    pr_body: str = "Generated by kernel-forge-aws-transform.",
) -> dict[str, Any]:
    """Emit a unified diff between target_file and new_source. Optionally open a PR."""
    return emit_diff(
        repo_root=repo_root,
        target_file=target_file,
        new_source=new_source,
        open_pr=open_pr,
        pr_title=pr_title,
        pr_body=pr_body,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
