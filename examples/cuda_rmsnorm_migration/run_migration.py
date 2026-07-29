#!/usr/bin/env python3
"""Drive the full CUDA->NKI migration loop end to end.

Shows the real chain `atx custom def exec` drives:

    atx CLI -> skill (SKILL.md) -> MCP tools -> AgentCore runtime -> Trn1 reward server

The agent GENERATES an @nki.jit kernel via the deployed AgentCore runtime in
multi-turn mode (its compile-verify-fix loop calls the compile/verify/profile
MCP tools on the real Trn1 reward server), then reports the on-device result.
Multi-turn is used because its fix loop reliably lands a verified kernel even
when the first generation doesn't compile — that recovery IS the point.

Usage:
    AGENTCORE_ARN=arn:aws:bedrock-agentcore:us-east-1:<acct>:runtime/<id> \\
    AWS_REGION=us-east-1 \\
    python3 examples/cuda_rmsnorm_migration/run_migration.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import boto3
from botocore.config import Config as BotoConfig

_ARN = os.environ.get("AGENTCORE_ARN", "")
_REGION = os.environ.get("AWS_REGION", "us-east-1")
_MAX_TURNS = int(os.environ.get("DEMO_MAX_TURNS", "10"))

_GREEN, _CYAN, _DIM, _BOLD, _RESET = "\033[32m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"


def _reference_source() -> str:
    # Functional reference(*args) the Trn1 reward server runs as ground truth.
    return (
        "import torch\n"
        "def reference(x):\n"
        "    v = x.pow(2).mean(dim=-1, keepdim=True)\n"
        "    return x * torch.rsqrt(v + 1e-6)\n"
    )


def main() -> int:
    if not _ARN:
        print("error: set AGENTCORE_ARN (and AWS_REGION) first", file=sys.stderr)
        return 2

    payload = {
        "task_spec": {
            "description": (
                "Convert this RMSNorm (originally a hand-written CUDA custom kernel) "
                "to an @nki.jit kernel named `nki_kernel` for AWS Trainium. Match the "
                "PyTorch reference numerically."
            ),
            "reference_source": _reference_source(),
            "input_specs": [{"shape": [4096, 4096], "dtype": "float32"}],
            "constraints": {"atol": 1e-3, "rtol": 1e-3},
        },
        "mode": "multi_turn",
        "max_turns": _MAX_TURNS,
    }

    print(f"{_DIM}  MCP nki_generate_kernel -> AgentCore runtime (Strands, Opus 4.8)…{_RESET}")
    print(f"{_DIM}  the runtime's compile-verify-fix loop calls compile/verify/profile{_RESET}")
    print(f"{_DIM}  on the Trn1 reward server (@nki.baremetal, real NeuronCore)…{_RESET}")

    client = boto3.Session().client(
        "bedrock-agentcore", region_name=_REGION,
        config=BotoConfig(read_timeout=400, connect_timeout=30, retries={"max_attempts": 1}),
    )
    sid = f"atx-demo-rmsnorm-{int(time.time())}".ljust(33, "x")[:33]
    t0 = time.time()
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=_ARN, qualifier="DEFAULT", runtimeSessionId=sid,
        payload=json.dumps(payload).encode(),
    )
    body = (b"".join(resp["response"].iter_chunks())
            if hasattr(resp["response"], "iter_chunks") else resp["response"].read())
    result = json.loads(body)
    elapsed = time.time() - t0

    kernel = result.get("kernel_source") or ""
    ver = result.get("verify") or {}
    prof = result.get("profile") or {}
    ok = bool(result.get("compiled")) and bool(result.get("correct"))

    if kernel:
        print(f"\n{_BOLD}{_CYAN}  generated @nki.jit kernel:{_RESET}")
        for line in kernel.splitlines()[:14]:
            print(f"    {line}")
        n = len(kernel.splitlines())
        if n > 14:
            print(f"    {_DIM}… ({n} lines total){_RESET}")

    mark = f"{_GREEN}✅{_RESET}" if ok else "❌"
    print(f"\n{_BOLD}  on-device result ({result.get('model_used')}, "
          f"{result.get('turns_used')} turns / {result.get('tool_calls_used')} tool calls, {elapsed:.0f}s):{_RESET}")
    print(f"    compile : {'✅ success' if result.get('compiled') else '❌ failed'}")
    print(f"    verify  : {'✅ correct' if result.get('correct') else '❌ mismatch'}"
          f"  max_abs_error={ver.get('max_abs_error')}"
          f"  mismatched={ver.get('mismatched_elements')}/{ver.get('total_elements')}")
    if prof:
        print(f"    profile : median={prof.get('latency_median_us')}µs  p99={prof.get('latency_p99_us')}µs")
    print(f"\n  {mark} migration {'verified on real Trainium silicon' if ok else 'did not verify'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
