"""P1.5 — multi-turn fix-loop demo.

Drives a client-side fix loop directly against Bedrock + the Trn1 reward
server, logging every attempt so the trace shows attempt-by-attempt
progression (compile error → fix → verify error → fix → success).

Why client-side instead of the deployed AgentCore multi-turn path:
the deployed agent's `_parse_result` returns only the final kernel +
final verification result; intermediate tool calls aren't exposed in the
response. Client-side gives a full trace.

Output:
  results/fix_loop_trace_<task>_<date>.json
    {
      "task": {...},
      "attempts": [
        {"attempt": 1, "prompt_excerpt", "kernel_source",
         "compile": {...}, "verify": {...}, "outcome": "compile_failed"},
        ...
      ],
      "final": {"converged": bool, "attempts_used": int, "wall_time_s"}
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

# ---------------------------------------------------------------------------
# Task: GELU. nl.gelu doesn't exist on Neuron, so the agent must implement it
# manually as 0.5*x*(1 + erf(x/sqrt(2))) or via tanh approx — either of which
# requires non-trivial primitives. Likely to need at least one fix iteration.
# ---------------------------------------------------------------------------

TASK_NAME = "rmsnorm_3d"
# 3D-input task. NKI kernels can only TILE_M=128 along a partition dim and
# require 2D (M, N) inputs — a model that takes the input verbatim as 3D will
# either fail to compile or produce wrong results. Recovery: reshape to 2D,
# run the kernel, reshape back. Real failure mode (48% of v3 SFT outputs hit
# this).
TASK_DESCRIPTION = (
    "Implement an @nki.jit kernel `nki_kernel(x)` that applies RMSNorm "
    "(no learned weight) along the last axis of a **3D** input shaped "
    "(B, S, H). Formula:  x * rsqrt(mean(x^2, axis=-1, keepdims=True) + eps), "
    "with eps=1e-6. The input shape will be (8, 128, 768)."
)

REFERENCE_SOURCE = '''\
import torch

def reference(x):
    eps = 1e-6
    # x is shape (B, S, H); normalize along H
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)
'''

INPUT_SPECS = [{"shape": [8, 128, 768], "dtype": "float32"}]


SYSTEM_PROMPT_STRONG = """You are an NKI kernel author for AWS Trainium.

Hard rules:
- The function MUST be named `nki_kernel`, decorated with `@nki.jit`.
- Inputs are 2-D (M, N); M is a multiple of 128.
- Use TILE_M = 128 and `nl.affine_range(M // TILE_M)`.
- Output via `result = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)` and `return result`.

Available NKI primitives:
  nl.load, nl.store, nl.ndarray, nl.zeros, nl.full, nl.arange, nl.affine_range
  nl.add, nl.subtract, nl.multiply, nl.divide, nl.negative, nl.abs
  nl.exp, nl.log, nl.rsqrt, nl.reciprocal
  nl.maximum, nl.minimum, nl.max, nl.min, nl.sum
  nisa.nc_matmul

DO NOT USE (these don't exist on Neuron NKI):
  nl.gelu, nl.sigmoid, nl.relu, nl.tanh, nl.sqrt, nl.pow, nl.erf, nl.exp2

If you need a missing primitive, implement it from the available ones.

Output ONLY a Python code block with the kernel — no prose, no JSON wrapper."""


SYSTEM_PROMPT_WEAK = """You are an NKI kernel author for AWS Trainium.

Hard rules:
- The function MUST be named `nki_kernel`, decorated with `@nki.jit`.
- Output via `result = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)` and `return result`.

Use the standard `import neuronxcc.nki as nki; import neuronxcc.nki.language as nl`.

Output ONLY a Python code block with the kernel — no prose, no JSON wrapper."""


# Default to STRONG; --weak-prompt switches to the lean variant that doesn't
# pre-disclose which NKI primitives exist (so the model is more likely to try
# `nl.gelu` etc. and trigger the compile-error fix loop).
SYSTEM_PROMPT = SYSTEM_PROMPT_STRONG


# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------

def call_bedrock(model_id: str, region: str, system_prompt: str,
                 messages: list[dict]) -> str:
    """Invoke a Claude model via Bedrock; return assistant text."""
    import boto3

    client = boto3.Session().client("bedrock-runtime", region_name=region)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "system": system_prompt,
        "messages": messages,
    }
    r = client.invoke_model(modelId=model_id, body=json.dumps(body))
    payload = json.loads(r["body"].read())
    return payload["content"][0]["text"]


def extract_kernel(text: str) -> str:
    """Pull the first ```python ... ``` block; fall back to the whole text."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Reward-server client
# ---------------------------------------------------------------------------

def _sigv4_headers(method: str, url: str, body: bytes, region: str) -> dict:
    """Sign a request with the caller's own AWS credential chain against
    the `execute-api` service.

    This signing is currently a no-op against the reward server: the private
    API Gateway that used to verify these signatures was removed (see
    reward_server_cdk/lib/reward-server-stack.ts and reward_server/auth.py) —
    the reward server's actual (and only) access control is network position,
    reachable solely from the AgentCore runtime's security group inside the
    VPC. Calling this against the reward server requires a real network path
    into that VPC; the AWS credentials below are still needed to construct a
    valid request but do not currently gate anything on the receiving end.
    This is an open item pending a decision on what MCP-direct access to the
    reward server should actually require."""
    creds = boto3.Session(region_name=region).get_credentials()
    if creds is None:
        raise RuntimeError(
            "no AWS credentials found; configure env vars, ~/.aws/credentials, "
            "or IAM Identity Center. Note: this only produces a well-formed "
            "SigV4-signed request; it does not by itself grant reward-server "
            "access, which is gated by network position, not IAM (see "
            "reward_server/auth.py)."
        )
    req = AWSRequest(method=method, url=url, data=body)
    SigV4Auth(creds, "execute-api", region).add_auth(req)
    return dict(req.headers)


def reward_post(url: str, region: str, payload: dict, timeout: int = 600) -> dict:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"refusing non-http(s) reward-server URL: {url}")
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    headers.update(_sigv4_headers("POST", url, body, region))
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 — url scheme validated above
        return json.loads(r.read().decode())


def compile_kernel(url, region, source: str) -> dict:
    return reward_post(url, region, {
        "op": "compile",
        "kernel_source": source,
        "input_specs": INPUT_SPECS,
    })


def verify_kernel(url, region, kernel_source: str, ref_source: str,
                  atol: float = 1e-3, rtol: float = 1e-3) -> dict:
    return reward_post(url, region, {
        "op": "verify",
        "kernel_source": kernel_source,
        "reference_source": ref_source,
        "input_specs": INPUT_SPECS,
        "atol": atol,
        "rtol": rtol,
    })


# ---------------------------------------------------------------------------
# Fix-loop driver
# ---------------------------------------------------------------------------

def initial_user_prompt() -> str:
    return (
        f"Task: {TASK_DESCRIPTION}\n\n"
        f"PyTorch reference (semantics to match):\n```python\n{REFERENCE_SOURCE}```\n\n"
        f"Input shape: {INPUT_SPECS[0]['shape']}, dtype: {INPUT_SPECS[0]['dtype']}.\n\n"
        f"Output the kernel now."
    )


def feedback_prompt(failure: str, detail: str) -> str:
    return (
        f"That kernel {failure}. Fix the specific issue below and resubmit "
        f"the full kernel. Do not rewrite from scratch — preserve what works.\n\n"
        f"```\n{detail[:1500]}\n```\n\n"
        f"Output ONLY the corrected kernel as a Python code block."
    )


def run(args) -> int:
    system_prompt = SYSTEM_PROMPT_WEAK if args.weak_prompt else SYSTEM_PROMPT_STRONG

    suffix = "_weak" if args.weak_prompt else ""
    repo_root = Path(__file__).resolve().parent.parent
    out_path = args.output or str(
        repo_root / "results"
        / f"fix_loop_trace_{TASK_NAME}{suffix}_{datetime.now().strftime('%Y-%m-%d_%H%M')}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    trace: dict[str, Any] = {
        "task": {
            "name": TASK_NAME,
            "description": TASK_DESCRIPTION,
            "reference_source": REFERENCE_SOURCE,
            "input_specs": INPUT_SPECS,
        },
        "model_id": args.model_id,
        "reward_url": args.reward_url,
        "max_attempts": args.max_attempts,
        "attempts": [],
        "final": {"converged": False, "attempts_used": 0, "wall_time_s": None},
    }

    t_start = time.time()
    messages: list[dict] = [{"role": "user", "content": initial_user_prompt()}]
    converged = False

    for attempt in range(1, args.max_attempts + 1):
        print(f"\n=== attempt {attempt}/{args.max_attempts} ===", flush=True)
        attempt_rec: dict[str, Any] = {
            "attempt": attempt,
            "prompt_excerpt": messages[-1]["content"][:400],
            "kernel_source": None,
            "compile": None,
            "verify": None,
            "outcome": None,
        }

        # 1) generate
        try:
            assistant_text = call_bedrock(
                args.model_id, args.region, system_prompt, messages,
            )
        except Exception as e:
            attempt_rec["outcome"] = f"bedrock_error: {e}"
            trace["attempts"].append(attempt_rec)
            print(f"  bedrock error: {e}")
            break

        kernel = extract_kernel(assistant_text)
        attempt_rec["kernel_source"] = kernel
        messages.append({"role": "assistant", "content": assistant_text})
        print(f"  generated kernel ({len(kernel)} chars)")

        # 2) compile
        comp = compile_kernel(args.reward_url, args.region, kernel)
        attempt_rec["compile"] = {
            "success": comp.get("success"),
            "errors": (comp.get("errors") or [])[:5],
            "stderr_tail": (comp.get("stderr_tail") or [])[-5:],
        }
        if not comp.get("success"):
            attempt_rec["outcome"] = "compile_failed"
            print(f"  compile FAILED")
            err_blob = "\n".join(
                comp.get("errors") or comp.get("stderr_tail") or []
            ) or "compile failed (no error detail)"
            messages.append({
                "role": "user",
                "content": feedback_prompt("failed to compile", err_blob),
            })
            trace["attempts"].append(attempt_rec)
            _save(trace, out_path)
            continue

        print(f"  compile OK")

        # 3) verify
        ver = verify_kernel(
            args.reward_url, args.region, kernel, REFERENCE_SOURCE,
        )
        attempt_rec["verify"] = {
            "correct": ver.get("correct"),
            "max_abs_error": ver.get("max_abs_error"),
            "max_rel_error": ver.get("max_rel_error"),
            "error": ver.get("error"),
        }
        if ver.get("correct"):
            attempt_rec["outcome"] = "success"
            print(f"  verify OK — converged on attempt {attempt}")
            trace["attempts"].append(attempt_rec)
            converged = True
            break

        attempt_rec["outcome"] = "verify_failed"
        print(f"  verify FAILED  max_abs_error={ver.get('max_abs_error')}")
        diff_blob = (
            f"max_abs_error={ver.get('max_abs_error')}, "
            f"max_rel_error={ver.get('max_rel_error')}, "
            f"detail={ver.get('error')}"
        )
        messages.append({
            "role": "user",
            "content": feedback_prompt(
                "compiled but produced wrong output", diff_blob),
        })
        trace["attempts"].append(attempt_rec)
        _save(trace, out_path)

    trace["final"]["converged"] = converged
    trace["final"]["attempts_used"] = len(trace["attempts"])
    trace["final"]["wall_time_s"] = round(time.time() - t_start, 1)
    _save(trace, out_path)

    print(f"\n=== summary ===")
    print(f"converged: {converged}")
    print(f"attempts:  {len(trace['attempts'])}")
    print(f"wall:      {trace['final']['wall_time_s']}s")
    print(f"trace:     {out_path}")
    return 0 if converged else 1


def _save(trace: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-url", default=os.environ.get(
        "REWARD_URL", "http://REWARD_SERVER_IP:5050/reward"))
    parser.add_argument("--model-id", default=os.environ.get(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-8"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--weak-prompt", action="store_true",
                        help="Use a leaner system prompt that doesn't pre-list "
                             "missing NKI primitives — more likely to trigger "
                             "compile-error fix loops.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
