"""AgentCore entry point — Strands Agent + tools + model router.

This is the heavy multi-turn agent. The MCP server's `nki_generate_kernel`
tool calls into this endpoint; the four `@tool`s here delegate to the
Trn1 reward server for compile / verify / profile.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any

import httpx
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from router import BackendConfig, ModelRouter, TaskSpec

app = BedrockAgentCoreApp()
logger = app.logger

# REWARD_SERVER_URL is the reward server's private IP:5050 (RewardUrlPrivate
# stack output) — e.g. http://10.42.0.51:5050. The runtime calls it directly.
# Auth is enforced at the network layer — the reward server sits in a
# PRIVATE_ISOLATED subnet (no public IP, no internet route) and its security
# group accepts 5050 only from this runtime's client SG, so the runtime's
# ENIs are the only thing that can reach the port. A direct call has no
# request-size timeout ceiling to work around, so compile/verify ops that
# run for minutes complete normally (bounded only by REQUEST_TIMEOUT_S).
#
# There is intentionally no fallback default here. agent_config.yaml ships
# REWARD_SERVER_URL as a placeholder ("http://REPLACE_WITH_REWARD_SERVER_
# PRIVATE_IP:5050"), which repoint-agentcore.sh overwrites with the real
# stack output once the reward server is deployed. _validate_reward_url()
# rejects an unset value, that placeholder, or any other malformed URL
# immediately at import time, so a misconfigured deploy fails with a clear
# cause instead of surfacing later as a DNS lookup error on the first tool
# call.
_UNCONFIGURED_URL = "http://REPLACE_WITH_REWARD_SERVER_PRIVATE_IP:5050"


class RewardServerNotConfiguredError(RuntimeError):
    """Raised when REWARD_SERVER_URL is unset, a known placeholder, or malformed."""


def _validate_reward_url(url: str | None) -> str:
    """Fails fast if REWARD_SERVER_URL is unset, unconfigured, or malformed.

    Args:
        url (str | None): the value of the REWARD_SERVER_URL environment
            variable (`None` if unset).

    Returns:
        str: `url`, unchanged, if it passes validation.

    Raises:
        RewardServerNotConfiguredError: if `url` is unset, empty, matches the
            committed placeholder from agent_config.yaml, or is not a
            syntactically valid `http://`/`https://` URL with a host and port.
    """
    from urllib.parse import urlsplit

    if not url or url == _UNCONFIGURED_URL:
        raise RewardServerNotConfiguredError(
            f"REWARD_SERVER_URL is not configured (got {url!r}). "
            "Run infrastructure/reward_server_cdk/repoint-agentcore.sh after "
            "deploying the reward server, or set REWARD_SERVER_URL directly."
        )
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or not parsed.port:
        raise RewardServerNotConfiguredError(
            f"REWARD_SERVER_URL is not a valid http(s)://host:port URL: {url!r}"
        )
    return url


REWARD_URL = _validate_reward_url(os.environ.get("REWARD_SERVER_URL"))
REQUEST_TIMEOUT_S = int(os.environ.get("REWARD_TIMEOUT_S", "600"))
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Backend definitions — wired from agent_config.yaml at deploy time.
# ---------------------------------------------------------------------------
def _load_backends() -> dict[str, BackendConfig]:
    return {
        "opus-4-8": BackendConfig(
            name="opus-4-8",
            provider="bedrock",
            # Cross-region inference-profile ID ("us." prefix), not the bare
            # model ID — Bedrock rejects on-demand InvokeModel/Converse calls
            # against Opus 4.8 with just "anthropic.claude-opus-4-8"
            # (ValidationException: "on-demand throughput isn't supported").
            # Must match the fallback literals in _run_single_shot/
            # _run_multi_turn below, which already use the profile ID.
            model_id=os.environ.get("OPUS_MODEL_ID", "us.anthropic.claude-opus-4-8"),
        ),
        "qwen3-sft-v4": BackendConfig(
            name="qwen3-sft-v4",
            provider="bedrock-cmi",
            model_arn=os.environ.get("QWEN_CMI_ARN", ""),
        ),
        "vllm": BackendConfig(
            name="vllm",
            provider="openai-compat",
            endpoint=os.environ.get("VLLM_ENDPOINT", ""),
        ),
    }


def _skill_score(_query: str) -> float:
    """Hook for the router. The real implementation calls the MCP skill DB
    or a vector store; deployments that don't ship one return 0.0 so the
    router defaults to opus-4-8."""
    return 0.0


def _build_model(backend: BackendConfig, region: str) -> Any:
    if backend.provider == "bedrock":
        return BedrockModel(model_id=backend.model_id, region_name=region)
    if backend.provider == "bedrock-cmi":
        return BedrockModel(model_id=backend.model_arn, region_name=region)
    if backend.provider == "openai-compat":
        # Strands ships an OpenAI-compatible client; the import is deferred so
        # the deployment does not require it when only Bedrock is used.
        from strands.models.openai import OpenAIModel  # type: ignore

        return OpenAIModel(
            base_url=backend.endpoint,
            model=backend.model_id or "qwen3-coder-30b-sft-v4",
        )
    raise ValueError(f"unknown backend provider: {backend.provider}")


# ---------------------------------------------------------------------------
# Tools — thin HTTP clients to the reward server (direct, network-isolated).
# ---------------------------------------------------------------------------
def _post_reward(payload: dict[str, Any]) -> dict[str, Any]:
    """POST to the reward server directly over the private VPC path.

    No SigV4 / API Gateway: the reward server is only reachable from this
    runtime's ENIs (SG-to-SG on 5050, in a no-internet subnet), so the network
    is the auth boundary. The long timeout matters — a real NKI compile/verify
    can run for minutes, and there is no gateway ceiling to trip.
    """
    url = f"{REWARD_URL.rstrip('/')}/reward"
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


@tool
def compile_nki_kernel(
    kernel_source: str,
    kernel_name: str = "nki_kernel",
    optimization_level: int = 2,
) -> dict[str, Any]:
    """Compile an NKI kernel via neuronx-cc. Returns success, errors, neff_path."""
    return _post_reward(
        {
            "op": "compile",
            "kernel_source": kernel_source,
            "kernel_name": kernel_name,
            "optimization_level": optimization_level,
        }
    )


@tool
def verify_nki_kernel(
    kernel_source: str,
    reference_source: str,
    input_specs: list[dict[str, Any]],
    kernel_name: str = "nki_kernel",
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Verify an NKI kernel against a PyTorch reference on real Trainium silicon."""
    return _post_reward(
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


@tool
def profile_nki_kernel(
    kernel_source: str,
    input_specs: list[dict[str, Any]],
    kernel_name: str = "nki_kernel",
    warmup_runs: int = 5,
    measure_runs: int = 20,
) -> dict[str, Any]:
    """Profile a compiled NKI kernel on Trainium. Returns latency stats and utilization."""
    return _post_reward(
        {
            "op": "profile",
            "kernel_source": kernel_source,
            "input_specs": input_specs,
            "kernel_name": kernel_name,
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
        }
    )


@tool
def lookup_nki_skill(
    query: str,
    top_k: int = 5,
    skill_category: str | None = None,
) -> dict[str, Any]:
    """Search the NKI skill DB for relevant patterns and recipes."""
    return _post_reward(
        {
            "op": "skill",
            "query": query,
            "top_k": top_k,
            "skill_category": skill_category,
        }
    )


_TOOLS = [compile_nki_kernel, verify_nki_kernel, profile_nki_kernel, lookup_nki_skill]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an NKI kernel optimization agent for AWS Trainium.

You convert PyTorch and Triton kernels into @nki.jit kernels that compile,
verify against a PyTorch reference on real Trainium silicon, and run faster
than the eager baseline.

Hard rules:
- The kernel function MUST be named `nki_kernel` and decorated with `@nki.jit`.
- The partition dim of every TILE MUST be exactly 128 (128 lanes).
- READ each input's shape first — inputs are most often 3-D (B, M, N) or
  4-D (N, C, H, W); only a minority are 2-D. Do NOT default to `M, N = x.shape`.
  For higher-rank inputs, reshape or stride-iterate the leading dims so each
  tile's leading axis is 128.
- EVERY output element must be written by an `nl.store`; leave no region unwritten
  (else "Output ... has no store def").
- Output via `result = nl.ndarray(<full output shape>, buffer=nl.shared_hbm)` and `return result`.
- `nl.relu / sigmoid / tanh / gelu / erf / sqrt / rsqrt / exp / log` DO exist — call them.
  `nl.pow` / `nl.exp2` do NOT — implement via `nl.exp(b*nl.log(a))` etc.

PRE-FLIGHT CHECKLIST — verify every item BEFORE your first compile_nki_kernel call:
  [ ] Input rank read from the shapes block and unpacked correctly (2-D/3-D/4-D) — never default to (M, N).
  [ ] Every TILE's partition dim (axis 0) is exactly 128; leading dims > 128 are tiled with an affine_range loop.
  [ ] Every element of `result` is written by some nl.store (no un-stored output regions).
  [ ] Reductions (nl.sum/max/min) pass axis=1 (the FREE dim), not axis=0.
  [ ] Only real ops used: no nl.pow / nl.exp2; no invented helper APIs.
  [ ] result = nl.ndarray(<full output shape>, dtype=..., buffer=nl.shared_hbm); return result.

HARDWARE ANTI-PATTERNS (the errors that kill L2/L3 compiles):
  - PSUM cannot be stored to HBM directly. Matmul writes PSUM -> copy PSUM->SBUF -> then nl.store.
  - Accumulate a matmul's K dimension in ONE PSUM region across an nl.affine_range K-loop; do not
    round-trip PSUM to HBM between K-steps.
  - PSUM free dim <= 512; SBUF free dim <= ~32K; matmul K <= 2048. Tile any dimension that exceeds these.
  - Reduction axis must be the free dim of a 2-D tile; transpose first if the axis you need is the partition.

Workflow per task:
  1. lookup_nki_skill FIRST — fetch the matching VERIFIED recipe (section 11) and adapt it.
  2. emit a candidate kernel (adapting the recipe).
  3. compile_nki_kernel — on failure, apply the diagnosis and fix ONLY that issue; do not rewrite or change imports.
  4. verify_nki_kernel — on a numerical mismatch, fix ONLY the numerics (reduction axis, scale, mask-before-softmax); keep the working structure.
  5. Once verify PASSES, STOP — do not keep editing a correct kernel. Then profile_nki_kernel.

Be terse. Surface errors verbatim. Do not invent NKI APIs.
"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def _build_agent(backend: BackendConfig, region: str) -> Agent:
    model = _build_model(backend, region)
    return Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=_TOOLS)


# ---------------------------------------------------------------------------
# Single-shot generation via raw Bedrock — deterministic, terminates in ~10s.
# Mirrors run_agent.py: model emits {nki_kernel, reference, input_specs} JSON,
# we run compile -> verify -> profile sequentially against the reward server.
# ---------------------------------------------------------------------------
SINGLE_SHOT_SYSTEM = """You are an expert NKI (Neuron Kernel Interface) kernel generator targeting AWS Trainium.

Output STRICT JSON with three string fields:
  - nki_kernel: a Python module containing exactly one function named `nki_kernel`,
    decorated with `@nki.jit`. Use `import neuronxcc.nki as nki`,
    `import neuronxcc.nki.language as nl`, and `import neuronxcc.nki.isa as nisa` if needed.
    Allocate the FULL output shape via `result = nl.ndarray(<full output shape>, dtype=x.dtype, buffer=nl.shared_hbm)`
    and `return result`. Every element of `result` MUST be written by an `nl.store(...)`
    somewhere — otherwise compilation fails with "Output ... has no store def".
  - reference: a Python module containing `def reference(*args) -> torch.Tensor` whose semantics
    match `nki_kernel`. May import torch.
  - input_specs: a list of {"shape": [...], "dtype": "float32"} entries, one per input.

### Input rank — NOT always 2-D
Read the task's `Input shapes:` block carefully and match the function signature
to the ACTUAL rank. NKIBench task inputs are most often 3-D `(B, M, N)` or
4-D `(N, C, H, W)`; only a small minority are 2-D. Do NOT default to
`M, N = x.shape` — read the spec.
  # Conv2d task — inputs are 4-D:        N, C_in, H, W = x.shape
  # Element-wise on a 3-D batched tensor: B, M, N = x.shape
  # Genuine 2-D (matmul, linear):         M, K = x.shape

### Tiling pattern
The PARTITION dim of every TILE (not necessarily of the input tensor) MUST be
exactly 128. For higher-rank inputs, tile by reshaping or stride-iterating the
leading dims so each tile's leading axis is 128:
    TILE_M = 128
    for i in nl.affine_range(M // TILE_M):
        ix = nl.mgrid[i * TILE_M : (i + 1) * TILE_M, 0 : N]
        tile = nl.load(x[ix])
        out = some_operation(tile)
        nl.store(result[ix], value=out)

### Available NKI primitives (verified on the deployed Neuron SDK)
`nl.add`, `nl.subtract`, `nl.multiply`, `nl.divide`, `nl.exp`, `nl.log`,
`nl.sqrt`, `nl.rsqrt`, `nl.reciprocal`, `nl.maximum`, `nl.minimum`, `nl.abs`,
`nl.negative`, `nl.max`, `nl.min`, `nl.sum`, and the activations
`nl.relu`, `nl.sigmoid`, `nl.tanh`, `nl.gelu`, `nl.erf` — these DO exist, call
them (do not hand-roll). NOT available: `nl.pow`, `nl.exp2` — implement via
`nl.exp(b * nl.log(a))` etc.

### Common failure modes to avoid
- Forgetting to `nl.store` part of the output (typical when you tile a leading
  dim and only store one tile slice).
- Writing `M, N = x.shape` on a 4-D conv input — see "Input rank" above.
- Reduction `axis` must be on the FREE dim (e.g. `axis=[1]` for a 2-D tile).

Output dtype matches input dtype. Do NOT include any extra prose. Output ONLY the JSON object.
"""


def _bedrock_single_shot(task: TaskSpec, model_id: str, region: str) -> dict[str, Any]:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    user_parts = [f"Task: {task.description}"]
    if task.reference_source:
        user_parts.append(
            f"PyTorch reference (semantics to match):\n```python\n{task.reference_source}\n```"
        )
    if task.input_specs:
        user_parts.append(
            f"Input shapes: {json.dumps(task.input_specs)}"
        )
    if task.nki_metadata:
        user_parts.append(f"NKI hints (engine + tile suggestions): {json.dumps(task.nki_metadata)}")
    user_parts.append("Output the JSON object now.")
    user_prompt = "\n\n".join(user_parts)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 6000,
        "system": SINGLE_SHOT_SYSTEM,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    r = client.invoke_model(modelId=model_id, body=json.dumps(body))
    payload = json.loads(r["body"].read())
    text = payload["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Handler — invoked by AgentCore (or a Lambda shim) on each request.
# ---------------------------------------------------------------------------
@app.entrypoint
def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Entry point.

    event:
      task_spec: { description, reference_source, input_specs[], constraints, model_pin }
      mode: "single_shot" (default — deterministic, ~30s) | "multi_turn" (Strands tool loop)
      max_turns: int — hard cap on tool calls in multi_turn mode
      session_id: optional
    """
    start = time.time()
    try:
        task_spec_raw = event.get("task_spec")
        if not task_spec_raw:
            return _error("missing task_spec", start)

        task = TaskSpec(
            description=task_spec_raw.get("description", ""),
            reference_source=task_spec_raw.get("reference_source", ""),
            input_specs=task_spec_raw.get("input_specs", []),
            operator_signature=task_spec_raw.get("operator_signature", ""),
            constraints=task_spec_raw.get("constraints", {}),
            model_pin=task_spec_raw.get("model_pin"),
            nki_metadata=task_spec_raw.get("nki_metadata", {}) or {},
        )

        backends = _load_backends()
        router = ModelRouter(backends=backends, skill_lookup=_skill_score)
        backend = router.resolve(task, recent_failures=[])
        region = os.environ.get("AWS_REGION", "us-east-1")

        mode = event.get("mode", "single_shot")
        if mode == "single_shot":
            return _run_single_shot(task, backend, region, start)
        return _run_multi_turn(task, backend, region, event, start)
    except Exception as exc:
        logger.error("agent invocation failed:\n%s", traceback.format_exc())
        return _error(f"{type(exc).__name__}: {exc}", start)


def _run_single_shot(
    task: TaskSpec, backend: BackendConfig, region: str, start: float
) -> dict[str, Any]:
    """Generate kernel + reference once, then compile/verify/profile."""
    model_id = backend.model_id or os.environ.get("OPUS_MODEL_ID", "us.anthropic.claude-opus-4-8")
    spec = _bedrock_single_shot(task, model_id, region)

    response: dict[str, Any] = {
        "status": "success",
        "mode": "single_shot",
        "model_used": backend.name,
        "model_id": model_id,
        "kernel_source": spec["nki_kernel"],
        "reference_source": spec["reference"],
        "input_specs": spec["input_specs"],
        "compile": None,
        "verify": None,
        "profile": None,
        "wall_time_s": None,
    }

    response["compile"] = _post_reward({
        "op": "compile",
        "kernel_source": spec["nki_kernel"],
        "input_specs": spec["input_specs"],
    })
    if not response["compile"].get("success"):
        response["status"] = "compile_failed"
        response["wall_time_s"] = round(time.time() - start, 2)
        return response

    constraints = task.constraints or {}
    response["verify"] = _post_reward({
        "op": "verify",
        "kernel_source": spec["nki_kernel"],
        "reference_source": spec["reference"],
        "input_specs": spec["input_specs"],
        "atol": float(constraints.get("atol", 1e-3)),
        "rtol": float(constraints.get("rtol", 1e-3)),
    })
    if not response["verify"].get("correct"):
        response["status"] = "verify_failed"
        response["wall_time_s"] = round(time.time() - start, 2)
        return response

    response["profile"] = _post_reward({
        "op": "profile",
        "kernel_source": spec["nki_kernel"],
        "input_specs": spec["input_specs"],
        "warmup_runs": 5,
        "measure_runs": 50,
    })
    response["wall_time_s"] = round(time.time() - start, 2)
    return response


# Bedrock Converse tool schema for the multi-turn NKIAgent loop. Mirrors the
# paper's eval (compile / verify / skill) — native tool use, executed against
# the Trn1 reward server inside the runtime (reachable over PrivateLink).
_CONVERSE_TOOLS = {
    "tools": [
        {"toolSpec": {
            "name": "compile_nki_kernel",
            "description": "Compile an NKI kernel on real Trainium hardware. Returns compilation success/failure with error details.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"kernel_code": {"type": "string", "description": "Complete NKI kernel source including imports, @nki.jit decorator, and the nki_kernel function."}},
                "required": ["kernel_code"],
            }},
        }},
        {"toolSpec": {
            "name": "verify_nki_kernel",
            "description": "Compile AND verify an NKI kernel against the PyTorch reference on real Trainium. Returns correctness within tolerance and error magnitudes.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"kernel_code": {"type": "string", "description": "Complete NKI kernel source code to verify."}},
                "required": ["kernel_code"],
            }},
        }},
        {"toolSpec": {
            "name": "skill_lookup",
            "description": "Look up NKI programming patterns and best practices before writing code.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "Topic, e.g. 'softmax', 'matmul', 'tile constraints', 'memory hierarchy'."}},
                "required": ["topic"],
            }},
        }},
    ]
}


# Compiler-error decoder — maps raw neuronx-cc error text to a one-line, actionable
# diagnosis, injected at failure time so the fix loop repairs the specific issue
# instead of regenerating (the technique kernel-forge found most effective).
_ERROR_DECODER = [
    ("has no store def", "An output region is never written — nl.store EVERY tile of `result` (iterate ALL leading batch/head/channel dims)."),
    ("psum", "PSUM misuse — allocate a fresh nl.zeros(..., buffer=nl.psum) per matmul, accumulate the K-loop in ONE psum region, copy PSUM->SBUF before nl.store; never store PSUM->HBM directly."),
    ("not enough values to unpack", "Wrong rank unpack — read the Inputs block: unpack N,C,H,W (4-D) or B,M,N (3-D), do NOT default to M,N."),
    ("too many values to unpack", "Wrong rank unpack — the input has fewer dims than you unpacked; read the Inputs block."),
    ("non-affine", "Non-affine index — index as loop_var*TILE + const; no data-dependent indexing."),
    ("partition dimension", "Partition (axis-0) of every TILE must be exactly 128 — tile leading dims by 128 with nl.affine_range."),
    ("sbuf", "Tile too large for SBUF (~24MB) — shrink the free dimension TILE_F and re-tile."),
    ("unsupported", "Op does not exist in NKI — no nl.pow/nl.exp2; use nl.exp(b*nl.log(a)); activations nl.relu/sigmoid/tanh/gelu DO exist."),
    ("dtype", "Reduction dtype — cast bf16 inputs to fp32 for the reduction, then back."),
    ("statically known", "Loop bound must be a compile-time constant — derive it from x.shape, not a runtime value."),
]


def _diagnose_compile(errors: list) -> str:
    blob = "\n".join(str(e) for e in errors).lower()
    for needle, diag in _ERROR_DECODER:
        if needle in blob:
            return diag
    return ""


def _execute_converse_tool(
    tool_name: str, tool_input: dict[str, Any], task: TaskSpec
) -> dict[str, Any]:
    """Execute a Converse tool call against the reward server (op-based API)."""
    if tool_name == "compile_nki_kernel":
        out = _post_reward({
            "op": "compile",
            "kernel_source": tool_input.get("kernel_code", ""),
            "kernel_name": "nki_kernel",
            # Compile against the task's REAL input shapes. Without this the
            # reward server falls back to a dummy (128,256), so a 3-D/4-D L2/L3
            # kernel is compiled against wrong-shaped inputs — passing/failing
            # for the wrong reason and diverging from what verify then sees.
            "input_specs": task.input_specs,
        })
        ok = bool(out.get("success"))
        errs = out.get("errors") or []
        if ok:
            msg = "Compiled successfully on Trainium."
        else:
            diag = _diagnose_compile(errs)
            head = f"Compilation FAILED. Diagnosis: {diag}\n" if diag else "Compilation FAILED.\n"
            msg = (head + "Fix ONLY this specific issue; do not rewrite the rest of the kernel "
                   "or change the imports (local repair, not regeneration).\nCompiler error:\n"
                   + "\n".join(str(e) for e in errs[:3])[:1500])
        return {"compiled": ok, "message": msg, **out}
    if tool_name == "verify_nki_kernel":
        out = _post_reward({
            "op": "verify",
            "kernel_source": tool_input.get("kernel_code", ""),
            "reference_source": task.reference_source,
            "input_specs": task.input_specs,
            "kernel_name": "nki_kernel",
            "atol": (task.constraints or {}).get("atol", 1e-3),
            "rtol": (task.constraints or {}).get("rtol", 1e-3),
        })
        correct = bool(out.get("correct"))
        # A populated `error` means it did not compile; else it compiled and we
        # got numerical metrics back (correct or a real mismatch).
        compiled = out.get("error") is None
        if correct:
            msg = "CORRECT — output matches the PyTorch reference within tolerance. You are done."
        elif not compiled:
            msg = ("Did NOT compile. Fix ONLY the specific error below:\n"
                   + str(out.get("error"))[:1500])
        else:
            mae = out.get("max_abs_error")
            mm = out.get("mismatched_elements")
            tot = out.get("total_elements")
            msg = (f"Compiled but NUMERICALLY WRONG: max_abs_error={mae}, "
                   f"mismatched {mm}/{tot} elements (tol atol={out.get('atol')}, rtol={out.get('rtol')}). "
                   "The tiling/algorithm is off — check reduction axes, tile/store coverage of the FULL "
                   "output, and that leading (batch/channel) dims are all iterated. Do NOT change the imports.")
        return {"correct": correct, "compiled": compiled, "message": msg, **out}
    if tool_name == "skill_lookup":
        return _post_reward({"op": "skill", "query": tool_input.get("topic", ""), "top_k": 5})
    return {"error": f"unknown tool: {tool_name}"}


def _run_multi_turn(
    task: TaskSpec, backend: BackendConfig, region: str, event: dict[str, Any], start: float
) -> dict[str, Any]:
    """Native Bedrock Converse tool loop — the paper's NKIAgent approach.

    The model iterates generate -> compile -> verify -> fix for up to max_turns,
    calling compile/verify tools that run on the Trn1 reward server. Every tool
    result (including compiler errors and numerical divergence) is fed back so
    the model can fix the specific failure. This is what turns single-shot's low
    pass rate into the multi-turn headline rate.
    """
    import boto3

    max_turns = int(event.get("max_turns", 10))
    model_id = backend.model_id or backend.model_arn or os.environ.get(
        "OPUS_MODEL_ID", "us.anthropic.claude-opus-4-8"
    )
    client = boto3.client("bedrock-runtime", region_name=region)

    prompt = _format_prompt(task, max_turns)
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": prompt}]}]

    best_code = ""
    last_code = ""
    best_verify: dict[str, Any] | None = None
    last_compile: dict[str, Any] | None = None
    compiled = False
    correct = False
    turns_used = 0
    tool_calls = 0

    for turn in range(max_turns):
        turns_used = turn + 1
        try:
            resp = client.converse(
                modelId=model_id,
                messages=messages,
                system=[{"text": SYSTEM_PROMPT}],
                toolConfig=_CONVERSE_TOOLS,
                inferenceConfig={"maxTokens": 8000},
            )
        except Exception as exc:  # noqa: BLE001 — surface, keep looping/refresh
            err = str(exc)
            logger.warning("multi_turn converse error (turn %d): %s", turn + 1, err[:200])
            if "ExpiredToken" in err or "InvalidSignature" in err:
                client = boto3.client("bedrock-runtime", region_name=region)
                continue
            break

        out_msg = resp.get("output", {}).get("message", {})
        messages.append(out_msg)
        stop = resp.get("stopReason", "end_turn")

        if stop == "tool_use":
            tool_results = []
            for block in out_msg.get("content", []):
                tu = block.get("toolUse")
                if not tu:
                    continue
                tool_calls += 1
                name = tu.get("name", "")
                tinput = tu.get("input", {})
                result = _execute_converse_tool(name, tinput, task)
                if name in ("compile_nki_kernel", "verify_nki_kernel"):
                    code = tinput.get("kernel_code", "")
                    if code:
                        last_code = code
                    if result.get("compiled"):
                        compiled = True
                        best_code = code or best_code
                        last_compile = result
                    if name == "verify_nki_kernel" and result.get("correct"):
                        correct = True
                        best_code = code or best_code
                        best_verify = result
                tool_results.append({"toolResult": {
                    "toolUseId": tu.get("toolUseId", ""),
                    "content": [{"json": result}],
                }})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            if correct:  # solved — stop early
                break
        else:
            # Model ended its turn: extract any final kernel and check it once.
            for block in out_msg.get("content", []):
                text = block.get("text", "")
                if text and "@nki.jit" in text:
                    code = _extract_code_block(text)
                    if code:
                        last_code = code
                        v = _execute_converse_tool("verify_nki_kernel", {"kernel_code": code}, task)
                        if v.get("compiled"):
                            compiled = True
                            best_code = code
                        if v.get("correct"):
                            correct = True
                            best_verify = v
            break

    final_code = best_code or last_code
    profiling = None
    if correct and final_code:
        try:
            profiling = _post_reward({
                "op": "profile", "kernel_source": final_code,
                "input_specs": task.input_specs, "kernel_name": "nki_kernel",
                "warmup_runs": 5, "measure_runs": 50,
            })
        except Exception:  # noqa: BLE001 — profiling is best-effort
            profiling = None

    return {
        "status": "success" if correct else ("compiled" if compiled else "failed"),
        "mode": "multi_turn",
        "model_used": backend.name,
        "model_id": model_id,
        "kernel_source": final_code or None,
        "compile": last_compile,
        "verify": best_verify,
        "profile": profiling,
        "compiled": compiled,
        "correct": correct,
        "turns_used": turns_used,
        "tool_calls_used": tool_calls,
        "wall_time_s": round(time.time() - start, 2),
    }


def _extract_code_block(text: str) -> str:
    """Pull the last ```python fenced block (or a bare @nki.jit tail) from text."""
    import re

    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    for block in reversed(blocks):
        if "@nki.jit" in block:
            return block.strip()
    return ""


def _error(msg: str, start: float) -> dict[str, Any]:
    return {
        "status": "error",
        "kernel_source": None,
        "verification": None,
        "profiling": None,
        "turns_used": 0,
        "trajectory": [],
        "error": msg,
        "wall_time_s": round(time.time() - start, 2),
    }


def _format_prompt(task: TaskSpec, max_turns: int) -> str:
    parts = [f"## Task\n{task.description}"]
    if task.reference_source:
        parts.append(f"## Reference\n```python\n{task.reference_source}\n```")
    if task.input_specs:
        parts.append(
            f"## Inputs\n```json\n{json.dumps(task.input_specs, indent=2)}\n```"
        )
    if task.constraints:
        parts.append(
            f"## Constraints\n```json\n{json.dumps(task.constraints, indent=2)}\n```"
        )
    if task.nki_metadata:
        parts.append(
            "## NKI hints (engine + tile suggestions for this task)\n"
            f"```json\n{json.dumps(task.nki_metadata, indent=2)}\n```"
        )
    parts.append(f"## Budget\nMax turns: {max_turns}")
    parts.append(
        "## Instructions\n"
        "STEP 1 (MANDATORY, do this FIRST before writing any code): call skill_lookup with the "
        "operator/pattern name (e.g. 'softmax', 'attention scaled dot product', 'rmsnorm', "
        "'matmul tiling', 'layer normalization'). The skill DB contains VERIFIED, copy-paste-correct "
        "2.23 kernels (section 11) — adapt the matching recipe rather than inventing structure. "
        "Most numerical failures come from getting softmax/reduction details subtly wrong, and the "
        "verified recipe fixes exactly that.\n"
        "STEP 2: write the kernel (adapting the recipe), call compile_nki_kernel, then verify_nki_kernel.\n"
        "STEP 3 (fix loop): On a compile error, quote the exact error and change ONLY what it names — "
        "do not rewrite or change the imports. On a NUMERICAL mismatch, the structure compiled but the "
        "math is wrong: re-check the softmax max-subtraction, that reductions use axis=1 on the free "
        "dimension, the 1/sqrt(D) scale placement, mask-before-softmax, and that every output element "
        "(across ALL batch/head/channel dims) is written by an nl.store. If unsure, skill_lookup again."
    )
    return "\n\n".join(parts)


def _parse_result(result: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "kernel_source": None,
        "verification": None,
        "profiling": None,
        "turns_used": 0,
        "trajectory": [],
    }
    try:
        messages = getattr(result, "messages", [])
        turns = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                turns += 1
            for tr in msg.get("tool_results", []) or []:
                tool_name = tr.get("tool_name", "")
                output = tr.get("output", {})
                if tool_name == "verify_nki_kernel" and isinstance(output, dict):
                    response["verification"] = output
                elif tool_name == "profile_nki_kernel" and isinstance(output, dict):
                    response["profiling"] = output
            content = msg.get("content", "")
            if isinstance(content, str) and "@nki.jit" in content:
                import re

                blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
                for block in reversed(blocks):
                    if "@nki.jit" in block or "nki_kernel" in block:
                        response["kernel_source"] = block.strip()
                        break
        response["turns_used"] = turns
    except Exception as e:
        logger.warning("failed to parse agent result: %s", e)
    return response


if __name__ == "__main__":
    app.run()
