#!/usr/bin/env python3
"""
NKIGen-Bench evaluation in NKIAgent mode (full agentic workflow with tools).

This is the paper's actual evaluation approach: the model has access to tools
(compile_nki_kernel, verify_nki_kernel, profile_nki_kernel, skill_lookup)
and can call them strategically.

Supports:
  - Claude via Bedrock Converse API (native tool use)
  - vLLM models (simulated tool calls — extract code, compile, feed back results)

Usage:
  # Claude Sonnet NKIAgent mode
  python3 scripts/eval_nkibench_agentcore.py --backend claude \
    --model-id us.anthropic.claude-sonnet-4-20250514-v1:0 --num-tasks 10

  # Claude Opus NKIAgent mode (full 250 tasks)
  python3 scripts/eval_nkibench_agentcore.py --backend claude \
    --model-id us.anthropic.claude-opus-4-5-20251101-v1:0 --num-tasks 0

  # vLLM base model NKIAgent mode
  python3 scripts/eval_nkibench_agentcore.py --backend vllm \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --num-tasks 10
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nkibench import NKIBenchTask, load_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
sys.stdout.reconfigure(line_buffering=True)
logger = logging.getLogger("nkibench-nkiagent")


# ---------------------------------------------------------------------------
# Tool definitions for the NKIAgent
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "compile_nki_kernel",
        "description": "Compile an NKI kernel on real Trainium hardware. Returns compilation success/failure with error details.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "kernel_code": {
                        "type": "string",
                        "description": "Complete NKI kernel source code including imports, @nki.jit decorator, and function definition."
                    }
                },
                "required": ["kernel_code"]
            }
        }
    },
    {
        "name": "verify_nki_kernel",
        "description": "Compile and verify an NKI kernel against the PyTorch reference implementation. Returns whether the kernel produces correct output within tolerance.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "kernel_code": {
                        "type": "string",
                        "description": "Complete NKI kernel source code to verify."
                    }
                },
                "required": ["kernel_code"]
            }
        }
    },
    {
        "name": "skill_lookup",
        "description": "Look up NKI programming documentation, patterns, and best practices. Use this to learn about NKI APIs before writing code.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to look up (e.g., 'softmax', 'matmul', 'tile constraints', 'memory hierarchy')."
                    }
                },
                "required": ["topic"]
            }
        }
    },
]

SYSTEM_PROMPT = """\
You are an expert NKI (Neuron Kernel Interface) kernel developer for AWS Trainium.

You have access to tools to compile and verify NKI kernels on real hardware:
- compile_nki_kernel: Compile an NKI kernel (fast, checks syntax and hardware constraints)
- verify_nki_kernel: Compile AND verify correctness against PyTorch reference
- skill_lookup: Look up NKI documentation and patterns

Strategy:
1. Read the task and understand what PyTorch operation to implement
2. Optionally use skill_lookup to review relevant NKI patterns
3. Write the NKI kernel
4. Use compile_nki_kernel to check for errors
5. If errors, fix and retry
6. Once compiling, use verify_nki_kernel to check correctness

Key NKI rules:
- Partition dimension (axis 0) MUST be exactly 128 (TILE_M = 128)
- Import: neuronxcc.nki as nki, neuronxcc.nki.language as nl
- Decorator: @nki.jit
- Function name: nki_kernel
- Return result via nl.ndarray(..., buffer=nl.shared_hbm) + return result
- Available ops: nl.add, nl.subtract, nl.multiply, nl.divide, nl.exp, nl.log, nl.rsqrt, nl.reciprocal, nl.maximum, nl.minimum, nl.abs, nl.negative, nl.max, nl.min, nl.sum
- CRITICAL: nl.sigmoid/relu/tanh/gelu/sqrt/pow do NOT exist"""

TASK_PROMPT_TEMPLATE = """\
Write an NKI kernel equivalent to this PyTorch model:

```python
{model_code}
```

Inputs: `{inputs_code}`
{nki_metadata}

Start by writing the kernel, then use compile_nki_kernel to check it, and fix any errors."""


# ---------------------------------------------------------------------------
# Tool execution via reward server
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, tool_input: dict, reward_url: str,
                 shapes: list, dtypes: list) -> dict:
    """Execute a tool call via the reward server."""
    if tool_name == "compile_nki_kernel":
        code = tool_input.get("kernel_code", "")
        r = _check_reward(code, shapes, dtypes, reward_url)
        reward = r.get("reward", -1.0)
        breakdown = r.get("breakdown", {})
        if reward >= 0:
            return {"status": "success", "compiled": True, "message": "Kernel compiled successfully on Trainium."}
        errors = breakdown.get("errors", [])
        error_stage = breakdown.get("error_stage", "unknown")
        return {
            "status": "error", "compiled": False,
            "error_stage": error_stage,
            "errors": errors[:3],
            "message": f"Compilation failed at stage: {error_stage}. " + (" ".join(errors[:2])[:500] if errors else "")
        }

    elif tool_name == "verify_nki_kernel":
        code = tool_input.get("kernel_code", "")
        r = _check_reward(code, shapes, dtypes, reward_url)
        reward = r.get("reward", -1.0)
        breakdown = r.get("breakdown", {})
        if reward > 0:
            return {"status": "success", "correct": True, "message": "Kernel is correct! Output matches PyTorch reference."}
        elif reward >= 0:
            return {"status": "success", "correct": False, "compiled": True, "message": "Kernel compiled but output does not match reference."}
        errors = breakdown.get("errors", [])
        error_stage = breakdown.get("error_stage", "unknown")
        return {
            "status": "error", "correct": False, "compiled": False,
            "error_stage": error_stage,
            "errors": errors[:3],
            "message": f"Compilation failed: {error_stage}. " + (" ".join(errors[:2])[:500] if errors else "")
        }

    elif tool_name == "skill_lookup":
        topic = tool_input.get("topic", "")
        return _skill_lookup(topic)

    return {"status": "error", "message": f"Unknown tool: {tool_name}"}


def _check_reward(code: str, shapes: list, dtypes: list, reward_url: str) -> dict:
    if not reward_url.lower().startswith(("http://", "https://")):
        raise ValueError(f"refusing non-http(s) reward-server URL: {reward_url}")
    payload = {"code": code, "input_shapes": shapes, "expected_output_shapes": [], "dtypes": dtypes}
    try:
        req = urllib.request.Request(
            f"{reward_url}/reward", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310 — reward_url scheme validated above
            return json.loads(resp.read())
    except Exception as e:
        return {"reward": -1.0, "breakdown": {"errors": [str(e)], "error_stage": "network"}}


def _skill_lookup(topic: str) -> dict:
    """Return NKI documentation for a topic."""
    docs = {
        "tile constraints": "Partition dimension (axis 0) must be exactly 128. Free dimension (axis 1) is flexible. Use TILE_M=128.",
        "memory hierarchy": "HBM (GB) for inputs/outputs via nl.load/nl.store. SBUF (24MB) for working memory. PSUM (4MB) for matmul accumulation.",
        "matmul": "Use nisa.nc_matmul(a_tile, b_tile) with tiles in SBUF. Result goes to PSUM. Copy to SBUF before VectorE ops.",
        "softmax": "3-pass: (1) max = nl.max(x, axis=[1]), (2) exp_sum = nl.add(nl.exp(x - max), axis=[1]), (3) result = nl.exp(x - max) / exp_sum",
        "activations": "relu: nl.maximum(x, zero). sigmoid: 1/(1+nl.exp(-x)). tanh: 2*sigmoid(2x)-1. gelu: x*0.5*(1+erf(x/sqrt(2))) approx x*sigmoid(1.702*x)",
        "reductions": "nl.max(x, axis=[1]), nl.min(x, axis=[1]), nl.add(x, axis=[1]) for sum. Reduction only along free dim (axis 1).",
        "loop constructs": "nl.affine_range(N) for independent iterations (unrolled, pipelined). nl.sequential_range(N) for dependent iterations.",
        "common patterns": "Elementwise: load tile, compute, store. Matmul: outer M,N loops, inner K accumulation in PSUM. Norm: 2-pass statistics then normalize.",
    }
    content = docs.get(topic.lower(), "")
    if not content:
        for key, val in docs.items():
            if topic.lower() in key:
                content = val
                break
    if not content:
        content = "\n".join(f"- {k}: {v[:80]}..." for k, v in docs.items())
    return {"status": "success", "topic": topic, "content": content}


# ---------------------------------------------------------------------------
# Claude NKIAgent evaluation (native tool use)
# ---------------------------------------------------------------------------

def eval_claude_nkiagent(
    task: NKIBenchTask,
    client,
    model_id: str,
    reward_url: str,
    max_turns: int,
    shapes: list,
    dtypes: list,
) -> dict:
    """Evaluate a task using Claude with native tool use."""
    t0 = time.time()
    meta = json.dumps(task.nki_metadata, indent=2) if task.nki_metadata else ""
    meta_str = f"\nHints: {meta}" if meta else ""
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        model_code=task.model_class_code,
        inputs_code=task.get_inputs_code.replace("\n", "; "),
        nki_metadata=meta_str,
    )

    messages = [{"role": "user", "content": [{"text": task_prompt}]}]
    best_reward = -1.0
    best_code = ""
    compiled = False
    correct = False
    turns_used = 0
    tool_calls_made = 0

    # Bedrock Converse API tool format
    tool_config = {"tools": [
        {"toolSpec": {
            "name": t["name"],
            "description": t["description"],
            "inputSchema": t["inputSchema"],
        }} for t in TOOL_DEFINITIONS
    ]}

    for turn in range(max_turns):
        turns_used = turn + 1
        try:
            response = client.converse(
                modelId=model_id,
                messages=messages,
                system=[{"text": SYSTEM_PROMPT}],
                toolConfig=tool_config,
                inferenceConfig={"maxTokens": 4096} if "opus-4-8" in model_id else {"maxTokens": 4096, "temperature": 0.0},
            )
        except Exception as e:
            logger.warning("  Turn %d: API error: %s", turn + 1, str(e)[:100])
            time.sleep(5)
            continue

        stop_reason = response.get("stopReason", "end_turn")
        output_msg = response.get("output", {}).get("message", {})
        messages.append(output_msg)

        if stop_reason == "tool_use":
            # Process tool calls
            tool_results = []
            for block in output_msg.get("content", []):
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_name = tool_use.get("name", "")
                    tool_input = tool_use.get("input", {})
                    tool_use_id = tool_use.get("toolUseId", "")
                    tool_calls_made += 1

                    logger.debug("  Tool: %s", tool_name)
                    result = execute_tool(tool_name, tool_input, reward_url, shapes, dtypes)

                    # Track best code
                    if tool_name in ("compile_nki_kernel", "verify_nki_kernel"):
                        code = tool_input.get("kernel_code", "")
                        r = _check_reward(code, shapes, dtypes, reward_url)
                        reward = r.get("reward", -1.0)
                        if reward > best_reward:
                            best_reward = reward
                            best_code = code
                        if reward >= 0:
                            compiled = True
                        if reward > 0:
                            correct = True

                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"json": result}],
                        }
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break
        else:
            # End turn — extract any code from the final response
            for block in output_msg.get("content", []):
                if "text" in block:
                    code = _extract_code(block["text"])
                    if code:
                        r = _check_reward(code, shapes, dtypes, reward_url)
                        reward = r.get("reward", -1.0)
                        if reward > best_reward:
                            best_reward = reward
                            best_code = code
                        if reward >= 0:
                            compiled = True
                        if reward > 0:
                            correct = True
            break

    return {
        "task_id": task.task_id,
        "name": task.name,
        "level": task.level,
        "category": task.category,
        "compiled": compiled,
        "correct": correct,
        "reward": best_reward,
        "turns_used": turns_used,
        "tool_calls": tool_calls_made,
        "code_preview": best_code[:300],
        "time": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# vLLM NKIAgent evaluation (native tool calling via OpenAI API)
# ---------------------------------------------------------------------------

# OpenAI-format tool definitions for vLLM
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["inputSchema"]["json"],
        }
    }
    for t in TOOL_DEFINITIONS
]


def eval_vllm_nkiagent(
    task: NKIBenchTask,
    vllm_url: str,
    model: str,
    reward_url: str,
    max_turns: int,
    shapes: list,
    dtypes: list,
) -> dict:
    """Evaluate using vLLM with native tool calling (requires --enable-auto-tool-choice).

    The model decides which tools to call via the OpenAI tool calling API.
    vLLM parses the model's tool call tokens and returns structured tool_calls.
    """
    t0 = time.time()
    meta = json.dumps(task.nki_metadata, indent=2) if task.nki_metadata else ""
    meta_str = f"\nHints: {meta}" if meta else ""
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        model_code=task.model_class_code,
        inputs_code=task.get_inputs_code.replace("\n", "; "),
        nki_metadata=meta_str,
    )

    best_reward = -1.0
    best_code = ""
    compiled = False
    correct = False
    turns_used = 0
    tool_calls_made = 0
    prev_error_stage = None
    same_error_count = 0

    if not vllm_url.lower().startswith(("http://", "https://")):
        raise ValueError(f"refusing non-http(s) vllm URL: {vllm_url}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    for turn in range(max_turns):
        turns_used = turn + 1
        turn_temp = 0.0 if turn == 0 else 0.3

        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": turn_temp,
                "tools": OPENAI_TOOLS,
                "tool_choice": "auto",
            }
            req = urllib.request.Request(
                f"{vllm_url}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:  # nosec B310 — vllm_url scheme validated above
                resp_data = json.loads(resp.read())
            choice = resp_data["choices"][0]
            msg = choice["message"]
        except Exception as e:
            logger.warning("  Turn %d: generation error: %s", turn + 1, str(e)[:80])
            continue

        # Append assistant message to conversation
        messages.append(msg)

        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            # Process each tool call
            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    tool_input = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_input = {}
                tool_call_id = tc.get("id", "")
                tool_calls_made += 1

                logger.debug("  Tool: %s", tool_name)
                result = execute_tool(tool_name, tool_input, reward_url, shapes, dtypes)

                # Track best code
                if tool_name in ("compile_nki_kernel", "verify_nki_kernel"):
                    code = tool_input.get("kernel_code", "")
                    r = _check_reward(code, shapes, dtypes, reward_url)
                    reward = r.get("reward", -1.0)
                    if reward > best_reward:
                        best_reward = reward
                        best_code = code
                    if reward >= 0:
                        compiled = True
                    if reward > 0:
                        correct = True

                # Feed tool result back
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result),
                })

            if compiled:
                logger.info("  Turn %d: SUCCESS reward=%.2f (tools=%d)", turn + 1, best_reward, tool_calls_made)
                break

            # Check for repeated errors
            last_error_stage = None
            for tc in tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") in ("compile_nki_kernel", "verify_nki_kernel"):
                    code = json.loads(fn.get("arguments", "{}")).get("kernel_code", "")
                    r = _check_reward(code, shapes, dtypes, reward_url)
                    bd = r.get("breakdown", {})
                    last_error_stage = bd.get("error_stage")

            if last_error_stage:
                logger.info("  Turn %d: FAIL stage=%s (tools=%d)", turn + 1, last_error_stage, tool_calls_made)
                if last_error_stage == prev_error_stage:
                    same_error_count += 1
                    if same_error_count >= 2:
                        logger.info("  Stopping: same error repeated")
                        break
                else:
                    same_error_count = 0
                prev_error_stage = last_error_stage
        else:
            # No tool calls — model generated text. Extract code and auto-compile.
            content = msg.get("content", "")
            code = _extract_code(content) if content else ""
            if code:
                tool_calls_made += 1
                r = _check_reward(code, shapes, dtypes, reward_url)
                reward = r.get("reward", -1.0)
                if reward > best_reward:
                    best_reward = reward
                    best_code = code
                if reward >= 0:
                    compiled = True
                if reward > 0:
                    correct = True
                if compiled:
                    logger.info("  Turn %d: SUCCESS reward=%.2f (tools=%d)", turn + 1, reward, tool_calls_made)
                    break
                error_stage = r.get("breakdown", {}).get("error_stage", "unknown")
                logger.info("  Turn %d: FAIL stage=%s (text, tools=%d)", turn + 1, error_stage, tool_calls_made)
                # Feed error back as user message for next turn
                result = execute_tool("compile_nki_kernel", {"kernel_code": code}, reward_url, shapes, dtypes)
                messages.append({"role": "user", "content": f"Your code was auto-compiled. Result:\n{json.dumps(result)[:1000]}\n\nPlease fix and use compile_nki_kernel tool."})
            else:
                logger.info("  Turn %d: no code or tool calls", turn + 1)
                messages.append({"role": "user", "content": "Please write the NKI kernel and use the compile_nki_kernel tool to check it."})

    return {
        "task_id": task.task_id,
        "name": task.name,
        "level": task.level,
        "category": task.category,
        "compiled": compiled,
        "correct": correct,
        "reward": best_reward,
        "turns_used": turns_used,
        "tool_calls": tool_calls_made,
        "code_preview": best_code[:300],
        "time": time.time() - t0,
    }


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(("import ", "from ", "@nki", "def ")):
            in_code = True
        if in_code:
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else ""


def extract_shapes_from_task(task: NKIBenchTask) -> tuple:
    import torch
    namespace = {"torch": torch}
    try:
        exec(task.get_inputs_code, namespace)  # nosec B102 — task.get_inputs_code is from the static in-repo nkibench registry (nkibench/level{1,2,3}.py string literals), not runtime/attacker input
        get_inputs = namespace.get("get_inputs")
        if get_inputs:
            inputs = get_inputs()
            return [list(t.shape) for t in inputs], [str(t.dtype).replace("torch.", "") for t in inputs]
    except Exception as e:
        logger.debug("extract_shapes_from_task: get_inputs_code failed for %s, falling back to default shape: %s", task.task_id, str(e)[:200])
    return [[128, 128]], ["float32"]


# ---------------------------------------------------------------------------
# AgentCore Runtime backend (deployed endpoint)
# ---------------------------------------------------------------------------

def _build_reference_source(task: NKIBenchTask) -> str:
    """Wrap the task's PyTorch `Model` class into a `reference(*args)` function.

    The reward server's verify runner requires the reference module to define
    `reference(*args)` (it feeds torch tensors positionally). NKIBench ships the
    reference as a `class Model(nn.Module)`, so we append a thin adapter that
    instantiates it and calls forward. Without this, multi_turn verify always
    fails with 'reference defines no reference() function' even for kernels that
    are numerically correct.
    """
    # Seed before constructing Model so any default-random weights are at least
    # reproducible run-to-run. NOTE: tasks whose Model owns *learned* weights
    # (Conv/Linear/Embedding) still cannot be matched from inputs alone — the
    # kernel never receives those weights — so they are inherently unverifiable
    # and are tagged separately (see _has_learned_params).
    return (
        f"{task.model_class_code}\n\n"
        "def reference(*args):\n"
        "    import torch\n"
        "    torch.manual_seed(0)\n"
        "    _m = Model()\n"
        "    _m.eval()\n"
        "    with torch.no_grad():\n"
        "        return _m(*args)\n"
    )


def _weight_specs_and_reference(task: NKIBenchTask):
    """Option B: make a learned-weight task verifiable by passing weights as inputs.

    Instantiate Model() with a fixed seed, read its parameters+buffers in
    state_dict order, and return (weight_input_specs, reference_source) such that:
      - the kernel/reference receive (x..., w0, w1, ...) positionally, and
      - reference() loads those exact passed tensors into a fresh Model, then runs
        forward — so kernel and reference agree on the weights.

    Returns (None, None) if the model can't be introspected (caller falls back to
    the plain, weight-hidden reference and keeps the unverifiable tag).
    """
    import torch  # local import; only needed when building weighted refs

    try:
        ns: dict = {}
        exec(task.model_class_code, ns)  # nosec B102 — task.model_class_code is from the static in-repo nkibench registry (nkibench/level{1,2,3}.py string literals), not runtime/attacker input
        Model = ns.get("Model")
        if Model is None:
            return None, None
        torch.manual_seed(0)
        m = Model().eval()
        sd = m.state_dict()  # ordered dict, deterministic under the seed
        names = list(sd.keys())
        if not names:
            return None, None
        weight_specs = []
        for k in names:
            t = sd[k]
            weight_specs.append({"shape": list(t.shape), "dtype": "float32"})
    except Exception:
        return None, None

    # reference(x..., *weights): rebuild Model, overwrite its params with the
    # passed tensors (in state_dict order), then forward the activation inputs.
    n_act = len(task.input_specs) if getattr(task, "input_specs", None) else 1
    names_repr = repr(names)
    ref = (
        f"{task.model_class_code}\n\n"
        f"_PARAM_NAMES = {names_repr}\n"
        f"_N_ACT = {n_act}\n"
        "def reference(*args):\n"
        "    import torch\n"
        "    torch.manual_seed(0)\n"
        "    _m = Model().eval()\n"
        "    acts = args[:_N_ACT]\n"
        "    weights = args[_N_ACT:]\n"
        "    sd = _m.state_dict()\n"
        "    for name, w in zip(_PARAM_NAMES, weights):\n"
        "        sd[name] = w.reshape(sd[name].shape).to(sd[name].dtype)\n"
        "    _m.load_state_dict(sd)\n"
        "    with torch.no_grad():\n"
        "        return _m(*acts)\n"
    )
    return weight_specs, ref


def _has_learned_params(task: NKIBenchTask) -> bool:
    """True if the task's Model owns weights the kernel can't see from inputs.

    LayerNorm/BatchNorm default to affine identity (weight=1,bias=0) so they are
    still matchable; Conv/Linear/Embedding carry random learned weights and are
    not verifiable by input-only comparison. Used to tag honest failures.
    """
    import re
    return bool(re.search(r"nn\.(Conv\d?d|Linear|Embedding|Bilinear)\(", task.model_class_code))


def eval_agentcore_nkiagent(
    task: NKIBenchTask,
    runtime_arn: str,
    region: str,
    shapes: list,
    dtypes: list,
    qualifier: str = "DEFAULT",
    timeout_s: int = 300,
    mode: str = "single_shot",
    max_turns: int = 10,
    pass_weights: bool = False,
) -> dict:
    """Evaluate a task by invoking the deployed AgentCore Runtime.

    The runtime container handles the model call + compile/verify/profile against
    the Trn1 reward server, so this driver does NO direct reward-server I/O. In
    `single_shot` mode the runtime does one generate+compile+verify pass; in
    `multi_turn` mode it runs the paper's compile-verify-fix tool loop (up to
    `max_turns`), which is what reaches the multi-turn headline pass rate.
    """
    import boto3
    from botocore.config import Config
    from uuid import uuid4

    t0 = time.time()
    config = Config(read_timeout=timeout_s, connect_timeout=30, retries={"max_attempts": 1})
    # Fresh Session per call so creds rewritten on disk by `ada credentials update`
    # are picked up — the module-level default Session would cache stale creds and
    # eventually raise AccessDeniedException once the STS token expires.
    client = boto3.Session().client("bedrock-agentcore", region_name=region, config=config)

    act_specs = [{"shape": list(s), "dtype": d} for s, d in zip(shapes, dtypes)]
    reference_source = _build_reference_source(task)
    input_specs = act_specs
    weights_note = ""
    made_verifiable = False

    # Option B: for learned-weight models, pass the weights as extra kernel inputs
    # so the task becomes genuinely verifiable (kernel + reference get identical
    # weights). Only kicks in with --pass-weights AND when the model has weights.
    if pass_weights and _has_learned_params(task):
        wspecs, wref = _weight_specs_and_reference(task)
        if wspecs and wref:
            input_specs = act_specs + wspecs
            reference_source = wref
            n_act = len(act_specs)
            weights_note = (
                f"\n\nIMPORTANT: nki_kernel receives {len(input_specs)} inputs positionally: "
                f"the first {n_act} are the activation tensor(s); the remaining {len(wspecs)} "
                f"are the model's learned weights/buffers in state_dict order "
                f"(shapes {[w['shape'] for w in wspecs]}). Use them — do NOT invent weights."
            )
            made_verifiable = True

    payload = {
        "task_spec": {
            "description": (
                f"Convert this PyTorch module to an @nki.jit kernel named `nki_kernel`.\n"
                f"```python\n{task.model_class_code}\n```\n"
                f"Inputs:\n```python\n{task.get_inputs_code}\n```"
                f"{weights_note}"
            ),
            "reference_source": reference_source,
            "input_specs": input_specs,
            "constraints": {"atol": 1e-3, "rtol": 1e-3},
            # Per-task NKI hints (engine + tile suggestions) — the paper injects
            # these into the prompt; passing them lifts L2/L3 tiling choices.
            "nki_metadata": getattr(task, "nki_metadata", None) or {},
        },
        "mode": mode,
        "max_turns": max_turns,
    }

    compiled = False
    correct = False
    best_reward = -1.0
    best_code = ""
    p50_us = None
    p99_us = None
    error = None
    turns_used = 1
    tool_calls = 0

    try:
        resp = client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            qualifier=qualifier,
            runtimeSessionId=f"nkibench-{task.task_id}-{uuid4().hex[:8]}".ljust(33, "x"),
            payload=json.dumps(payload).encode("utf-8"),
        )
        body = b"".join(chunk for chunk in resp["response"].iter_chunks()) if hasattr(resp["response"], "iter_chunks") else resp["response"].read()
        result = json.loads(body)

        best_code = result.get("kernel_source") or ""
        comp = result.get("compile") or {}
        ver = result.get("verify") or {}
        prof = result.get("profile") or {}

        # Robust across single_shot (compile.success / verify.correct) and
        # multi_turn (top-level compiled / correct) response shapes.
        if comp.get("success") or result.get("compiled"):
            compiled = True
            best_reward = 0.0
        if ver.get("correct") or result.get("correct"):
            correct = True
            best_reward = 1.0
        if prof:
            p50_us = prof.get("latency_median_us")
            p99_us = prof.get("latency_p99_us")
        turns_used = result.get("turns_used", 1) or 1
        tool_calls = result.get("tool_calls_used", 0) or 0
        if result.get("status") not in ("success", "compiled", "compile_failed", "verify_failed", "failed"):
            error = result.get("error") or f"status={result.get('status')}"
    except Exception as exc:
        turns_used = 1
        tool_calls = 0
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.warning("  agentcore invoke failed for %s: %s", task.task_id, error)

    return {
        "task_id": task.task_id,
        "name": task.name,
        "level": task.level,
        "category": task.category,
        "compiled": compiled,
        "correct": correct,
        "reward": best_reward,
        "turns_used": turns_used,
        "tool_calls": tool_calls,
        "p50_us": p50_us,
        "p99_us": p99_us,
        "error": error,
        "code_preview": best_code[:300],
        # Honest tag: a correctness failure on a learned-weight task (Conv/Linear/
        # Embedding) is expected — the kernel never receives those weights — not a
        # model miss. Under Option B (--pass-weights) the weights ARE passed, so the
        # task becomes genuinely verifiable and the tag flips to False.
        "unverifiable_params": _has_learned_params(task) and not made_verifiable,
        "weights_passed": made_verifiable,
        "time": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NKIGen-Bench NKIAgent eval (agentic with tools)")
    parser.add_argument("--backend", choices=["claude", "vllm", "agentcore"], required=True)
    parser.add_argument("--model-id", default="us.anthropic.claude-sonnet-4-20250514-v1:0",
                        help="Claude model ID (for --backend claude)")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                        help="vLLM model name (for --backend vllm)")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--reward-server", default="http://REWARD_SERVER_IP:5050")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--agentcore-arn", default="",
                        help="AgentCore Runtime ARN (for --backend agentcore)")
    parser.add_argument("--agentcore-qualifier", default="DEFAULT")
    parser.add_argument("--agentcore-timeout", type=int, default=300,
                        help="Per-task client read timeout in seconds")
    parser.add_argument("--agentcore-mode", choices=["single_shot", "multi_turn"],
                        default="single_shot",
                        help="Runtime execution mode. multi_turn runs the paper's "
                             "compile-verify-fix tool loop (the multi-turn headline).")
    parser.add_argument("--task-ids", nargs="+", default=None,
                        help="Explicit task_ids to run (overrides the per-level slice).")
    parser.add_argument("--pass-weights", action="store_true",
                        help="Option B: pass learned weights (Conv/Linear/etc.) to the "
                             "kernel as extra inputs so param-model tasks become genuinely "
                             "verifiable, instead of tagging them unverifiable.")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--num-tasks", type=int, default=0, help="Tasks per level (0 = all)")
    parser.add_argument("--max-turns", type=int, default=10, help="Max agent turns")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Concurrent invocations for --backend agentcore (default 1=serial)")
    parser.add_argument("--resume-from", default="",
                        help="Path to prior *.details.json — skip task_ids whose prior result was a non-auth success")
    parser.add_argument("--output", default="results/nkibench_nkiagent.json")
    args = parser.parse_args()

    # Load tasks
    all_tasks = []
    for level in args.levels:
        level_tasks = load_tasks(level=level)
        if args.num_tasks > 0:
            level_tasks = level_tasks[:args.num_tasks]
        all_tasks.extend(level_tasks)
    # Optional explicit task-id selection (overrides the per-level slice) — used
    # to target specific tasks, e.g. param-models for Option B validation.
    if args.task_ids:
        wanted = set(args.task_ids)
        all_tasks = [t for lvl in args.levels for t in load_tasks(level=lvl)
                     if t.task_id in wanted]

    # Resume: keep prior results for task_ids that returned a real (non-auth) outcome,
    # and skip those tasks for this run. Anything that previously failed with
    # AccessDeniedException / token-expired is re-queued.
    prior_results = []
    if args.resume_from:
        try:
            with open(args.resume_from) as f:
                prior_blob = json.load(f)
            prior_details = prior_blob.get("details") or prior_blob.get("results") or []
            keep_ids = set()
            for r in prior_details:
                err = (r.get("error") or "").lower()
                if "accessdenied" in err or "token" in err and "expired" in err:
                    continue
                prior_results.append(r)
                keep_ids.add(r["task_id"])
            before = len(all_tasks)
            all_tasks = [t for t in all_tasks if t.task_id not in keep_ids]
            logger.info("resume: kept %d prior results, %d tasks remain to run (was %d)",
                        len(prior_results), len(all_tasks), before)
        except Exception as e:
            logger.error("resume failed: %s — running all tasks", e)
            prior_results = []

    if args.backend == "agentcore":
        if not args.agentcore_arn:
            logger.error("--agentcore-arn is required for --backend agentcore")
            sys.exit(2)
        backend_name = f"agentcore/{args.agentcore_arn.split('/')[-1][:40]}"
    elif args.backend == "claude":
        backend_name = f"claude/{args.model_id.split('/')[-1][:30]}"
    else:
        backend_name = f"vllm/{args.model}"
    logger.info("NKIAgent eval: %d tasks, backend=%s, max_turns=%d", len(all_tasks), backend_name, args.max_turns)

    # Check reward server (skipped for agentcore — runtime container reaches it directly)
    if args.backend != "agentcore":
        if not args.reward_server.lower().startswith(("http://", "https://")):
            raise ValueError(f"refusing non-http(s) reward-server URL: {args.reward_server}")
        try:
            with urllib.request.urlopen(f"{args.reward_server}/health", timeout=5) as resp:  # nosec B310 — args.reward_server scheme validated above
                logger.info("Reward server: %s", json.loads(resp.read()))
        except Exception as e:
            logger.error("Reward server unreachable: %s", e)
            sys.exit(1)

    # Set up client
    client = None
    if args.backend == "claude":
        import boto3
        client = boto3.client("bedrock-runtime", region_name=args.region)

    def _run_one(idx_task):
        idx, task = idx_task
        shapes, dtypes = extract_shapes_from_task(task)
        logger.info("[%d/%d] L%d %s: %s", idx + 1, len(all_tasks), task.level, task.category, task.name[:50])
        if args.backend == "claude":
            return eval_claude_nkiagent(task, client, args.model_id, args.reward_server, args.max_turns, shapes, dtypes)
        elif args.backend == "agentcore":
            return eval_agentcore_nkiagent(
                task, args.agentcore_arn, args.region, shapes, dtypes,
                qualifier=args.agentcore_qualifier, timeout_s=args.agentcore_timeout,
                mode=args.agentcore_mode, max_turns=args.max_turns,
                pass_weights=args.pass_weights,
            )
        else:
            return eval_vllm_nkiagent(task, args.vllm_url, args.model, args.reward_server, args.max_turns, shapes, dtypes)

    results = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    details_path = out.with_suffix(".details.json")

    def _checkpoint(current_results):
        merged = list(prior_results) + list(current_results)
        try:
            with open(details_path, "w") as f:
                json.dump({"summary": {"in_progress": True, "done": len(merged)}, "details": merged}, f, indent=2, default=str)
        except Exception as e:
            logger.warning("checkpoint write failed: %s", e)

    if args.backend == "agentcore" and args.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info("running with max_workers=%d (parallel agentcore invocations)", args.max_workers)
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(_run_one, (i, t)): i for i, t in enumerate(all_tasks)}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                done = len(results)
                if done % 10 == 0 or done == len(all_tasks):
                    n_c = sum(1 for x in results if x["compiled"])
                    n_ok = sum(1 for x in results if x["correct"])
                    logger.info("--- Progress: %d/%d | compiled=%d (%.1f%%) | correct=%d (%.1f%%) ---",
                                done, len(all_tasks), n_c, 100 * n_c / done, n_ok, 100 * n_ok / done)
    else:
        for i, task in enumerate(all_tasks):
            r = _run_one((i, task))
            results.append(r)
            _checkpoint(results)
            done = len(results)
            if done % 10 == 0 or done == len(all_tasks):
                n_c = sum(1 for x in results if x["compiled"])
                n_ok = sum(1 for x in results if x["correct"])
                avg_turns = sum(x["turns_used"] for x in results) / max(done, 1)
                avg_tools = sum(x["tool_calls"] for x in results) / max(done, 1)
                logger.info("--- Progress: %d/%d | compiled=%d (%.1f%%) | correct=%d (%.1f%%) | avg_turns=%.1f | avg_tools=%.1f ---",
                            done, len(all_tasks), n_c, 100 * n_c / done, n_ok, 100 * n_ok / done, avg_turns, avg_tools)

    # Merge prior (resumed) + new results before computing metrics
    results = list(prior_results) + list(results)

    # Compute metrics
    by_level = defaultdict(lambda: {"total": 0, "compiled": 0, "correct": 0})
    for r in results:
        by_level[r["level"]]["total"] += 1
        if r["compiled"]:
            by_level[r["level"]]["compiled"] += 1
        if r["correct"]:
            by_level[r["level"]]["correct"] += 1

    total = len(results)
    n_compiled = sum(1 for r in results if r["compiled"])
    n_correct = sum(1 for r in results if r["correct"])

    # Verifiable subset: tasks whose learned weights aren't hidden from the kernel.
    # Tasks that carry Conv/Linear/Embedding weights the kernel never receives are
    # structurally unverifiable by input-only comparison; compile is their ceiling.
    verifiable = [r for r in results if not r.get("unverifiable_params")]
    v_total = len(verifiable)
    v_correct = sum(1 for r in verifiable if r["correct"])
    n_unverifiable = total - v_total

    logger.info("\n--- NKIAgent: %s ---", backend_name)
    logger.info("Total: %d/%d compiled (%.1f%%), %d/%d correct (%.1f%%)",
                n_compiled, total, 100 * n_compiled / max(total, 1),
                n_correct, total, 100 * n_correct / max(total, 1))
    logger.info("Verified-correct on VERIFIABLE subset: %d/%d (%.1f%%)  [%d tasks excluded: learned weights hidden from kernel]",
                v_correct, v_total, 100 * v_correct / max(v_total, 1), n_unverifiable)
    for lvl in sorted(by_level.keys()):
        s = by_level[lvl]
        logger.info("  L%d: %d/%d compiled (%.0f%%), %d/%d correct (%.0f%%)",
                    lvl, s["compiled"], s["total"], 100 * s["compiled"] / max(s["total"], 1),
                    s["correct"], s["total"], 100 * s["correct"] / max(s["total"], 1))

    # Save
    if args.backend == "claude":
        model_label = args.model_id
    elif args.backend == "agentcore":
        model_label = args.agentcore_arn
    else:
        model_label = args.model
    summary = {
        "backend": args.backend,
        "model": model_label,
        "mode": "nkiagent",
        "max_turns": args.max_turns,
        "total": total, "compiled": n_compiled, "correct": n_correct,
        "compile_rate": round(100 * n_compiled / max(total, 1), 1),
        "correct_rate": round(100 * n_correct / max(total, 1), 1),
        # Verified-correct restricted to tasks that CAN be verified (weights not
        # hidden from the kernel). This is the fair quality bar vs the paper.
        "verifiable_total": v_total,
        "verifiable_correct": v_correct,
        "verifiable_correct_rate": round(100 * v_correct / max(v_total, 1), 1),
        "unverifiable_excluded": n_unverifiable,
        "by_level": {str(k): v for k, v in by_level.items()},
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    with open(details_path, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2, default=str)
    logger.info("\nSaved: %s", out)


if __name__ == "__main__":
    main()
