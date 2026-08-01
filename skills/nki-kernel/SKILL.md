---
name: nki-kernel
description: Convert PyTorch and Triton kernels to optimized NKI kernels for AWS Trainium and Inferentia. Use when the user says "convert this kernel to NKI", "optimize for Trainium", "PyTorch to NKI", "Triton to NKI", "rewrite this kernel for Inferentia", or "make this kernel run on Neuron". Don't use for general PyTorch optimization, model compilation via torch-neuronx, or non-Neuron hardware (CUDA, ROCm).
---

# NKI Kernel Optimization

## Overview

Domain expertise for rewriting PyTorch and Triton kernels as `@nki.jit` kernels that run on AWS Trainium / Inferentia NeuronCores. Orchestrates a generate → compile → verify → profile loop with two human-in-the-loop checkpoints, using a real Trainium device for ground-truth correctness and performance signals.

## Prerequisites

This skill requires the `kernelforge-nki-mcp` MCP server. Configure it in your agent's MCP settings (this plugin ships an `.mcp.json` that does this for you):

```json
{
  "mcpServers": {
    "kernelforge-nki-mcp": {
      "command": "uvx",
      "args": ["kernelforge-nki-mcp@latest"]
    }
  }
}
```

The MCP server holds AWS credentials via the standard chain (env vars, `~/.aws/credentials`, IAM Identity Center). Authentication is just-in-time — only when a tool that hits AWS is actually called.

## Mandatory workflow

Follow these phases in order. Do NOT skip ahead.

```
Resume        → Check .nki-agent/context.json
Intent        → Ask user what they want to do
Discovery     → Scan workspace for hot kernels
Scope         → User selects kernels + model backend (GATE 1)
Assessment    → Skill lookup + dry compile of stub
Requirements  → Draft accuracy/perf budget from assessment
Approval      → User approves requirements (GATE 2)
Tasks         → Generate per-kernel task list
Execute       → Generate / compile / verify / profile loop
Diff          → Emit patch or PR
```

**Discovery finds kernels. Assessment evaluates feasibility. Requirements come from the assessment — NOT from discovery.**

You MUST NOT generate a kernel without an accepted requirements document.
You MUST NOT emit a diff without a passing verification report.

## Resuming a prior session

Check for `.nki-agent/context.json` (workspace-relative). This is silent — never narrate the check, never produce preamble. On a fresh install, the first visible output is the intent question.

- **No context found:** proceed directly to intent.
- **Context found with active job:** name the kernel(s) the prior session was working on and the phase reached. Offer: continue (reuse prior assessment + requirements) or start fresh (delete `.nki-agent/`).

## Determining user intent

Ask: *"What would you like to do?"*

Options:
- **Optimize a single kernel** — user points at one file or function.
- **Audit this repo for kernel candidates** — full Discovery scan.
- **Continue a prior session** — only shown if `.nki-agent/context.json` is present.

Do NOT inspect any files until the user picks an intent.

## Discovery

Call `nki_discover_kernels(repo_root)`. It returns a list of candidates: each `@triton.jit` function and each `nn.Module.forward` with a hot-path signature (matmul, softmax, layernorm, attention, RMSNorm, conv).

For each candidate, surface:

| Kernel | File:line | Signature | Hot? | Already NKI? |
|---|---|---|---|---|

If a candidate is already `@nki.jit`, skip it.

## Scoping (GATE 1)

Show the candidate table. User multi-selects kernels.

Then ask the user to pick a **model backend** — one of:

- **`auto`** (recommended) — router picks Opus 4.8 vs Qwen3+SFT-v4 per kernel based on `nki_skill_lookup` confidence. Default.
- **`opus-4-8`** — Claude Opus 4.8 via Bedrock. Best for novel ops, unusual shapes, reasoning-heavy refactors.
- **`qwen3-sft-v4`** — Qwen3-Coder-30B + SFT-v4 LoRA via Bedrock Custom Model Import. Specialist for in-distribution NKI shapes; cheaper.
- **`vllm`** — failover only; same Qwen3+SFT-v4 weights, self-hosted. Use when CMI throughput / region constraints don't fit.

Save selection to `.nki-agent/context.json`.

## Assessment

For each selected kernel:

1. Call `nki_skill_lookup` with the kernel's operator signature. Record the top-1 score; >= 0.75 means we have an in-distribution NKI pattern for this op.
2. Generate a stub kernel with the chosen model and call `nki_compile` on it. Record compile success / failure mode.
3. Aggregate into an assessment table:

| Kernel | Top skill match | Skill score | Stub compiles? | Confidence | Recommended model |
|---|---|---|---|---|---|

Confidence is `high` (skill ≥ 0.75 AND stub compiles), `medium` (one of the two), `low` (neither). Recommended model is `qwen3-sft-v4` for high, `opus-4-8` for medium/low.

## Requirements (GATE 2)

Draft `.nki-agent/requirements.md`:

- Per-kernel accuracy budget — default `atol=1e-3, rtol=1e-3`.
- Per-kernel perf budget — default "match or beat the eager baseline on Trn1; matching `torch.compile` is acceptable on first iteration."
- Retry budget — default 5 compile retries, 3 verify retries per kernel.
- Total wall-clock budget — default 30 minutes per kernel.
- Failure handoff — kernels that don't meet the budget come back with a structured analysis (compiler errors, where verification diverged, last candidate source) for human review.

User approves or edits. Save accepted version.

## Tasks

Emit `.nki-agent/tasks.md` — one task per kernel. Each task records:

- input file:line, target shape, target dtype, target hardware (`trn1` | `trn2`)
- chosen model
- accuracy + perf budget
- status (`pending` | `in_progress` | `done` | `failed`)

## Execute

For each pending task:

1. `nki_skill_lookup(kernel.operator_signature)` — fetch top-K patterns and inject into the prompt.
2. `nki_generate_kernel(spec, model)` — multi-turn agent loop in AgentCore. Returns candidate NKI source.
3. `nki_compile(candidate)` — neuronx-cc on Trn1.
   - On failure: feed compiler stderr back to the model. Retry up to budget.
4. `nki_verify(candidate, reference, input_specs)` — torch.allclose on real device.
   - On failure: feed `max_abs_error`, `max_rel_error`, mismatched-elements summary back. Retry up to budget.
5. `nki_profile(candidate, input_specs)` — neuron-profile latency + engine utilization.
   - Compare to eager / `torch.compile` baseline.

**Router escalation:** if `qwen3-sft-v4` fails compile twice in a row OR fails verify twice in a row, the next attempt swaps to `opus-4-8` carrying the error feedback. Reverse escalation does not happen. The customer's pinned model overrides escalation.

Stream progress to the user — they should see "compile attempt 2/5" not silence.

## Diff / PR

When a task completes, call `nki_emit_diff(repo, kernel)` with the final source. The user sees:

- the diff,
- the verification report (`max_abs_error`, `max_rel_error`),
- the perf comparison (latency µs vs baseline, engine utilization %).

User accepts → applied to working tree. Optionally `nki_emit_diff(..., open_pr=true)` opens a PR.

For tasks that exhausted their budget without converging, do NOT emit a diff. Surface the structured handoff and mark the task `failed` in `tasks.md` with the reason.

## References

Read these for deeper guidance during execution:

- [tile-model.md](references/tile-model.md) — partition / free dimension rules, SBUF/PSUM, allowed tile shapes.
- [layout-rules.md](references/layout-rules.md) — what's a 2D vs 3D tensor in NKI, why `TILE_M = 128` is non-negotiable.
- [common-ops.md](references/common-ops.md) — matmul, softmax, layernorm, RMSNorm, attention recipes.
- [compiler-errors.md](references/compiler-errors.md) — decoder ring for `neuronx-cc` error messages.
- [workflow.md](references/workflow.md) — the long-form version of this workflow with edge cases.
- [KernelForgeDeployAccess.json](references/KernelForgeDeployAccess.json) — least-privilege IAM
  policy for deploying the stack (AgentCore runtime + Trn1 reward server). Attach to the
  deploying principal; scoped to the `TrainiumRewardServer` / `AgentCore-*` / `CDKToolkit`
  stacks and `us-east-1`. Not needed to *use* the agent, only to run `make deploy`.

## Hard rules (apply at all times)

1. The kernel function MUST be named `nki_kernel` and decorated with `@nki.jit`.
2. The partition dimension (`TILE_M`) MUST be exactly 128.
3. Input tensors are 2D `(M, N)` — never 3D `(B, M, N)`.
4. Output goes through `result = nl.ndarray(..., buffer=nl.shared_hbm)` and `return result`.
5. `nl.sigmoid`, `nl.relu`, `nl.tanh`, `nl.gelu`, `nl.sqrt`, `nl.pow` do NOT exist — implement manually using `nl.exp`, `nl.multiply`, etc.
6. Do not invent NKI APIs. Verify each call site against `nki_skill_lookup` results.
