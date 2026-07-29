# PyTorch/Triton → NKI kernel migration

Convert PyTorch (`nn.Module.forward`) and Triton (`@triton.jit`) kernels in a
repository into `@nki.jit` NKI kernels for AWS Trainium / Inferentia, then
**compile, numerically verify, and profile every candidate on a real Trainium
device** before writing the change. Nothing is claimed that wasn't measured on
hardware.

This is the AWS Transform (ATX Custom) transformation definition for the agent
in this repository. It is the ATX-CLI equivalent of `skills/nki-kernel/SKILL.md`
— the same workflow and the same seven MCP tools, expressed as a transformation
definition the `atx` CLI executes. The tools live in the `kernelforge-nki-mcp`
MCP server, which you must register in `~/.aws/atx/mcp.json` first (see the
`atx/` README).

## Tools used

This transformation drives the `kernelforge-nki-mcp` MCP server. Call the tools
in this order; do not invent NKI APIs, and do not skip the verification step.

| Tool | Purpose |
|---|---|
| `nki_discover_kernels` | Find `@triton.jit` and hot `nn.Module.forward` (incl. CUDA custom kernels) |
| `nki_skill_lookup` | Pull the NKI pattern/recipe for the operator |
| `nki_generate_kernel` | AgentCore multi-turn generation (returns candidate `@nki.jit` source) |
| `nki_compile` | `neuronx-cc` compile on the Trn1 reward server |
| `nki_verify` | `np.allclose` against the PyTorch reference on the real device |
| `nki_profile` | latency p50/p99 + engine utilization on the device |
| `nki_emit_diff` | write the migrated kernel / open a PR |

## Workflow

Follow these phases in order. Do NOT skip ahead. This mirrors the AWS Transform
standard shape (Discovery → Scope → Assessment → Requirements → Tasks → Execute
→ Diff/PR) with two human-in-the-loop gates.

1. **Discovery** — call `nki_discover_kernels(repo_root)` on the code repository.
   Surface each candidate: file:line, operator signature, whether it is hot, and
   whether it is already `@nki.jit` (skip those). CUDA custom kernels
   (`load_inline` / `__global__`) are in scope and report `kind="cuda"`.
2. **Scope (GATE 1)** — the user selects which kernels to convert and the model
   backend (`auto` default, or `opus-4-8` / `qwen3-sft-v4` / `vllm`). Do not
   proceed without an explicit selection.
3. **Assessment** — for each selected kernel, call `nki_skill_lookup` with the
   operator signature (record the top-1 score; ≥ 0.75 means an in-distribution
   pattern exists), then generate a stub and `nki_compile` it to gauge
   feasibility.
4. **Requirements (GATE 2)** — draft the per-kernel accuracy budget (default
   `atol = rtol = 1e-3`) and performance budget (match or beat the eager /
   `torch.compile` baseline on Trn1), plus retry budget. The user signs off.
   Kernel equivalence is not a routine refactor — an owner must approve.
5. **Tasks** — one task per selected kernel: input file:line, target shape and
   dtype, target hardware, chosen model, and the accepted budget.
6. **Execute** — per kernel:
   `nki_skill_lookup → nki_generate_kernel → nki_compile → nki_verify →
   nki_profile`. On a compile failure feed the `neuronx-cc` stderr back and
   retry; on a verify failure feed `max_abs_error` / mismatched-element summary
   back and retry, up to the retry budget. Stream progress ("compile attempt
   2/5"), never silence.
7. **Diff / PR** — only after a passing verification, call `nki_emit_diff` with
   the final source. Show the diff, the verification report, and the
   profile-vs-baseline comparison. For kernels that exhausted their budget
   without verifying, do NOT emit a diff — surface the structured handoff
   (compiler errors, where verification diverged, last candidate source) for
   human review.

## Hard rules

You MUST NOT generate a kernel without an accepted requirements document.
You MUST NOT emit a diff without a passing verification report.

1. The kernel function MUST be named `nki_kernel` and decorated with `@nki.jit`.
2. The partition dimension (`TILE_M`) MUST be exactly 128.
3. Input tensors are 2D `(M, N)` — never 3D `(B, M, N)`.
4. Output goes through `result = nl.ndarray(..., buffer=nl.shared_hbm)` and
   `return result`.
5. `nl.sigmoid`, `nl.relu`, `nl.tanh`, `nl.gelu`, `nl.sqrt`, `nl.pow` do NOT
   exist — implement manually using `nl.exp`, `nl.multiply`, etc.
6. Do not invent NKI APIs. Verify each call site against `nki_skill_lookup`
   results.

## Validation

The optional build/validation command for this transformation should run the
repository's kernel checks. This transformation's ground truth is the on-device
`nki_verify` result, not a local build — the `neuronx-cc` compile and the
`np.allclose` numerical check both run on the Trn1 reward server behind the
`nki_compile` / `nki_verify` tools. A migrated kernel is only considered done
when `nki_verify` reports `correct: true` within the accuracy budget.
