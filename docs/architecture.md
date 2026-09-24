# Architecture

End-state design for the ATX NKI agent — packaged as an Agent Plugin (Skill + MCP) for AWS Transform.

## Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│ CUSTOMER SURFACES                                                     │
│   IDE: Claude Code · Kiro · Codex                                     │
│   Console: console.aws.amazon.com/transform                           │
│   CLI: atx custom def exec -n pytorch-triton-to-nki -p <repo>         │
│   Bulk: AWS Batch (multi-repo, parallel)                              │
└────────────────────────────────┬──────────────────────────────────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STEERING — Agent Plugin (Skill + MCP)                                 │
│   skills/nki-kernel/SKILL.md                                          │
│   skills/nki-kernel/references/{tile-model,layout-rules,common-ops,   │
│                                  compiler-errors,workflow}.md         │
│   Workflow: Resume → Intent → Discovery → Scope (G1) →                │
│             Assessment → Requirements (G2) → Tasks → Execute → Diff  │
└────────────────────────────────┬──────────────────────────────────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│ TOOLS — kernelforge-nki-mcp (in-repo mcp/, uvx --from <path>)         │
│   nki_discover_kernels   nki_generate_kernel   nki_compile            │
│   nki_verify             nki_profile           nki_skill_lookup       │
│   nki_emit_diff                                                       │
└───────────────┬───────────────────────────────────┬───────────────────┘
                │ SigV4                             │ SigV4 / mTLS
                ▼                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────────┐
│ AGENT — AgentCore Runtime     │   │ EXECUTION — Trn1 reward server    │
│   Strands Agent + 4 tools     │   │   Flask, /reward, /reward/batch   │
│   Multi-turn fix loop         │   │   neuronx-cc + on-device verify   │
│   Cedar policy /tmp/nki-agent │   │   neuron-profile                  │
│   Model router                │   │   Long-lived Trn1.2xlarge         │
└──────────────┬────────────────┘   └───────────────────────────────────┘
               ▼
┌───────────────────────────────────────────────────────────────────────┐
│ MODEL — pluggable, per-invocation                                     │
│   Claude Opus 4.8 (Bedrock)        — default for `auto` & off-dist    │
│   Bedrock Custom Model Import      — Qwen3-Coder-30B + SFT-v4 LoRA    │
│   Self-hosted vLLM                  — failover only                   │
└───────────────────────────────────────────────────────────────────────┘
```

## Layers

### 1. Customer surfaces

Three entry points, all clients of the same MCP server:

- **IDE / agent** — Claude Code, Kiro, Codex via `/plugin install kernel-forge-aws-transform`.
- **Console** — `console.aws.amazon.com/transform` registers a "PyTorch/Triton → NKI" Custom transformation tile.
- **CLI / Batch** — `atx custom def exec` (AWS Transform Custom) runs the transformation definition in `atx/` against a repo; AWS Batch fans it out across many repos. See [`atx/README.md`](../atx/README.md).

Surfaces come and go (Kiro Power → Skill+MCP just happened). The MCP layer is the durable contract.

### 2. Steering — Skill

`SKILL.md` frontmatter triggers on: *"convert this kernel to NKI"*, *"optimize for Trainium"*, *"PyTorch → NKI"*, *"Triton → NKI"*, *"rewrite this kernel for Inferentia"*.

The workflow mirrors `awslabs/agent-plugins/plugins/aws-transform`'s `Resume → Intent → Discovery → Scope → Assessment → Requirements → Tasks → Execute` shape. Two human gates:

- **GATE 1 (Scope)** — user picks which kernels to convert and which model backend.
- **GATE 2 (Requirements)** — user signs off on accuracy + perf budget. Kernel equivalence is not a routine refactor; an owner must approve.

References split the existing `kernel-forge/src/env/skill_library/SKILL.md` into:

- `tile-model.md`
- `layout-rules.md`
- `common-ops.md`
- `compiler-errors.md`
- `workflow.md`

### 3. Tools — MCP server

`kernelforge-nki-mcp` is a stateless Python package that ships in this repo under `mcp/`, launched via `uvx --from <path-to-mcp> kernelforge-nki-mcp`. It is deliberately **not** published to public PyPI, and the launcher always names a path: a bare `uvx kernelforge-nki-mcp` would resolve the unregistered name from public PyPI, giving whoever registered it code execution inside the agent process. Seven tools:

| Tool | Backend | Purpose |
|---|---|---|
| `nki_discover_kernels` | local repo walk | Find `@triton.jit` and hot `nn.Module.forward` |
| `nki_generate_kernel`  | AgentCore endpoint | Multi-turn agentic generation |
| `nki_compile`          | Trn1 reward server | `neuronx-cc` compile |
| `nki_verify`           | Trn1 reward server | torch.allclose on real device |
| `nki_profile`          | Trn1 reward server | `neuron-profile` latency + util |
| `nki_skill_lookup`     | in-package skill DB | NKI patterns / examples |
| `nki_emit_diff`        | local repo write | Generate patch / open PR |

Auth: SigV4 to AgentCore and to the Trn1 reward server. No state in this layer.

### 4. Agent — AgentCore Runtime

Owns the multi-turn fix loop:

```
1. parse_pytorch_input(repo_or_file)
2. skill_lookup(operator_signature)
3. router.resolve(task_spec)            ← picks Opus / Qwen / vLLM
4. generate_nki_kernel(prompt, model)
5. compile_kernel        → reward server (retry on fail)
6. verify_kernel         → reward server (retry on fail)
7. profile_kernel        → reward server
8. emit_pr / emit_diff
```

Cedar policy scopes the agent's filesystem to `/tmp/nki-agent`. Network egress is gated to the explicit allowlist.

### 5. Execution — Trn1 reward server

Long-lived Flask on a Trn1.2xlarge. The only component in the system that touches real silicon, and the same endpoint the project's training pipeline uses — the runtime reward signal is identical to the training reward signal.

`/reward` and `/reward/batch` accept compile / verify / profile / baseline / skill ops. Returns ground-truth signals:

```json
{ "compile_ok": true, "verify_ok": true, "max_abs_error": 4.2e-4,
  "latency_us": 142.7, "flops_utilization": 0.71 }
```

**Provisioning is declarative** (`infrastructure/reward_server_cdk/`): a CDK stack launches a stock
Neuron DLAMI in a Trn1-capable AZ, pulls the reward server from a bundled S3 asset,
and runs it under systemd (4 workers; the persisted-NEFF write path is
request-scoped so concurrent workers can't collide — see `reward_server/compile.py`).
The auth token is generated on the instance at boot into SSM
SecureString; the security group locks port 5050 to the VPC CIDR (the stack refuses
`0.0.0.0/0`). No hand-built AMIs or manual SSH — `cdk deploy` reproduces it from zero,
and `repoint-agentcore.sh` wires the new endpoint into the AgentCore runtime in-place.

### 6. Model — pluggable backends

Three backends. The agent's router picks per task; customers can pin per invocation.

| Backend | When the router picks it | Why |
|---|---|---|
| **Claude Opus 4.8** (Bedrock) | `auto` default; `nki_skill_lookup` < 0.75; reasoning-heavy refactors | Strongest general code reasoning. Tracks the latest Claude release without re-training. |
| **Bedrock Custom Model Import** (Qwen3-Coder-30B + SFT-v4 LoRA merged) | `nki_skill_lookup` ≥ 0.75; in-distribution NKI shapes; cost-sensitive bulk runs | Specialist. +10pp over base. Cheaper. Serverless — no GPU babysitting. |
| **Self-hosted vLLM** on G5/P4d | Failover when CMI quota / region constraints don't fit | Same SFT-v4 weights; kept as backstop, not a fixture. |

**Router policy:**

```python
def resolve(task_spec, recent_failures) -> Backend:
    if task_spec.model_pin:
        return backends[task_spec.model_pin]
    if len(recent_failures) >= 2 and recent_failures[-1] == "qwen3-sft-v4":
        return backends["opus-4-8"]                         # escalate
    if skill_lookup(task_spec.operator_signature) >= 0.75:
        return backends["qwen3-sft-v4"]
    return backends["opus-4-8"]
```

`opus-4-8` does not escalate. A user pin disables escalation.

## Trust & isolation

| Boundary | Auth | Scope |
|---|---|---|
| Customer surface ↔ MCP server | local process | uvx-launched, customer machine |
| MCP server ↔ AgentCore | SigV4 | IAM gates `bedrock:InvokeAgent` |
| MCP server ↔ Trn1 reward server | SigV4 / mTLS | VPC-private |
| AgentCore ↔ Bedrock (Opus + CMI) | IAM | VPC-private |
| AgentCore ↔ vLLM (failover) | IAM-signed HTTP | VPC-private |
| AgentCore filesystem | Cedar | `/tmp/nki-agent` only |
| Trn1 reward server | network policy | accepts AgentCore + ATX Batch only |

## Why this shape

1. **One contract — the MCP server — every surface depends on.** AWS is standardizing on Skill+MCP across `awslabs/agent-plugins` and the Transform agent builder toolkit.
2. **Skill + MCP is the post-Power packaging.** `awslabs/agent-plugins/plugins/aws-transform` is the canonical example.
3. **AgentCore = state, MCP = verbs.** Same split AWS Transform itself uses. Adding a first-party Transform tile later is a registration step, not a re-architecture.
4. **Trn1 reward server is the single source of truth.** No mocked validation path.
5. **Pluggable models with a router.** Opus 4.8 covers the long tail; Qwen3+SFT-v4 wins on in-distribution shapes; vLLM is failover. Customers can override.
6. **Two human gates** match awslabs' shape and are appropriate for kernel equivalence.

## Out of scope (intentionally)

- Generating NKI kernels offline / batch without Trainium hardware in the loop.
- A Transform tile that bypasses the MCP server (would fork the contract).
- Hosting models anywhere other than Bedrock or vLLM (e.g., raw SageMaker endpoints).
- Whole-program PyTorch → NKI compilation. Kernel-level rewrites only; full-program is `torch-neuronx`'s job.
