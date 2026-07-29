# Example: migrating a CUDA custom kernel to NKI

This example shows the headline use case end to end: an open-source repo ships a
**hand-written CUDA custom kernel** (RMSNorm — the normalization in LLaMA /
Mistral / Qwen), and the ATX NKI agent migrates it to an `@nki.jit` kernel that
is **compiled, numerically verified, and profiled on a real Trainium device**.

## Watch the demo

![ATX NKI migration demo](atx-nki-demo.gif)

A real recording of the full chain — **atx CLI → skill (SKILL.md) → MCP tools →
AgentCore runtime → Trn1 reward server** — driving a live migration. The agent
generates the `@nki.jit` kernel via the deployed runtime's multi-turn fix loop
(2 turns here) and it **compiles, verifies (max_abs_error 2.1e-4, 0 / 16.7M
mismatched), and profiles at 544 µs median** on a real `trn1.2xlarge`.

## Files

| File | Role |
|---|---|
| `rmsnorm_cuda_input.py` | The input: a `torch.nn.Module` backed by a `__global__` CUDA C kernel (`load_inline`). NVIDIA-only — this is the wall a Trainium migration hits. |
| `rmsnorm_reference.py` | The ground-truth PyTorch numerics the migrated kernel must match. |
| `nki_rmsnorm_output.py` | The migrated `@nki.jit` kernel the agent produced. |
| `migration_report.json` | The on-device evidence: compile + verify + profile results from the Trn1 reward server. |
| `demo.sh` / `run_migration.py` | The recordable terminal demo that drives the migration live (see below). |
| `atx-nki-demo.gif` | The recorded demo of the full live migration loop. |

## Run the migration via the ATX CLI

The [`atx` CLI](https://aws.amazon.com/transform/) is one client of the
`kernelforge-nki-mcp` MCP server (the durable contract; the IDE, Kiro, and AWS
Batch are other clients). ATX runs a **transformation definition** against a
repo; this agent ships one in [`atx/`](../../atx/). Install `atx`, register the
MCP server, then execute the transformation:

```bash
# 1. Install the atx CLI (official AWS Transform install script) and register
#    the MCP server for it — full steps in atx/README.md.
curl -fsSL https://transform-cli.awsstatic.com/install.sh | bash
sed "s#REPLACE_WITH_ABSOLUTE_PATH#$(cd ../.. && pwd)#" ../../atx/mcp.json > ~/.aws/atx/mcp.json

# 2. Execute the transformation definition against this repo.
atx custom def exec \
  --code-repository-path ./examples/cuda_rmsnorm_migration \
  --configuration "additionalPlanContext=Follow atx/transformation-definition/transformation_definition.md"

#    Under the hood the transformation definition drives the MCP tools:
#      nki_discover_kernels  -> finds the CUDA-backed RMSNorm forward (kind="cuda")
#      nki_skill_lookup      -> pulls the RMSNorm NKI pattern
#      nki_generate_kernel   -> AgentCore multi-turn loop (Opus 4.8)
#      nki_compile / nki_verify / nki_profile  -> on the Trn1 reward server
#      nki_emit_diff         -> writes the migrated kernel / opens a PR

# 3. Two human-in-the-loop gates: Scope (pick kernels + backend) and
#    Requirements (sign off the accuracy + performance budget).
```

> The `atx` CLI drives the exact MCP tool sequence above. If you don't have the
> CLI, the same chain runs from any MCP-aware agent (Claude Code, Kiro, Codex)
> or directly against the tools — see `tests/test_integration_chain.py` and
> `tests/test_atx_cli.py`.

## Record the demo as a video

`demo.sh` is a guided, recordable walkthrough: it shows the CUDA input, runs
discovery, then invokes the deployed runtime and prints the real compile / verify
/ profile numbers from Trainium. Point it at the runtime and capture it:

```bash
export AGENTCORE_ARN=arn:aws:bedrock-agentcore:us-east-1:<acct>:runtime/<id>
export AWS_REGION=us-east-1

# Option A — asciinema cast (shareable, tiny), optionally converted to a gif:
asciinema rec atx-nki-demo.cast -c "bash examples/cuda_rmsnorm_migration/demo.sh"
agg atx-nki-demo.cast atx-nki-demo.gif       # optional cast -> animated gif

# Option B — screen-record any terminal window while running:
bash examples/cuda_rmsnorm_migration/demo.sh
```

A captured run is saved in [`demo_transcript.txt`](demo_transcript.txt).

## Result — measured on a real `trn1.2xlarge`

Model: **Claude Opus 4.8** (Bedrock). Input: `(4096, 4096)` fp32.

| Stage | Result |
|---|---|
| **Compile** (`neuronx-cc`) | ✅ success — NEFF produced, 9.05 s, no errors |
| **Verify** (`np.allclose`, atol=rtol=1e-3) | ✅ **correct** — `max_abs_error ≈ 2.11e-4`, **0 / 16,777,216** elements mismatched |
| **Profile** (50 runs) | median **544 µs**, p99 546 µs |

Nothing here is claimed that wasn't measured on hardware — that is the whole
point of the compile-verify-profile loop.
