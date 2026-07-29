#!/usr/bin/env bash
#
# Reproducible terminal demo: migrate a CUDA custom kernel to NKI with the ATX
# NKI agent, driving the same MCP tool chain `atx custom def exec` runs, and
# showing the real on-device result from the Trn1 reward server.
#
# The green "$ atx ..." lines below are the real atx commands this migration
# maps to (see atx/README.md); the demo executes the underlying MCP tool chain
# directly so it is fast and self-contained to record. For a true end-to-end
# `atx custom def exec` run, follow atx/README.md.
#
# This is written to be *recorded*. To capture an actual video:
#
#   # asciinema cast (then upload or convert):
#   asciinema rec atx-nki-demo.cast -c "bash examples/cuda_rmsnorm_migration/demo.sh"
#   # optional: cast -> animated gif
#   agg atx-nki-demo.cast atx-nki-demo.gif
#   # or screen-record any terminal while running the line below.
#
# Prereqs (the live migration hits real hardware through the deployed runtime):
#   export AGENTCORE_ARN=arn:aws:bedrock-agentcore:us-east-1:<acct>:runtime/<id>
#   export AWS_REGION=us-east-1
#   valid AWS credentials with bedrock-agentcore:InvokeAgentRuntime
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${AGENTCORE_ARN:?export AGENTCORE_ARN=arn:aws:bedrock-agentcore:...:runtime/...}"
: "${AWS_REGION:=us-east-1}"
export AWS_REGION

# Small helpers so the recording reads like a guided walkthrough.
c_reset=$'\033[0m'; c_dim=$'\033[2m'; c_cyan=$'\033[36m'; c_green=$'\033[32m'; c_bold=$'\033[1m'
say()  { printf '\n%s%s%s\n' "$c_bold$c_cyan" "$1" "$c_reset"; }
note() { printf '%s%s%s\n' "$c_dim" "$1" "$c_reset"; }
run()  { printf '%s$ %s%s\n' "$c_green" "$1" "$c_reset"; eval "$2"; }

clear || true
say "ATX NKI Agent — migrate a CUDA custom kernel to Trainium (NKI)"
note "Full chain: atx CLI → skill (SKILL.md) → MCP tools → AgentCore runtime → Trn1 reward server"
sleep 1

say "1) The input — a hand-written CUDA custom kernel (RMSNorm, NVIDIA-only)"
run "sed -n '30,60p' rmsnorm_cuda_input.py" \
    "sed -n '30,60p' '$HERE/rmsnorm_cuda_input.py'"
sleep 1

say "2) atx CLI → transformation definition: the workflow the agent follows"
note "The transformation definition steers the agent through discover → skill_lookup → generate → compile → verify → profile."
run "atx custom def get -n pytorch-triton-to-nki" \
    "grep -nE '^#|discover|skill_lookup|compile|verify|profile' '$REPO/atx/transformation-definition/transformation_definition.md' | head -8"
sleep 1

say "3) transformation definition → MCP: discover kernels (nki_discover_kernels)"
note "A generic tool sees a .py module; the agent recognizes a CUDA custom kernel."
run "atx custom def exec -p ./examples/cuda_rmsnorm_migration  # step: discover" "python3 - <<'PY'
import sys; sys.path.insert(0, '$REPO/mcp/src')
from kernelforge_nki_mcp.discover import discover
r = discover('$HERE')
for c in r.candidates:
    if c.already_nki:
        continue
    print(f'  found  kind={c.kind:6s}  {c.function}()  in {c.file.split(\"/\")[-1]}:{c.line}')
PY"
sleep 1

say "4) MCP → AgentCore → Trn1: generate + compile + verify + profile on real silicon"
note "nki_generate_kernel invokes the deployed AgentCore runtime (Opus 4.8). Its multi-turn"
note "fix loop calls compile/verify/profile on the Trn1 reward server until the kernel verifies."
run "atx custom def exec -p ./examples/cuda_rmsnorm_migration  # step: generate+compile+verify+profile" \
    "python3 '$HERE/run_migration.py'"

say "Done — the agent generated the NKI kernel and every number was measured on a real trn1.2xlarge."
