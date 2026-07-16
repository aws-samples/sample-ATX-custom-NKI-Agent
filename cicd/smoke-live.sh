#!/usr/bin/env bash
#
# Exercise the actual core topic of this repo end-to-end, against the live
# deployment: generate an @nki.jit kernel via the deployed AgentCore runtime,
# and assert it COMPILES and VERIFIES on real Trainium silicon through the
# Trn1 reward server. This is not a unit test — it is a real hardware run
# that makes billed Bedrock + Trainium calls, the same "ground truth in the
# loop" path the README's headline numbers are measured against.
#
# What this actually proves, that the hardware-free pytest suite (make prep)
# cannot: that AgentCore can reach the reward server, the reward server can
# reach the NeuronCore, and neuronx-cc + np.allclose on real hardware agree
# with the model's generated kernel — the full chain, not a mocked link.
#
# Two tiers, cheapest first — a full non-zero exit on either means the core
# loop is broken, not just one kernel:
#   1. Fast smoke: one trivial L1 task (relu) via scripts/smoke_opus48.sh.
#      Finishes in well under a minute; if THIS fails, don't bother with #2.
#   2. Real migration: the CUDA->NKI RMSNorm example via
#      examples/cuda_rmsnorm_migration/run_migration.py — the same multi-turn
#      compile-verify-fix loop a real migration uses, on a real custom kernel.
#
# Usage:
#   ./cicd/smoke-live.sh [--region us-east-1] [--agent-name atxnkiagent_nki_agent] [--skip-migration]
#
# AGENTCORE_ARN is resolved automatically the same way cicd/deploy.sh does
# (list-agent-runtimes by name) unless already exported. CICD_YES=1 skips
# the confirmation prompt, for CI use.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REGION="${AWS_REGION:-us-east-1}"
AGENT_NAME="${AGENT_NAME:-atxnkiagent_nki_agent}"
SKIP_MIGRATION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --agent-name) AGENT_NAME="$2"; shift 2 ;;
    --skip-migration) SKIP_MIGRATION=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

section "smoke-live: resolve AgentCore runtime ARN"
if [[ -z "${AGENTCORE_ARN:-}" ]]; then
  AGENTCORE_ARN="$(aws bedrock-agentcore-control list-agent-runtimes \
    --region "${REGION}" \
    --query "agentRuntimes[?agentRuntimeName=='${AGENT_NAME}'].agentRuntimeArn | [0]" \
    --output text)"
  if [[ -z "${AGENTCORE_ARN}" || "${AGENTCORE_ARN}" == "None" ]]; then
    echo "could not find an agent runtime named '${AGENT_NAME}' in ${REGION}." >&2
    echo "Deploy first (make deploy), or pass --agent-name / export AGENTCORE_ARN directly." >&2
    exit 1
  fi
fi
export AGENTCORE_ARN
export AWS_REGION="${REGION}"
echo "runtime: ${AGENTCORE_ARN}"

echo ""
echo "This makes real, billed calls: Bedrock model invocations and Trn1"
echo "compile/verify/profile calls on real Trainium hardware. It does NOT"
echo "create or destroy infrastructure — only exercises what's already"
echo "deployed."
confirm "Proceed with the live smoke run against ${AGENT_NAME}?" || { echo "aborted"; exit 1; }

section "smoke-live: 1/2 fast smoke (single L1 task via AgentCore)"
# scripts/eval_nkibench_agentcore.py (which smoke_opus48.sh drives) imports
# torch lazily to read task shapes — a real runtime dependency of the script
# that isn't declared in any project's pyproject.toml (it lives in scripts/,
# outside all of them). `uv run --with` provisions it ad hoc rather than
# requiring a separate dev-environment setup step, matching how this repo's
# other one-off diagnostic/eval scripts are already run.
(
  cd "${REPO_ROOT}" && uv run --with torch --with boto3 bash "${REPO_ROOT}/scripts/smoke_opus48.sh"
)
echo "fast smoke: PASSED"

if [[ "${SKIP_MIGRATION}" == "1" ]]; then
  section "smoke-live: 2/2 SKIPPED (--skip-migration)"
  section "smoke-live: PASSED (fast smoke only)"
  exit 0
fi

section "smoke-live: 2/2 real migration (CUDA RMSNorm -> NKI, multi-turn fix loop)"
(
  cd "${REPO_ROOT}/examples/cuda_rmsnorm_migration"
  uv run --with boto3 python run_migration.py
)
echo "migration: PASSED (compiled + verified on real Trainium silicon)"

section "smoke-live: PASSED"
