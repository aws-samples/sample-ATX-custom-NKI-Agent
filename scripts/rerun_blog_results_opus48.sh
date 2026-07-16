#!/usr/bin/env bash
#
# Re-run the three blog-backing result sets on Claude Opus 4.8.
#
# This must run with the Trn1 reward server UP and reachable, because every
# compile/verify/profile call hits real Trainium silicon. It cannot run from a
# laptop with no Neuron SDK.
#
# Prerequisites (export before running):
#   REWARD_URL     the reward server's direct URL — http://<private-ip>:5050/reward
#                  (RewardUrlPrivate stack output from reward_server_cdk).
#                  Reachability requires a real network path into the VPC
#                  (VPN / Direct Connect / SSM tunnel) since the Trn1 has no
#                  public IP.
#   AWS_REGION     default us-east-1 — used for the SigV4 signature this script
#                  computes and sends. The reward server does not verify that
#                  signature (see reward_server/auth.py); network position is
#                  its access control. Sending it is kept for
#                  forward-compatibility, not because anything checks it today.
#                  This is an open item pending a decision on what MCP-direct
#                  access to the reward server should actually require.
#   AGENTCORE_ARN  the deployed AgentCore runtime ARN (for the NKIGen-Bench run)
#
# Produces (dated), replacing the retired opus-4-7 / haiku / sample runs:
#   results/e2e_softmax_<date>.json                 (Table 2 — single kernel)
#   results/p1_4_shape_sweep_<date>.json            (Table 3 — NKI vs baseline)
#   results/nkibench250_agentcore_v4_<date>.json    (150-task subset headline)
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DATE="$(date +%Y-%m-%d)"
REGION="${AWS_REGION:-us-east-1}"
OPUS_ID="us.anthropic.claude-opus-4-8"

: "${REWARD_URL:?export REWARD_URL=http://<private-ip>:5050/reward}"

echo "== 1/3: shape sweep (NKI vs torch.compile(openxla) baseline) =="
python "$HERE/p1_4_shape_sweep.py" \
  --reward-url "$REWARD_URL" \
  --region "$REGION" \
  --warmup 5 --measure 50 \
  --output "$REPO/results/p1_4_shape_sweep_${DATE}.json"

echo "== 2/3: fix-loop trace on Opus 4.8 (single-kernel loop illustration) =="
python "$HERE/p1_5_fix_loop_demo.py" \
  --reward-url "$REWARD_URL" \
  --model-id "$OPUS_ID" \
  --region "$REGION" \
  --output "$REPO/results/fix_loop_trace_opus48_${DATE}.json"

if [[ -n "${AGENTCORE_ARN:-}" ]]; then
  echo "== 3/3: NKIGen-Bench 150-task subset via AgentCore (Opus 4.8) =="
  OPUS_MODEL_ID="$OPUS_ID" python "$HERE/eval_nkibench_agentcore.py" \
    --backend agentcore \
    --agentcore-arn "$AGENTCORE_ARN" \
    --region "$REGION" \
    --levels 1 2 3 --num-tasks 50 --max-turns 10 \
    --output "$REPO/results/nkibench250_agentcore_v4_${DATE}.json"
else
  echo "== 3/3: SKIPPED — set AGENTCORE_ARN to run the NKIGen-Bench subset =="
fi

echo "Done. New results written under results/ dated ${DATE}."
