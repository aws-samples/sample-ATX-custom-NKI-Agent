#!/usr/bin/env bash
#
# Independent drift check for the AgentCore runtime's live
# network configuration.
#
# The reward server's ONLY access control is a security-group rule scoped to
# AgentCoreClientSg (see infrastructure/reward_server_cdk/lib/reward-server-stack.ts) —
# there is no application-layer auth by default and no API Gateway/IAM
# authorizer in front of it anymore. That means the single thing standing
# between the reward server and an unrestricted network path is the AgentCore
# runtime staying in VPC mode, attached to exactly that security group. This
# script provides an independent signal: it queries the live
# runtime directly via bedrock-agentcore-control (not the CDK template, which
# only describes intent, not what's actually deployed) and fails loudly if the
# runtime's real networkConfiguration has drifted from what the reward-server
# stack expects.
#
# This does NOT replace a preventive control (an IAM condition-key policy on
# whoever is authorized to call CreateAgentRuntime/UpdateAgentRuntime — that
# has to be an account-level IAM action, not CDK code in
# this repo). This script is a detective control: it tells you drift already
# happened. A preventive control stops the drift from being
# possible in the first place. Both are recommended; this repo can only
# provide the former as code.
#
# Usage:
#   ./cicd/check-agentcore-network-drift.sh [--region us-east-1] [--agent-name atxnkiagent_nki_agent]
#
# Exit codes: 0 = matches expected config, 1 = drift detected or lookup failed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REGION="${AWS_REGION:-us-east-1}"
# Matches cicd/deploy.sh's own default -- Bedrock AgentCore derives the
# runtime name as "<agentcore.json 'name'>_<runtime 'name'>", not an
# arbitrary string. See cicd/deploy.sh's own comment on AGENT_NAME for why.
AGENT_NAME="${AGENT_NAME:-atxnkiagent_nki_agent}"
REWARD_STACK_NAME="${REWARD_STACK_NAME:-TrainiumRewardServer}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --agent-name) AGENT_NAME="$2"; shift 2 ;;
    --reward-stack) REWARD_STACK_NAME="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

section "resolve expected network config from the reward_server_cdk stack outputs"
EXPECTED_CLIENT_SG="$(aws cloudformation describe-stacks --region "${REGION}" \
  --stack-name "${REWARD_STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='AgentCoreClientSgId'].OutputValue | [0]" \
  --output text 2>/dev/null || true)"
EXPECTED_SUBNETS_RAW="$(aws cloudformation describe-stacks --region "${REGION}" \
  --stack-name "${REWARD_STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='VpcSubnets'].OutputValue | [0]" \
  --output text 2>/dev/null || true)"

if [[ -z "${EXPECTED_CLIENT_SG}" || "${EXPECTED_CLIENT_SG}" == "None" ]]; then
  echo "could not read AgentCoreClientSgId from stack '${REWARD_STACK_NAME}' in ${REGION}." >&2
  echo "Has reward_server_cdk been deployed? Set REWARD_STACK_NAME if it uses a different name." >&2
  exit 1
fi
echo "expected security group: ${EXPECTED_CLIENT_SG}"
echo "expected subnets: ${EXPECTED_SUBNETS_RAW}"

section "resolve the live AgentCore runtime's actual network configuration"
AGENT_RUNTIME_ID="$(aws bedrock-agentcore-control list-agent-runtimes \
  --region "${REGION}" \
  --query "agentRuntimes[?agentRuntimeName=='${AGENT_NAME}'].agentRuntimeId | [0]" \
  --output text 2>/dev/null || true)"
if [[ -z "${AGENT_RUNTIME_ID}" || "${AGENT_RUNTIME_ID}" == "None" ]]; then
  echo "could not find an agent runtime named '${AGENT_NAME}' in ${REGION} via" >&2
  echo "list-agent-runtimes. Check the name/region, or that it has been deployed." >&2
  exit 1
fi

RUNTIME_JSON="$(aws bedrock-agentcore-control get-agent-runtime \
  --region "${REGION}" --agent-runtime-id "${AGENT_RUNTIME_ID}" --output json)"

# Real API field names (confirmed via `aws bedrock-agentcore-control
# get-agent-runtime help`), NOT the agentcore.json CDK-project-schema field
# names (networkMode/networkConfig) -- the two are different naming
# conventions for the same concept, and the API is the ground truth here.
LIVE_NETWORK_MODE="$(echo "${RUNTIME_JSON}" | python3 -c \
  "import json,sys; print(json.load(sys.stdin).get('networkConfiguration', {}).get('networkMode', 'UNKNOWN'))")"
LIVE_SGS="$(echo "${RUNTIME_JSON}" | python3 -c \
  "import json,sys; print(','.join(json.load(sys.stdin).get('networkConfiguration', {}).get('networkModeConfig', {}).get('securityGroups', [])))")"
LIVE_SUBNETS="$(echo "${RUNTIME_JSON}" | python3 -c \
  "import json,sys; print(','.join(json.load(sys.stdin).get('networkConfiguration', {}).get('networkModeConfig', {}).get('subnets', [])))")"

echo "live networkMode: ${LIVE_NETWORK_MODE}"
echo "live security groups: ${LIVE_SGS}"
echo "live subnets: ${LIVE_SUBNETS}"

section "compare"
DRIFT=0

if [[ "${LIVE_NETWORK_MODE}" != "VPC" ]]; then
  echo "DRIFT: networkMode is '${LIVE_NETWORK_MODE}', expected 'VPC'." >&2
  echo "  In PUBLIC mode the runtime's ENIs are not in AgentCoreClientSg at all --" >&2
  echo "  the reward server's sole access control no longer restricts this runtime." >&2
  DRIFT=1
fi

if [[ ",${LIVE_SGS}," != *",${EXPECTED_CLIENT_SG},"* ]]; then
  echo "DRIFT: live security groups (${LIVE_SGS}) do not include the expected" >&2
  echo "  AgentCoreClientSg (${EXPECTED_CLIENT_SG})." >&2
  DRIFT=1
fi

if [[ "${DRIFT}" == "1" ]]; then
  echo "" >&2
  echo "==> AgentCore runtime '${AGENT_NAME}' network configuration has drifted" >&2
  echo "    from what infrastructure/reward_server_cdk expects. The reward server's" >&2
  echo "    ONLY access control (the RewardSg-to-AgentCoreClientSg ingress rule) may" >&2
  echo "    no longer be restricting who can reach it. Investigate before this runtime" >&2
  echo "    is invoked again." >&2
  exit 1
fi

echo "no drift detected: networkMode=VPC, AgentCoreClientSg present in live security groups."
exit 0
