#!/usr/bin/env bash
# Deploy the ATX NKI agent to Bedrock AgentCore Runtime.
#
# Prereqs:
#   - `uv sync` run in this directory (agentcore/) — pulls in
#     bedrock-agentcore-cli as a pinned [dependency-groups].dev entry, so
#     `agentcore` below resolves via `uv run` from this project's own
#     lockfile instead of assuming a global pip install on PATH.
#   - AWS credentials with bedrock:Invoke* + iam:CreateRole + bedrock-agentcore:* perms
#   - REWARD_SERVER_URL exported — the reward server's direct private IP:5050
#     (RewardUrlPrivate stack output from reward_server_cdk), e.g.
#     http://10.42.0.51:5050. The runtime calls this DIRECTLY: there is no
#     API Gateway in the path (see reward_server_cdk/README.md).
#   - No REWARD_AUTH_TOKEN and no SigV4 signing needed: auth is enforced at
#     the network layer. The reward server sits in a PRIVATE_ISOLATED subnet
#     (no public IP, no internet route) and its security group accepts 5050
#     only from this runtime's client SG — the runtime's ENIs are the only
#     thing that can reach the port. There is no IAM grant to attach.
#   - Optional: QWEN_CMI_ARN exported (Bedrock CMI ARN for Qwen3-Coder-30B+SFT-v4)
#
# NOTE: this is the pip-distributed `bedrock-agentcore-cli` — a different
# package from the npm-distributed `@aws/agentcore` CLI that
# agentcore_cdk/atxnkiagent/ (and cicd/deploy.sh's primary path) uses. Both
# install a same-named `agentcore` binary; see cicd/README.md's note on the two competing deployment surfaces.
# This script's `agentcore` always means the pip one, resolved via `uv run`.
#
# Usage:
#   ./deploy.sh create   # one-time scaffold + IAM role + endpoint
#   ./deploy.sh deploy   # subsequent code-only deploys
#   ./deploy.sh destroy  # tear down endpoint + role
#
set -euo pipefail

AGENT_NAME="${AGENT_NAME:-atx-nki-agent}"
REGION="${AWS_REGION:-us-east-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ACTION="${1:-deploy}"

: "${REWARD_SERVER_URL:?must export REWARD_SERVER_URL}"

case "$ACTION" in
    create)
        echo "=> scaffolding agentcore project for $AGENT_NAME in $REGION"
        cd "$HERE"
        uv run agentcore create \
            --name "$AGENT_NAME" \
            --region "$REGION" \
            --entrypoint app:handler \
            --requirements "$HERE/../../requirements-agentcore.txt"
        ;;
    deploy)
        echo "=> deploying $AGENT_NAME to $REGION"
        cd "$HERE"
        uv run agentcore deploy \
            --name "$AGENT_NAME" \
            --region "$REGION" \
            --env "REWARD_SERVER_URL=$REWARD_SERVER_URL" \
            --env "OPUS_MODEL_ID=${OPUS_MODEL_ID:-anthropic.claude-opus-4-8}" \
            --env "QWEN_CMI_ARN=${QWEN_CMI_ARN:-}"
        echo "=> done. invoke with:"
        echo "   uv run agentcore invoke --name $AGENT_NAME --payload-file payload.json"
        ;;
    destroy)
        echo "=> destroying $AGENT_NAME in $REGION"
        cd "$HERE"
        uv run agentcore destroy --name "$AGENT_NAME" --region "$REGION"
        ;;
    *)
        echo "usage: $0 {create|deploy|destroy}"
        exit 1
        ;;
esac
