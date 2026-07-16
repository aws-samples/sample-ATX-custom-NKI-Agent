#!/usr/bin/env bash
# Automate the AgentCore re-point after the reward server is deployed.
#
# Reads the reward server's direct private IP:5050, client security group,
# and subnets from the CDK stack outputs, writes them into the AgentCore
# project's agentcore.json, and runs an in-place `agentcore deploy`. No
# manual editing.
#
# The runtime is put in VPC network mode (client SG + subnets) so it can reach
# the PrivateLink endpoints (Bedrock, CloudWatch, SSM, CodeArtifact) — never
# the public internet. Reward-server calls themselves go DIRECTLY to the Trn1
# over SG-to-SG networking: there is no API Gateway, VPC Link, or NLB in the
# path (removed — its only contribution was SigV4/IAM auth, redundant given
# the network posture, plus a 29s integration timeout that capped real
# compile/verify calls). The runtime's security-group membership is now the
# entire access control on that connection; there is no shared secret and no
# IAM grant to manage for it either.
#
# Usage:
#   ./repoint-agentcore.sh [--region us-east-1] [--stack TrainiumRewardServer]
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
STACK="TrainiumRewardServer"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --stack)  STACK="$2";  shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
AGENTCORE_DIR="$(cd "$HERE/../agentcore_cdk/atxnkiagent" && pwd)"
AGENTCORE_JSON="$AGENTCORE_DIR/agentcore/agentcore.json"

echo "==> reading stack outputs from $STACK ($REGION)"
# Direct reward server URL (private IP:5050) — the runtime calls it directly,
# no API Gateway. This is the RewardUrlPrivate stack output.
REWARD_API_URL="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='RewardUrlPrivate'].OutputValue" --output text)"
STATUS_PARAM="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='BootstrapStatusParam'].OutputValue" --output text)"
CLIENT_SG="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='AgentCoreClientSgId'].OutputValue" --output text)"
SUBNETS="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='VpcSubnets'].OutputValue" --output text)"
[[ -n "$REWARD_API_URL" && -n "$CLIENT_SG" && -n "$SUBNETS" ]] \
  || { echo "stack outputs not found"; exit 1; }

# Gate on bootstrap status so we never re-point at a half-provisioned server.
STATUS_PARAM="${STATUS_PARAM:-/nki-agent/bootstrap-status}"
STATUS="$(aws ssm get-parameter --region "$REGION" --name "$STATUS_PARAM" \
  --query 'Parameter.Value' --output text 2>/dev/null || echo PENDING)"
if [[ "$STATUS" != "OK" ]]; then
  echo "reward server bootstrap not OK (status: $STATUS)."
  echo "  If PENDING, the instance is still installing — wait and retry."
  echo "  If FAILED, check /var/log/nki-bootstrap.log on the instance."
  exit 1
fi

echo "==> reward server (direct): $REWARD_API_URL (VPC mode, client SG $CLIENT_SG)"
echo "==> patching $AGENTCORE_JSON (env + VPC networkConfig)"
python3 - "$AGENTCORE_JSON" "$REWARD_API_URL" "$CLIENT_SG" "$SUBNETS" <<'PY'
import json, sys
path, url, client_sg, subnets = sys.argv[1:5]
d = json.load(open(path))
for rt in d.get("runtimes", []):
    env_vars = rt.get("envVars", [])
    found = False
    for ev in env_vars:
        if ev["name"] == "REWARD_SERVER_URL":
            ev["value"] = url
            found = True
    if not found:
        # A freshly-registered runtime (agentcore add agent) starts with
        # envVars: [] — there is nothing to update, so REWARD_SERVER_URL
        # must be inserted, not just patched in place. Without this, the
        # runtime falls back to app.py's dummy default hostname
        # ("reward-server.internal") and every reward-server call fails
        # with a DNS resolution error, even though everything else (VPC
        # mode, subnets, SG, IAM grant) is correctly wired.
        env_vars.append({"name": "REWARD_SERVER_URL", "value": url})
    rt["envVars"] = env_vars
    # Legacy cleanup: strip any REWARD_AUTH_TOKEN env var left over from an
    # agentcore.json written before the token was retired entirely. Nothing
    # generates, stores, or reads this token anymore — reward_server/auth.py
    # has no bearer-token fallback and reward_server_cdk no longer provisions
    # an SSM parameter for it — so this is a no-op on any current deployment,
    # kept only so re-running this script against an old config migrates it.
    rt["envVars"] = [ev for ev in rt.get("envVars", []) if ev["name"] != "REWARD_AUTH_TOKEN"]
    # Put the runtime in VPC mode so it reaches the PrivateLink endpoints
    # (execute-api, Bedrock, CloudWatch) privately. NetworkConfig schema is
    # exactly {subnets, securityGroups} — no nested networkMode (an extra key
    # silently drops the whole config -> null ENIs).
    rt["networkMode"] = "VPC"
    rt["networkConfig"] = {
        "subnets": [s for s in subnets.split(",") if s],
        "securityGroups": [client_sg],
    }
json.dump(d, open(path, "w"), indent=2)
open(path, "a").write("\n")
print("  updated (networkMode=VPC, REWARD_AUTH_TOKEN removed)")
PY

echo "==> agentcore deploy (in-place)"
cd "$AGENTCORE_DIR"
# npx can't see agentcore/cdk/'s node_modules from this project-root cwd (a
# parent dir) — call the locally-installed @aws/agentcore binary by its
# resolved path instead, matching cicd/deploy.sh's convention.
AGENTCORE_BIN="$AGENTCORE_DIR/agentcore/cdk/node_modules/.bin/agentcore"
( cd "$AGENTCORE_DIR/agentcore/cdk" && npm install )
"$AGENTCORE_BIN" validate
"$AGENTCORE_BIN" deploy --yes

echo "==> done. Runtime now calls the reward server directly at $REWARD_API_URL"
echo "    (SG-to-SG, no signing, no gateway — the runtime's security-group"
echo "    membership in AgentCoreClientSg is the entire access control here)."
