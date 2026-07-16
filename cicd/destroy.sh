#!/usr/bin/env bash
#
# Tear down everything cicd/deploy.sh created, in reverse order: AgentCore
# runtime first (so nothing is left pointing at a reward server about to
# disappear), then the reward_server_cdk stack (Trn1, VPC, API Gateway).
#
# HIGH-RISK / DESTRUCTIVE. Confirms before each stage unless CICD_YES=1 is
# set. Per this repo's own reward_server_cdk/README.md, the reward server is
# stateless (artifacts go to S3/CloudWatch) so destroying it is safe from a
# data-loss standpoint — but it is still a live-infrastructure deletion.
#
# Two out-of-band resource types recurringly block the reward_server_cdk
# stack's own VPC/subnet/security-group deletion (the stack is named
# TrainiumRewardServer on this branch, NkiRewardServer if first stood up from
# main — this script discovers whichever exists; see discover_reward_stack) —
# neither is in the CDK template, so CloudFormation can't tear them down itself and
# a plain `cdk destroy` gets stuck in DELETE_FAILED:
#
#   1. AgentCore's own VPC-mode ENIs (type "agentic_ai", tagged
#      AmazonBedrockAgentCoreManaged=true) on AgentCoreClientSg. These are
#      AWS-service-managed ("ela-attach" attachments — not force-detachable
#      via the EC2 API, confirmed) and release asynchronously sometime AFTER
#      the AgentCore runtime itself finishes deleting, not necessarily
#      within CloudFormation's own delete-retry window.
#   2. GuardDuty's auto-provisioned resources in any VPC it monitors (a
#      GuardDutyManagedSecurityGroup-* SG, and — if Runtime Monitoring is on
#      — a com.amazonaws.<region>.guardduty-data interface endpoint). These
#      exist outside the CDK template entirely and block subnet/VPC deletion
#      on every teardown in a GuardDuty-monitored account, not just
#      occasionally.
#
# This script handles both: it waits (bounded) for AgentCore's ENIs to clear
# before touching the reward server stack, and proactively removes any
# GuardDuty-managed resources on that stack's VPC before deleting it. If the
# stack still ends up DELETE_FAILED (the ENIs didn't release in time, or
# something otherwise unexpected), it retries a few times with backoff
# before failing with a specific, actionable message — not a raw CDK trace.
#
# Usage:
#   ./cicd/destroy.sh [--region us-east-1] [--agent-name atx-nki-agent]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REGION="${AWS_REGION:-us-east-1}"
AGENT_NAME="${AGENT_NAME:-atx-nki-agent}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --agent-name) AGENT_NAME="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Discover the reward-server stack rather than hardcoding a single name. The
# stack this branch's reward_server_cdk deploys is 'TrainiumRewardServer', but
# an environment first stood up from `main` is named 'NkiRewardServer' — a
# hardcoded name silently skips the wrong one and leaves an orphaned
# trn1.2xlarge billing. Discovery order:
#   1. Explicit override: REWARD_STACK_NAME env var, if the caller set one.
#   2. The CloudFormation stack that owns the reward-server EC2 instance,
#      found by its 'project=atx-nki-agent' tag (both naming eras tag it) via
#      the aws:cloudformation:stack-name tag on the instance.
#   3. Fall back to the two known literals, newest-first, whichever EXISTS.
# Fails loudly if none is found, instead of proceeding to delete nothing.
discover_reward_stack() {
  if [[ -n "${REWARD_STACK_NAME:-}" ]]; then
    echo "${REWARD_STACK_NAME}"
    return 0
  fi
  local from_instance
  from_instance="$(aws ec2 describe-instances --region "${REGION}" \
    --filters "Name=tag:project,Values=atx-nki-agent" \
              "Name=instance-state-name,Values=running,stopped,pending,stopping" \
    --query "Reservations[].Instances[].Tags[?Key=='aws:cloudformation:stack-name'].Value | [0][0]" \
    --output text 2>/dev/null || true)"
  if [[ -n "${from_instance}" && "${from_instance}" != "None" ]]; then
    echo "${from_instance}"
    return 0
  fi
  local candidate
  for candidate in TrainiumRewardServer NkiRewardServer; do
    if aws cloudformation describe-stacks --region "${REGION}" \
        --stack-name "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! REWARD_STACK_NAME="$(discover_reward_stack)"; then
  echo "could not find a reward-server stack in ${REGION} (looked for the" >&2
  echo "instance tagged project=atx-nki-agent, then the TrainiumRewardServer /" >&2
  echo "NkiRewardServer stack names). Nothing to destroy, or set" >&2
  echo "REWARD_STACK_NAME explicitly if it uses a different name." >&2
  exit 1
fi
echo "==> reward-server stack resolved to: ${REWARD_STACK_NAME}"

# Delete any GuardDuty-managed security group and PrivateLink endpoint found
# in the given VPC. Safe to call unconditionally (no-ops if GuardDuty VPC
# protection isn't enabled, or nothing has been created yet) — these
# resources are always safe to remove: they carry no customer data and
# GuardDuty recreates them automatically if it needs to.
clear_guardduty_vpc_resources() {
  local vpc_id="$1"
  [[ -z "${vpc_id}" ]] && return 0

  local endpoint_ids
  endpoint_ids="$(aws ec2 describe-vpc-endpoints --region "${REGION}" \
    --filters "Name=vpc-id,Values=${vpc_id}" "Name=tag:GuardDutyManaged,Values=true" \
    --query 'VpcEndpoints[].VpcEndpointId' --output text 2>/dev/null || true)"
  if [[ -n "${endpoint_ids}" ]]; then
    echo "removing GuardDuty-managed VPC endpoint(s) in ${vpc_id}: ${endpoint_ids}"
    # shellcheck disable=SC2086 # word-splitting is intentional: multiple IDs
    aws ec2 delete-vpc-endpoints --region "${REGION}" --vpc-endpoint-ids ${endpoint_ids} >/dev/null || true
  fi

  local sg_ids
  sg_ids="$(aws ec2 describe-security-groups --region "${REGION}" \
    --filters "Name=vpc-id,Values=${vpc_id}" "Name=tag:GuardDutyManaged,Values=true" \
    --query 'SecurityGroups[].GroupId' --output text 2>/dev/null || true)"
  for sg_id in ${sg_ids}; do
    echo "removing GuardDuty-managed security group in ${vpc_id}: ${sg_id}"
    # May still fail if the endpoint's ENIs haven't detached yet — the
    # retry loop around delete-stack (below) covers that; a single
    # best-effort attempt here just saves a retry cycle in the common case.
    aws ec2 delete-security-group --region "${REGION}" --group-id "${sg_id}" >/dev/null 2>&1 || true
  done
}

# Poll (bounded) for the given security group to have zero attached network
# interfaces. AgentCore's own VPC-mode ENIs release asynchronously after the
# runtime is deleted — this gives that cleanup a chance to finish before
# reward_server_cdk tries to delete the VPC those ENIs sit in, instead of
# finding out via a DELETE_FAILED several minutes into `cdk destroy`.
wait_for_enis_to_clear() {
  local sg_id="$1" max_attempts="${2:-20}" delay_s="${3:-30}"
  [[ -z "${sg_id}" ]] && return 0

  local attempt=1
  while (( attempt <= max_attempts )); do
    local remaining
    remaining="$(aws ec2 describe-network-interfaces --region "${REGION}" \
      --filters "Name=group-id,Values=${sg_id}" \
      --query 'length(NetworkInterfaces)' --output text 2>/dev/null || echo 0)"
    if [[ "${remaining}" == "0" ]]; then
      echo "no network interfaces remain on ${sg_id}"
      return 0
    fi
    echo "waiting for ${remaining} network interface(s) on ${sg_id} to release (attempt ${attempt}/${max_attempts})..."
    sleep "${delay_s}"
    ((attempt++))
  done
  echo "warning: ${sg_id} still has attached network interfaces after $((max_attempts * delay_s))s — proceeding anyway; the delete-stack retry loop below will keep trying." >&2
  return 0
}

section "destroy: AgentCore runtime '${AGENT_NAME}' — region ${REGION}"
if confirm "Destroy the AgentCore runtime '${AGENT_NAME}' in ${REGION}?"; then
  if [[ -d "${AGENTCORE_CDK_DIR}" ]]; then
    echo "Tearing down via the agentcore_cdk/atxnkiagent/ path:"
    echo "  'agentcore remove all --yes' resets agentcore.json's runtime/resource"
    echo "  entries to empty (confirmed via --dry-run: it targets exactly the"
    echo "  deployed runtime, not the project's own agentcore.json/aws-targets.json"
    echo "  files themselves), then 'agentcore deploy --yes' applies that removal"
    echo "  to AWS — tearing down the AgentCore Runtime and its execution role."
    AGENTCORE_PROJECT_DIR="$(cd "$(dirname "${AGENTCORE_CDK_DIR}")/.." && pwd)"
    ( cd "${AGENTCORE_CDK_DIR}" && npm install )
    # See cicd/deploy.sh's matching note: npx can't see AGENTCORE_CDK_DIR's
    # node_modules from AGENTCORE_PROJECT_DIR (a parent dir), so call the
    # installed binary by its resolved path instead.
    AGENTCORE_BIN="${AGENTCORE_CDK_DIR}/node_modules/.bin/agentcore"
    ( cd "${AGENTCORE_PROJECT_DIR}" && "${AGENTCORE_BIN}" remove all --yes )
    # `agentcore deploy --yes` fails with "No resources defined in project" if
    # the runtime was already torn down by an earlier destroy attempt (e.g.
    # this script being re-run after a partial failure below) — that is a
    # successful end state for this step, not an error, so don't let it abort
    # the rest of the script (set -e would otherwise kill the whole run here).
    ( cd "${AGENTCORE_PROJECT_DIR}" && "${AGENTCORE_BIN}" deploy --yes ) \
      || echo "agentcore deploy reported nothing to deploy — runtime already removed, continuing."
  else
    echo "agentcore_cdk/.../cdk/ not found — falling back to agentcore/deploy.sh's raw-CLI path."
    "${AGENTCORE_DIR}/deploy.sh" destroy
  fi

  # AgentCore's VPC-mode ENIs on AgentCoreClientSg release asynchronously,
  # sometimes well after the runtime deletion call above returns — give that
  # a bounded window to finish before touching the VPC those ENIs sit in.
  CLIENT_SG_ID="$(aws cloudformation describe-stacks --region "${REGION}" \
    --stack-name "${REWARD_STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='AgentCoreClientSgId'].OutputValue | [0]" \
    --output text 2>/dev/null || true)"
  if [[ -n "${CLIENT_SG_ID}" && "${CLIENT_SG_ID}" != "None" ]]; then
    section "waiting for AgentCore's VPC-mode ENIs to release from ${CLIENT_SG_ID}"
    wait_for_enis_to_clear "${CLIENT_SG_ID}"
  fi
else
  echo "skipped AgentCore teardown"
fi

section "destroy: reward_server_cdk (Trn1, VPC, API Gateway) — region ${REGION}"
echo "This deletes the trn1.2xlarge, its VPC, PrivateLink endpoints, NLB,"
echo "and API Gateway. The reward server is stateless (results already"
echo "written go to S3/CloudWatch), so this is safe from a data-loss"
echo "standpoint, but it IS a live-infrastructure deletion."
if confirm "Destroy the reward_server_cdk stack (${REWARD_STACK_NAME}) in ${REGION}?"; then
  REWARD_VPC_ID="$(aws cloudformation describe-stack-resources --region "${REGION}" \
    --stack-name "${REWARD_STACK_NAME}" \
    --query "StackResources[?ResourceType=='AWS::EC2::VPC'].PhysicalResourceId | [0]" \
    --output text 2>/dev/null || true)"
  if [[ -n "${REWARD_VPC_ID}" && "${REWARD_VPC_ID}" != "None" ]]; then
    clear_guardduty_vpc_resources "${REWARD_VPC_ID}"
  fi

  DESTROY_OK=0
  ( cd "${REWARD_SERVER_CDK_DIR}" && npx cdk destroy --force ) && DESTROY_OK=1

  if [[ "${DESTROY_OK}" != "1" ]]; then
    section "reward_server_cdk destroy failed — retrying (GuardDuty resources + DELETE_FAILED stack)"
    echo "cdk destroy did not complete cleanly. Re-checking for the two known,"
    echo "recurring blockers (GuardDuty-managed VPC resources, and AgentCore ENIs"
    echo "that hadn't released yet) before retrying the stack deletion directly."
    RETRY_MAX=6
    RETRY_DELAY_S=60
    retry=1
    while (( retry <= RETRY_MAX )); do
      STACK_STATUS="$(aws cloudformation describe-stacks --region "${REGION}" \
        --stack-name "${REWARD_STACK_NAME}" \
        --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "MISSING")"
      if [[ "${STACK_STATUS}" == "MISSING" ]]; then
        echo "${REWARD_STACK_NAME} no longer exists — destroy succeeded."
        DESTROY_OK=1
        break
      fi
      echo "retry ${retry}/${RETRY_MAX}: stack status is ${STACK_STATUS}"
      if [[ -n "${REWARD_VPC_ID}" && "${REWARD_VPC_ID}" != "None" ]]; then
        clear_guardduty_vpc_resources "${REWARD_VPC_ID}"
      fi
      if [[ -n "${CLIENT_SG_ID:-}" && "${CLIENT_SG_ID}" != "None" ]]; then
        wait_for_enis_to_clear "${CLIENT_SG_ID}" 1 0
      fi
      aws cloudformation delete-stack --region "${REGION}" --stack-name "${REWARD_STACK_NAME}" || true
      sleep "${RETRY_DELAY_S}"
      ((retry++))
    done
  fi

  if [[ "${DESTROY_OK}" != "1" ]]; then
    STACK_STATUS="$(aws cloudformation describe-stacks --region "${REGION}" \
      --stack-name "${REWARD_STACK_NAME}" \
      --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "MISSING")"
    if [[ "${STACK_STATUS}" != "MISSING" ]]; then
      echo "" >&2
      echo "==> ${REWARD_STACK_NAME} did not finish deleting (status: ${STACK_STATUS})." >&2
      echo "    This is almost always caused by AWS-service-managed network interfaces" >&2
      echo "    (type 'agentic_ai', tag AmazonBedrockAgentCoreManaged=true) that hadn't" >&2
      echo "    released yet — they are not force-detachable via the EC2 API and can" >&2
      echo "    take longer than this script's wait window to clear." >&2
      echo "" >&2
      echo "    What to do:" >&2
      echo "      1. Wait a few more minutes, then re-run: ./cicd/destroy.sh --region ${REGION}" >&2
      echo "         (safe to re-run — it will skip the already-deleted AgentCore runtime" >&2
      echo "         and pick up straight at the reward_server_cdk retry)." >&2
      echo "      2. If it is still stuck after several retries, check for lingering ENIs:" >&2
      echo "           aws ec2 describe-network-interfaces --region ${REGION} \\" >&2
      echo "             --filters Name=vpc-id,Values=<vpc-id-from-stack> Name=status,Values=in-use" >&2
      echo "         If any are tagged AmazonBedrockAgentCoreManaged=true and still attached" >&2
      echo "         after 15-20+ minutes, open an AWS Support case referencing the ENI IDs" >&2
      echo "         and the stack name — this is a known AWS-side cleanup lag, not something" >&2
      echo "         this script or the CDK app can force." >&2
      exit 1
    fi
  fi
else
  echo "skipped reward_server_cdk teardown"
fi

section "destroy complete"
