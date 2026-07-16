#!/usr/bin/env bash
#
# Orchestrate a full deploy, AgentCore-first: AgentCore runtime -> its
# execution role ARN -> reward_server_cdk (Trn1, SG-to-SG, no gateway),
# then repoint-agentcore.sh to wire the two together.
#
# Deploying AgentCore first means its execution role ARN is known and fixed
# before reward_server_cdk ever runs. reward_server_cdk grants no IAM
# permissions on that ARN — access to the reward server is controlled
# entirely by security groups (see reward_server_cdk/README.md), and
# reward-server-stack.ts does not read the `-c agentCoreExecutionRoleArn=...`
# context that this script passes; it is accepted only for compatibility
# with callers that still pass it. The dependency this ordering can't
# remove: AgentCore's REWARD_SERVER_URL / VPC network config still isn't
# known until AFTER the reward server exists, so AgentCore is redeployed
# once, in place, by repoint-agentcore.sh at the end. That's the same "the
# thing deployed second can't be configured with its dependency's real
# values yet" constraint, just moved to the other side — it doesn't
# disappear, but this ordering means only ONE resource (AgentCore) is ever
# deployed twice, instead of reward_server_cdk needing two full `cdk deploy`
# passes.
#
# Steps:
#   1. Deploy AgentCore (agentcore/deploy.sh's raw-CLI path, or the
#      agentcore_cdk/atxnkiagent/ CLI-scaffolded path — see the ordering
#      note below on which one this script drives and why).
#   2. Look up its execution role ARN via the bedrock-agentcore-control API
#      directly (list-agent-runtimes -> agentRuntimeId, then
#      get-agent-runtime -> roleArn) — NOT by parsing `agentcore status`
#      text/JSON output, which does not surface the execution role ARN as a
#      documented field. Passed through to reward_server_cdk's `-c` context
#      for callers that expect it; the stack itself does not act on it.
#   3. Single reward_server_cdk deploy (Trn1, SG-to-SG, no gateway — there
#      is no IAM grant to attach).
#   4. repoint-agentcore.sh patches AgentCore's REWARD_SERVER_URL and VPC
#      network config to point at the now-fully-provisioned reward server,
#      and redeploys the runtime in place.
#
# This is a HIGH-RISK operation (creates/modifies live AWS resources,
# including a trn1.2xlarge — check quota and cost before running). Confirms
# before each stage unless CICD_YES=1 is set.
#
# Usage:
#   ./cicd/deploy.sh [--region us-east-1] [--agent-name atx-nki-agent]
#
# For a fully unattended run (no prompts), set CICD_YES=1.
#   CICD_YES=1 ./cicd/deploy.sh                   # unattended
#
# AgentCore deployment path used for step 1/2: this script drives
# agentcore_cdk/atxnkiagent/ (the `agentcore` CLI project scaffold) when
# present, because that is the same path repoint-agentcore.sh (step 4)
# already assumes and patches — see cicd/README.md's note on the two
# competing AgentCore deployment surfaces exist in this repo with no single
# declared canonical path; this script picks the one repoint-agentcore.sh
# can actually wire up automatically). Falls back to agentcore/deploy.sh's
# raw-CLI path only if agentcore_cdk/.../cdk/ isn't present, in which case
# you'll need to export REWARD_SERVER_URL yourself for repoint-agentcore.sh's
# equivalent step, since that script only patches the CDK-scaffolded project.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REGION="${AWS_REGION:-us-east-1}"
# Must match what the CDK-scaffolded path (agentcore_cdk/atxnkiagent/) actually
# deploys: Bedrock AgentCore derives the runtime name as
# "<agentcore.json 'name'>_<runtime 'name'>" — i.e. "atxnkiagent" + "nki_agent"
# (see infrastructure/agentcore_cdk/atxnkiagent/agentcore/agentcore.json) —
# not an arbitrary string. AGENT_NAME is only honored by the agentcore/deploy.sh
# raw-CLI fallback path below; the CDK path ignores it and reads agentcore.json
# instead, so this default has to track that file's name/runtime fields, or the
# post-deploy list-agent-runtimes lookup a few lines down will find nothing.
AGENT_NAME="${AGENT_NAME:-atxnkiagent_nki_agent}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --agent-name) AGENT_NAME="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

section "deploy: AgentCore runtime '${AGENT_NAME}' — region ${REGION}"
echo "This creates the Bedrock AgentCore Runtime and its execution role."
echo "REWARD_SERVER_URL isn't known yet (the reward server doesn't exist"
echo "yet) — AgentCore is redeployed once more at the end, in place, once"
echo "it does."
confirm "Proceed with AgentCore deploy in ${REGION}?" || { echo "aborted"; exit 1; }

if [[ -d "${AGENTCORE_CDK_DIR}" ]]; then
  echo "Deploying via the agentcore_cdk/atxnkiagent/ path (agentcore create/deploy)."
  echo "Installing the @aws/agentcore CLI as a local devDependency of"
  echo "agentcore/cdk/ (no global npm install required)."
  AGENTCORE_PROJECT_DIR="$(cd "$(dirname "${AGENTCORE_CDK_DIR}")/.." && pwd)"
  ( cd "${AGENTCORE_CDK_DIR}" && npm install )

  AGENTCORE_BIN="${AGENTCORE_CDK_DIR}/node_modules/.bin/agentcore"
  AGENTCORE_JSON="${AGENTCORE_PROJECT_DIR}/agentcore/agentcore.json"
  AWS_TARGETS_JSON="${AGENTCORE_PROJECT_DIR}/agentcore/aws-targets.json"

  # Validate any existing agentcore.json/aws-targets.json BEFORE calling into
  # any CDK CLI command (including `cdk bootstrap` below) — cdk.json's
  # "app": "node dist/bin/cdk.js" means EVERY cdk invocation, bootstrap
  # included, loads this project's own app entrypoint to discover stacks/
  # environments, and that entrypoint itself throws "No deployment targets
  # configured" if aws-targets.json has no targets. So this check has to run
  # first, or bootstrap fails with the exact same opaque error this
  # validation exists to catch and explain.
  #
  # A prior interrupted/partial run can leave both files present but empty
  # (agentcore.json's runtimes: [] and/or aws-targets.json's []), which used
  # to be silently treated as "already initialized, skip regeneration".
  # validate_agentcore_config prints the reason and returns non-zero for any
  # of: missing, invalid JSON, empty/missing aws-targets.json array, missing
  # nki_agent runtime entry, or a target with no valid 12-digit account id.
  if [[ -f "${AGENTCORE_JSON}" || -f "${AWS_TARGETS_JSON}" ]]; then
    if ! validate_agentcore_config "${AGENTCORE_JSON}" "${AWS_TARGETS_JSON}"; then
      echo
      echo "agentcore.json/aws-targets.json exist but failed validation (see above)."
      if [[ "${CICD_YES:-0}" == "1" ]]; then
        echo "CICD_YES=1 — regenerating automatically (non-interactive)."
        choice="regenerate"
      else
        echo "Choose how to proceed:"
        echo "  [r]egenerate — delete and re-scaffold both files automatically (recommended;"
        echo "                 both are gitignored, locally-generated config, not deployed state)"
        echo "  [e]dit       — stop here so you can fix the file(s) by hand, then re-run this script"
        echo "  [a]bort      — stop without changing anything"
        while true; do
          read -r -p "regenerate / edit / abort? [r/e/a] " reply
          case "${reply}" in
            r|R|regenerate) choice="regenerate"; break ;;
            e|E|edit) choice="edit"; break ;;
            a|A|abort) choice="abort"; break ;;
            *) echo "please answer r, e, or a" ;;
          esac
        done
      fi
      case "${choice}" in
        edit)
          echo
          echo "Leaving the files as-is. Fix them by hand and re-run this script:"
          echo "  agentcore.json:    ${AGENTCORE_JSON}"
          echo "  aws-targets.json:  ${AWS_TARGETS_JSON}"
          echo "See infrastructure/agentcore_cdk/atxnkiagent/agentcore/.llm-context/aws-targets.ts"
          echo "for the aws-targets.json schema (name/description/account/region)."
          exit 1
          ;;
        abort)
          echo "aborted"
          exit 1
          ;;
        regenerate)
          echo "regenerating: removing the invalid config so it can be re-scaffolded."
          rm -f "${AGENTCORE_JSON}" "${AWS_TARGETS_JSON}"
          ;;
      esac
    fi
  fi

  # Locally-installed cdk binary, used for bootstrap further below once
  # agentcore.json/aws-targets.json are known-valid. Bootstraps this
  # account/region's default CDK environment (qualifier "hnb659fds", stack
  # "CDKToolkit") before `agentcore deploy` runs — idempotent, so safe on
  # every deploy (no-ops once current). This is deliberately the *default*
  # bootstrap, not a dedicated qualifier: `agentcore deploy`'s own internal
  # CDK invocation always targets the default qualifier regardless of what
  # this project's cdk.json declares, so a dedicated qualifier here bought no
  # real isolation — it just left a second, unused toolkit stack in the
  # account every time. Standardizing on one shared CDKToolkit means
  # `agentcore deploy` deploys on top of the same bootstrap this script
  # already ensured exists, instead of silently auto-bootstrapping its own.
  # Called by its resolved node_modules/.bin path rather than via `npx` (same
  # reasoning as AGENTCORE_BIN below: npx can silently resolve a different,
  # incompatible cdk than the one pinned in agentcore/cdk/package.json).
  AGENTCORE_CDK_BIN="${AGENTCORE_CDK_DIR}/node_modules/.bin/cdk"
  if [[ -f "${AGENTCORE_JSON}" ]]; then
    echo "agentcore.json already present — project already initialized, skipping 'create'."
  else
    # agentcore.json/aws-targets.json are gitignored on purpose (they hold real
    # account IDs/ARNs/endpoints once deployed) and must be regenerated on a
    # fresh checkout. `agentcore create` always scaffolds a brand-new project
    # DIRECTORY under its cwd — it can't write config into an existing
    # agentcore/cdk/ in place — so we scaffold into a throwaway temp dir and
    # move ONLY the generated config (agentcore.json / aws-targets.json /
    # .env.local) into the existing project, keeping the existing cdk/ app and
    # ported app/ code untouched. Then patch the two fields the scaffold can't
    # know: the runtime name (must be 'nki_agent' so this stack deploys as
    # 'atxnkiagent_nki_agent', which repoint-agentcore.sh + the lookups here
    # expect) and the deploy target's account/region.
    echo "agentcore.json missing — regenerating config via a temp scaffold (keeps existing cdk/ + app/)."
    ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
    TMP_SCAFFOLD="$(mktemp -d)"
    # Non-interactive scaffold; same framework/provider as the real project.
    "${AGENTCORE_BIN}" create --name atxnkiagent --defaults --framework Strands \
      --model-provider Bedrock --memory none --skip-git \
      --output-dir "${TMP_SCAFFOLD}" >/dev/null
    cp "${TMP_SCAFFOLD}/atxnkiagent/agentcore/agentcore.json"  "${AGENTCORE_JSON}"
    cp "${TMP_SCAFFOLD}/atxnkiagent/agentcore/aws-targets.json" "${AWS_TARGETS_JSON}"
    [[ -f "${TMP_SCAFFOLD}/atxnkiagent/agentcore/.env.local" ]] && \
      cp "${TMP_SCAFFOLD}/atxnkiagent/agentcore/.env.local" "${AGENTCORE_PROJECT_DIR}/agentcore/.env.local"
    rm -rf "${TMP_SCAFFOLD}"
    # Patch runtime name + deploy target with python (json in-place).
    python3 - "${AGENTCORE_JSON}" "${AWS_TARGETS_JSON}" "${ACCOUNT_ID}" "${REGION}" <<'PY'
import json, sys
ac_path, tgt_path, account, region = sys.argv[1:5]
d = json.load(open(ac_path))
for rt in d.get("runtimes", []):
    # deploy.sh + repoint-agentcore.sh key off runtime name 'nki_agent'
    # (-> runtime 'atxnkiagent_nki_agent'); the scaffold names it 'atxnkiagent'.
    rt["name"] = "nki_agent"
    ev = {e["name"]: e for e in rt.get("envVars", [])}
    ev.setdefault("OPUS_MODEL_ID",   {"name": "OPUS_MODEL_ID",   "value": "us.anthropic.claude-opus-4-8"})
    ev.setdefault("REWARD_TIMEOUT_S",{"name": "REWARD_TIMEOUT_S","value": "600"})
    rt["envVars"] = list(ev.values())
json.dump(d, open(ac_path, "w"), indent=2); open(ac_path, "a").write("\n")
json.dump([{"name": "default", "description": "ATX NKI agent runtime",
            "account": account, "region": region}],
          open(tgt_path, "w"), indent=2); open(tgt_path, "a").write("\n")
print(f"  regenerated agentcore.json (runtime nki_agent) + aws-targets.json ({account}/{region})")
PY
  fi
  # Only now — after agentcore.json/aws-targets.json are known-valid, either
  # because they already were, or because the scaffold step just regenerated
  # them — is it safe to invoke any CDK CLI command. Every `cdk` invocation
  # loads cdk.json's "app": "node dist/bin/cdk.js" entrypoint (bootstrap
  # included), which reads these same two files and throws "No deployment
  # targets configured" if aws-targets.json has no targets — exactly the
  # failure this script's own validation above exists to catch first, with a
  # specific, actionable reason, instead of surfacing here as an opaque CDK
  # synthesis error.
  ( cd "${AGENTCORE_CDK_DIR}" && "${AGENTCORE_CDK_BIN}" bootstrap )
  ( cd "${AGENTCORE_PROJECT_DIR}" && "${AGENTCORE_BIN}" deploy --yes )
else
  echo "agentcore_cdk/.../cdk/ not found — falling back to agentcore/deploy.sh's raw-CLI path."
  echo "NOTE: repoint-agentcore.sh (step 4) only patches the CDK-scaffolded project's"
  echo "agentcore.json — it will NOT wire this path up automatically. You'll need to"
  echo "export REWARD_SERVER_URL yourself and re-run '${AGENTCORE_DIR}/deploy.sh deploy'"
  echo "after the reward server exists."
  # No REWARD_SERVER_URL yet on this first deploy — agentcore/deploy.sh hard-requires
  # it, so give it an obvious placeholder rather than letting an unset-variable error
  # look like a script bug. The real URL is applied by the fallback re-run above.
  REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://REWARD_SERVER_NOT_YET_DEPLOYED:5050}" \
    AGENT_NAME="${AGENT_NAME}" AWS_REGION="${REGION}" \
    "${AGENTCORE_DIR}/deploy.sh" create
fi

section "resolve AgentCore's execution role ARN"
echo "Querying bedrock-agentcore-control directly (not parsing 'agentcore"
echo "status' output — it does not expose the execution role ARN as a"
echo "documented field)."
AGENT_RUNTIME_ID="$(aws bedrock-agentcore-control list-agent-runtimes \
  --region "${REGION}" \
  --query "agentRuntimes[?agentRuntimeName=='${AGENT_NAME}'].agentRuntimeId | [0]" \
  --output text)"
if [[ -z "${AGENT_RUNTIME_ID}" || "${AGENT_RUNTIME_ID}" == "None" ]]; then
  echo "could not find an agent runtime named '${AGENT_NAME}' in ${REGION} via" >&2
  echo "list-agent-runtimes — check the name/region, or that the deploy above" >&2
  echo "actually completed." >&2
  exit 1
fi
AGENT_CORE_ROLE_ARN="$(aws bedrock-agentcore-control get-agent-runtime \
  --region "${REGION}" --agent-runtime-id "${AGENT_RUNTIME_ID}" \
  --query roleArn --output text)"
if [[ -z "${AGENT_CORE_ROLE_ARN}" || "${AGENT_CORE_ROLE_ARN}" == "None" ]]; then
  echo "get-agent-runtime returned no roleArn for runtime id ${AGENT_RUNTIME_ID}" >&2
  exit 1
fi
echo "execution role: ${AGENT_CORE_ROLE_ARN}"

section "deploy: reward_server_cdk (Trn1 reward server) — region ${REGION}"
echo "This provisions a trn1.2xlarge (billed hourly), a private-isolated VPC,"
echo "and PrivateLink endpoints (Bedrock/CloudWatch/etc.). The AgentCore runtime"
echo "calls the reward server DIRECTLY on 5050 (SG-to-SG, no API Gateway); the"
echo "reward server is reachable only from the runtime's client SG inside a"
echo "subnet with no public IP or internet route. The -c context below is passed"
echo "for backward compatibility and is ignored by the current (gateway-less) stack."
confirm "Proceed with reward_server_cdk deploy in ${REGION}?" || { echo "aborted"; exit 1; }

(
  cd "${REWARD_SERVER_CDK_DIR}"
  npx cdk bootstrap
  npx cdk deploy --require-approval never \
    -c agentCoreExecutionRoleArn="${AGENT_CORE_ROLE_ARN}"
)

section "repoint-agentcore.sh — wire AgentCore at the reward server"
confirm "Run repoint-agentcore.sh (patches agentcore.json, runs 'agentcore deploy --yes')?" \
  || { echo "skipped — re-run reward_server_cdk/repoint-agentcore.sh manually when ready"; exit 0; }
"${REWARD_SERVER_CDK_DIR}/repoint-agentcore.sh" --region "${REGION}"

section "deploy complete"
