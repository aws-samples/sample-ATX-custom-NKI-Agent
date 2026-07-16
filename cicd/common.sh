#!/usr/bin/env bash
#
# Shared helpers for the cicd/ scripts. Sourced by every other script in
# this directory except itself.
#
# Resolves the repository root and the key sub-project directories referenced
# throughout cicd/ so every script operates from a stable, absolute location
# regardless of where it was invoked from.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Deployable / dependency-bearing surfaces in this repo, per its actual
# layout: the repo root has its own uv project (dev tooling only —
# package = false, no runtime code); mcp/ is a standalone uv project with
# real runtime deps and stays at the repo root (workstation-only — the MCP
# server is uvx-launched locally, never deployed to AWS). Everything that
# actually gets deployed to AWS — the two CDK apps and the app code they
# ship — lives under infrastructure/: agentcore/ and reward_server/ are
# each standalone uv/requirements-based projects (reward_server/ is
# requirements.txt-only, installed on the Trn1 itself via CodeArtifact, not
# a uv project run from a developer machine); agentcore_cdk/ and
# reward_server_cdk/ are the CDK apps that provision the AWS side. See
# docs/deployment-architecture.md for the full map.
MCP_DIR="${REPO_ROOT}/mcp"
INFRA_DIR="${REPO_ROOT}/infrastructure"
AGENTCORE_DIR="${INFRA_DIR}/agentcore"
AGENTCORE_CDK_DIR="${INFRA_DIR}/agentcore_cdk/atxnkiagent/agentcore/cdk"
REWARD_SERVER_CDK_DIR="${INFRA_DIR}/reward_server_cdk"
REWARD_SERVER_DIR="${INFRA_DIR}/reward_server"

# Print a section heading.
section() {
  printf '\n\033[1m==> %s\033[0m\n' "$*"
}

# True if the given command exists on PATH.
have() {
  command -v "$1" >/dev/null 2>&1
}

# Prompt for an explicit "yes" before a destructive action. Skipped entirely
# if CICD_YES=1 is set (non-interactive / CI use), matching the `--yes`
# convention the `agentcore` CLI itself already uses.
confirm() {
  local prompt="$1"
  if [[ "${CICD_YES:-0}" == "1" ]]; then
    return 0
  fi
  read -r -p "${prompt} [y/N] " reply
  [[ "${reply}" =~ ^[Yy]$ ]]
}

# Validate an existing agentcore.json + aws-targets.json pair before trusting
# them as "already initialized". A prior interrupted/partial scaffold (e.g.
# deploy.sh cancelled mid-run, or a hand-edited file) can leave both files
# present on disk but syntactically or semantically empty — agentcore.json
# with runtimes: [] and/or aws-targets.json as []  — which the CDK app only
# surfaces later, opaquely, as "No deployment targets configured" at synth
# time. Checking here gives a specific, actionable reason instead.
#
# Args: $1 = path to agentcore.json, $2 = path to aws-targets.json
# Returns 0 if both files are valid JSON AND semantically complete (at least
# one deployment target with a 12-digit account id, and an "nki_agent"
# runtime entry in agentcore.json); non-zero otherwise, printing the reason
# to stderr. Either file may be individually absent — a fresh checkout with
# neither file present is NOT a validation failure; only "present but
# broken" is. Requires python3 (already a hard dependency of this script's
# own regeneration path).
validate_agentcore_config() {
  local agentcore_json="$1"
  local aws_targets_json="$2"

  # Neither file exists yet — nothing to validate; the caller's own
  # regeneration path handles a from-scratch scaffold.
  if [[ ! -f "${agentcore_json}" && ! -f "${aws_targets_json}" ]]; then
    return 0
  fi

  python3 - "${agentcore_json}" "${aws_targets_json}" <<'PY'
import json
import re
import sys

agentcore_path, targets_path = sys.argv[1:3]
problems = []


def load_json(path, label):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        problems.append(f"{label} is missing: {path}")
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        problems.append(f"{label} is not valid JSON ({path}): {e}")
        return None


agentcore = load_json(agentcore_path, "agentcore.json")
targets = load_json(targets_path, "aws-targets.json")

if targets is not None:
    if not isinstance(targets, list) or len(targets) == 0:
        problems.append(
            f"aws-targets.json ({targets_path}) has no deployment targets "
            "(must be a non-empty JSON array)"
        )
    else:
        account_re = re.compile(r"^[0-9]{12}$")
        for i, t in enumerate(targets):
            if not isinstance(t, dict):
                problems.append(f"aws-targets.json entry {i} is not an object")
                continue
            account = t.get("account")
            if not isinstance(account, str) or not account_re.match(account):
                problems.append(
                    f"aws-targets.json entry {i} (name={t.get('name')!r}) has an "
                    f"invalid account id: {account!r} (must be exactly 12 digits)"
                )
            if not t.get("region"):
                problems.append(
                    f"aws-targets.json entry {i} (name={t.get('name')!r}) is missing 'region'"
                )

if agentcore is not None:
    runtimes = agentcore.get("runtimes")
    if not isinstance(runtimes, list) or not any(
        isinstance(rt, dict) and rt.get("name") == "nki_agent" for rt in runtimes
    ):
        problems.append(
            f"agentcore.json ({agentcore_path}) has no 'nki_agent' entry in "
            "'runtimes' (deploy.sh/repoint-agentcore.sh key off this exact name)"
        )

if problems:
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PY
}
