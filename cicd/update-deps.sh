#!/usr/bin/env bash
#
# Update dependencies across every Python and npm project in the repo, then
# re-scan for known vulnerabilities so a stale-but-vulnerable pin never
# silently survives an update run.
#
# Scope (mirrors common.sh's dependency-bearing surfaces):
#   - repo root       uv project (dev tooling only: pytest, ruff, pip-audit, flask)
#   - mcp/            uv project (runtime: mcp, boto3, pydantic, httpx)
#   - agentcore/      uv project (runtime: bedrock-agentcore, strands-agents, boto3)
#   - infrastructure/reward_server_cdk/                    npm project (CDK app)
#   - infrastructure/agentcore_cdk/atxnkiagent/agentcore/cdk/   npm project (CDK app, if present)
#   - infrastructure/reward_server/requirements.txt        requirements.txt-only
#     (installed on the Trn1 itself via CodeArtifact during bootstrap, not a
#     uv project — see common.sh, cicd/README.md). It has no lockfile, so
#     "update" here means resolving the current latest version of each
#     floor-pinned (>=) package and rewriting the floor to match, via the
#     cicd/update_requirements_txt.py helper (uses `uv pip compile --no-deps`
#     per package, so comments and the >= floor style are preserved instead
#     of being replaced with a fully resolved, ==-pinned lockfile). torch /
#     torch-neuronx / neuronx-cc are commented out on purpose (installed
#     separately via the Neuron SDK) and are left alone.
#
# Policy, matching security-check.sh: pip-audit is FATAL (drives the exit
# code); npm audit is ADVISORY (reported, never blocks). Re-scan runs after
# every update below, and the script still exits non-zero at the end if any
# pip-audit pass found a vulnerability — so the repo's deps get updated
# either way, but CI/you still find out. NOTICE/SBOM.json are regenerated
# from the *updated* lockfiles/node_modules before that final exit, so they
# stay current even on a failing pip-audit run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

status=0

section "uv lock --upgrade: repo root (dev tooling)"
(cd "${REPO_ROOT}" && uv lock --upgrade && uv sync)

section "uv lock --upgrade: mcp/"
(cd "${MCP_DIR}" && uv lock --upgrade && uv sync)

section "uv lock --upgrade: agentcore/"
(cd "${AGENTCORE_DIR}" && uv lock --upgrade && uv sync)

section "update: infrastructure/reward_server/requirements.txt"
(cd "${REPO_ROOT}" && uv run python cicd/update_requirements_txt.py "${REWARD_SERVER_DIR}/requirements.txt")

section "npm update: infrastructure/reward_server_cdk/"
(cd "${REWARD_SERVER_CDK_DIR}" && npm update --save && npm install)

if [[ -d "${AGENTCORE_CDK_DIR}" ]]; then
  section "npm update: infrastructure/agentcore_cdk/atxnkiagent/agentcore/cdk/"
  (cd "${AGENTCORE_CDK_DIR}" && npm update --save && npm install)
else
  section "skip: agentcore_cdk/.../cdk/ not present (agentcore create not yet run)"
fi

section "re-scan: pip-audit (repo root, fatal)"
(cd "${REPO_ROOT}" && uv run pip-audit --local) || status=$?

section "re-scan: pip-audit (mcp/, fatal)"
(cd "${MCP_DIR}" && uv run pip-audit --local) || status=$?

section "re-scan: pip-audit (agentcore/, fatal)"
(cd "${AGENTCORE_DIR}" && uv run pip-audit --local) || status=$?

section "re-scan: pip-audit (reward_server/requirements.txt, fatal)"
(
  cd "${REPO_ROOT}"
  # pip-audit's `-r requirements.txt` mode always builds its own throwaway
  # venv via the stdlib `venv.EnvBuilder`, which invokes `ensurepip` as an
  # in-process subprocess. On some macOS/uv-managed-Python combinations that
  # subprocess call segfaults/aborts (SIGABRT) regardless of the uv-managed
  # CPython version — a known class of issue with relocated/standalone
  # Python builds and macOS code signing, not specific to this repo. Side-
  # step it entirely by building the venv with `uv venv`/`uv pip install`
  # (uv's own installer, no `venv.EnvBuilder`/`ensurepip` involved) and
  # pointing pip-audit's `--local` mode at it via `PIPAPI_PYTHON_LOCATION`
  # instead of using `-r` — same audit coverage, no ensurepip in the path.
  tmp_venv="$(mktemp -d)/venv"
  trap 'rm -rf "$(dirname "${tmp_venv}")"' EXIT
  uv venv --python 3.12 "${tmp_venv}"
  uv pip install --python "${tmp_venv}/bin/python" pip -r "${REWARD_SERVER_DIR}/requirements.txt"
  PIPAPI_PYTHON_LOCATION="${tmp_venv}/bin/python" uvx pip-audit --local
) || status=$?

section "re-scan: npm audit (reward_server_cdk/, advisory)"
(cd "${REWARD_SERVER_CDK_DIR}" && npm audit --omit=dev) || true

if [[ -d "${AGENTCORE_CDK_DIR}" ]]; then
  section "re-scan: npm audit (agentcore_cdk/.../cdk/, advisory, filtered)"
  bash "$(dirname "${BASH_SOURCE[0]}")/npm_audit_filtered.sh" "${AGENTCORE_CDK_DIR}"
fi

section "regenerate SBOM.json + NOTICE from the updated dependencies"
(cd "${REPO_ROOT}" && uv run python cicd/generate_sbom.py)

if [[ "${status}" -ne 0 ]]; then
  echo
  echo "dependencies were updated, but pip-audit still found a vulnerability" >&2
  echo "in the new lock — see output above. Failing." >&2
fi
exit "${status}"
