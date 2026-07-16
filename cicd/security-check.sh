#!/usr/bin/env bash
#
# Dependency vulnerability scanning across every Python and npm project in
# the repo, plus an ASH pass over the CDK/IaC surfaces (local mode).
#
# Policy: pip-audit findings are FATAL (drive the exit code so CI can gate
# on them). npm audit and ASH are ADVISORY — reported, never block.
#
# ASH mode: this script runs ASH unconditionally in `--mode local`, NOT
# container mode. Local mode uses a different, weaker-isolation scanner set than container mode and does
# not require Docker/Podman on the machine running this script. This
# script's scan is a fast pre-deploy gate — a more thorough, container-mode
# ASH pass (Docker, falling back to Podman) is recommended for a full
# periodic review, run separately.
#
# pip-audit runs three times, once per *runtime* venv, because each has a
# distinct dependency set: the root project's own dev-tooling group (pytest,
# ruff, pip-audit itself, flask), and mcp/ and agentcore/'s actual runtime
# dependencies (mcp, boto3, bedrock-agentcore, strands-agents, etc.) in their
# own uv-managed venvs. Auditing root doesn't cover mcp/ or agentcore/, and
# vice versa — none of these three runs is redundant. pip-audit is a pinned
# [dependency-groups].dev entry in all three pyproject.toml files, so `uv
# run pip-audit` resolves from each project's own lockfile rather than an
# ephemeral `--with` install — reproducible, and covered by prep.sh's `uv
# sync` in each directory.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

status=0

section "pip-audit: repo root (dev tooling)"
(cd "${REPO_ROOT}" && uv run pip-audit --local) || status=$?

section "pip-audit: mcp/"
(cd "${MCP_DIR}" && uv run pip-audit --local) || status=$?

section "pip-audit: agentcore/"
(cd "${AGENTCORE_DIR}" && uv run pip-audit --local) || status=$?

section "npm audit: infrastructure/reward_server_cdk/ (advisory)"
(cd "${REWARD_SERVER_CDK_DIR}" && npm audit --omit=dev) || true

if [[ -d "${AGENTCORE_CDK_DIR}" ]]; then
  section "npm audit: infrastructure/agentcore_cdk/atxnkiagent/agentcore/cdk/ (advisory, filtered)"
  bash "$(dirname "${BASH_SOURCE[0]}")/npm_audit_filtered.sh" "${AGENTCORE_CDK_DIR}"
fi

section "ASH (local mode, advisory)"
(
  cd "${REPO_ROOT}"
  rm -rf .ash/ash_output
  uvx "git+https://github.com/awslabs/automated-security-helper.git@v3" \
    --mode local \
    --source-dir reward_server/ --source-dir mcp/ --source-dir agentcore/ \
    --source-dir agentcore_cdk/ --source-dir reward_server_cdk/ --source-dir scripts/
) || true
echo "ASH output: ${REPO_ROOT}/.ash/ash_output/reports/ash.summary.md"

if [[ "${status}" -ne 0 ]]; then
  echo
  echo "pip-audit found vulnerabilities — see output above. Failing." >&2
fi
exit "${status}"
