#!/usr/bin/env bash
#
# Prepare the repo for deployment/development: sync the root dev-tooling
# project plus every Python runtime project's own dependencies (mcp/,
# agentcore/ — each a standalone uv project with real runtime deps), install
# npm dependencies for the two CDK apps (reward_server_cdk/,
# agentcore_cdk/.../cdk/), and run the test suite.
#
# This is the "green check" target — run it before deploy.sh or a PR.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

section "uv sync: repo root (dev tooling — pytest, ruff, pip-audit, flask)"
(cd "${REPO_ROOT}" && uv sync)

section "uv sync: mcp/"
(cd "${MCP_DIR}" && uv sync)

section "uv sync: agentcore/"
(cd "${AGENTCORE_DIR}" && uv sync)

section "npm install: infrastructure/reward_server_cdk/"
(cd "${REWARD_SERVER_CDK_DIR}" && npm install)

if [[ -d "${AGENTCORE_CDK_DIR}" ]]; then
  section "npm install: infrastructure/agentcore_cdk/atxnkiagent/agentcore/cdk/"
  (cd "${AGENTCORE_CDK_DIR}" && npm install)
else
  section "skip: agentcore_cdk/.../cdk/ not present (agentcore create not yet run)"
fi

section "cdk build: reward_server_cdk/"
(cd "${REWARD_SERVER_CDK_DIR}" && npm run build)

section "pytest (repo root — reward_server/, agentcore/, mcp/, tests/)"
# The root pyproject.toml (package = false, dev-only dependency group) pins
# pytest/pytest-cov/flask so this is reproducible from the lockfile instead
# of an ephemeral `uv run --with` resolution. pytest discovers
# reward_server/tests, agentcore/tests, mcp/tests, and tests/ via
# conftest.py's sys.path wiring, same as before. flask is test-only, for the
# reward-server auth/validation branches (Neuron itself is not required).
(cd "${REPO_ROOT}" && uv run pytest -v --cov)

section "prep complete"
