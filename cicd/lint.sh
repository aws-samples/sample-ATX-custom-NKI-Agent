#!/usr/bin/env bash
#
# Lint every Python project in the repo with ruff (default settings — no
# per-project [tool.ruff] config in this repo, so don't add one here either).
#
# Scope (mirrors common.sh's dependency-bearing surfaces): repo root,
# mcp/, infrastructure/agentcore/. reward_server/ has no uv project of its
# own (installed via CodeArtifact on the Trn1, not from a dev machine) —
# lint it through the root project's ruff instead, since ruff needs no
# per-project install to check arbitrary paths.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

section "ruff check: repo root (., excluding mcp/ and infrastructure/agentcore/ — linted separately below)"
(cd "${REPO_ROOT}" && uv run ruff check --exclude mcp --exclude infrastructure/agentcore .)

section "ruff check: mcp/"
(cd "${MCP_DIR}" && uv run ruff check src tests)

section "ruff check: infrastructure/agentcore/"
(cd "${AGENTCORE_DIR}" && uv run ruff check .)

section "lint complete"
