#!/usr/bin/env bash
#
# Remove caches and build artifacts created during development. Non-
# destructive by default: leaves .venv / node_modules / uv-managed
# environments in place so you don't have to re-download/re-install
# anything just to clear caches. Pass --deep to also remove those.
#
# Usage:
#   ./cicd/clean.sh          # caches only
#   ./cicd/clean.sh --deep   # caches + all venvs/node_modules (forces a
#                             # full re-`prep.sh` next time)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DEEP=0
if [[ "${1:-}" == "--deep" ]]; then
  DEEP=1
fi

section "removing Python caches (__pycache__, .pytest_cache, .ruff_cache, *.egg-info)"
find "${REPO_ROOT}" \
  \( -path "*/node_modules/*" -o -path "*/.git/*" \) -prune -o \
  \( -type d -name "__pycache__" -o \
     -type d -name ".pytest_cache" -o \
     -type d -name ".ruff_cache" -o \
     -type d -name ".hypothesis" -o \
     -type d -name "*.egg-info" \) -print0 \
  | xargs -0 rm -rf

section "removing coverage artifacts"
find "${REPO_ROOT}" \
  \( -path "*/node_modules/*" -o -path "*/.git/*" \) -prune -o \
  \( -name ".coverage" -o -name "coverage.xml" -o -type d -name "htmlcov" \) -print0 \
  | xargs -0 rm -rf

section "removing CDK synth output (cdk.out) and TS build artifacts"
rm -rf "${REWARD_SERVER_CDK_DIR}/cdk.out"
# -print0/xargs -0: paths in this repo can contain spaces (verified — this
# workspace's own path does), so newline-delimited `find | xargs` output
# silently mis-splits and misses files. NUL-delimited is the only safe form.
# `|| true` on the grep stage: under `pipefail`, grep exits 1 when nothing
# matches (e.g. no build artifacts present yet), which would otherwise abort
# the whole script — that's a normal "nothing to clean" case, not an error.
find "${REWARD_SERVER_CDK_DIR}" -maxdepth 2 \( -name "*.d.ts" -o -name "*.js" \) -print0 2>/dev/null \
  | { grep -zv jest.config.js || true; } \
  | xargs -0 rm -f
if [[ -d "${AGENTCORE_CDK_DIR}" ]]; then
  rm -rf "${AGENTCORE_CDK_DIR}/cdk.out"
fi

section "removing ASH scan output (.ash/ash_output)"
rm -rf "${REPO_ROOT}/.ash/ash_output"

if [[ "${DEEP}" -eq 1 ]]; then
  section "--deep: removing virtual environments and node_modules"
  rm -rf "${MCP_DIR}/.venv" "${AGENTCORE_DIR}/.venv"
  rm -rf "${REWARD_SERVER_CDK_DIR}/node_modules"
  [[ -d "${AGENTCORE_CDK_DIR}" ]] && rm -rf "${AGENTCORE_CDK_DIR}/node_modules"
  echo "removed. run cicd/prep.sh to reinstall before next use."
fi

section "clean complete"
