#!/usr/bin/env bash
# Local smoke test for the MCP server.
#
# Spins up the MCP server, calls nki_discover_kernels on the examples folder,
# and prints the result. Does NOT exercise AgentCore or the reward server.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/mcp"

echo "=> installing kernelforge-nki-mcp in dev mode"
uv pip install -e .

echo "=> running discover smoke"
python -c "
from kernelforge_nki_mcp.discover import discover
r = discover('$ROOT/examples')
print('candidates:', len(r.candidates))
for c in r.candidates:
    print(f'  {c.kind:16}  {c.function:24}  {c.file}:{c.line}')
"

echo "=> running pytest"
pytest -q tests/
