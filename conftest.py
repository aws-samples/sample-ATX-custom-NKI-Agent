"""Root pytest config: make the three packages importable from the repo root.

Lets a bare `pytest` (or `python -m pytest`) run the full suite without per-
package PYTHONPATH juggling:
  - repo root                  -> `nkibench`
  - mcp/src                    -> `kernelforge_nki_mcp`
  - infrastructure/reward_server -> `reward_server` (as a top-level module, so
    its own internal `from reward_server.sandbox_env import ...` imports resolve)
  - infrastructure/agentcore   -> `router`, `app`
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
for rel in (".", "mcp/src", "infrastructure", "infrastructure/agentcore"):
    p = str((_ROOT / rel).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
