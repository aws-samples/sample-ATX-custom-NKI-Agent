#!/usr/bin/env bash
# Fast connectivity smoke test: one trivial L1 task (relu) through the deployed
# AgentCore runtime on the live reward path. A correct run finishes in well
# under a minute (historical median ~38s); if this hangs > ~2 min or returns
# 0 compiled, the reward path is broken — do NOT proceed to the full benchmark.
#
# Usage: AGENTCORE_ARN=... ./scripts/smoke_opus48.sh
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ARN="${AGENTCORE_ARN:?export AGENTCORE_ARN=arn:aws:bedrock-agentcore:...:runtime/...}"
HERE="$(cd "$(dirname "$0")" && pwd)"

python3 "$HERE/eval_nkibench_agentcore.py" \
  --backend agentcore \
  --agentcore-arn "$ARN" \
  --region "$REGION" --agentcore-timeout 180 \
  --levels 1 --num-tasks 1 --max-turns 10 \
  --output "$HERE/../results/_smoke_opus48.json"

python3 - "$HERE/../results/_smoke_opus48.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
compiled = d.get("compiled", 0)
print(f"smoke: compiled={compiled}/{d.get('total')} correct={d.get('correct')}")
sys.exit(0 if compiled and compiled > 0 else 1)
PY
