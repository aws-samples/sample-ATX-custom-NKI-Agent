#!/usr/bin/env bash
#
# Wraps `npm audit --json` for a CDK app and filters out a fixed allowlist
# of advisories that are known, upstream-in-@aws/agentcore, and have no
# viable local fix (see cicd/README.md's "Known npm advisory exclusions"
# section for the full writeup and the re-check trigger). Any advisory NOT
# on the allowlist is still reported and still advisory-only (this script
# always exits 0 — security-check.sh/update-deps.sh treat all npm audit
# output as advisory, never fatal, matching their existing policy). This
# script only changes what's *displayed*, not the pass/fail gate, which was
# already non-blocking.
#
# Usage: bash cicd/npm_audit_filtered.sh <cdk-app-dir>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <cdk-app-dir>" >&2
  exit 1
fi
cdk_app_dir="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allowlist: GHSA ID -> short reason. Keep in sync with cicd/README.md's
# "Known npm advisory exclusions" section — that section is the source of
# truth for *why*; this list is only the *which*, kept minimal on purpose.
excluded_advisories=(
  "GHSA-g7r4-m6w7-qqqr:esbuild dev-server arbitrary file read (Windows)"
  "GHSA-xrhx-7g5j-rcj5:hono IP-restriction bypass"
  "GHSA-3hrh-pfw6-9m5x:hono cookie sameSite/priority injection"
  "GHSA-f577-qrjj-4474:hono JWT auth-scheme confusion"
  "GHSA-2gcr-mfcq-wcc3:hono app.mount() path decoding"
  "GHSA-wwfh-h76j-fc44:hono serve-static path traversal (Windows)"
  "GHSA-j6c9-x7qj-28xf:hono Lambda adapter Set-Cookie merge"
  "GHSA-88fw-hqm2-52qc:hono CORS wildcard-with-credentials"
  "GHSA-rv63-4mwf-qqc2:hono Body Limit Middleware bypass on Lambda"
  "GHSA-wgpf-jwqj-8h8p:hono Lambda@Edge repeated-header handling"
  "GHSA-q8mj-m7cp-5q26:qs.stringify DoS crash"
  "GHSA-4x5r-pxfx-6jf8:@babel/core arbitrary file read via sourceMappingURL"
  "GHSA-8988-4f7v-96qf:@opentelemetry/core unbounded memory in W3C Baggage propagation"
  "GHSA-jxxr-4gwj-5jf2:brace-expansion max-DoS-protection bypass"
  "GHSA-gh4j-gqv2-49f6:fast-xml-parser XMLBuilder comment/CDATA injection"
  "GHSA-h67p-54hq-rp68:js-yaml quadratic-complexity DoS in merge-key handling"
)
# All excluded advisories above are bundled inside @aws/agentcore's own
# node_modules/ (not this repo's direct deps), and @aws/agentcore@0.24.1 is
# the latest published version with no fix — see cicd/README.md.
excluded_ids="$(IFS=,; echo "${excluded_advisories[*]}" | tr ',' '\n' | cut -d: -f1 | paste -sd, -)"

section "npm audit (filtered): ${cdk_app_dir}"
(
  cd "${cdk_app_dir}"
  audit_json_file="$(mktemp)"
  trap 'rm -f "${audit_json_file}"' EXIT
  npm audit --json > "${audit_json_file}" || true
  python3 "${script_dir}/npm_audit_filter.py" "${audit_json_file}" "${excluded_ids}"
) || true

