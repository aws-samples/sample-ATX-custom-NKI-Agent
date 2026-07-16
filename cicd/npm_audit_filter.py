"""Filters `npm audit --json` output against a fixed allowlist of known,
tracked GHSA advisory IDs, so advisories with no viable local fix (e.g.
bundled inside a third-party CLI's own node_modules/) don't need re-triage
every run. Excluded advisories are still printed, just separated from the
"needs attention" list — this never hides a finding outright.

The allowlist itself lives in cicd/npm_audit_filtered.sh (the caller), which
is the single source of truth for *which* GHSA IDs are excluded; this script
only does the filtering. See cicd/README.md's "Known npm advisory
exclusions" section for the full writeup and the re-check trigger.

Usage: python3 npm_audit_filter.py <audit-json-file> <comma-separated-ghsa-ids>
"""

from __future__ import annotations

import json
import sys


def filter_audit_report(audit_json_path: str, excluded_ids: set[str]) -> int:
    """Filters an npm audit JSON report against an excluded-GHSA-ID set.

    Args:
        audit_json_path (str): path to a file containing `npm audit --json` output.
        excluded_ids (set[str]): GHSA IDs to exclude from the "needs attention" list.

    Returns:
        int: process exit code (always 0 — this is an advisory-only report).
    """
    with open(audit_json_path, encoding="utf-8") as f:
        data = json.load(f)

    vulnerabilities = data.get("vulnerabilities", {})
    reported: dict[str, dict] = {}
    excluded_hits: list[tuple[str, str, str]] = []

    for name, vuln in vulnerabilities.items():
        via_advisories = [v for v in vuln.get("via", []) if isinstance(v, dict)]
        remaining = []
        for advisory in via_advisories:
            url = advisory.get("url", "")
            ghsa_id = url.rsplit("/", 1)[-1] if url else ""
            if ghsa_id in excluded_ids:
                excluded_hits.append((name, ghsa_id, advisory.get("title", "")))
            else:
                remaining.append(advisory)
        if remaining:
            reported[name] = {
                "severity": vuln.get("severity"),
                "advisories": [
                    (a.get("url", ""), a.get("title", "")) for a in remaining
                ],
            }

    if excluded_hits:
        print(
            f"Excluded {len(excluded_hits)} known advisory hit(s) (see cicd/README.md):"
        )
        for name, ghsa_id, title in excluded_hits:
            print(f"  - {name}: {ghsa_id} - {title}")
        print()

    if reported:
        print(f"{len(reported)} package(s) with advisories NOT on the exclusion list:")
        for name, info in reported.items():
            print(f"  - {name} ({info['severity']}):")
            for url, title in info["advisories"]:
                print(f"      {url} - {title}")
    else:
        print("No advisories outside the known-exclusions allowlist.")

    return 0


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <audit-json-file> <comma-separated-ghsa-ids>",
            file=sys.stderr,
        )
        return 1

    audit_json_path, excluded_ids_arg = sys.argv[1], sys.argv[2]
    excluded_ids = set(excluded_ids_arg.split(",")) if excluded_ids_arg else set()
    return filter_audit_report(audit_json_path, excluded_ids)


if __name__ == "__main__":
    sys.exit(main())
