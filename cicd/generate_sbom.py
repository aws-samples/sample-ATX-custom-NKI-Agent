"""Generate a CycloneDX 1.5 SBOM and a NOTICE file for the repo's dependencies.

Stdlib-only (no new dependency): uses ``importlib.metadata`` to enumerate the
installed distributions in each Python project's own venv, and walks each
CDK app's ``node_modules`` for npm package metadata.

Scope (mirrors cicd/common.sh's dependency-bearing surfaces):
  - repo root uv project                          (.venv)
  - mcp/ uv project                               (mcp/.venv)
  - infrastructure/agentcore/ uv project          (infrastructure/agentcore/.venv)
  - infrastructure/reward_server_cdk/ npm project  (node_modules)
  - infrastructure/agentcore_cdk/.../cdk/ npm project (node_modules, if present)

reward_server/ is intentionally excluded — it is requirements.txt-only,
installed on the Trn1 itself via CodeArtifact during bootstrap, not from a
developer machine.

The repo's own packages (FIRST_PARTY_SUBPATHS) are installed into those venvs by
``uv sync`` and so appear in the scan, but they are not published to any index.
They get a ``pkg:generic/...?vcs_url=...`` purl instead of ``pkg:pypi/...``, and
are left out of NOTICE, which covers third-party attribution only.

Run via ``uv run python cicd/generate_sbom.py`` from the repo root.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from importlib.metadata import Distribution
from pathlib import Path
from typing import Any

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

REPO_ROOT = Path(__file__).parent.parent
INFRA_DIR = REPO_ROOT / "infrastructure"

PYTHON_VENVS: list[tuple[str, Path]] = [
    ("root", REPO_ROOT / ".venv"),
    ("mcp", REPO_ROOT / "mcp" / ".venv"),
    ("agentcore", INFRA_DIR / "agentcore" / ".venv"),
]

NODE_MODULES_DIRS: list[tuple[str, Path]] = [
    ("reward_server_cdk", INFRA_DIR / "reward_server_cdk" / "node_modules"),
    (
        "agentcore_cdk",
        INFRA_DIR
        / "agentcore_cdk"
        / "atxnkiagent"
        / "agentcore"
        / "cdk"
        / "node_modules",
    ),
]

SBOM_PATH = REPO_ROOT / "SBOM.json"
NOTICE_PATH = REPO_ROOT / "NOTICE"

REPO_VCS_URL = "git+https://github.com/aws-samples/sample-ATX-custom-NKI-Agent"

# Distributions that live in this repo rather than on an index. They show up in
# the scanned venvs because `uv sync` installs them (editable), but they are NOT
# published anywhere: giving them a `pkg:pypi/...` purl asserts a public PyPI
# origin that does not exist. An SBOM consumer that resolves such a purl either
# 404s or — the dependency-confusion failure mode — fetches whatever a third
# party has since registered under that name. Mapped to the repo subdirectory
# each one is built from, which becomes the purl subpath.
FIRST_PARTY_SUBPATHS: dict[str, str] = {
    "kernelforge-nki-mcp": "mcp",
    "atx-nki-agentcore": "infrastructure/agentcore",
    "atx-nki-agent-dev": ".",
    "atxnkiagent": "infrastructure/agentcore_cdk/atxnkiagent/app/atxnkiagent",
}


@dataclass(frozen=True)
class Component:
    """A single third-party dependency, deduplicated by name@version.

    Args:
        name (str): Package name.
        version (str): Package version.
        license (str): Best-effort license identifier, or "UNKNOWN".
        purl (str): Package URL — `pkg:pypi/...` / `pkg:npm/...` for published
            dependencies, `pkg:generic/...?vcs_url=...` for the repo's own
            packages (see FIRST_PARTY_SUBPATHS).
        source (str): Which project/venv or node_modules tree this came from.
        first_party (bool): True if this package is built from this repo and is
            not available from any public index.
    """

    name: str
    version: str
    license: str
    purl: str
    source: str
    first_party: bool = False


def _python_purl(name: str, version: str) -> tuple[str, bool]:
    """Builds the purl for an installed Python distribution.

    Args:
        name (str): Distribution name as declared in its metadata.
        version (str): Distribution version.

    Returns:
        tuple[str, bool]: The purl, and whether the package is first-party.
    """
    normalized = name.lower().replace("_", "-")
    subpath = FIRST_PARTY_SUBPATHS.get(normalized)
    if subpath is None:
        return f"pkg:pypi/{name.lower()}@{version}", False
    # `pkg:generic` with a vcs_url qualifier is the purl spec's form for a
    # package that is not fetchable from a package registry. `+` is
    # percent-encoded because a literal `+` in a query string decodes to a space.
    purl = f"pkg:generic/{normalized}@{version}?vcs_url={REPO_VCS_URL.replace('+', '%2B')}"
    return (purl if subpath == "." else f"{purl}#{subpath}"), True


def _python_license(dist: Distribution) -> str:
    """Resolves a Python distribution's license per PEP 639 precedence.

    Precedence: PEP 639 ``License-Expression`` -> OSI ``Classifier`` ->
    legacy ``License`` field -> "UNKNOWN".

    Args:
        dist (Distribution): The installed distribution to inspect.

    Returns:
        str: The resolved license identifier, or "UNKNOWN".
    """
    meta = dist.metadata
    license_expression = meta.get("License-Expression")
    if license_expression:
        return license_expression

    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License :: OSI Approved"):
            parts = classifier.split(" :: ")
            if len(parts) >= 3:
                return parts[-1]

    legacy_license = meta.get("License")
    if legacy_license and legacy_license.strip() and legacy_license != "UNKNOWN":
        return legacy_license.strip()

    return "UNKNOWN"


def _collect_python_components(label: str, venv: Path) -> list[Component]:
    """Collects installed Python distributions from a venv's site-packages.

    Args:
        label (str): Human-readable label for this venv (used as `source`).
        venv (Path): Path to the venv root (containing lib/.../site-packages).

    Returns:
        list[Component]: One Component per installed distribution found.
    """
    if not venv.is_dir():
        logger.info("skipping %s: %s not found (run `uv sync` first)", label, venv)
        return []

    site_packages_dirs = list(venv.glob("lib/python*/site-packages")) + list(
        venv.glob("Lib/site-packages")
    )
    components: list[Component] = []
    seen: set[str] = set()
    for site_packages in site_packages_dirs:
        for dist in Distribution.discover(path=[str(site_packages)]):
            name = dist.metadata.get("Name") or dist.metadata.get("Summary")
            version = dist.version
            if not name or not version:
                continue
            key = f"{name}@{version}"
            if key in seen:
                continue
            seen.add(key)
            purl, first_party = _python_purl(name, version)
            components.append(
                Component(
                    name=name,
                    version=version,
                    license=_python_license(dist),
                    purl=purl,
                    source=label,
                    first_party=first_party,
                )
            )
    return components


def _npm_license(package_json: dict[str, Any]) -> str:
    """Resolves an npm package's license from its package.json.

    Args:
        package_json (dict[str, Any]): Parsed package.json contents.

    Returns:
        str: The resolved license identifier, or "UNKNOWN".
    """
    license_field = package_json.get("license")
    if isinstance(license_field, str) and license_field.strip():
        return license_field.strip()
    if isinstance(license_field, dict) and license_field.get("type"):
        return str(license_field["type"])

    licenses_field = package_json.get("licenses")
    if isinstance(licenses_field, list) and licenses_field:
        first = licenses_field[0]
        if isinstance(first, dict) and first.get("type"):
            return str(first["type"])

    return "UNKNOWN"


def _collect_npm_components(label: str, node_modules: Path) -> list[Component]:
    """Collects npm package metadata by walking a node_modules tree.

    Args:
        label (str): Human-readable label for this npm project (used as
            `source`).
        node_modules (Path): Path to the node_modules directory.

    Returns:
        list[Component]: One Component per package.json found.
    """
    if not node_modules.is_dir():
        logger.info(
            "skipping %s: %s not found (run `npm install` first)", label, node_modules
        )
        return []

    components: list[Component] = []
    seen: set[str] = set()
    for package_json_path in node_modules.glob("**/package.json"):
        # Skip nested node_modules-within-node_modules package.json files that
        # belong to a dependency's own bundled copy of something else's
        # metadata directory (rare, but avoids double-counting via symlinks).
        if "node_modules" in package_json_path.parent.parts[
            len(node_modules.parts) :
        ]:
            continue
        try:
            package_json = json.loads(package_json_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not parse %s: %s", package_json_path, exc)
            continue

        name = package_json.get("name")
        version = package_json.get("version")
        if not name or not version:
            continue
        key = f"{name}@{version}"
        if key in seen:
            continue
        seen.add(key)
        components.append(
            Component(
                name=name,
                version=version,
                license=_npm_license(package_json),
                purl=f"pkg:npm/{name}@{version}",
                source=label,
            )
        )
    return components


def _collect_all_components() -> list[Component]:
    """Collects and deduplicates components across all configured surfaces.

    Deduplication is by `name@version` globally, across Python and npm
    surfaces alike, keeping the first occurrence encountered.

    Returns:
        list[Component]: The deduplicated, sorted list of components.
    """
    components: list[Component] = []
    seen: set[str] = set()

    for label, venv in PYTHON_VENVS:
        for component in _collect_python_components(label, venv):
            key = f"{component.name}@{component.version}"
            if key in seen:
                continue
            seen.add(key)
            components.append(component)

    for label, node_modules in NODE_MODULES_DIRS:
        for component in _collect_npm_components(label, node_modules):
            key = f"{component.name}@{component.version}"
            if key in seen:
                continue
            seen.add(key)
            components.append(component)

    return sorted(components, key=lambda c: (c.name.lower(), c.version))


def _sbom_component(c: Component) -> dict[str, Any]:
    """Renders one Component as a CycloneDX 1.5 component object.

    Args:
        c (Component): The component to render.

    Returns:
        dict[str, Any]: The CycloneDX component entry.
    """
    entry: dict[str, Any] = {
        "type": "library",
        "name": c.name,
        "version": c.version,
        "purl": c.purl,
        "licenses": [{"license": {"id": c.license}}]
        if c.license != "UNKNOWN"
        else [{"license": {"name": "UNKNOWN"}}],
        "properties": [{"name": "source", "value": c.source}],
    }
    if c.first_party:
        # Make the "do not try to fetch this from an index" signal explicit for
        # consumers that key off properties rather than parsing the purl type.
        entry["properties"].append({"name": "first-party", "value": "true"})
        entry["externalReferences"] = [{"type": "vcs", "url": REPO_VCS_URL}]
    return entry


def _write_sbom(components: list[Component]) -> None:
    """Writes a CycloneDX 1.5 SBOM.json listing every component.

    Args:
        components (list[Component]): The components to include.
    """
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "components": [_sbom_component(c) for c in components],
    }
    SBOM_PATH.write_text(json.dumps(sbom, indent=2) + "\n")
    logger.info("wrote %s (%d components)", SBOM_PATH, len(components))


def _write_notice(components: list[Component]) -> None:
    """Writes a human-readable NOTICE file listing every component + license.

    Args:
        components (list[Component]): The components to include.
    """
    lines = [
        "NOTICE",
        "",
        "This file lists third-party dependencies bundled with or used to",
        "build the ATX NKI Agent, and their licenses, as declared by each",
        "package's own metadata. Generated by cicd/generate_sbom.py — do not",
        "edit by hand; re-run the generator instead.",
        "",
        "This repo's own packages are excluded (they are MIT-0, covered by",
        "LICENSE, and are not third-party); SBOM.json lists them with a",
        "pkg:generic purl.",
        "",
    ]
    by_source: dict[str, list[Component]] = {}
    for component in components:
        if component.first_party:
            continue
        by_source.setdefault(component.source, []).append(component)

    for source in sorted(by_source):
        lines.append(f"## {source}")
        lines.append("")
        for component in by_source[source]:
            lines.append(f"- {component.name} {component.version} ({component.license})")
        lines.append("")

    NOTICE_PATH.write_text("\n".join(lines).rstrip() + "\n")
    logger.info(
        "wrote %s (%d third-party components)",
        NOTICE_PATH,
        sum(len(v) for v in by_source.values()),
    )


def main() -> None:
    """Collects dependencies across the repo and writes SBOM.json + NOTICE.

    Raises:
        SystemExit: If no components were collected at all (refuses to write
            empty output files).
    """
    components = _collect_all_components()
    if not components:
        logger.error(
            "no components collected — refusing to write an empty SBOM/NOTICE. "
            "Run `uv sync` / `npm install` across the repo first."
        )
        sys.exit(1)

    _write_sbom(components)
    _write_notice(components)


if __name__ == "__main__":
    main()
