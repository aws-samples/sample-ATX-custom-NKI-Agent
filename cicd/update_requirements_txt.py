"""Bump the >= floors in a requirements.txt to each package's current latest
version, in place, without touching comments, blank lines, or any line that
isn't a plain `name>=version` pin (e.g. the reward_server/requirements.txt
lines noting torch/torch-neuronx/neuronx-cc are installed separately via the
Neuron SDK are left exactly as written).

Resolves each package's latest version with `uv pip compile --no-deps`
(no venv required, no transitive packages pulled in) rather than a full
`uv pip compile --upgrade`, which would turn this intentionally minimal,
floor-pinned file into a fully resolved lockfile with exact (==) pins and
transitive entries never listed in the source file.

Usage: uv run python cicd/update_requirements_txt.py <path/to/requirements.txt>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)>=([A-Za-z0-9.*+!-]+)\s*$")


def latest_version(package: str) -> str:
    """Resolves the latest available version of a single package.

    Args:
        package (str): PyPI package name, with no version specifier.

    Returns:
        str: the resolved version string.

    Raises:
        RuntimeError: if `uv pip compile` fails or produces no resolvable pin
            for the package.
    """
    result = subprocess.run(
        ["uv", "pip", "compile", "-", "--no-deps", "--no-header", "-q"],
        input=f"{package}\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"uv pip compile failed for {package!r}: {result.stderr.strip()}"
        )
    for line in result.stdout.splitlines():
        match = re.match(rf"^{re.escape(package)}==([A-Za-z0-9.*+!-]+)", line)
        if match:
            return match.group(1)
    raise RuntimeError(f"could not resolve a version for {package!r}")


def update_requirements(path: Path) -> None:
    """Rewrites every `name>=version` line in-place with the latest version.

    Args:
        path (Path): path to the requirements.txt to update.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=False)
    updated_lines = []
    changed = False

    for line in lines:
        match = PIN_RE.match(line)
        if not match:
            updated_lines.append(line)
            continue

        package, current_version = match.group(1), match.group(2)
        new_version = latest_version(package)
        if new_version != current_version:
            changed = True
            print(f"  {package}: {current_version} -> {new_version}")
        else:
            print(f"  {package}: {current_version} (already latest)")
        updated_lines.append(f"{package}>={new_version}")

    trailing_newline = "\n" if path.read_text(encoding="utf-8").endswith("\n") else ""
    path.write_text("\n".join(updated_lines) + trailing_newline, encoding="utf-8")

    if not changed:
        print(f"{path}: no floor changes needed")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/requirements.txt>", file=sys.stderr)
        return 1

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1

    update_requirements(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
