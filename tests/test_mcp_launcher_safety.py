"""Dependency-confusion guards for the MCP launcher configs.

`kernelforge-nki-mcp` is not published on public PyPI. `uvx` resolves a bare
package name from public PyPI by default, so a config of the form

    {"command": "uvx", "args": ["kernelforge-nki-mcp@latest"]}

hands code execution inside the agent process to whoever registers that name on
PyPI first — with the process's filesystem access, AWS credentials, environment
variables and tool permissions.

Every launcher config in the repo must therefore name a *path* for the package
(`uvx --from <path> kernelforge-nki-mcp`) or an explicit trusted index, so
resolution can never fall through to public PyPI. These tests pin that for all
three client surfaces — Claude Code / Codex (`.mcp.json`), Kiro
(`.kiro/settings/mcp.json`), and the `atx` CLI (`atx/mcp.json`) — and for the
config snippets the docs tell users to copy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SERVER = "kernelforge-nki-mcp"

# Every launcher config that can spawn the MCP server.
_CONFIGS = (
    Path(".mcp.json"),
    Path(".kiro/settings/mcp.json"),
    Path("atx/mcp.json"),
)

# Resolution modes that do NOT touch public PyPI's bare-name namespace.
_PINNING_FLAGS = ("--from", "--index-url", "--default-index", "--index")


def _entry(config: Path) -> dict:
    cfg = json.loads((_REPO / config).read_text())
    assert _SERVER in cfg["mcpServers"], f"{config} does not register {_SERVER}"
    return cfg["mcpServers"][_SERVER]


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda p: str(p))
def test_launcher_does_not_resolve_from_public_pypi(config: Path) -> None:
    entry = _entry(config)
    args = entry.get("args", [])

    # A uvx launch must pin where the package comes from. Without one of these
    # flags, `uvx <name>` is a public-PyPI lookup.
    if entry.get("command") == "uvx":
        assert any(flag in args for flag in _PINNING_FLAGS), (
            f"{config} launches {_SERVER} via bare `uvx` with no {' / '.join(_PINNING_FLAGS)}; "
            "uvx would resolve the name from public PyPI, where it is unregistered — "
            "anyone who claims it gets code execution in the agent process"
        )

    # Belt and braces: no argument may be the bare package name with a version
    # specifier attached (`name@latest`, `name==1.2.3`), which is the shape that
    # forces an index lookup even when other flags are present.
    for arg in args:
        assert not re.fullmatch(rf"{re.escape(_SERVER)}[@=<>!~].*", arg), (
            f"{config} passes {arg!r}: a version-specified bare name resolves "
            "through an index; use `--from <path>` instead"
        )


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda p: str(p))
def test_launcher_from_target_is_a_path_not_a_package_name(config: Path) -> None:
    entry = _entry(config)
    args = entry.get("args", [])
    if "--from" not in args:
        pytest.skip(f"{config} pins resolution by index, not by path")

    target = args[args.index("--from") + 1]
    # `uvx --from kernelforge-nki-mcp` is still a PyPI lookup; the target has to
    # look like a filesystem path (possibly via a placeholder the client or the
    # setup instructions substitute).
    assert target != _SERVER, f"{config} passes `--from {_SERVER}` — that is a PyPI lookup"
    assert "/" in target, f"{config} `--from {target}` is not a path"
    assert target.endswith("/mcp"), (
        f"{config} `--from {target}` should point at this repo's `mcp/` package directory"
    )


def test_ide_and_atx_configs_register_the_same_server() -> None:
    # The surfaces must stay in lockstep: same server id, same launch mechanism,
    # so a future edit can't quietly re-introduce bare-name resolution on one of
    # them while the others stay pinned.
    ids = [set(json.loads((_REPO / c).read_text())["mcpServers"]) for c in _CONFIGS]
    assert all(s == ids[0] for s in ids), "launcher configs register different MCP servers"


@pytest.mark.parametrize(
    "doc",
    (
        Path("README.md"),
        Path("mcp/README.md"),
        Path("atx/README.md"),
        Path("skills/nki-kernel/SKILL.md"),
        Path("docs/architecture.md"),
        Path("docs/codebase.md"),
        Path("docs/deployment-architecture.md"),
    ),
    ids=lambda p: str(p),
)
def test_docs_do_not_advertise_a_bare_uvx_invocation(doc: Path) -> None:
    """Docs are copy-paste sources; a bare `uvx <name>` in one re-opens the hole."""
    text = (_REPO / doc).read_text()
    # Match `uvx kernelforge-nki-mcp` / `uvx kernelforge-nki-mcp@latest` where no
    # pinning flag sits between the command and the package name.
    bare = re.compile(rf"uvx(?!\s+--)\s+{re.escape(_SERVER)}\S*")
    # Prose that explicitly calls the bare form out as unsafe is allowed, so scan
    # line by line and require the warning to sit on the offending line.
    warned = re.compile(r"\bbare\b|\bnot\b|\bnever\b|\bwould\b|\bdo not\b", re.I)
    unwarned = [
        match.group()
        for line in text.splitlines()
        for match in bare.finditer(line)
        if not warned.search(line)
    ]
    assert not unwarned, (
        f"{doc} shows {unwarned} as a usable command; document "
        f"`uvx --from <path> {_SERVER}` instead"
    )
