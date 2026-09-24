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
four client surfaces — Claude Code / Codex in-repo (`.mcp.json`), the installed
plugin (`.mcp.plugin.json`), Kiro (`.kiro/settings/mcp.json`), and the `atx` CLI
(`atx/mcp.json`) — and for the config snippets the docs tell users to copy.

Why two Claude configs rather than one with a defaulted placeholder: Claude Code
substitutes `${CLAUDE_PLUGIN_ROOT}` only for that exact token. Verified
empirically against a locally installed copy of this plugin:

    | config file        | `${CLAUDE_PLUGIN_ROOT}` | `${CLAUDE_PLUGIN_ROOT:-.}` |
    | installed plugin   | plugin dir              | `.`  (wrong dir)           |
    | project repo root  | unset -> literal        | `.`  (repo root, correct)  |

So a single file cannot serve both: the `:-` form silently sends the installed
plugin to the caller's cwd, and the bare form leaves an unexpanded literal in a
project config. `.claude-plugin/plugin.json` therefore points `mcpServers` at
`.mcp.plugin.json` (bare token, plugin dir), while `.mcp.json` stays relative
for in-repo use. Both are `--from <path>`; neither can reach an index.
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
    Path(".mcp.plugin.json"),
    Path(".kiro/settings/mcp.json"),
    Path("atx/mcp.json"),
)

# The config Claude Code loads when the plugin is installed from a marketplace,
# and the manifest key that selects it.
_PLUGIN_CONFIG = Path(".mcp.plugin.json")
_PLUGIN_MANIFEST = Path(".claude-plugin/plugin.json")
_PLUGIN_ROOT_TOKEN = "${CLAUDE_PLUGIN_ROOT}"

# Resolution modes that do NOT touch public PyPI's bare-name namespace.
_PINNING_FLAGS = ("--from", "--index-url", "--default-index", "--index")


def _entry(config: Path) -> dict:
    cfg = json.loads((_REPO / config).read_text())
    assert _SERVER in cfg["mcpServers"], f"{config} does not register {_SERVER}"
    return cfg["mcpServers"][_SERVER]


def _from_target(config: Path) -> str:
    args = _entry(config).get("args", [])
    assert "--from" in args, f"{config} does not pass --from"
    return args[args.index("--from") + 1]


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


def test_plugin_manifest_selects_the_plugin_root_config() -> None:
    # Without this key Claude Code auto-discovers `.mcp.json`, whose relative
    # `./mcp` resolves against the *caller's* cwd once the plugin is installed
    # outside this checkout — the server then fails to start from anywhere else.
    manifest = json.loads((_REPO / _PLUGIN_MANIFEST).read_text())
    assert manifest.get("mcpServers") == f"./{_PLUGIN_CONFIG}", (
        f"{_PLUGIN_MANIFEST} must point mcpServers at ./{_PLUGIN_CONFIG}; "
        "otherwise the installed plugin falls back to the repo-relative config"
    )


def test_plugin_config_anchors_on_the_plugin_root() -> None:
    target = _from_target(_PLUGIN_CONFIG)
    assert target == f"{_PLUGIN_ROOT_TOKEN}/mcp", (
        f"{_PLUGIN_CONFIG} must use the bare {_PLUGIN_ROOT_TOKEN} token. Claude Code "
        f"substitutes it only in that exact form: {_PLUGIN_ROOT_TOKEN[:-1]}:-.}} is treated "
        "as an ordinary unset variable with a default, so it collapses to the "
        "caller's cwd instead of the installed plugin directory."
    )


def test_project_config_does_not_depend_on_the_plugin_root() -> None:
    # Mirror image of the above: in a project-scope config the variable is never
    # set, and with no default Claude Code passes the literal `${...}` through as
    # a path, so the server cannot start inside this checkout.
    target = _from_target(Path(".mcp.json"))
    assert "CLAUDE_PLUGIN_ROOT" not in target, (
        ".mcp.json is loaded as a project config where CLAUDE_PLUGIN_ROOT is unset; "
        f"it must stay repo-relative and leave {_PLUGIN_ROOT_TOKEN} to {_PLUGIN_CONFIG}"
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
