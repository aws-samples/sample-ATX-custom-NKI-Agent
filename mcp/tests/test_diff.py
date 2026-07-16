"""Tests for emit_diff — especially the path-traversal guard.

emit_diff writes files into a caller-supplied repo, so the "target must stay
inside repo_root" check is a security boundary and gets explicit coverage.
"""

from __future__ import annotations

from pathlib import Path

from kernelforge_nki_mcp.diff import emit_diff


def test_rejects_missing_repo(tmp_path: Path) -> None:
    r = emit_diff(
        repo_root=str(tmp_path / "nope"),
        target_file="k.py",
        new_source="x = 1\n",
    )
    assert r["ok"] is False
    assert "repo_root" in r["error"]


def test_rejects_path_traversal_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "secret.py").write_text("SECRET = 1\n")
    r = emit_diff(
        repo_root=str(repo),
        target_file="../secret.py",   # escapes repo_root
        new_source="SECRET = 2\n",
    )
    assert r["ok"] is False
    assert "escapes repo_root" in r["error"]
    # The out-of-repo file must be untouched.
    assert (tmp_path / "secret.py").read_text() == "SECRET = 1\n"


def test_rejects_absolute_path_outside_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("KEEP = 1\n")
    r = emit_diff(
        repo_root=str(repo),
        target_file=str(outside),
        new_source="KEEP = 2\n",
    )
    assert r["ok"] is False
    assert "escapes repo_root" in r["error"]
    assert outside.read_text() == "KEEP = 1\n"


def test_missing_target_reported(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    r = emit_diff(repo_root=str(repo), target_file="ghost.py", new_source="x\n")
    assert r["ok"] is False
    assert "not found" in r["error"]


def test_happy_path_writes_diff_and_applies(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "kernel.py"
    target.write_text("def f():\n    return 1\n")
    r = emit_diff(
        repo_root=str(repo),
        target_file="kernel.py",
        new_source="def f():\n    return 2\n",
    )
    assert r["ok"] is True
    assert r["pr_url"] is None            # open_pr defaults to False
    assert "return 1" in r["diff"] and "return 2" in r["diff"]
    assert target.read_text() == "def f():\n    return 2\n"   # applied to tree


def test_diff_uses_repo_relative_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    target = repo / "sub" / "k.py"
    target.write_text("a\n")
    r = emit_diff(repo_root=str(repo), target_file="sub/k.py", new_source="b\n")
    assert r["ok"] is True
    # Header should be repo-relative, not an absolute tmp path.
    assert "sub/k.py" in r["diff"]
    assert str(tmp_path) not in r["diff"]
