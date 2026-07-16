"""Tests for build_subprocess_env: the subprocess env passed to
untrusted, LLM-generated kernel/reference code must be an explicit allowlist,
never a full copy of the parent process's environment.
"""

from __future__ import annotations

from reward_server.sandbox_env import _ALLOWED_EXACT, build_subprocess_env


def test_secret_like_parent_vars_are_excluded(monkeypatch) -> None:
    # Simulate a secret-bearing variable present in the parent process (the
    # exact scenario that motivated this design: os.environ.copy() would have leaked
    # this straight into the untrusted child).
    monkeypatch.setenv("SOME_FUTURE_SECRET_TOKEN", "super-secret-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-leak")

    env = build_subprocess_env({"NKI_KERNEL_FILE": "kernel_under_test.py"})

    assert "SOME_FUTURE_SECRET_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_allowlisted_parent_vars_are_preserved(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/opt/neuron/bin:/usr/bin")
    monkeypatch.setenv("HOME", "/home/nki")

    env = build_subprocess_env({})

    assert env["PATH"] == "/opt/neuron/bin:/usr/bin"
    assert env["HOME"] == "/home/nki"


def test_neuron_prefixed_vars_are_preserved(monkeypatch) -> None:
    monkeypatch.setenv("NEURON_RT_LOG_LEVEL", "INFO")
    monkeypatch.setenv("AWS_NEURON_VISIBLE_CORES", "0")
    monkeypatch.setenv("UNRELATED_APP_VAR", "should-not-appear")

    env = build_subprocess_env({})

    assert env["NEURON_RT_LOG_LEVEL"] == "INFO"
    assert env["AWS_NEURON_VISIBLE_CORES"] == "0"
    assert "UNRELATED_APP_VAR" not in env


def test_extra_vars_are_always_included_even_if_not_allowlisted(monkeypatch) -> None:
    # The handler's own NKI_* payload variables must always be passed through,
    # regardless of the allowlist (they aren't parent secrets, they're the
    # explicit inputs this call is handing to the child).
    env = build_subprocess_env({"NKI_INPUT_SPECS": "[]", "NKI_ATOL": "0.001"})

    assert env["NKI_INPUT_SPECS"] == "[]"
    assert env["NKI_ATOL"] == "0.001"


def test_extra_vars_take_precedence_over_parent_env(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/parent/path")

    env = build_subprocess_env({"PATH": "/overridden/path"})

    assert env["PATH"] == "/overridden/path"


def test_no_full_environ_copy_leakage(monkeypatch) -> None:
    # Broad regression guard: nothing outside the allowlist/prefixes/extras
    # should ever appear in the built env, however many vars the parent has.
    monkeypatch.setenv("RANDOM_VAR_1", "x")
    monkeypatch.setenv("RANDOM_VAR_2", "y")
    monkeypatch.setenv("DATABASE_PASSWORD", "z")

    env = build_subprocess_env({"NKI_KERNEL_NAME": "k"})

    unexpected = set(env) - _ALLOWED_EXACT - {"NKI_KERNEL_NAME"}
    unexpected = {k for k in unexpected if not k.startswith(("NEURON_", "AWS_NEURON_"))}
    assert unexpected == set()
