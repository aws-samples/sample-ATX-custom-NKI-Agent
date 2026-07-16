"""Auth middleware + request dispatch tests.

The reward server's real trust boundary is the network (private-isolated subnet
+ SG-to-SG on 5050), so the app-layer check defaults OFF. These tests pin both
that default (a request with no header is accepted) AND the opt-in belt-and-
suspenders mode (REWARD_REQUIRE_AUTH=true requires a well-formed-ARN
X-Verified-Caller header). Dispatch tests confirm unknown ops are rejected and
batch size is capped.
"""

from __future__ import annotations

import importlib

import pytest
from flask import Flask

from reward_server import auth


def _client(monkeypatch, **env):
    """Build a fresh Flask app with auth installed under the given env."""
    monkeypatch.delenv("REWARD_REQUIRE_AUTH", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app = Flask(__name__)
    auth.install(app)

    @app.post("/reward")
    def _reward():  # pyright: ignore[reportUnusedFunction]
        return {"ok": True}

    @app.get("/health")
    def _health():  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    return app.test_client()


def test_health_is_unauthenticated(monkeypatch) -> None:
    c = _client(monkeypatch)
    assert c.get("/health").status_code == 200


def test_health_deep_reports_import_status(monkeypatch) -> None:
    # Exercises the real endpoint via the server module. numpy is present in the
    # test venv but neuronxcc is not, so we assert on structure/keys rather than
    # a specific 200/503 (which depends on what's installed where tests run).
    monkeypatch.setenv("REWARD_REQUIRE_AUTH", "false")
    server = importlib.reload(importlib.import_module("reward_server.server"))
    resp = server.app.test_client().get("/health/deep")
    body = resp.get_json()
    assert resp.status_code in (200, 503)
    assert set(body["imports"]) == {"numpy", "torch", "neuronxcc.nki"}
    assert (body["status"] == "ok") == (resp.status_code == 200)


def test_health_deep_is_unauthenticated(monkeypatch) -> None:
    # The provisioning readiness gate curls /health/deep with no credential,
    # so it must be auth-exempt like /health.
    c = _client(monkeypatch)

    @c.application.get("/health/deep")
    def _deep():  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    assert c.get("/health/deep").status_code == 200


def test_default_is_network_trust_no_header_required(monkeypatch) -> None:
    # Shipped design: no gateway in front, so the app-layer check defaults OFF
    # and a request with no credential header is accepted — the network (SG +
    # private-isolated subnet) is the boundary.
    c = _client(monkeypatch)
    assert c.post("/reward", json={}).status_code == 200


def test_optin_missing_header_is_rejected(monkeypatch) -> None:
    # With REWARD_REQUIRE_AUTH=true (opt-in proxy mode), a request carrying no
    # X-Verified-Caller header is rejected.
    c = _client(monkeypatch, REWARD_REQUIRE_AUTH="true")
    assert c.post("/reward", json={}).status_code == 401


def test_optin_verified_caller_header_is_trusted(monkeypatch) -> None:
    c = _client(monkeypatch, REWARD_REQUIRE_AUTH="true")
    r = c.post(
        "/reward",
        json={},
        headers={
            "X-Verified-Caller": "arn:aws:sts::123456789012:assumed-role/AgentCoreRole/session"
        },
    )
    assert r.status_code == 200


def test_optin_empty_verified_caller_header_is_rejected(monkeypatch) -> None:
    # A present-but-empty value must fail closed (falsy -> 401), same as missing.
    c = _client(monkeypatch, REWARD_REQUIRE_AUTH="true")
    r = c.post("/reward", json={}, headers={"X-Verified-Caller": ""})
    assert r.status_code == 401


def test_optin_non_arn_verified_caller_header_is_rejected(monkeypatch) -> None:
    # A present, non-empty, but non-ARN-shaped value is rejected — a caller
    # with direct network access but no identity-injecting proxy in front of
    # this server cannot satisfy the check with an arbitrary string.
    c = _client(monkeypatch, REWARD_REQUIRE_AUTH="true")
    r = c.post("/reward", json={}, headers={"X-Verified-Caller": "fake-arn"})
    assert r.status_code == 401


# --------------------------------------------------------------------------- #
# dispatch + batch limits (server module has no top-level Neuron imports)
# --------------------------------------------------------------------------- #
def test_unknown_op_is_reported() -> None:
    server = importlib.import_module("reward_server.server")
    out = server._dispatch({"op": "does-not-exist"})
    assert "unknown op" in out["error"]


def test_missing_op_is_reported() -> None:
    server = importlib.import_module("reward_server.server")
    out = server._dispatch({})
    assert "unknown op" in out["error"]


def test_batch_over_limit_rejected(monkeypatch) -> None:
    monkeypatch.setenv("REWARD_REQUIRE_AUTH", "false")
    server = importlib.reload(importlib.import_module("reward_server.server"))
    client = server.app.test_client()
    resp = client.post("/reward/batch", json={"items": [{"op": "skill"}] * 65})
    assert resp.status_code == 400
    assert "batch size" in resp.get_json()["error"]


def test_batch_items_must_be_list(monkeypatch) -> None:
    monkeypatch.setenv("REWARD_REQUIRE_AUTH", "false")
    server = importlib.reload(importlib.import_module("reward_server.server"))
    client = server.app.test_client()
    resp = client.post("/reward/batch", json={"items": "notalist"})
    assert resp.status_code == 400


def test_batch_within_limit_dispatches(monkeypatch) -> None:
    monkeypatch.setenv("REWARD_REQUIRE_AUTH", "false")
    server = importlib.reload(importlib.import_module("reward_server.server"))
    client = server.app.test_client()
    # `skill` op is a pure stub (no Neuron), so a small batch returns results.
    resp = client.post("/reward/batch", json={"items": [{"op": "skill"}] * 2})
    assert resp.status_code == 200
    assert len(resp.get_json()["results"]) == 2
