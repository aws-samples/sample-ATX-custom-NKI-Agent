"""Tests for the AgentCore data-plane client.

The bug these pin: `AgentCoreClient` used to build its own request as
``POST {AGENTCORE_ENDPOINT}/invoke`` signed for service ``bedrock-agent``. The
real AgentCore data plane is
``POST /runtimes/{url-encoded-arn}/invocations?qualifier={qualifier}`` on
``bedrock-agentcore``, so no value of ``AGENTCORE_ENDPOINT`` could ever reach a
deployed runtime — every call 404'd, and `nki_generate_kernel` was unreachable
in practice while the hardware-free tools kept working.

So the tests assert the *call shape* against a stubbed boto3 client, not just
that some request went out.
"""

from __future__ import annotations

import json

import pytest
from kernelforge_nki_mcp.clients import AgentCoreClient, ConfigError
from kernelforge_nki_mcp.config import Config

_ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/nki_agent-abc123"


def _cfg(**over) -> Config:
    base = dict(
        agentcore_arn=_ARN,
        agentcore_qualifier="DEFAULT",
        reward_server_url="http://127.0.0.1:5050",
        aws_region="us-east-1",
        request_timeout_s=60,
    )
    base.update(over)
    return Config(**base)


class _StubBody:
    """Mimics botocore's streaming response body."""

    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def iter_chunks(self):
        # Split across chunks so a client that reads only the first one fails.
        yield self._raw[:3]
        yield self._raw[3:]


class _StubRuntime:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def invoke_agent_runtime(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": _StubBody(self.payload)}


def _wire(client: AgentCoreClient, stub: _StubRuntime) -> None:
    client._client = stub  # bypass credential lookup / real client construction


def test_invoke_targets_the_runtime_by_arn_with_a_qualifier() -> None:
    stub = _StubRuntime({"status": "success", "turns_used": 2})
    client = AgentCoreClient(_cfg())
    _wire(client, stub)

    client.invoke({"mode": "multi_turn", "max_turns": 3})

    (call,) = stub.calls
    assert call["agentRuntimeArn"] == _ARN
    assert call["qualifier"] == "DEFAULT"
    assert json.loads(call["payload"]) == {"mode": "multi_turn", "max_turns": 3}


def test_invoke_sends_a_session_id_the_api_will_accept() -> None:
    # InvokeAgentRuntime rejects runtimeSessionId shorter than 33 characters, so
    # a bare uuid4().hex (32) is one character too short.
    stub = _StubRuntime({"status": "success"})
    client = AgentCoreClient(_cfg())
    _wire(client, stub)

    client.invoke({})

    session_id = stub.calls[0]["runtimeSessionId"]
    assert 33 <= len(session_id) <= 100, f"runtimeSessionId is {len(session_id)} chars"


def test_invoke_reassembles_a_chunked_response_body() -> None:
    stub = _StubRuntime({"status": "success", "kernel_source": "import neuronxcc"})
    client = AgentCoreClient(_cfg())
    _wire(client, stub)

    assert client.invoke({}) == {
        "status": "success",
        "kernel_source": "import neuronxcc",
    }


def test_each_invoke_uses_a_fresh_session_id() -> None:
    # Reusing a session id would fold two independent generation runs into one
    # server-side conversation.
    stub = _StubRuntime({"status": "success"})
    client = AgentCoreClient(_cfg())
    _wire(client, stub)

    client.invoke({})
    client.invoke({})

    assert stub.calls[0]["runtimeSessionId"] != stub.calls[1]["runtimeSessionId"]


def test_honors_a_non_default_qualifier() -> None:
    stub = _StubRuntime({"status": "success"})
    client = AgentCoreClient(_cfg(agentcore_qualifier="canary"))
    _wire(client, stub)

    client.invoke({})

    assert stub.calls[0]["qualifier"] == "canary"


def test_missing_arn_fails_before_calling_aws() -> None:
    stub = _StubRuntime({"status": "success"})
    client = AgentCoreClient(_cfg(agentcore_arn=""))
    _wire(client, stub)

    with pytest.raises(ConfigError, match="AGENTCORE_ARN"):
        client.invoke({})
    assert stub.calls == [], "called AWS despite having no runtime ARN"


def test_config_reads_the_runtime_arn_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("AGENTCORE_ARN", _ARN)
    monkeypatch.delenv("AGENTCORE_QUALIFIER", raising=False)

    cfg = Config.from_env()

    assert cfg.agentcore_arn == _ARN
    assert cfg.agentcore_qualifier == "DEFAULT"


def test_config_defaults_to_unconfigured_rather_than_a_fake_endpoint(monkeypatch) -> None:
    # The old default was a bogus URL ("https://agentcore.local.invalid"), which
    # turned a missing-config mistake into an opaque connection error at call
    # time. Empty means AgentCoreClient can say what's actually wrong.
    monkeypatch.delenv("AGENTCORE_ARN", raising=False)

    assert Config.from_env().agentcore_arn == ""
