"""HTTP clients for the two backend services the MCP tools delegate to.

`AgentCoreClient` calls the real Bedrock AgentCore `InvokeAgentRuntime` API
through boto3, which signs and routes the request. That API genuinely verifies
the signature; this path is unaffected by anything below.

`RewardServerClient` still computes and sends SigV4 headers too, but as of
the reward server's SG-to-SG redesign (no more API Gateway — see
reward_server_cdk/lib/reward-server-stack.ts and reward_server/auth.py)
those headers are sent to a plain Flask process that performs no signature
verification at all. The signing there is currently a no-op: it neither
grants nor blocks anything. What MCP-direct access
to the reward server should actually require is still an open decision —
until that lands, calling
`RewardServerClient` needs a real network path into the VPC (there is no
internet-reachable API Gateway fallback anymore either); the `execute-api:Invoke`
IAM grant this class's signing implies is not what gates access today.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config as BotoConfig

from .config import Config


class AuthError(RuntimeError):
    """Raised when the AWS credential chain is empty."""


def _sigv4_headers(
    cfg: Config,
    method: str,
    url: str,
    body: bytes,
    service: str,
) -> dict[str, str]:
    session = boto3.Session(region_name=cfg.aws_region)
    creds = session.get_credentials()
    if creds is None:
        raise AuthError(
            "no AWS credentials found; configure env, ~/.aws/credentials, "
            "or IAM Identity Center"
        )
    req = AWSRequest(method=method, url=url, data=body)
    SigV4Auth(creds, service, cfg.aws_region).add_auth(req)
    return dict(req.headers)


class ConfigError(RuntimeError):
    """Raised when a required piece of runtime configuration is missing."""


class AgentCoreClient:
    """Invokes the AgentCore runtime that runs the Strands agent loop.

    Goes through boto3's `bedrock-agentcore` data-plane client rather than a
    hand-rolled signed POST. The runtime is addressed by ARN, and the real
    invocation path is
    `POST /runtimes/{url-encoded-arn}/invocations?qualifier={qualifier}` —
    a shape no `{base_url}/invoke` construction can produce, so hand-rolling it
    silently 404s. Using the SDK also keeps the signing service name
    (`bedrock-agentcore`) and the response's streaming body handling correct.
    This is the same call `cicd/smoke-live.sh`,
    `examples/cuda_rmsnorm_migration/run_migration.py`, and the live tier of
    `tests/test_integration_chain.py` make, so every surface drives the runtime
    identically.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client: Any | None = None

    def _runtime(self) -> Any:
        # Built lazily: importing the server must not require credentials or a
        # configured runtime, so that the hardware-free tools stay callable.
        if self._client is None:
            session = boto3.Session(region_name=self.cfg.aws_region)
            if session.get_credentials() is None:
                raise AuthError(
                    "no AWS credentials found; configure env, ~/.aws/credentials, "
                    "or IAM Identity Center"
                )
            self._client = session.client(
                "bedrock-agentcore",
                config=BotoConfig(
                    read_timeout=self.cfg.request_timeout_s,
                    connect_timeout=30,
                    # The multi-turn loop is not idempotent — a retried invoke
                    # starts a second generation run and bills for it.
                    retries={"max_attempts": 1},
                ),
            )
        return self._client

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.cfg.agentcore_arn:
            raise ConfigError(
                "AGENTCORE_ARN is not set, so there is no AgentCore runtime to "
                "call. Set it to the deployed runtime's ARN (see "
                "`aws bedrock-agentcore-control list-agent-runtimes`) in the "
                "`env` block of your MCP server config or in the environment "
                "that launches it."
            )
        # runtimeSessionId must be 33-100 characters; uuid4().hex is 32, so the
        # prefix is load-bearing rather than cosmetic.
        session_id = f"mcp-{uuid.uuid4().hex}"
        resp = self._runtime().invoke_agent_runtime(
            agentRuntimeArn=self.cfg.agentcore_arn,
            qualifier=self.cfg.agentcore_qualifier,
            runtimeSessionId=session_id,
            payload=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )
        body = resp["response"]
        raw = (
            b"".join(body.iter_chunks())
            if hasattr(body, "iter_chunks")
            else body.read()
        )
        return json.loads(raw)


class RewardServerClient:
    """Calls the Trn1 reward server for compile / verify / profile.

    Still computes and sends a SigV4 signature against the `execute-api`
    service on every call, but this is currently a no-op: the reward server
    no longer sits behind an API Gateway (removed — see
    reward_server_cdk/lib/reward-server-stack.ts), and reward_server/auth.py
    performs no signature verification. Nothing checks these headers today.

    The reward server's actual (and only) access control is network position:
    it is reachable solely from the AgentCore runtime's security group inside
    a private-isolated VPC subnet with no internet route. A caller using this
    class needs a real network path into that VPC (VPN / Direct Connect /
    SSM tunnel) — there is no internet-reachable fallback mode. What
    MCP-direct access should require
    is still an open decision; until then,
    `mcpOperatorPrincipalArns` and `execute-api:Invoke` (both referenced in
    older docs) do not gate anything — that CDK prop was removed as dead
    code.

    `REWARD_SERVICE_NAME` overrides the signed service name; retained for
    forward-compatibility with a design that reintroduces a verifying
    gateway, not because anything currently reads it.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._http = httpx.Client(timeout=cfg.request_timeout_s)
        self._service = os.environ.get("REWARD_SERVICE_NAME", "execute-api")

    def _auth_headers(self, method: str, url: str, body: bytes) -> dict[str, str]:
        return _sigv4_headers(self.cfg, method, url, body, service=self._service)

    def reward(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.cfg.reward_server_url.rstrip('/')}/reward"
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {"content-type": "application/json"}
        headers.update(self._auth_headers("POST", url, body))
        r = self._http.post(url, content=body, headers=headers)
        r.raise_for_status()
        return r.json()

    def reward_batch(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        url = f"{self.cfg.reward_server_url.rstrip('/')}/reward/batch"
        body = json.dumps({"items": payloads}, separators=(",", ":")).encode("utf-8")
        headers = {"content-type": "application/json"}
        headers.update(self._auth_headers("POST", url, body))
        r = self._http.post(url, content=body, headers=headers)
        r.raise_for_status()
        return r.json()["results"]
