"""Runtime configuration for the MCP server.

All knobs are environment-variable-driven so the server is trivially deployable
in different AWS accounts, regions, and dev/prod modes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    agentcore_arn: str
    agentcore_qualifier: str
    reward_server_url: str
    aws_region: str
    request_timeout_s: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            # The deployed runtime is addressed by ARN through the AgentCore
            # data-plane API (`InvokeAgentRuntime`), not by a base URL: the
            # invocation path is
            # `/runtimes/{url-encoded-arn}/invocations?qualifier=...`, which no
            # single "endpoint" prefix can express. Resolve it with
            #   aws bedrock-agentcore-control list-agent-runtimes \
            #     --query "agentRuntimes[?agentRuntimeName=='<name>'].agentRuntimeArn"
            # the same way cicd/smoke-live.sh does. Empty means "not configured";
            # AgentCoreClient raises a directed error rather than calling AWS.
            agentcore_arn=os.environ.get("AGENTCORE_ARN", ""),
            agentcore_qualifier=os.environ.get("AGENTCORE_QUALIFIER", "DEFAULT"),
            reward_server_url=os.environ.get(
                # Defaults to loopback for local dev against a stub server.
                # For a real deployment, set this to the reward server's
                # RewardApiInvokeUrl stack output (the API Gateway invoke
                # URL) — every RewardServerClient call is SigV4-signed
                # against the `execute-api` service, so this must point at
                # API Gateway, not the Trn1's raw IP:5050.
                "REWARD_SERVER_URL",
                "http://127.0.0.1:5050",
            ),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            request_timeout_s=int(os.environ.get("MCP_REQUEST_TIMEOUT_S", "1800")),
        )
