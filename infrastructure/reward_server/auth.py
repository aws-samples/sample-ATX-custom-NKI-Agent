"""Authentication / access control for the reward server.

The reward server's trust boundary is the NETWORK, not this process. The Trn1
runs in a PRIVATE_ISOLATED subnet (no public IP, no internet gateway, no NAT)
and its security group accepts port 5050 only from the AgentCore runtime's
client security group (SG-to-SG). So the single AgentCore runtime is the only
thing that can reach this server at all — there is no gateway, no VPC Link, and
no shared secret in the path. See reward_server_cdk/lib/reward-server-stack.ts.

Because reachability is already restricted to one peer SG inside a no-egress
subnet, this module does not add a second application-layer credential check by
default. It keeps two things:

  * `REWARD_REQUIRE_AUTH` — when explicitly set truthy, require the
    `X-Verified-Caller` header (an optional belt-and-suspenders hook for a
    deployment that DOES front the server with an identity-injecting proxy).
    Defaults to OFF, since the network is the boundary in the shipped design.
    This check validates that the header's *value* is a well-formed AWS ARN —
    it does not, and cannot, cryptographically verify that the caller actually
    is that principal. The real authentication has to happen upstream (e.g.
    API Gateway IAM/SigV4 authorizer, or an equivalent identity-aware proxy)
    that injects this header only after verifying the caller and that this
    server can trust not to be bypassable by the caller setting the header
    itself. Format validation narrows the accepted values to syntactically
    valid ARNs — it is a defense-in-depth check against a misconfigured or
    missing upstream proxy, not a substitute for one.
  * The `/health` and `/health/deep` endpoints are always unauthenticated (the
    provisioning readiness gate curls them with no credential).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable

from flask import Flask, abort, request

logger = logging.getLogger(__name__)

_HEALTH_PATHS = ("/health", "/health/deep")

# arn:<partition>:<service>:<region>:<account-id>:<resource>
# Region and account-id may be empty for some resource types (e.g. IAM), so
# both are optional rather than fixed-width. This validates *shape* only, not
# that the ARN refers to a real, authorized principal — see the module
# docstring for why that distinction matters here.
_ARN_RE = re.compile(
    r"^arn:(aws|aws-cn|aws-us-gov):[a-zA-Z0-9-]+:[a-zA-Z0-9-]*:[0-9]*:.+$"
)


def _is_valid_arn(value: str) -> bool:
    """Checks whether a string is a well-formed AWS ARN.

    Args:
        value (str): the header value to validate.

    Returns:
        bool: True if `value` matches the ARN shape
            `arn:<partition>:<service>:<region>:<account-id>:<resource>`.
    """
    return bool(_ARN_RE.match(value))


def install(app: Flask) -> None:
    # Default OFF: network isolation (SG-to-SG + private-isolated subnet) is the
    # access control in the shipped design. Set REWARD_REQUIRE_AUTH=true only if
    # this server is deployed behind a proxy that injects X-Verified-Caller.
    require_auth = os.environ.get("REWARD_REQUIRE_AUTH", "false").lower() == "true"

    if not require_auth:
        logger.info(
            "reward server app-layer auth OFF (network isolation is the trust "
            "boundary: private-isolated subnet + SG-to-SG on 5050). Set "
            "REWARD_REQUIRE_AUTH=true to also require an X-Verified-Caller header."
        )
        return

    @app.before_request
    def _check_auth() -> None:  # pyright: ignore[reportUnusedFunction]
        # Health checks are always unauthenticated (the provisioning readiness
        # gate curls them with no credential).
        if request.path in _HEALTH_PATHS:
            return None
        # Opt-in mode: require an X-Verified-Caller header injected by an
        # upstream identity-aware proxy, with a well-formed ARN value. A
        # caller with direct network access but no such proxy in front of
        # this server cannot satisfy this by supplying an arbitrary string
        # (e.g. "fake-arn"). A missing, empty, or malformed value fails
        # closed (401).
        caller = request.headers.get("X-Verified-Caller", "")
        if caller and _is_valid_arn(caller):
            return None
        abort(401, description="missing or invalid authentication")
        return None


def require(_func: Callable[..., object]) -> Callable[..., object]:
    """Decorator stub kept for API parity with future per-route auth."""
    return _func
