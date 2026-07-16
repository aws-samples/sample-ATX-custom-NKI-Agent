"""Trn1 reward server.

Single source of truth for compile / verify / profile signals. Long-lived
Flask service running on a Trn1.2xlarge with the Neuron SDK installed.

Security defaults:
  - Bind to 127.0.0.1; operator must opt into 0.0.0.0 explicitly.
  - Auth required by default (see auth.py).
  - 4 MB request body cap.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from flask import Flask, jsonify, request

from . import auth
from .baseline import baseline_kernel
from .compile import compile_kernel
from .profile import profile_kernel
from .skill import skill_search
from .verify import verify_kernel

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB cap
auth.install(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DISPATCH = {
    "compile": compile_kernel,
    "verify": verify_kernel,
    "profile": profile_kernel,
    "baseline": baseline_kernel,
    "skill": skill_search,
}


def _dispatch(item: dict[str, Any]) -> dict[str, Any]:
    op = item.get("op")
    handler = _DISPATCH.get(op or "")
    if handler is None:
        return {"error": f"unknown op: {op}"}
    return handler(item)


@app.post("/reward")
def reward() -> Any:
    payload = request.get_json(force=True, silent=True) or {}
    return jsonify(_dispatch(payload))


@app.post("/reward/batch")
def reward_batch() -> Any:
    payload = request.get_json(force=True, silent=True) or {}
    items = payload.get("items", [])
    if not isinstance(items, list):
        return jsonify({"error": "items must be a list"}), 400
    if len(items) > 64:
        return jsonify({"error": "batch size exceeds 64"}), 400
    return jsonify({"results": [_dispatch(i) for i in items]})


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/health/deep")
def health_deep() -> Any:
    """Verify the compile/verify toolchain is importable in *this* interpreter.

    A plain /health only proves Flask is up; it would still pass if the service
    were running under a Python that can't import numpy/torch/neuronxcc (the
    subprocess handlers would then fail every request). This endpoint imports
    the toolchain so provisioning can fail loudly instead of silently serving
    100%-failing compiles.
    """
    checks: dict[str, str] = {}
    ok = True
    for mod in ("numpy", "torch", "neuronxcc.nki"):
        try:
            __import__(mod)
            checks[mod] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks[mod] = f"{type(exc).__name__}: {exc}"
            ok = False
    return jsonify({"status": "ok" if ok else "degraded", "imports": checks}), (
        200 if ok else 503
    )


@app.errorhandler(401)
def _unauthorized(e: Any) -> Any:  # type: ignore[no-untyped-def]
    return jsonify({"error": "unauthorized"}), 401


@app.errorhandler(413)
def _too_large(e: Any) -> Any:  # type: ignore[no-untyped-def]
    return jsonify({"error": "request body too large"}), 413


if __name__ == "__main__":
    # SECURITY: bind to loopback by default. Operator opts into 0.0.0.0
    # explicitly via REWARD_HOST=0.0.0.0 (which should only happen behind a
    # firewall / SG that locks the port to the AgentCore VPC).
    host = os.environ.get("REWARD_HOST", "127.0.0.1")
    port = int(os.environ.get("REWARD_PORT", "5050"))
    if host == "0.0.0.0":  # nosec B104 — warning branch only; default is 127.0.0.1, operator must opt in explicitly via REWARD_HOST
        logger.warning(
            "reward server binding to 0.0.0.0 — ensure security group locks "
            "this port to the AgentCore VPC and auth is enabled."
        )
    app.run(host=host, port=port, threaded=True)
