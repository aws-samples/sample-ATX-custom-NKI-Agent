"""Model router for the AgentCore deployment.

Picks among Opus 4.8, Qwen3-Coder-30B + SFT-v4 LoRA (via Bedrock CMI), and a
self-hosted vLLM failover. Customers can pin a backend per invocation; the
router defaults to picking based on `nki_skill_lookup` confidence.

Escalation: if the specialist model fails compile/verify twice in a row, the
next attempt swaps to Opus 4.8 carrying the error feedback. Reverse escalation
does not happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TaskSpec:
    description: str
    reference_source: str
    input_specs: list[dict[str, Any]]
    operator_signature: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    model_pin: str | None = None
    nki_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendConfig:
    name: str
    provider: str
    model_id: str | None = None
    model_arn: str | None = None
    endpoint: str | None = None
    inference_config: dict[str, Any] = field(default_factory=dict)


class ModelRouter:
    """Pure-policy router. Holds no state across invocations.

    `failure_history` is supplied per-call by the agent loop so escalation can
    be deterministic.
    """

    def __init__(
        self,
        backends: dict[str, BackendConfig],
        skill_lookup: Callable[[str], float],
        skill_score_threshold: float = 0.75,
        escalate_after_failures: int = 2,
        pinned_disables_escalation: bool = True,
    ) -> None:
        self.backends = backends
        self.skill_lookup = skill_lookup
        self.threshold = skill_score_threshold
        self.escalate_after = escalate_after_failures
        self.pinned_disables_escalation = pinned_disables_escalation

    def resolve(
        self,
        task: TaskSpec,
        recent_failures: list[str] | None = None,
    ) -> BackendConfig:
        """Pick a backend.

        recent_failures is the list of backend names that failed in the last
        consecutive attempts. Used to drive escalation.
        """
        recent_failures = recent_failures or []

        # 1. Honour explicit pin (and disable escalation if configured).
        if task.model_pin:
            if (
                self.pinned_disables_escalation
                or len(recent_failures) < self.escalate_after
            ):
                return self.backends[task.model_pin]
            # If the user opted into escalation, fall through to escalation logic.

        # 2. Escalation: if the specialist failed >= N times in a row, swap to opus.
        if (
            len(recent_failures) >= self.escalate_after
            and recent_failures[-1] == "qwen3-sft-v4"
        ):
            return self.backends["opus-4-8"]

        # 3. Default: skill confidence determines specialist vs generalist.
        score = self.skill_lookup(task.operator_signature or task.description)
        if score >= self.threshold:
            return self.backends.get("qwen3-sft-v4", self.backends["opus-4-8"])
        return self.backends["opus-4-8"]
