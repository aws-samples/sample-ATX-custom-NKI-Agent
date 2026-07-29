"""Tests for the model router's backend-selection policy.

The router is pure policy (no I/O), so its escalation and pin behavior can be
pinned exactly. These guard the rules documented in SKILL.md:
  - skill score >= threshold -> specialist, else Opus
  - N consecutive specialist failures -> escalate to Opus
  - a user pin overrides routing (and, by default, disables escalation)
"""

from __future__ import annotations

from router import BackendConfig, ModelRouter, TaskSpec

OPUS = "opus-4-8"
QWEN = "qwen3-sft-v4"


def _router(score: float, **kw) -> ModelRouter:
    backends = {
        OPUS: BackendConfig(name=OPUS, provider="bedrock", model_id="anthropic.claude-opus-4-8"),
        QWEN: BackendConfig(name=QWEN, provider="bedrock-cmi", model_arn="arn:..."),
        "vllm": BackendConfig(name="vllm", provider="openai-compat", endpoint="http://x"),
    }
    return ModelRouter(backends=backends, skill_lookup=lambda _q: score, **kw)


def _task(**kw) -> TaskSpec:
    base = dict(description="softmax", reference_source="", input_specs=[])
    base.update(kw)
    return TaskSpec(**base)


def test_high_skill_score_routes_to_specialist() -> None:
    r = _router(0.9)
    assert r.resolve(_task()).name == QWEN


def test_low_skill_score_routes_to_opus() -> None:
    r = _router(0.10)
    assert r.resolve(_task()).name == OPUS


def test_threshold_is_inclusive() -> None:
    r = _router(0.75)  # exactly at the default threshold
    assert r.resolve(_task()).name == QWEN


def test_just_below_threshold_routes_to_opus() -> None:
    r = _router(0.7499)
    assert r.resolve(_task()).name == OPUS


def test_explicit_pin_overrides_routing() -> None:
    r = _router(0.9)  # would otherwise pick the specialist
    assert r.resolve(_task(model_pin=OPUS)).name == OPUS
    assert r.resolve(_task(model_pin="vllm")).name == "vllm"


def test_two_specialist_failures_escalate_to_opus() -> None:
    r = _router(0.9)  # high score would keep picking the specialist
    picked = r.resolve(_task(), recent_failures=[QWEN, QWEN])
    assert picked.name == OPUS


def test_single_failure_does_not_escalate() -> None:
    r = _router(0.9)
    assert r.resolve(_task(), recent_failures=[QWEN]).name == QWEN


def test_opus_does_not_escalate_on_its_own_failures() -> None:
    r = _router(0.10)  # routes to Opus
    # Two Opus failures must not flip to the specialist (no reverse escalation).
    assert r.resolve(_task(), recent_failures=[OPUS, OPUS]).name == OPUS


def test_pin_disables_escalation_by_default() -> None:
    r = _router(0.9)
    picked = r.resolve(_task(model_pin=QWEN), recent_failures=[QWEN, QWEN])
    assert picked.name == QWEN  # pinned, so no escalation


def test_pin_can_allow_escalation_when_configured() -> None:
    r = _router(0.9, pinned_disables_escalation=False)
    picked = r.resolve(_task(model_pin=QWEN), recent_failures=[QWEN, QWEN])
    assert picked.name == OPUS


def test_configurable_escalation_threshold() -> None:
    r = _router(0.9, escalate_after_failures=3)
    assert r.resolve(_task(), recent_failures=[QWEN, QWEN]).name == QWEN  # not yet
    assert r.resolve(_task(), recent_failures=[QWEN, QWEN, QWEN]).name == OPUS


def test_missing_specialist_falls_back_to_opus() -> None:
    # If the specialist backend isn't configured, a high score still resolves.
    backends = {OPUS: BackendConfig(name=OPUS, provider="bedrock", model_id="x")}
    r = ModelRouter(backends=backends, skill_lookup=lambda _q: 0.9)
    assert r.resolve(_task()).name == OPUS
