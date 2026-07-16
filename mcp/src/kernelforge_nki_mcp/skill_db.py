"""Tiny in-process skill DB.

In production this should be backed by a vector store (OpenSearch, Pinecone)
or a curated lookup table; here we ship a minimal pattern-match implementation
so the MCP server is self-contained for installs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Skill:
    skill_id: str
    title: str
    category: str
    keywords: tuple[str, ...]
    content: str


_SKILLS: list[Skill] = [
    Skill(
        skill_id="matmul-2d",
        title="2D matmul with PSUM accumulator",
        category="compute",
        keywords=("matmul", "gemm", "linear", "@", "torch.matmul"),
        content="See references/common-ops.md § Matmul.",
    ),
    Skill(
        skill_id="softmax-rowwise",
        title="Numerically stable row-wise softmax",
        category="compute",
        keywords=("softmax", "F.softmax", "logits"),
        content="See references/common-ops.md § Softmax.",
    ),
    Skill(
        skill_id="rmsnorm",
        title="RMSNorm row-wise",
        category="compute",
        keywords=("rmsnorm", "rms_norm", "layernorm", "F.layer_norm"),
        content="See references/common-ops.md § RMSNorm.",
    ),
    Skill(
        skill_id="add-pointwise",
        title="Elementwise add (pointwise)",
        category="compute",
        keywords=("add", "elementwise", "pointwise", "torch.add"),
        content="See references/common-ops.md § Pointwise.",
    ),
    Skill(
        skill_id="flash-attention",
        title="Flash-attention via online softmax",
        category="compute",
        keywords=("attention", "scaled_dot_product_attention", "flash"),
        content="See examples/flash_attention.py.",
    ),
    Skill(
        skill_id="tile-rules",
        title="TILE_M=128 partition rule",
        category="layout",
        keywords=("tile", "partition", "shape", "layout"),
        content="See references/tile-model.md § Tile shape rules.",
    ),
    Skill(
        skill_id="missing-ops",
        title="Manual implementations for sigmoid/relu/tanh/gelu",
        category="api",
        keywords=("sigmoid", "relu", "tanh", "gelu", "sqrt", "pow"),
        content="See references/layout-rules.md § Operator availability.",
    ),
]


@dataclass
class SkillMatch:
    skill_id: str
    title: str
    content: str
    category: str
    relevance_score: float


def search(query: str, top_k: int = 5, category: str | None = None) -> list[SkillMatch]:
    q = query.lower()
    scored: list[tuple[float, Skill]] = []
    for s in _SKILLS:
        if category and s.category != category:
            continue
        score = sum(1.0 for kw in s.keywords if kw in q) / max(1, len(s.keywords))
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        SkillMatch(
            skill_id=s.skill_id,
            title=s.title,
            content=s.content,
            category=s.category,
            relevance_score=round(score, 3),
        )
        for score, s in scored[:top_k]
    ]
