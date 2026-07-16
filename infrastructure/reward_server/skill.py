"""Server-side skill DB endpoint.

Keyword-scored retrieval over an NKI knowledge base (``skill_library/SKILL.md``)
of tiling patterns, memory-hierarchy rules, engine mapping, and worked recipes
(matmul, softmax, layernorm, flash-attention, activations) plus anti-patterns.

The agent's multi-turn loop calls this before writing a kernel, so returning
real, on-point NKI guidance — not a stub — is what lets the model pick a correct
tiling/layout for higher-rank L2/L3 tasks instead of guessing.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_SKILL_MD_PATH = os.environ.get(
    "NKI_SKILL_MD_PATH",
    str(Path(__file__).resolve().parent / "skill_library" / "SKILL.md"),
)

# Cap per returned section so a lookup can't blow the model's context.
_MAX_SECTION_CHARS = 4000


@lru_cache(maxsize=1)
def _load_sections(path: str) -> tuple[tuple[str, str], ...]:
    """Load SKILL.md and split into (heading, body) sections by Markdown headings."""
    p = Path(path)
    if p.suffix.lower() not in (".md", ".markdown", ".txt") or not p.is_file():
        return ()
    doc = p.read_text(encoding="utf-8", errors="ignore")
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_re.finditer(doc))
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        sections.append((heading, doc[start:end].strip()))
    if not sections and doc.strip():
        sections.append(("(root)", doc.strip()))
    return tuple(sections)


def _score(heading: str, body: str, keywords: list[str]) -> float:
    """TF-like relevance: heading matches weigh more than body matches (capped)."""
    hl, bl = heading.lower(), body.lower()
    score = 0.0
    for kw in keywords:
        k = kw.lower()
        if k in hl:
            score += 3.0
        count = bl.count(k)
        if count:
            score += min(count, 5)
    return score


def skill_search(payload: dict[str, Any]) -> dict[str, Any]:
    query = (payload.get("query") or payload.get("topic") or "").strip()
    top_k = int(payload.get("top_k", 5))

    sections = _load_sections(_SKILL_MD_PATH)
    if not sections:
        return {"results": [], "total_matches": 0,
                "note": f"skill document not found at {_SKILL_MD_PATH}"}
    if not query:
        return {"results": [], "total_matches": 0, "note": "empty query"}

    keywords = re.findall(r"[a-zA-Z0-9_]+", query)
    scored = sorted(
        ((h, b, _score(h, b, keywords)) for h, b in sections),
        key=lambda x: x[2],
        reverse=True,
    )
    hits = [(h, b, s) for h, b, s in scored[:top_k] if s > 0]

    results = [
        {
            "heading": h,
            "content": b[:_MAX_SECTION_CHARS] + ("... [truncated]" if len(b) > _MAX_SECTION_CHARS else ""),
            "score": s,
        }
        for h, b, s in hits
    ]
    return {"results": results, "total_matches": len(results), "query": query}
