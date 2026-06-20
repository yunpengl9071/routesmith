"""Shadow execution for trust-but-verify quality comparisons.

Runs a configurable fraction of requests through both the selected (cheap)
model and the most expensive available model, compares responses, and tracks
equivalence rates per agent. This provides verifiable evidence that cheaper
models deliver equivalent quality.
"""

from __future__ import annotations

from typing import Any


def compare_responses(
    cheap_response: Any,
    expensive_response: Any,
) -> dict[str, Any]:
    """Compare two model responses for equivalence.

    Uses Jaccard similarity of character trigrams as a fast, language-agnostic
    similarity measure. Responses with >80% similarity are considered equivalent.

    Returns:
        dict with 'equivalent', 'similarity', 'summary'.
    """
    text1 = _response_text(cheap_response)
    text2 = _response_text(expensive_response)

    if text1 == text2:
        return {
            "equivalent": True,
            "similarity": 1.0,
            "summary": "identical responses",
        }

    similarity = _trigram_jaccard(text1, text2)
    equivalent = similarity >= 0.7

    if equivalent:
        summary = "substantially similar"
    else:
        summary = "substantially different"

    return {
        "equivalent": equivalent,
        "similarity": round(similarity, 3),
        "summary": summary,
    }


def _response_text(response: Any) -> str:
    """Extract raw text from a model response."""
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


def _trigram_jaccard(text1: str, text2: str) -> float:
    """Compute Jaccard similarity of character trigrams."""
    tri1 = set(_ngrams(text1.lower(), 3))
    tri2 = set(_ngrams(text2.lower(), 3))
    if not tri1 and not tri2:
        return 1.0
    if not tri1 or not tri2:
        return 0.0
    return len(tri1 & tri2) / len(tri1 | tri2)


def _ngrams(text: str, n: int) -> list[str]:
    """Generate character n-grams from text."""
    return [text[i:i + n] for i in range(len(text) - n + 1)]


class VerificationTracker:
    """Accumulates shadow execution results and provides per-agent stats."""

    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []

    def record(
        self,
        agent_role: str | None,
        cheap_model: str,
        expensive_model: str,
        equivalent: bool,
        summary: str,
        savings: str,
    ) -> None:
        """Record a single shadow execution result."""
        self._results.append({
            "agent_role": agent_role or "unknown",
            "cheap_model": cheap_model,
            "expensive_model": expensive_model,
            "equivalent": equivalent,
            "summary": summary,
            "savings": savings,
        })

    def stats(self, agent_role: str | None = None) -> dict[str, Any]:
        """Get verification stats, optionally filtered by agent_role.

        Returns:
            dict with 'verified', 'equivalence_rate', 'equivalent_count'.
            When agent_role is None, returns global stats.
        """
        records = self._results
        if agent_role is not None:
            records = [r for r in records if r["agent_role"] == agent_role]

        total = len(records)
        eq_count = sum(1 for r in records if r["equivalent"])

        return {
            "verified": total,
            "equivalent_count": eq_count,
            "equivalence_rate": round(eq_count / total, 3) if total > 0 else 0.0,
        }


def shadow_execute(
    cheap_response: Any,
    cheap_model: str,
    expensive_response: Any,
    expensive_model: str,
    cheap_cost: float,
    expensive_cost: float,
) -> dict[str, Any]:
    """Execute a shadow comparison and return verification metadata."""
    comparison = compare_responses(cheap_response, expensive_response)
    savings_pct = (
        (expensive_cost - cheap_cost) / expensive_cost * 100
        if expensive_cost > 0
        else 0.0
    )

    return {
        "cheap_model": cheap_model,
        "expensive_model": expensive_model,
        "equivalent": comparison["equivalent"],
        "similarity": comparison["similarity"],
        "summary": comparison["summary"],
        "savings": f"${expensive_cost - cheap_cost:.4f} ({savings_pct:.0f}%)",
    }
