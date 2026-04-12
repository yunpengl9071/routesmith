"""Lightweight agent role inference from system prompts."""

from __future__ import annotations

import hashlib

# Ordered list of roles. Index 0 is reserved for unknown (role=None).
# Indices 1-8 correspond to these roles (used by role_ordinal()).
AGENT_ROLES: list[str] = [
    "research",
    "coding",
    "summarizer",
    "qa",
    "customer_service",
    "planning",
    "creative",
    "general",
]

_ROLE_KEYWORDS: dict[str, frozenset[str]] = {
    "research": frozenset([
        "research", "analyze", "analysis", "findings", "papers", "literature",
        "study", "investigate", "evidence", "scholar", "academic", "survey",
        "hypothesis", "data", "experiment", "citation",
    ]),
    "coding": frozenset([
        "code", "coding", "programming", "python", "javascript", "typescript",
        "debug", "implement", "algorithm", "function", "class", "developer",
        "software", "bug", "compile", "syntax", "script", "repository",
    ]),
    "summarizer": frozenset([
        "summarize", "summary", "condense", "shorten", "brief",
        "key", "highlight", "extract", "distill", "compress",
    ]),
    "qa": frozenset([
        "answer", "question", "faq", "quiz", "knowledge", "factual",
        "accurate", "correct", "truth", "fact",
    ]),
    "customer_service": frozenset([
        "customer", "support", "service", "issue", "complaint",
        "resolve", "ticket", "refund", "account", "order", "product",
        "satisfaction", "escalate", "politely", "assist",
    ]),
    "planning": frozenset([
        "plan", "planning", "schedule", "roadmap", "task", "project",
        "milestone", "timeline", "organize", "strategy", "objective",
        "goal", "workflow", "prioritize",
    ]),
    "creative": frozenset([
        "creative", "write", "story", "poem", "fiction", "imagine",
        "character", "narrative", "dialogue", "compose", "draft",
        "essay", "blog", "lyric", "brainstorm",
    ]),
}

_CONFIDENCE_THRESHOLD = 0.15


class AgentInferencer:
    """Infer agent role from system prompt using keyword density.

    Returns (role, confidence) where role is None and confidence is 0.0
    when keyword density falls below threshold — never forces a category.
    Results are cached by system-prompt hash; inference is <1ms.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str | None, float]] = {}

    def infer(self, messages: list[dict]) -> tuple[str | None, float]:
        """Infer agent role from messages.

        Uses system prompt if present; falls back to first user message.
        """
        if not messages:
            return None, 0.0

        text = self._extract_text(messages)
        if not text:
            return None, 0.0

        cache_key = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._classify(text)
        self._cache[cache_key] = result
        return result

    def _extract_text(self, messages: list[dict]) -> str:
        for msg in messages:
            if msg.get("role") == "system":
                return str(msg.get("content", "")).lower()
        for msg in messages:
            if msg.get("role") == "user":
                return str(msg.get("content", "")).lower()
        return ""

    def _classify(self, text: str) -> tuple[str | None, float]:
        # Special case: "helpful assistant" or "general purpose" phrases
        if "helpful assistant" in text or "general purpose" in text:
            return "general", 0.7

        words = set(text.split())
        word_count = max(len(words), 1)
        best_role: str | None = None
        best_score = 0.0

        for role, keywords in _ROLE_KEYWORDS.items():
            matches = len(words & keywords)
            score = min(1.0, matches / word_count * 8)
            if score > best_score:
                best_score = score
                best_role = role

        if best_score < _CONFIDENCE_THRESHOLD:
            return None, 0.0

        return best_role, best_score

    @staticmethod
    def role_ordinal(role: str | None) -> int:
        """Return 1-based ordinal for role, or 0 for unknown/None."""
        if role is None:
            return 0
        try:
            return AGENT_ROLES.index(role) + 1
        except ValueError:
            return 0
