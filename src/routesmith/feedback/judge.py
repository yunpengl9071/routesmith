"""LLM-as-judge quality evaluator."""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are a strict grader of AI assistant responses. Score how well the response "
    "answers the user's request: correctness, completeness, instruction-following. "
    'Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, "reason": "<max 15 words>"}'
)

_SCORE_RE = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?)')


class LLMJudge:
    def __init__(self, model: str, timeout_s: float = 20.0) -> None:
        self.model = model
        self.timeout_s = timeout_s

    def score(self, messages: list[dict], response_text: str) -> float | None:
        """Return quality in [0,1], or None on any failure (never raises)."""
        try:
            import litellm
            user_prompt = self._render(messages, response_text)
            result = litellm.completion(
                model=self.model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=80,
                timeout=self.timeout_s,
            )
            text = result.choices[0].message.content or ""
            return self._parse(text)
        except Exception as e:
            logger.warning(f"Judge scoring failed: {e}")
            return None

    @staticmethod
    def _render(messages: list[dict], response_text: str) -> str:
        last_user = next((m.get("content", "") for m in reversed(messages)
                          if m.get("role") == "user"), "")
        return (f"USER REQUEST:\n{last_user[:4000]}\n\n"
                f"ASSISTANT RESPONSE:\n{response_text[:4000]}\n\nGrade it.")

    @staticmethod
    def _parse(text: str) -> float | None:
        try:
            score = float(json.loads(text)["score"])
        except Exception:
            m = _SCORE_RE.search(text)
            if not m:
                return None
            score = float(m.group(1))
        return max(0.0, min(1.0, score))
