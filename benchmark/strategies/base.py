# benchmark/strategies/base.py
"""BaseStrategy ABC for all routing strategies."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from benchmark.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, RATE_LIMIT_S
from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


class BaseStrategy(ABC):
    """
    Abstract base for routing strategies.

    Each strategy receives a query dict and returns a result dict.
    Results are saved incrementally — experiments can resume after interruption.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier used in result filenames."""

    @abstractmethod
    def route(self, query: dict) -> dict:
        """
        Route a single query. Returns a result dict that must include:
        query_id, strategy, model, correct, cost_usd,
        prompt_tokens, completion_tokens.
        """

    def run(self, queries: list[dict], results_path: Path) -> list[dict]:
        """
        Run all queries with incremental save and resume support.
        Calls self.route() for each query not yet in results_path.
        """
        # Resume from existing results
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            done_ids = {r["query_id"] for r in results}
            if done_ids:
                print(f"  Resuming {self.name}: {len(done_ids)}/{len(queries)} done")
                self._on_resume(results)
        else:
            results = []
            done_ids = set()

        for q in queries:
            if q["query_id"] in done_ids:
                continue

            result = self.route(q)
            results.append(result)
            done_ids.add(q["query_id"])

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            n = len(results)
            acc = sum(r["correct"] for r in results) / n
            cost = sum(r["cost_usd"] for r in results)
            print(
                f"  [{n:4d}/{len(queries)}] {self.name:35s} | "
                f"{'✓' if result['correct'] else '✗'} | acc={acc:.1%} | cost=${cost:.4f}"
            )

            time.sleep(RATE_LIMIT_S)

        return results

    def _on_resume(self, existing_results: list[dict]) -> None:
        """Called when resuming — subclasses can restore internal state here."""


def call_llm(
    model: str,
    prompt: str,
    system: str = "You are a helpful assistant.",
    max_tokens: int = 10,
    temperature: float = 0.0,
) -> tuple[str, int, int]:
    """
    Call OpenRouter LLM.
    Returns (response_text, prompt_tokens, completion_tokens).
    """
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=60.0,
    )
    content = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    return content, pt, ct
