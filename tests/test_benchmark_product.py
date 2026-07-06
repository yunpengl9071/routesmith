"""Test that ProductRouterStrategy can run offline (no API calls)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from benchmark.config import STRONG_MODEL, WEAK_MODEL
from benchmark.strategies.product_router import ProductRouterStrategy


def _make_query(query_id: int, answer_letter: str = "A") -> dict:
    return {
        "query_id": query_id,
        "dataset": "mmlu",
        "category": "STEM",
        "prompt": (
            "Test question?\n"
            "A. First\nB. Second\nC. Third\nD. Fourth\n\n"
            "Answer with a single letter (A, B, C, or D):"
        ),
        "question": "Test question?",
        "answer_letter": answer_letter,
        "subject": "test",
    }


@pytest.mark.parametrize("n_queries", [1, 10])
def test_product_strategy_runs_offline(n_queries: int) -> None:
    """Mock call_llm, run queries, verify selections are from registered models."""
    mock_resp = ("A", 10, 5)  # (text, prompt_tokens, completion_tokens)

    with patch("benchmark.strategies.product_router.call_llm", return_value=mock_resp):
        strategy = ProductRouterStrategy()
        queries = [_make_query(i) for i in range(n_queries)]

        for q in queries:
            result = strategy.route(q)

            assert result["model"] in [STRONG_MODEL, WEAK_MODEL]
            assert result["final_model"] in [STRONG_MODEL, WEAK_MODEL]
            assert result["correct"] is True
            assert result["cost_usd"] > 0
            assert result["prompt_tokens"] == 10
            assert result["completion_tokens"] == 5
            assert result["strategy"] == "product_router_lints"
            assert result["query_id"] == q["query_id"]
            assert result["dataset"] == "mmlu"


def test_product_strategy_resume_state() -> None:
    """Verify resume saves and restores router state correctly."""
    mock_resp = ("A", 10, 5)

    with patch("benchmark.strategies.product_router.call_llm", return_value=mock_resp):
        strategy = ProductRouterStrategy()
        q = _make_query(0)
        result1 = strategy.route(q)

        state1 = result1["router_state"]
        assert state1["t"] >= 0
        assert len(state1["arms"]) >= 2

        # Route another query to advance state
        q2 = _make_query(1)
        result2 = strategy.route(q2)
        state2 = result2["router_state"]
        assert state2["t"] >= state1["t"]

        # Simulate resume: create a fresh strategy, load state
        strategy2 = ProductRouterStrategy()
        strategy2._on_resume([result2])
        assert strategy2.router.predictor._router._t == state2["t"]


def test_product_strategy_name() -> None:
    assert ProductRouterStrategy().name == "product_router_lints"
