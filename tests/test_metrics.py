"""Unit tests for metrics computation — no API calls needed."""
from __future__ import annotations
import pytest


def test_accuracy():
    from benchmark.metrics import accuracy
    results = [{"correct": True}, {"correct": True}, {"correct": False}]
    assert abs(accuracy(results) - 2/3) < 1e-9


def test_accuracy_empty():
    from benchmark.metrics import accuracy
    assert accuracy([]) == 0.0


def test_total_cost():
    from benchmark.metrics import total_cost
    results = [{"cost_usd": 0.01}, {"cost_usd": 0.02}]
    assert abs(total_cost(results) - 0.03) < 1e-9


def test_pgr_midpoint():
    """PGR=(router-weak)/(strong-weak). Router at midpoint → PGR=0.5."""
    from benchmark.metrics import performance_gap_recovery
    assert abs(performance_gap_recovery(
        router_acc=0.80, weak_acc=0.70, strong_acc=0.90
    ) - 0.5) < 1e-9


def test_pgr_at_weak():
    from benchmark.metrics import performance_gap_recovery
    assert performance_gap_recovery(0.70, 0.70, 0.90) == pytest.approx(0.0)


def test_pgr_at_strong():
    from benchmark.metrics import performance_gap_recovery
    assert performance_gap_recovery(0.90, 0.70, 0.90) == pytest.approx(1.0)


def test_pgr_degenerate():
    """When strong==weak, PGR is undefined — return 0."""
    from benchmark.metrics import performance_gap_recovery
    assert performance_gap_recovery(0.80, 0.80, 0.80) == 0.0


def test_cumulative_regret():
    from benchmark.metrics import cumulative_regret
    oracle = [1.0, 1.0, 1.0]
    actual = [0.8, 1.0, 0.6]
    reg = cumulative_regret(oracle, actual)
    assert len(reg) == 3
    assert abs(reg[0] - 0.2) < 1e-9
    assert abs(reg[1] - 0.2) < 1e-9   # 0.2 + 0.0
    assert abs(reg[2] - 0.6) < 1e-9   # 0.2 + 0.0 + 0.4


def test_convergence_query_converges():
    from benchmark.metrics import convergence_query
    # rolling window of 10 with perfect rewards → converges at query 10
    rewards = [1.0] * 100
    result = convergence_query(rewards, oracle_reward=1.0, window=10, threshold=0.02)
    assert result == 10


def test_convergence_query_never():
    from benchmark.metrics import convergence_query
    # Oscillating rewards → never converge within threshold
    rewards = [0.0, 1.0] * 50
    result = convergence_query(rewards, oracle_reward=1.0, window=10, threshold=0.02)
    assert result is None


def test_strong_usage_pct():
    from benchmark.metrics import strong_usage_pct
    results = [
        {"final_model": "openai/gpt-4o"},
        {"final_model": "openai/gpt-4o-mini"},
        {"final_model": "openai/gpt-4o"},
    ]
    assert strong_usage_pct(results, "openai/gpt-4o") == pytest.approx(2/3)
