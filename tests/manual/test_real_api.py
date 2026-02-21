#!/usr/bin/env python3
"""
Real API test - Requires API keys.

This test makes actual API calls to validate end-to-end functionality.
Set environment variables before running:
  export OPENAI_API_KEY=sk-...

Run with: python tests/manual/test_real_api.py

NOTE: These tests are skipped by pytest by default unless API keys are set.
"""

import os
import sys

import pytest

# Skip all tests in this module unless API keys are set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires API keys. Set OPENAI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY to run."
)


def check_api_keys():
    """Check if required API keys are set."""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not any([openai_key, anthropic_key, groq_key]):
        print("ERROR: No API keys found!")
        print()
        print("Set at least one of:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export GROQ_API_KEY=gsk_...")
        print()
        print("Tip: Groq has a generous free tier for testing.")
        sys.exit(1)

    return {
        "openai": bool(openai_key),
        "anthropic": bool(anthropic_key),
        "groq": bool(groq_key),
    }


def test_openai_routing():
    """Test routing with OpenAI models."""
    from routesmith import RouteSmith

    print("Testing OpenAI routing...")

    rs = RouteSmith()
    rs.register_model(
        "gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.95,
    )
    rs.register_model(
        "gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=0.85,
    )

    # Simple query - should route to mini
    response = rs.completion(
        messages=[{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
        min_quality=0.8,
    )

    print(f"  Response: {response.choices[0].message.content}")
    print(f"  Stats: {rs.stats}")
    print("  OpenAI routing test PASSED")
    return True


def test_groq_routing():
    """Test routing with Groq models (free tier)."""
    from routesmith import RouteSmith

    print("Testing Groq routing...")

    rs = RouteSmith()
    rs.register_model(
        "groq/llama-3.3-70b-versatile",
        cost_per_1k_input=0.00059,
        cost_per_1k_output=0.00079,
        quality_score=0.90,
    )
    rs.register_model(
        "groq/llama-3.3-70b-specdec",
        cost_per_1k_input=0.00059,
        cost_per_1k_output=0.00079,
        quality_score=0.88,
    )

    response = rs.completion(
        messages=[{"role": "user", "content": "What is the capital of France? Reply briefly."}],
        min_quality=0.7,
    )

    print(f"  Response: {response.choices[0].message.content}")
    print(f"  Stats: {rs.stats}")
    print("  Groq routing test PASSED")
    return True


def test_cost_savings_demo():
    """Demonstrate cost savings with routing."""
    from routesmith import RouteSmith

    print("Demonstrating cost savings...")

    rs = RouteSmith()
    rs.register_model(
        "gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.95,
    )
    rs.register_model(
        "gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=0.85,
    )

    # Run multiple queries
    queries = [
        "What is 2+2?",
        "Hello!",
        "What day is today?",
        "Say 'hi' back to me.",
        "What is the capital of France?",
    ]

    for query in queries:
        rs.completion(
            messages=[{"role": "user", "content": query}],
            min_quality=0.8,
        )

    stats = rs.stats
    print(f"  Requests: {stats['request_count']}")
    print(f"  Total cost: ${stats['total_cost_usd']:.6f}")

    # Calculate what it would have cost with only gpt-4o
    # (This is a simplified estimate)
    estimated_without_routing = stats["total_cost_usd"] * 10  # Rough estimate
    savings = estimated_without_routing - stats["total_cost_usd"]

    print(f"  Estimated without routing: ${estimated_without_routing:.4f}")
    print(f"  Estimated savings: ${savings:.4f}")
    print("  Cost savings demo PASSED")
    return True


def test_feedback_system():
    """Test feedback collection: request IDs, outcome recording, and SQLite storage."""
    from routesmith import RouteSmith, RouteSmithConfig

    print("Testing feedback system...")

    config = RouteSmithConfig(
        feedback_enabled=True,
        feedback_sample_rate=1.0,
        feedback_storage_path=":memory:",
    )
    rs = RouteSmith(config=config)
    rs.register_model(
        "gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=0.85,
    )

    # 1. Completion should generate request_id
    response = rs.completion(
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        include_metadata=True,
    )

    rid = response._routesmith_request_id
    assert rid is not None and len(rid) == 16, f"Bad request_id: {rid}"
    assert response.routesmith_metadata["request_id"] == rid
    assert rs.last_routing_metadata.request_id == rid
    print(f"  request_id: {rid}")
    print(f"  Response: {response.choices[0].message.content}")

    # 2. Record should be in storage
    stored = rs.feedback._storage.get_record(rid)
    assert stored is not None, "Record not found in storage"
    print(f"  Stored in SQLite: model={stored['model_id']}, latency={stored['latency_ms']:.1f}ms")

    # 3. Implicit signals should be persisted
    conn = rs.feedback._storage._get_conn()
    signals = conn.execute(
        "SELECT signal_name, signal_value FROM outcome_signals WHERE request_id = ?",
        (rid,),
    ).fetchall()
    assert len(signals) == 6, f"Expected 6 signals, got {len(signals)}"
    print(f"  Implicit signals: {[(s['signal_name'], s['signal_value']) for s in signals]}")

    # 4. Record outcome and verify predictor update
    old_prior = rs.router.predictor.model_quality_priors.get("gpt-4o-mini")
    rs.record_outcome(rid, score=0.95, feedback="correct answer")
    new_prior = rs.router.predictor.model_quality_priors.get("gpt-4o-mini")
    assert new_prior != old_prior, "Predictor was not updated"
    print(f"  Predictor updated: {old_prior:.4f} -> {new_prior:.4f}")

    # 5. Verify outcome in storage
    stored = rs.feedback._storage.get_record(rid)
    assert stored["quality_score"] == 0.95
    assert stored["user_feedback"] == "correct answer"
    print(f"  Outcome stored: score={stored['quality_score']}, feedback='{stored['user_feedback']}'")

    # 6. Training data should be available
    training = rs.feedback._storage.get_training_data()
    assert len(training) >= 1
    print(f"  Training data rows: {len(training)}")

    print("  Feedback system test PASSED")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("RouteSmith Real API Tests")
    print("=" * 60)
    print()

    # Check API keys
    available = check_api_keys()
    print(f"Available providers: {[k for k, v in available.items() if v]}")
    print()

    tests_run = 0
    tests_passed = 0

    # Run tests based on available keys
    if available["openai"]:
        tests_run += 1
        try:
            if test_openai_routing():
                tests_passed += 1
        except Exception as e:
            print(f"  OpenAI test FAILED: {e}")
        print()

    if available["groq"]:
        tests_run += 1
        try:
            if test_groq_routing():
                tests_passed += 1
        except Exception as e:
            print(f"  Groq test FAILED: {e}")
        print()

    if available["openai"]:
        tests_run += 1
        try:
            if test_cost_savings_demo():
                tests_passed += 1
        except Exception as e:
            print(f"  Cost savings demo FAILED: {e}")
        print()

    if available["openai"]:
        tests_run += 1
        try:
            if test_feedback_system():
                tests_passed += 1
        except Exception as e:
            print(f"  Feedback system test FAILED: {e}")
        print()

    print("=" * 60)
    print(f"Results: {tests_passed}/{tests_run} tests passed")
    print("=" * 60)
