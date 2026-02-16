#!/usr/bin/env python3
"""
Basic routing test - No API keys required.

This test uses mocked LiteLLM responses to validate routing logic
without making real API calls.

Run with: python tests/manual/test_basic_routing.py
"""

from unittest.mock import Mock, patch


def test_basic_routing_mocked():
    """Test that routing selects the cheapest model meeting quality threshold."""

    # Mock LiteLLM to avoid real API calls
    with patch("litellm.completion") as mock_completion:
        # Set up mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello! How can I help?"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
        mock_response.model = "gpt-4o-mini"
        mock_completion.return_value = mock_response

        # Import after patching
        from routesmith import RouteSmith

        # Create router and register models
        rs = RouteSmith()

        # Expensive, high-quality model
        rs.register_model(
            "gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            quality_score=0.95,
        )

        # Cheap, good-enough model
        rs.register_model(
            "gpt-4o-mini",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            quality_score=0.85,
        )

        # Make a request with min_quality=0.8 (should use mini)
        response = rs.completion(
            messages=[{"role": "user", "content": "Hello!"}],
            min_quality=0.8,
        )

        # Verify response
        assert response.choices[0].message.content == "Hello! How can I help?"

        # Check which model was called
        call_args = mock_completion.call_args
        model_used = call_args.kwargs.get("model") or call_args.args[0]
        print(f"Model used: {model_used}")

        # Should have picked the cheaper model since it meets quality threshold
        assert model_used == "gpt-4o-mini", f"Expected gpt-4o-mini, got {model_used}"

        # Check stats
        print(f"Stats: {rs.stats}")
        assert rs.stats["request_count"] == 1

        print("Basic routing test PASSED")


def test_high_quality_routing_mocked():
    """Test that high quality requirement routes to premium model."""

    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Complex answer here"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=50)
        mock_response.model = "gpt-4o"
        mock_completion.return_value = mock_response

        from routesmith import RouteSmith

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

        # Request with high quality requirement (should use gpt-4o)
        response = rs.completion(
            messages=[{"role": "user", "content": "Explain quantum computing"}],
            min_quality=0.9,  # Higher than mini's 0.85
        )

        call_args = mock_completion.call_args
        model_used = call_args.kwargs.get("model") or call_args.args[0]
        print(f"Model used for high-quality request: {model_used}")

        assert model_used == "gpt-4o", f"Expected gpt-4o for high quality, got {model_used}"
        print("High quality routing test PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("RouteSmith Basic Routing Tests (Mocked)")
    print("=" * 60)
    print()

    test_basic_routing_mocked()
    print()

    test_high_quality_routing_mocked()
    print()

    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)
