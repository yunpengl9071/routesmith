"""Tests for RouteSmith client, cost tracking, and metadata."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from routesmith import RouteSmith, RouteSmithConfig, RoutingStrategy, RoutingMetadata


class TestRoutingMetadata:
    """Tests for RoutingMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating routing metadata."""
        metadata = RoutingMetadata(
            request_id="abc123",
            model_selected="gpt-4o-mini",
            routing_strategy="direct",
            routing_reason="cheapest model meeting quality threshold",
            routing_latency_ms=1.5,
            estimated_cost_usd=0.001,
            counterfactual_cost_usd=0.01,
            cost_savings_usd=0.009,
            models_considered=["gpt-4o", "gpt-4o-mini"],
        )

        assert metadata.model_selected == "gpt-4o-mini"
        assert metadata.routing_strategy == "direct"
        assert metadata.cost_savings_usd == 0.009

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = RoutingMetadata(
            request_id="def456",
            model_selected="gpt-4o-mini",
            routing_strategy="direct",
            routing_reason="test reason",
            routing_latency_ms=1.5,
            estimated_cost_usd=0.001,
            counterfactual_cost_usd=0.01,
            cost_savings_usd=0.009,
            models_considered=["gpt-4o", "gpt-4o-mini"],
        )

        d = metadata.to_dict()
        assert isinstance(d, dict)
        assert d["model_selected"] == "gpt-4o-mini"
        assert d["models_considered"] == ["gpt-4o", "gpt-4o-mini"]


class TestRouteSmithClient:
    """Tests for RouteSmith client."""

    @pytest.fixture
    def client(self):
        """Create a RouteSmith client with test models."""
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
        return rs

    def test_client_initialization(self, client):
        """Test client initializes with correct defaults."""
        assert len(client.registry) == 2
        assert client._request_count == 0
        assert client._total_cost == 0.0
        assert client._counterfactual_cost == 0.0

    def test_client_with_custom_config(self):
        """Test client with custom configuration."""
        config = RouteSmithConfig(
            default_strategy=RoutingStrategy.CASCADE,
        ).with_budget(quality_threshold=0.9)

        rs = RouteSmith(config=config)
        assert rs.config.default_strategy == RoutingStrategy.CASCADE
        assert rs.config.budget.quality_threshold == 0.9

    def test_register_model(self, client):
        """Test registering additional models."""
        client.register_model(
            "claude-3-sonnet",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            quality_score=0.90,
        )
        assert len(client.registry) == 3
        assert "claude-3-sonnet" in client.registry


class TestCostTracking:
    """Tests for cost tracking and counterfactual calculation."""

    @pytest.fixture
    def client(self):
        rs = RouteSmith()
        rs.register_model(
            "expensive",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            quality_score=0.95,
        )
        rs.register_model(
            "cheap",
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0003,
            quality_score=0.70,
        )
        return rs

    def test_stats_initial_values(self, client):
        """Test initial stats values."""
        stats = client.stats
        assert stats["request_count"] == 0
        assert stats["total_cost_usd"] == 0.0
        assert stats["estimated_without_routing"] == 0.0
        assert stats["cost_savings_usd"] == 0.0
        assert stats["savings_percent"] == 0.0

    def test_reset_stats(self, client):
        """Test resetting stats."""
        # Simulate some activity
        client._request_count = 10
        client._total_cost = 1.5
        client._counterfactual_cost = 5.0
        client._last_routing_metadata = RoutingMetadata(
            request_id="test123",
            model_selected="test",
            routing_strategy="direct",
            routing_reason="test",
            routing_latency_ms=1.0,
            estimated_cost_usd=0.01,
            counterfactual_cost_usd=0.05,
            cost_savings_usd=0.04,
            models_considered=["test"],
        )

        client.reset_stats()

        assert client._request_count == 0
        assert client._total_cost == 0.0
        assert client._counterfactual_cost == 0.0
        assert client._last_routing_metadata is None

    @patch("routesmith.client.litellm")
    def test_completion_tracks_costs(self, mock_litellm, client):
        """Test that completion tracks actual and counterfactual costs."""
        # Mock response with usage
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_litellm.completion.return_value = mock_response

        client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            min_quality=0.6,  # Should use cheap model
        )

        # Check that costs were tracked
        assert client._request_count == 1
        assert client._total_cost > 0
        assert client._counterfactual_cost >= client._total_cost

    @patch("routesmith.client.litellm")
    def test_stats_shows_savings(self, mock_litellm, client):
        """Test that stats shows cost savings."""
        # Mock response
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 500
        mock_litellm.completion.return_value = mock_response

        # Force cheap model selection
        client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="cheap",  # Explicitly use cheap model
        )

        stats = client.stats

        # Cheap model costs: (1000/1000 * 0.0001) + (500/1000 * 0.0003) = 0.00025
        # Expensive model costs: (1000/1000 * 0.01) + (500/1000 * 0.03) = 0.025
        # Savings should be: 0.025 - 0.00025 = 0.02475

        assert stats["cost_savings_usd"] > 0
        assert stats["savings_percent"] > 0

    @patch("routesmith.client.litellm")
    def test_last_routing_metadata_property(self, mock_litellm, client):
        """Test last_routing_metadata property."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_litellm.completion.return_value = mock_response

        assert client.last_routing_metadata is None

        client.completion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert client.last_routing_metadata is not None
        assert isinstance(client.last_routing_metadata, RoutingMetadata)


class TestResponseMetadata:
    """Tests for response metadata attachment."""

    @pytest.fixture
    def client(self):
        rs = RouteSmith()
        rs.register_model("model", cost_per_1k_input=0.001, cost_per_1k_output=0.002, quality_score=0.85)
        return rs

    @patch("routesmith.client.litellm")
    def test_metadata_not_attached_by_default(self, mock_litellm, client):
        """Test metadata is not attached when include_metadata=False."""
        # Use a simple class instead of MagicMock to test attribute non-existence
        class MockResponse:
            class Usage:
                prompt_tokens = 100
                completion_tokens = 50
            usage = Usage()

        mock_response = MockResponse()
        mock_litellm.completion.return_value = mock_response

        response = client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            include_metadata=False,
        )

        # With include_metadata=False, the attribute should not be set
        assert not hasattr(response, "routesmith_metadata")

    @patch("routesmith.client.litellm")
    def test_metadata_attached_when_requested(self, mock_litellm, client):
        """Test metadata is attached when include_metadata=True."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_litellm.completion.return_value = mock_response

        response = client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            include_metadata=True,
        )

        assert hasattr(response, "routesmith_metadata")
        assert isinstance(response.routesmith_metadata, dict)
        assert "model_selected" in response.routesmith_metadata
        assert "routing_reason" in response.routesmith_metadata
        assert "estimated_cost_usd" in response.routesmith_metadata
        assert "cost_savings_usd" in response.routesmith_metadata

    @patch("routesmith.client.litellm")
    def test_metadata_contains_all_fields(self, mock_litellm, client):
        """Test metadata contains all expected fields."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_litellm.completion.return_value = mock_response

        response = client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            include_metadata=True,
        )

        metadata = response.routesmith_metadata
        expected_fields = [
            "model_selected",
            "routing_strategy",
            "routing_reason",
            "routing_latency_ms",
            "estimated_cost_usd",
            "counterfactual_cost_usd",
            "cost_savings_usd",
            "models_considered",
        ]
        for field in expected_fields:
            assert field in metadata, f"Missing field: {field}"

    @patch("routesmith.client.litellm")
    def test_explicit_model_shows_in_metadata(self, mock_litellm, client):
        """Test that using explicit model is reflected in metadata."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_litellm.completion.return_value = mock_response

        response = client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="model",
            include_metadata=True,
        )

        assert response.routesmith_metadata["model_selected"] == "model"
        assert "explicit model specified" in response.routesmith_metadata["routing_reason"]


class TestRoutingReasons:
    """Tests for routing reason generation."""

    @pytest.fixture
    def client(self):
        rs = RouteSmith()
        rs.register_model("model", cost_per_1k_input=0.001, cost_per_1k_output=0.002, quality_score=0.85)
        return rs

    @patch("routesmith.client.litellm")
    def test_direct_routing_reason_with_cost_constraint(self, mock_litellm, client):
        """Test routing reason for direct strategy with cost constraint."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_litellm.completion.return_value = mock_response

        response = client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            strategy=RoutingStrategy.DIRECT,
            max_cost=0.01,
            include_metadata=True,
        )

        assert "under $0.01/1k" in response.routesmith_metadata["routing_reason"]

    @patch("routesmith.client.litellm")
    def test_cascade_routing_reason(self, mock_litellm, client):
        """Test routing reason for cascade strategy."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_litellm.completion.return_value = mock_response

        response = client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            strategy=RoutingStrategy.CASCADE,
            include_metadata=True,
        )

        assert "cascade" in response.routesmith_metadata["routing_reason"]


class TestAsyncCompletion:
    """Tests for async completion."""

    @pytest.fixture
    def client(self):
        rs = RouteSmith()
        rs.register_model("model", cost_per_1k_input=0.001, cost_per_1k_output=0.002, quality_score=0.85)
        return rs

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_async_completion_tracks_costs(self, mock_litellm, client):
        """Test async completion tracks costs."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        # Use AsyncMock for async function
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        await client.acompletion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert client._request_count == 1
        assert client._total_cost > 0

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_async_completion_with_metadata(self, mock_litellm, client):
        """Test async completion with metadata."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        # Use AsyncMock for async function
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        response = await client.acompletion(
            messages=[{"role": "user", "content": "Hello"}],
            include_metadata=True,
        )

        assert hasattr(response, "routesmith_metadata")
