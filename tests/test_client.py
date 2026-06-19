"""Tests for RouteSmith client, cost tracking, and metadata."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routesmith import RouteSmith, RouteSmithConfig, RoutingMetadata, RoutingStrategy


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


class TestCapabilityAutoDetection:
    """Tests for automatic capability detection in the client."""

    def test_detect_tool_calling_from_tools_kwarg(self):
        """Client detects tool_calling requirement from tools kwarg."""
        caps = RouteSmith._detect_required_capabilities(
            messages=[{"role": "user", "content": "Hello"}],
            kwargs={"tools": [{"type": "function", "function": {"name": "test"}}]},
        )
        assert "tool_calling" in caps

    def test_detect_tool_calling_from_functions_kwarg(self):
        """Client detects tool_calling requirement from functions kwarg."""
        caps = RouteSmith._detect_required_capabilities(
            messages=[{"role": "user", "content": "Hello"}],
            kwargs={"functions": [{"name": "test"}]},
        )
        assert "tool_calling" in caps

    def test_detect_vision_from_image_content(self):
        """Client detects vision requirement from image_url in messages."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        caps = RouteSmith._detect_required_capabilities(messages, kwargs={})
        assert "vision" in caps

    def test_no_capabilities_for_simple_text(self):
        """No capabilities detected for plain text messages."""
        caps = RouteSmith._detect_required_capabilities(
            messages=[{"role": "user", "content": "Hello"}],
            kwargs={},
        )
        assert caps == set()

    def test_has_image_content_true(self):
        """_has_image_content returns True for image messages."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
        assert RouteSmith._has_image_content(msg) is True

    def test_has_image_content_false_for_text(self):
        """_has_image_content returns False for text-only messages."""
        msg = {"role": "user", "content": "Hello"}
        assert RouteSmith._has_image_content(msg) is False

    @patch("routesmith.client.litellm")
    def test_completion_routes_tool_calls_to_capable_model(self, mock_litellm):
        """Client routes tool-calling requests only to capable models."""
        rs = RouteSmith()
        rs.register_model(
            "capable", cost_per_1k_input=0.01, cost_per_1k_output=0.03,
            quality_score=0.90, supports_function_calling=True,
        )
        rs.register_model(
            "incapable", cost_per_1k_input=0.0001, cost_per_1k_output=0.0003,
            quality_score=0.70, supports_function_calling=False,
            supports_json_mode=False,
        )

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_litellm.completion.return_value = mock_response

        rs.completion(
            messages=[{"role": "user", "content": "Use a tool"}],
            tools=[{"type": "function", "function": {"name": "test", "parameters": {}}}],
        )

        # Should have called litellm with the capable model
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["model"] == "capable"

    @patch("routesmith.client.litellm")
    def test_completion_error_records_feedback(self, mock_litellm):
        """Client records failure feedback when litellm raises."""
        rs = RouteSmith()
        rs.register_model("model", cost_per_1k_input=0.001, cost_per_1k_output=0.002)

        mock_litellm.completion.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            rs.completion(messages=[{"role": "user", "content": "Hello"}])


def _mock_litellm_response():
    return MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="hi", tool_calls=None),
            finish_reason="stop",
        )],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        id="resp_1",
        model="gpt-4o-mini",
    )


def _make_rs():
    rs = RouteSmith()
    rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                     cost_per_1k_output=0.015, quality_score=0.9)
    rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                     cost_per_1k_output=0.0006, quality_score=0.7)
    return rs


class TestCompletionWithContext:
    @patch("litellm.completion")
    def test_accepts_context_param(self, mock_litellm):
        mock_litellm.return_value = _mock_litellm_response()
        rs = _make_rs()
        from routesmith.config import RouteContext
        ctx = RouteContext(agent_role="research", turn_index=2)
        response = rs.completion(
            messages=[{"role": "user", "content": "hello"}],
            context=ctx,
        )
        assert response is not None

    @patch("litellm.completion")
    def test_without_context_still_works(self, mock_litellm):
        mock_litellm.return_value = _mock_litellm_response()
        rs = _make_rs()
        response = rs.completion(messages=[{"role": "user", "content": "hello"}])
        assert response is not None

    @patch("litellm.completion")
    def test_agent_role_inferred_when_missing(self, mock_litellm):
        mock_litellm.return_value = _mock_litellm_response()
        rs = _make_rs()
        from routesmith.config import RouteContext
        msgs = [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "hello"},
        ]
        ctx = RouteContext()  # no agent_role
        rs.completion(messages=msgs, context=ctx)  # should not raise


class TestModelLifecycle:
    def test_deregister_removes_from_registry(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                         cost_per_1k_output=0.0006, quality_score=0.7)
        rs.deregister_model("gpt-4o-mini")
        ids = [m.model_id for m in rs.registry.list_models()]
        assert "gpt-4o-mini" not in ids

    def test_deregister_last_model_raises(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        with pytest.raises(ValueError, match="last registered model"):
            rs.deregister_model("gpt-4o")

    def test_deregister_nonexistent_is_noop(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.deregister_model("nonexistent")
        assert len(rs.registry.list_models()) == 1

    def test_register_model_adds_predictor_arm(self):
        rs = RouteSmith(config=RouteSmithConfig(predictor_type="lints"))
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.register_model("claude-haiku", cost_per_1k_input=0.00025,
                         cost_per_1k_output=0.00125, quality_score=0.75)
        predictor = rs.router.predictor
        assert "claude-haiku" in predictor._arm_index


class TestRecommendModelForAgent:
    def test_returns_none_for_none_role(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        assert rs.recommend_model_for_agent(None) is None

    def test_returns_none_when_insufficient_samples(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        assert rs.recommend_model_for_agent("research") is None

    def test_returns_recommendation_with_enough_samples(self):
        rs = RouteSmith(config=RouteSmithConfig(feedback_storage_path=":memory:"))
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                         cost_per_1k_output=0.0006, quality_score=0.7)
        storage = rs.feedback._storage
        for i in range(55):
            storage.store_record(
                request_id=f"req_{i}",
                model_id="gpt-4o",
                messages=[{"role": "user", "content": "research question"}],
                latency_ms=200.0,
                quality_score=0.9,
                agent_role="research",
            )
        result = rs.recommend_model_for_agent("research", min_samples=50)
        assert result is not None
        assert result["model"] == "gpt-4o"
        assert result["sample_count"] >= 50
        assert "confidence" in result
        assert "new_models_to_explore" in result

    def test_new_models_to_explore_includes_models_with_no_data(self):
        rs = RouteSmith(config=RouteSmithConfig(feedback_storage_path=":memory:"))
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                         cost_per_1k_output=0.0006, quality_score=0.7)
        storage = rs.feedback._storage
        for i in range(55):
            storage.store_record(
                request_id=f"req_{i}",
                model_id="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                latency_ms=100.0,
                quality_score=0.9,
                agent_role="research",
            )
        result = rs.recommend_model_for_agent("research", min_samples=50)
        assert "gpt-4o-mini" in result["new_models_to_explore"]

    def test_returns_none_when_all_records_are_unregistered_models(self):
        rs = RouteSmith(config=RouteSmithConfig(feedback_storage_path=":memory:"))
        rs.register_model(
            "gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.9
        )
        storage = rs.feedback._storage
        for i in range(55):
            storage.store_record(
                request_id=f"req_ghost_{i}",
                model_id="ghost-model",
                messages=[{"role": "user", "content": "hi"}],
                latency_ms=100.0,
                quality_score=0.9,
                agent_role="research",
            )
        assert rs.recommend_model_for_agent("research", min_samples=50) is None


class TestRegisterRewardFn:
    def test_adds_to_config(self):
        rs = RouteSmith()
        def _fn(r, m):
            return 0.95
        rs.register_reward_fn("research", _fn)
        assert rs.config.reward_fns["research"] is _fn

    def test_overrides_existing(self):
        def _fn1(r, m):
            return 0.8
        def _fn2(r, m):
            return 0.95
        rs = RouteSmith(config=RouteSmithConfig(reward_fns={"research": _fn1}))
        rs.register_reward_fn("research", _fn2)
        assert rs.config.reward_fns["research"] is _fn2


class TestWithAuto:
    def test_with_auto_returns_client(self):
        """with_auto() returns a RouteSmith with models registered."""
        rs = RouteSmith.with_auto()
        assert len(rs.registry) > 0
        assert rs.config.cache.enabled is False  # no cache by default

    def test_with_auto_registers_known_model(self):
        """with_auto() registers gpt-4o-mini."""
        rs = RouteSmith.with_auto()
        model = rs.registry.get("openai/gpt-4o-mini")
        assert model is not None
        assert model.cost_per_1k_input == pytest.approx(0.00015)

    def test_with_auto_accepts_tradeoff(self):
        """with_auto() accepts a default tradeoff."""
        rs = RouteSmith.with_auto(tradeoff=3)
        assert rs.config._auto_tradeoff == 3

    def test_with_auto_accepts_provider_filter(self):
        """with_auto() filters by provider."""
        rs = RouteSmith.with_auto(providers=["anthropic"])
        for m in rs.registry.list_models():
            assert m.model_id.startswith("anthropic/")

    def test_with_auto_accepts_cache_config(self):
        """with_auto() can enable caching."""
        rs = RouteSmith.with_auto(cache=True)
        assert rs._cache is not None

    def test_full_with_auto_flow(self):
        """Integration: with_auto() + completion + tradeoff works end-to-end."""
        from unittest.mock import MagicMock, patch

        rs = RouteSmith.with_auto(tradeoff=7)

        assert len(rs.registry) >= 3  # at least a few models registered

        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="response"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            resp = rs.completion(
                messages=[{"role": "user", "content": "Hello world"}],
                tradeoff=5,
                include_metadata=True,
            )

        assert resp.routesmith_metadata["model_selected"] is not None
        assert resp.routesmith_metadata["cache_hit"] is False
        assert "model_selected" in resp.routesmith_metadata


class TestConversationScopedRouting:
    def test_same_conversation_reuses_model(self):
        """Same conversation_id reuses the first-turn model."""
        from unittest.mock import MagicMock, patch

        from routesmith.config import RouteContext

        rs = RouteSmith()
        rs.register_model("gpt-4o", 0.005, 0.015, quality_score=0.95)
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006, quality_score=0.85)

        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )

            # First turn: picks a model
            ctx1 = RouteContext(conversation_id="conv-001", turn_index=1)
            resp1 = rs.completion(
                messages=[{"role": "user", "content": "Hello"}],
                context=ctx1,
                include_metadata=True,
            )
            first_model = resp1.routesmith_metadata["model_selected"]

            # Second turn: should reuse the same model
            ctx2 = RouteContext(conversation_id="conv-001", turn_index=2)
            resp2 = rs.completion(
                messages=[{"role": "user", "content": "Continue"}],
                context=ctx2,
                include_metadata=True,
            )

            assert resp2.routesmith_metadata["model_selected"] == first_model
            assert "stickiness" in resp2.routesmith_metadata.get("routing_reason", "")

    def test_different_conversations_explore_models(self):
        """Different conversation_ids get fresh exploration."""
        from unittest.mock import MagicMock, patch

        from routesmith.config import RouteContext

        rs = RouteSmith()
        rs.register_model("gpt-4o", 0.005, 0.015, quality_score=0.95)
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006, quality_score=0.85)

        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )

            # Conversation A
            ctx_a = RouteContext(conversation_id="conv-a", turn_index=1)
            resp_a = rs.completion(
                messages=[{"role": "user", "content": "Hi"}],
                context=ctx_a,
                include_metadata=True,
            )

            # Conversation B — fresh, should explore
            ctx_b = RouteContext(conversation_id="conv-b", turn_index=1)
            resp_b = rs.completion(
                messages=[{"role": "user", "content": "Hi"}],
                context=ctx_b,
                include_metadata=True,
            )

            # Both should get a model (not crash)
            assert resp_a.routesmith_metadata["model_selected"] is not None
            assert resp_b.routesmith_metadata["model_selected"] is not None
            # Models might differ (fresh exploration), or be same (no strong signal yet)
            # Either is acceptable


class TestTradeoff:
    def _make_client(self):
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

    def test_tradeoff_passed_to_route(self):
        """tradeoff parameter is accepted and affects routing."""
        from unittest.mock import MagicMock, patch

        client = self._make_client()
        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            resp = client.completion(
                messages=[{"role": "user", "content": "hello"}],
                tradeoff=3,
                include_metadata=True,
            )
            # tradeoff=3 should still pick a model (not crash)
            assert resp.routesmith_metadata["model_selected"] is not None

    def test_tradeoff_0_prefers_quality(self):
        """tradeoff=0 prefers highest quality model regardless of cost."""
        from unittest.mock import MagicMock, patch

        # Register two models with different quality
        client = RouteSmith()
        client.register_model("cheap-model", 0.0001, 0.0001, quality_score=0.70)
        client.register_model("expensive-model", 0.01, 0.01, quality_score=0.95)

        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            resp = client.completion(
                messages=[{"role": "user", "content": "hello"}],
                tradeoff=0,
                include_metadata=True,
            )
            assert resp.routesmith_metadata["model_selected"] == "expensive-model"

    def test_tradeoff_10_prefers_cost(self):
        """tradeoff=10 prefers cheapest model above min quality."""
        from unittest.mock import MagicMock, patch

        client = self._make_client()
        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            resp = client.completion(
                messages=[{"role": "user", "content": "hello"}],
                tradeoff=10,
                include_metadata=True,
            )
            assert resp.routesmith_metadata["model_selected"] == "gpt-4o-mini"


class TestPollInjection:
    def test_poll_attached_to_response(self):
        """Poll metadata is attached to response when sampled."""
        from unittest.mock import MagicMock, patch
        from routesmith.config import RouteSmithConfig

        config = RouteSmithConfig(poll_sample_rate=1.0)
        rs = RouteSmith(config=config)
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006)

        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            resp = rs.completion(
                messages=[{"role": "user", "content": "hi"}],
            )

        assert hasattr(resp, "routesmith_poll")
        poll = resp.routesmith_poll
        assert poll["type"] == "numbered"
        assert len(poll["options"]) == 5

    def test_poll_not_attached_when_rate_zero(self):
        """Poll is not attached when sample rate is 0."""
        from unittest.mock import MagicMock, patch
        from routesmith.config import RouteSmithConfig

        config = RouteSmithConfig(poll_sample_rate=0.0)
        rs = RouteSmith(config=config)
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006)

        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            resp = rs.completion(
                messages=[{"role": "user", "content": "hi"}],
            )

        assert not isinstance(getattr(resp, "routesmith_poll", None), dict)

    def test_on_poll_callback_invoked(self):
        """on_poll callback is invoked when poll is sampled."""
        from unittest.mock import MagicMock, patch
        from routesmith.config import RouteSmithConfig

        callback = MagicMock()
        config = RouteSmithConfig(poll_sample_rate=1.0, on_poll=callback)
        rs = RouteSmith(config=config)
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006)

        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            rs.completion(messages=[{"role": "user", "content": "hi"}])

        callback.assert_called_once()
        poll_dict = callback.call_args[0][0]
        assert poll_dict["type"] == "numbered"


class TestRecommendations:
    def test_recommendations_returns_dict(self):
        """recommendations() returns a dict with per-agent data."""
        rs = RouteSmith()
        recs = rs.recommendations()

        assert isinstance(recs, dict)
        assert "warnings" in recs
        assert "new_models_to_try" in recs
        assert "forecast" in recs

    def test_recommendations_forecast(self):
        """recommendations includes budget forecast."""
        rs = RouteSmith()
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006)
        rs._total_cost = 45.20
        rs._request_count = 100

        recs = rs.recommendations()
        forecast = recs["forecast"]

        assert "monthly_cost_current" in forecast
        assert forecast["monthly_cost_current"] == 45.20

    def test_recommendations_warnings(self):
        """recommendations includes anomaly warnings."""
        rs = RouteSmith()
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006)

        recs = rs.recommendations()
        warnings = recs["warnings"]

        assert isinstance(warnings, list)
        # With no data, warnings should be empty or informational

    def test_recommendations_new_models_to_try(self):
        """recommendations suggests new models to try."""
        rs = RouteSmith()
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006)

        recs = rs.recommendations()
        new_models = recs["new_models_to_try"]

        assert isinstance(new_models, list)


class TestAnswerPoll:
    def test_answer_poll_unknown_id(self):
        """answer_poll returns False for unknown poll ID."""
        rs = RouteSmith()
        result = rs.answer_poll("nonexistent", option=1)
        assert result is False

    def test_answer_poll_maps_option_to_signal(self):
        """answer_poll maps option to quality signal and feeds predictor."""
        from unittest.mock import MagicMock, patch
        from routesmith.config import RouteSmithConfig

        config = RouteSmithConfig(poll_sample_rate=1.0)
        rs = RouteSmith(config=config)
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006)

        request_id = None
        with patch("litellm.completion") as mock:
            mock.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            resp = rs.completion(
                messages=[{"role": "user", "content": "hi"}],
            )
            request_id = resp._routesmith_request_id

        # answer_poll should update predictor with quality signal
        result = rs.answer_poll(request_id, option=2)
        assert result is True
