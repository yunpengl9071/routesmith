"""Tests for semantic cache with model-aware lookups."""

import time
from unittest.mock import MagicMock, patch

import pytest

from routesmith.cache.semantic import SemanticCache
from routesmith.client import RouteSmith
from routesmith.config import RouteSmithConfig


class TestSemanticCacheModelAware:
    """Tests for model-aware caching behavior."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache instance."""
        return SemanticCache(similarity_threshold=0.95, ttl_seconds=3600)

    def test_put_stores_model_id(self, cache):
        """put() stores model_id in cache entry."""
        messages = [{"role": "user", "content": "Hello"}]
        response = "Hello there!"
        model_id = "gpt-4o"

        entry = cache.put(messages, response, model_id=model_id)
        assert entry.model_id == "gpt-4o"
        # Verify stored in cache (key is composite: hash|model_id)
        composite_key = cache._composite_key(messages, model_id)
        entry2 = cache._exact_cache.get(composite_key)
        assert entry2 is not None
        assert entry2.model_id == "gpt-4o"

    def test_get_exact_match_by_hash_and_model(self, cache):
        """get() with model_id returns exact match when both hash and model match."""
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "GPT response", model_id="gpt-4o")
        cache.put(messages, "Claude response", model_id="claude-sonnet")

        # Should get the right model's response
        gpt_entry = cache.get(messages, model_id="gpt-4o")
        assert gpt_entry is not None
        assert gpt_entry.response == "GPT response"
        assert gpt_entry.model_id == "gpt-4o"

        claude_entry = cache.get(messages, model_id="claude-sonnet")
        assert claude_entry is not None
        assert claude_entry.response == "Claude response"
        assert claude_entry.model_id == "claude-sonnet"

    def test_get_exact_match_model_mismatch_returns_none(self, cache):
        """get() with model_id returns None when hash matches but model differs."""
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "GPT response", model_id="gpt-4o")

        # Request with different model should MISS
        entry = cache.get(messages, model_id="claude-sonnet")
        assert entry is None

    def test_get_without_model_id_still_works_backward_compat(self, cache):
        """get() without model_id works as before (model-unaware)."""
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "response", model_id="gpt-4o")

        # Without model_id, should still find the entry
        entry = cache.get(messages)
        assert entry is not None
        assert entry.response == "response"

    def test_get_model_unaware_returns_first_match(self, cache):
        """get() without model_id returns any model's cached response."""
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "claude response", model_id="claude-sonnet")

        # Without model_id restriction, any model's cache is returned
        entry = cache.get(messages)
        assert entry is not None
        assert entry.model_id == "claude-sonnet"

    def test_get_model_aware_prevents_feedback_poisoning(self, cache):
        """Model-aware get() prevents returning wrong model's response,
        which would poison predictor feedback. See cache_stress_test.md §2B."""
        messages = [{"role": "user", "content": "Write a Python sort function"}]

        # T=1: gpt-4o-mini handles this query
        cache.put(messages, "GPT mini sort implementation", model_id="gpt-4o-mini")

        # T=2: router picks deepseek-chat for same query (exploration)
        # With model-aware, should MISS — forcing a real LLM call
        entry = cache.get(messages, model_id="deepseek-chat")
        assert entry is None, (
            "BUG: returned gpt-4o-mini's response for deepseek-chat request. "
            "This would poison predictor feedback — deepseek-chat's arm would "
            "get credit for gpt-4o-mini's quality."
        )

    def test_get_returns_none_when_cache_empty(self, cache):
        """get() returns None when cache is empty."""
        messages = [{"role": "user", "content": "Hello"}]
        entry = cache.get(messages, model_id="gpt-4o")
        assert entry is None

    def test_get_returns_none_when_timestamp_expired(self, cache):
        """get() returns None when entry TTL has expired."""
        messages = [{"role": "user", "content": "Hello"}]
        entry = cache.put(messages, "response", model_id="gpt-4o")

        # Artificially expire the entry
        entry.created_at = time.time() - 7200  # 2 hours ago
        assert cache.get(messages, model_id="gpt-4o") is None

    def test_get_increments_hit_count(self, cache):
        """get() increments hit_count on cache hits."""
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "response", model_id="gpt-4o")

        assert cache.get(messages, model_id="gpt-4o") is not None
        assert cache.get(messages, model_id="gpt-4o") is not None

        # Check stats show hits
        stats = cache.stats
        assert stats["total_hits"] >= 2

    def test_invalidate_removes_entry(self, cache):
        """invalidate() removes a cache entry."""
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "response", model_id="gpt-4o")

        assert cache.invalidate(messages) is True
        assert cache.get(messages, model_id="gpt-4o") is None

    def test_clear_removes_all_entries(self, cache):
        """clear() removes all cache entries."""
        cache.put(
            [{"role": "user", "content": "Q1"}], "A1", model_id="gpt-4o"
        )
        cache.put(
            [{"role": "user", "content": "Q2"}], "A2", model_id="claude"
        )

        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0

    def test_eviction_removes_oldest_when_over_capacity(self, cache):
        """Cache evicts oldest entries when over max_entries capacity."""
        cache.max_entries = 3

        cache.put(
            [{"role": "user", "content": "A"}], "resp", model_id="m1"
        )
        time.sleep(0.001)  # ensure different timestamps
        cache.put(
            [{"role": "user", "content": "B"}], "resp", model_id="m1"
        )
        time.sleep(0.001)
        cache.put(
            [{"role": "user", "content": "C"}], "resp", model_id="m1"
        )
        time.sleep(0.001)
        # This should evict "A" (oldest)
        cache.put(
            [{"role": "user", "content": "D"}], "resp", model_id="m1"
        )

        assert len(cache) <= 3
        assert cache.get(
            [{"role": "user", "content": "A"}], model_id="m1"
        ) is None  # evicted
        assert cache.get(
            [{"role": "user", "content": "D"}], model_id="m1"
        ) is not None  # kept

    def test_stats_reflects_cache_state(self, cache):
        """stats returns accurate cache state."""
        assert cache.stats["exact_entries"] == 0
        assert cache.stats["total_hits"] == 0

        cache.put(
            [{"role": "user", "content": "Q"}], "A", model_id="gpt-4o"
        )
        assert cache.stats["exact_entries"] == 1

        cache.get([{"role": "user", "content": "Q"}], model_id="gpt-4o")
        assert cache.stats["total_hits"] == 1


class TestSemanticCacheIntegration:
    """Integration-level tests for semantic cache in the client flow."""

    def test_route_then_cache_check_flow(self):
        """The cache check must happen AFTER routing to know model_id.

        This is the chicken-and-egg resolution: route first (cheap, <5ms),
        then check cache. The cache check is a simple dict lookup, even faster.
        """
        cache = SemanticCache()

        # Step 1: First request — executes LLM, caches result
        messages = [{"role": "user", "content": "What is a cache?"}]
        model_id = "gpt-4o"  # determined by router

        # Should be cache miss (first time)
        assert cache.get(messages, model_id=model_id) is None

        # Store after LLM call
        cache.put(messages, "Caching is...", model_id=model_id)

        # Step 2: Second request — cache hit
        entry = cache.get(messages, model_id=model_id)
        assert entry is not None
        assert entry.response == "Caching is..."
        assert entry.model_id == "gpt-4o"

    def test_different_model_bypasses_cache(self):
        """When router picks different model, cache must miss."""
        cache = SemanticCache()
        messages = [{"role": "user", "content": "Hello world"}]

        # First time: gpt-4o-mini handles it
        cache.put(messages, "Mini response", model_id="gpt-4o-mini")

        # Second time: router picks gpt-4o (budget increase, complex query, etc.)
        # Must miss so gpt-4o generates its own (potentially better) response
        entry = cache.get(messages, model_id="gpt-4o")
        assert entry is None

    def test_same_model_repeated_hits_cache(self):
        """When router picks same model, cache should hit."""
        cache = SemanticCache()
        messages = [{"role": "user", "content": "Ping"}]

        cache.put(messages, "Pong", model_id="claude-haiku")

        # Same model, same messages → hit
        for _ in range(5):
            entry = cache.get(messages, model_id="claude-haiku")
            assert entry is not None
            assert entry.response == "Pong"

    def test_similar_messages_different_models_separate_entries(self):
        """Semantically similar messages to different models are separate cache entries.

        When model_id matches, the semantic cache returns the right model's response.
        When model_id differs, exact cache will miss (separate composite keys).
        Semantic matching may find cross-model entries if content is similar enough
        — this is expected and desirable (quality-tolerant implicit fallback).
        """
        cache = SemanticCache(similarity_threshold=0.95)

        msg1 = [{"role": "user", "content": "How do I sort a list in Python?"}]
        msg2 = [{"role": "user", "content": "What is the capital of France?"}]

        cache.put(msg1, "GPT sort response", model_id="gpt-4o")
        cache.put(msg2, "Claude geography response", model_id="claude-sonnet")

        # Same query + same model = hit
        gpt_entry = cache.get(msg1, model_id="gpt-4o")
        assert gpt_entry is not None
        assert gpt_entry.response == "GPT sort response"

        claude_entry = cache.get(msg2, model_id="claude-sonnet")
        assert claude_entry is not None
        assert claude_entry.response == "Claude geography response"

        # Different query + same model = miss (content completely different)
        assert cache.get(msg1, model_id="claude-sonnet") is None
        assert cache.get(msg2, model_id="gpt-4o") is None


class TestClientCacheIntegration:
    """Tests for cache integration in RouteSmith client."""

    @pytest.fixture
    def client_with_cache(self):
        """Create a RouteSmith client with cache enabled and test models."""
        from routesmith.config import CacheConfig
        config = RouteSmithConfig(
            cache=CacheConfig(enabled=True, similarity_threshold=0.95, ttl_seconds=3600),
            feedback_enabled=False,
        )
        rs = RouteSmith(config=config)
        rs.register_model(
            "gpt-4o-mini",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            quality_score=0.85,
        )
        rs.register_model(
            "gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            quality_score=0.95,
        )
        return rs

    def test_cache_is_instantiated_when_enabled(self, client_with_cache):
        """SemanticCache is created when config.cache.enabled=True."""
        assert client_with_cache._cache is not None

    def test_cache_is_none_when_disabled(self):
        """No cache when config.cache.enabled=False (default)."""
        config = RouteSmithConfig()
        assert config.cache.enabled is False
        rs = RouteSmith(config=config)
        assert rs._cache is None

    @patch("litellm.completion")
    def test_cache_hit_skips_llm_call(self, mock_completion, client_with_cache):
        """When cache hits, the LLM is NOT called."""
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="LLM response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        messages = [{"role": "user", "content": "Hello world"}]

        # First call: cache miss → LLM called
        client_with_cache.completion(
            messages=messages, model="gpt-4o-mini"
        )
        assert mock_completion.call_count == 1

        # Second call: cache hit → LLM NOT called
        response2 = client_with_cache.completion(
            messages=messages, model="gpt-4o-mini", include_metadata=True
        )
        assert mock_completion.call_count == 1  # still only 1 LLM call
        assert response2.routesmith_metadata["cache_hit"] is True

    @patch("litellm.completion")
    def test_different_model_bypasses_cache(self, mock_completion, client_with_cache):
        """Same messages, different model → cache miss, LLM called."""
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        messages = [{"role": "user", "content": "Test"}]

        # First call with gpt-4o-mini
        client_with_cache.completion(messages=messages, model="gpt-4o-mini")
        assert mock_completion.call_count == 1

        # Second call with gpt-4o — different model, must miss
        client_with_cache.completion(messages=messages, model="gpt-4o")
        assert mock_completion.call_count == 2  # new LLM call

    @patch("litellm.completion")
    def test_cache_hit_includes_metadata(self, mock_completion, client_with_cache):
        """Cache hits have cache_hit=True in routesmith_metadata."""
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        messages = [{"role": "user", "content": "Metadata test"}]

        # First call: cache miss
        resp1 = client_with_cache.completion(
            messages=messages, model="gpt-4o-mini", include_metadata=True
        )
        assert resp1.routesmith_metadata["cache_hit"] is False

        # Second call: cache hit
        resp2 = client_with_cache.completion(
            messages=messages, model="gpt-4o-mini", include_metadata=True
        )
        assert resp2.routesmith_metadata["cache_hit"] is True

    @patch("litellm.completion")
    def test_cache_handles_routing_flow(self, mock_completion, client_with_cache):
        """Cache works when model is selected by router (not explicit)."""
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="routed response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        messages = [{"role": "user", "content": "Route me"}]

        # Let router pick the model
        resp1 = client_with_cache.completion(messages=messages, include_metadata=True)
        model_used = resp1.routesmith_metadata["model_selected"]

        # Same messages again — should hit cache for the routed model
        resp2 = client_with_cache.completion(messages=messages, include_metadata=True)
        assert resp2.routesmith_metadata["cache_hit"] is True
        assert resp2.routesmith_metadata["model_selected"] == model_used

    @patch("litellm.completion")
    def test_cache_miss_with_different_content(self, mock_completion, client_with_cache):
        """Different messages always miss cache."""
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        client_with_cache.completion(
            messages=[{"role": "user", "content": "Question A"}],
            model="gpt-4o-mini",
        )
        assert mock_completion.call_count == 1

        client_with_cache.completion(
            messages=[{"role": "user", "content": "Question B"}],
            model="gpt-4o-mini",
        )
        assert mock_completion.call_count == 2  # different content → miss

    @patch("litellm.acompletion")
    @pytest.mark.asyncio
    async def test_async_cache_hit_skips_llm_call(
        self, mock_acompletion, client_with_cache
    ):
        """Async completion with cache hit skips LLM call."""
        mock_response = MagicMock(
            choices=[MagicMock(message=MagicMock(content="async response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        mock_acompletion.return_value = mock_response

        messages = [{"role": "user", "content": "Async test"}]

        # First: cache miss
        await client_with_cache.acompletion(
            messages=messages, model="gpt-4o-mini"
        )
        assert mock_acompletion.call_count == 1

        # Second: cache hit
        await client_with_cache.acompletion(
            messages=messages, model="gpt-4o-mini"
        )
        assert mock_acompletion.call_count == 1  # no second LLM call
