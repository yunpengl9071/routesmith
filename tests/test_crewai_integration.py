"""Tests for CrewAI integration (routesmith_crewai_llm and routesmith_crewai_chat_model)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from routesmith import RouteSmith, RouteSmithConfig
from routesmith.integrations.crewai import routesmith_crewai_chat_model, routesmith_crewai_llm


# ---------------------------------------------------------------------------
# routesmith_crewai_llm (proxy-based)
# ---------------------------------------------------------------------------


class TestRoutesmithCrewaiLlm:
    def test_raises_import_error_when_crewai_missing(self):
        with patch.dict(sys.modules, {"crewai": None}):
            from routesmith.integrations import crewai as rs_crewai
            import importlib
            importlib.reload(rs_crewai)

            with pytest.raises(ImportError, match="crewai"):
                rs_crewai.routesmith_crewai_llm()

    def test_calls_crewai_llm_with_correct_args(self):
        mock_crewai = MagicMock()
        with patch.dict(sys.modules, {"crewai": mock_crewai}):
            from routesmith.integrations.crewai import routesmith_crewai_llm
            routesmith_crewai_llm(host="http://localhost:9119", model="auto")

        mock_crewai.LLM.assert_called_once_with(
            model="openai/auto",
            base_url="http://localhost:9119/v1",
            api_key="dummy",
        )

    def test_strips_trailing_slash_from_host(self):
        mock_crewai = MagicMock()
        with patch.dict(sys.modules, {"crewai": mock_crewai}):
            from routesmith.integrations.crewai import routesmith_crewai_llm
            routesmith_crewai_llm(host="http://localhost:9119/")

        call_kwargs = mock_crewai.LLM.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:9119/v1"

    def test_custom_host_and_model(self):
        mock_crewai = MagicMock()
        with patch.dict(sys.modules, {"crewai": mock_crewai}):
            from routesmith.integrations.crewai import routesmith_crewai_llm
            routesmith_crewai_llm(host="http://prod:9119", model="gpt-4o")

        call_kwargs = mock_crewai.LLM.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-4o"
        assert call_kwargs["base_url"] == "http://prod:9119/v1"

    def test_extra_kwargs_forwarded(self):
        mock_crewai = MagicMock()
        with patch.dict(sys.modules, {"crewai": mock_crewai}):
            from routesmith.integrations.crewai import routesmith_crewai_llm
            routesmith_crewai_llm(temperature=0.3, max_tokens=512)

        call_kwargs = mock_crewai.LLM.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 512

    def test_returns_crewai_llm_instance(self):
        mock_crewai = MagicMock()
        mock_llm_instance = MagicMock()
        mock_crewai.LLM.return_value = mock_llm_instance

        with patch.dict(sys.modules, {"crewai": mock_crewai}):
            from routesmith.integrations.crewai import routesmith_crewai_llm
            result = routesmith_crewai_llm()

        assert result is mock_llm_instance

    def test_default_api_key_is_dummy(self):
        mock_crewai = MagicMock()
        with patch.dict(sys.modules, {"crewai": mock_crewai}):
            from routesmith.integrations.crewai import routesmith_crewai_llm
            routesmith_crewai_llm()

        call_kwargs = mock_crewai.LLM.call_args[1]
        assert call_kwargs["api_key"] == "dummy"


# ---------------------------------------------------------------------------
# routesmith_crewai_chat_model (native, no proxy)
# ---------------------------------------------------------------------------


class TestRoutesmithCrewaiChatModel:
    def test_returns_chat_routesmith(self):
        from routesmith.integrations.langchain import ChatRouteSmith
        result = routesmith_crewai_chat_model()
        assert isinstance(result, ChatRouteSmith)

    def test_accepts_existing_routesmith(self):
        rs = RouteSmith()
        result = routesmith_crewai_chat_model(routesmith=rs)
        assert result.routesmith is rs

    def test_creates_fresh_routesmith_when_none(self):
        result = routesmith_crewai_chat_model()
        assert isinstance(result.routesmith, RouteSmith)

    def test_accepts_config(self):
        config = RouteSmithConfig()
        result = routesmith_crewai_chat_model(config=config)
        assert isinstance(result.routesmith, RouteSmith)

    def test_two_calls_return_independent_instances(self):
        a = routesmith_crewai_chat_model()
        b = routesmith_crewai_chat_model()
        assert a is not b
        assert a.routesmith is not b.routesmith

    def test_extra_kwargs_forwarded_to_chat_routesmith(self):
        result = routesmith_crewai_chat_model(strategy="direct")
        assert result.strategy == "direct"

    def test_registered_models_accessible(self):
        result = routesmith_crewai_chat_model()
        result.routesmith.register_model(
            "openai/gpt-4o-mini",
            cost_per_1k_input=0.15,
            cost_per_1k_output=0.60,
            quality_score=0.85,
        )
        assert len(result.routesmith.registry) == 1

    def test_chat_model_has_invoke_method(self):
        """Verify the returned object satisfies LangChain BaseChatModel interface."""
        result = routesmith_crewai_chat_model()
        assert callable(getattr(result, "invoke", None))
