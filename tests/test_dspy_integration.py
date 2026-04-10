"""Tests for DSPy integration (routesmith_lm and RouteSmithLM)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call

import pytest

from routesmith import RouteSmith, RouteSmithConfig
from routesmith.integrations.dspy import RouteSmithLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rs_with_mock(text="Mocked answer"):
    """Return a RouteSmith with completion() mocked."""
    rs = RouteSmith()
    rs.register_model(
        "openai/gpt-4o-mini",
        cost_per_1k_input=0.15,
        cost_per_1k_output=0.60,
        quality_score=0.80,
    )
    rs.register_model(
        "openai/gpt-4o",
        cost_per_1k_input=2.50,
        cost_per_1k_output=10.0,
        quality_score=0.95,
    )
    mock_response = MagicMock()
    mock_response.choices[0].message.content = text
    rs.completion = MagicMock(return_value=mock_response)
    return rs


# ---------------------------------------------------------------------------
# routesmith_lm (proxy-based factory)
# ---------------------------------------------------------------------------


class TestRoutesmithLmFactory:
    def test_raises_import_error_when_dspy_missing(self):
        with patch.dict(sys.modules, {"dspy": None}):
            from routesmith.integrations import dspy as rs_dspy
            import importlib
            importlib.reload(rs_dspy)

            with pytest.raises(ImportError, match="dspy-ai"):
                rs_dspy.routesmith_lm()

    def test_calls_dspy_lm_with_correct_args(self):
        mock_dspy = MagicMock()
        with patch.dict(sys.modules, {"dspy": mock_dspy}):
            from routesmith.integrations.dspy import routesmith_lm
            routesmith_lm(host="http://localhost:9119", model="auto")

        mock_dspy.LM.assert_called_once_with(
            "openai/auto",
            base_url="http://localhost:9119/v1",
            api_key="dummy",
        )

    def test_strips_trailing_slash_from_host(self):
        mock_dspy = MagicMock()
        with patch.dict(sys.modules, {"dspy": mock_dspy}):
            from routesmith.integrations.dspy import routesmith_lm
            routesmith_lm(host="http://localhost:9119/", model="auto")

        call_kwargs = mock_dspy.LM.call_args
        assert call_kwargs[1]["base_url"] == "http://localhost:9119/v1"

    def test_custom_model_name(self):
        mock_dspy = MagicMock()
        with patch.dict(sys.modules, {"dspy": mock_dspy}):
            from routesmith.integrations.dspy import routesmith_lm
            routesmith_lm(host="http://myserver:8080", model="gpt-4o-mini")

        mock_dspy.LM.assert_called_once_with(
            "openai/gpt-4o-mini",
            base_url="http://myserver:8080/v1",
            api_key="dummy",
        )

    def test_extra_kwargs_forwarded(self):
        mock_dspy = MagicMock()
        with patch.dict(sys.modules, {"dspy": mock_dspy}):
            from routesmith.integrations.dspy import routesmith_lm
            routesmith_lm(temperature=0.5, max_tokens=256)

        call_kwargs = mock_dspy.LM.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 256


# ---------------------------------------------------------------------------
# RouteSmithLM — init and registration
# ---------------------------------------------------------------------------


class TestRouteSmithLMInit:
    def test_creates_routesmith_if_none(self):
        lm = RouteSmithLM()
        assert isinstance(lm._rs, RouteSmith)

    def test_accepts_existing_routesmith(self):
        rs = RouteSmith()
        lm = RouteSmithLM(routesmith=rs)
        assert lm._rs is rs

    def test_accepts_config(self):
        config = RouteSmithConfig()
        lm = RouteSmithLM(config=config)
        assert isinstance(lm._rs, RouteSmith)

    def test_history_starts_empty(self):
        lm = RouteSmithLM()
        assert lm.history == []

    def test_register_model_delegates(self):
        lm = RouteSmithLM()
        lm.register_model("openai/gpt-4o-mini", cost_per_1k_input=0.15,
                          cost_per_1k_output=0.60, quality_score=0.85)
        assert len(lm._rs.registry) == 1

    def test_stats_property(self):
        lm = RouteSmithLM()
        stats = lm.stats
        assert "request_count" in stats


# ---------------------------------------------------------------------------
# RouteSmithLM.__call__ — prompt mode
# ---------------------------------------------------------------------------


class TestRouteSmithLMCallPrompt:
    def test_prompt_string_converted_to_messages(self):
        rs = _make_rs_with_mock("4")
        lm = RouteSmithLM(routesmith=rs)

        result = lm("What is 2+2?")

        call_kwargs = rs.completion.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "What is 2+2?"}]

    def test_returns_list_of_strings(self):
        rs = _make_rs_with_mock("4")
        lm = RouteSmithLM(routesmith=rs)
        result = lm("What is 2+2?")
        assert isinstance(result, list)
        assert result == ["4"]

    def test_response_text_extracted(self):
        rs = _make_rs_with_mock("Paris")
        lm = RouteSmithLM(routesmith=rs)
        result = lm("Capital of France?")
        assert result[0] == "Paris"

    def test_history_appended_after_call(self):
        rs = _make_rs_with_mock("42")
        lm = RouteSmithLM(routesmith=rs)
        lm("What is 6×7?")
        assert len(lm.history) == 1
        assert lm.history[0]["prompt"] == "What is 6×7?"
        assert lm.history[0]["response"] == ["42"]

    def test_multiple_calls_accumulate_history(self):
        rs = _make_rs_with_mock("ok")
        lm = RouteSmithLM(routesmith=rs)
        lm("Q1")
        lm("Q2")
        lm("Q3")
        assert len(lm.history) == 3

    def test_extra_kwargs_forwarded_to_completion(self):
        rs = _make_rs_with_mock()
        lm = RouteSmithLM(routesmith=rs)
        lm("Hello", max_tokens=128, temperature=0.1)
        call_kwargs = rs.completion.call_args[1]
        assert call_kwargs["max_tokens"] == 128
        assert call_kwargs["temperature"] == 0.1

    def test_raises_if_no_prompt_and_no_messages(self):
        lm = RouteSmithLM()
        with pytest.raises(ValueError, match="prompt or messages"):
            lm()


# ---------------------------------------------------------------------------
# RouteSmithLM.__call__ — messages mode
# ---------------------------------------------------------------------------


class TestRouteSmithLMCallMessages:
    def test_messages_list_passed_directly(self):
        rs = _make_rs_with_mock("Sure")
        lm = RouteSmithLM(routesmith=rs)
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        lm(messages=msgs)
        call_kwargs = rs.completion.call_args[1]
        assert call_kwargs["messages"] == msgs

    def test_messages_takes_priority_over_prompt(self):
        rs = _make_rs_with_mock("ok")
        lm = RouteSmithLM(routesmith=rs)
        msgs = [{"role": "user", "content": "From messages"}]
        lm(prompt="From prompt", messages=msgs)
        call_kwargs = rs.completion.call_args[1]
        assert call_kwargs["messages"][0]["content"] == "From messages"

    def test_multi_turn_messages(self):
        rs = _make_rs_with_mock("Berlin")
        lm = RouteSmithLM(routesmith=rs)
        msgs = [
            {"role": "user", "content": "Name a European capital."},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "Another one?"},
        ]
        result = lm(messages=msgs)
        assert result == ["Berlin"]

    def test_empty_response_content_handled(self):
        rs = RouteSmith()
        rs.register_model("openai/gpt-4o-mini", cost_per_1k_input=0.15,
                          cost_per_1k_output=0.60, quality_score=0.85)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = None
        rs.completion = MagicMock(return_value=mock_response)

        lm = RouteSmithLM(routesmith=rs)
        result = lm("Hello")
        assert result == [""]
