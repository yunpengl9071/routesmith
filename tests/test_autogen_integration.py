"""Tests for AutoGen integration (config helpers and agent factory)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call

import pytest

from routesmith.integrations.autogen import (
    routesmith_config_list,
    routesmith_autogen_llm_config,
    routesmith_autogen_agents,
)


# ---------------------------------------------------------------------------
# routesmith_config_list
# ---------------------------------------------------------------------------


class TestRoutesmithConfigList:
    def test_default_returns_single_auto_entry(self):
        result = routesmith_config_list()
        assert len(result) == 1
        assert result[0]["model"] == "auto"

    def test_base_url_constructed_correctly(self):
        result = routesmith_config_list(host="http://localhost:9119")
        assert result[0]["base_url"] == "http://localhost:9119/v1"

    def test_trailing_slash_stripped(self):
        result = routesmith_config_list(host="http://localhost:9119/")
        assert result[0]["base_url"] == "http://localhost:9119/v1"

    def test_api_key_is_dummy(self):
        result = routesmith_config_list()
        assert result[0]["api_key"] == "dummy"

    def test_custom_host(self):
        result = routesmith_config_list(host="http://prod.internal:9119")
        assert result[0]["base_url"] == "http://prod.internal:9119/v1"

    def test_custom_model(self):
        result = routesmith_config_list(model="gpt-4o-mini")
        assert result[0]["model"] == "gpt-4o-mini"

    def test_extra_models_appended(self):
        result = routesmith_config_list(
            model="auto",
            extra_models=["gpt-4o", "claude-3-haiku"],
        )
        assert len(result) == 3
        assert result[0]["model"] == "auto"
        assert result[1]["model"] == "gpt-4o"
        assert result[2]["model"] == "claude-3-haiku"

    def test_all_extra_models_share_same_base_url(self):
        result = routesmith_config_list(
            host="http://localhost:9119",
            extra_models=["gpt-4o"],
        )
        for entry in result:
            assert entry["base_url"] == "http://localhost:9119/v1"

    def test_no_extra_models_returns_single_entry(self):
        result = routesmith_config_list(extra_models=None)
        assert len(result) == 1

    def test_empty_extra_models_list(self):
        result = routesmith_config_list(extra_models=[])
        assert len(result) == 1

    def test_returns_list_of_dicts(self):
        result = routesmith_config_list()
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_each_entry_has_required_keys(self):
        result = routesmith_config_list(extra_models=["gpt-4o"])
        for entry in result:
            assert "model" in entry
            assert "base_url" in entry
            assert "api_key" in entry


# ---------------------------------------------------------------------------
# routesmith_autogen_llm_config
# ---------------------------------------------------------------------------


class TestRoutesmithAutogenLlmConfig:
    def test_returns_dict_with_config_list(self):
        result = routesmith_autogen_llm_config()
        assert isinstance(result, dict)
        assert "config_list" in result

    def test_config_list_has_auto_model(self):
        result = routesmith_autogen_llm_config()
        assert result["config_list"][0]["model"] == "auto"

    def test_base_url_in_config_list(self):
        result = routesmith_autogen_llm_config(host="http://localhost:9119")
        assert result["config_list"][0]["base_url"] == "http://localhost:9119/v1"

    def test_cache_seed_included_when_provided(self):
        result = routesmith_autogen_llm_config(cache_seed=42)
        assert result["cache_seed"] == 42

    def test_cache_seed_not_included_when_none(self):
        result = routesmith_autogen_llm_config(cache_seed=None)
        assert "cache_seed" not in result

    def test_extra_kwargs_merged_into_config(self):
        result = routesmith_autogen_llm_config(temperature=0.5, timeout=60)
        assert result["temperature"] == 0.5
        assert result["timeout"] == 60

    def test_extra_models_passed_to_config_list(self):
        result = routesmith_autogen_llm_config(extra_models=["gpt-4o"])
        assert len(result["config_list"]) == 2

    def test_custom_host_propagated(self):
        result = routesmith_autogen_llm_config(host="http://prod:9119")
        assert result["config_list"][0]["base_url"] == "http://prod:9119/v1"

    def test_custom_model_propagated(self):
        result = routesmith_autogen_llm_config(model="gpt-4o-mini")
        assert result["config_list"][0]["model"] == "gpt-4o-mini"

    def test_does_not_include_cache_seed_by_default(self):
        """Default behavior: no cache_seed (RouteSmith handles caching)."""
        result = routesmith_autogen_llm_config()
        assert "cache_seed" not in result

    def test_extra_kwargs_do_not_override_config_list(self):
        """Extra kwargs should not clobber the config_list key."""
        result = routesmith_autogen_llm_config(some_flag=True)
        assert "config_list" in result
        assert result["some_flag"] is True


# ---------------------------------------------------------------------------
# routesmith_autogen_agents
# ---------------------------------------------------------------------------


class _FakeAssistantAgent:
    def __init__(self, name, system_message, llm_config, **kwargs):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config
        self.extra = kwargs


class _FakeUserProxyAgent:
    def __init__(self, name, human_input_mode, max_consecutive_auto_reply, llm_config):
        self.name = name
        self.human_input_mode = human_input_mode
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.llm_config = llm_config


def _make_mock_autogen():
    mock = MagicMock()
    mock.AssistantAgent.side_effect = lambda **kw: _FakeAssistantAgent(**kw)
    mock.UserProxyAgent.side_effect = lambda **kw: _FakeUserProxyAgent(**kw)
    return mock


class TestRoutesmithAutogenAgents:
    def _patch_autogen(self):
        mock_ag = _make_mock_autogen()
        return patch.dict(sys.modules, {"autogen": mock_ag}), mock_ag

    def test_raises_import_error_when_autogen_missing(self):
        with patch.dict(sys.modules, {"autogen": None, "autogen_agentchat": None,
                                      "autogen_agentchat.agents": None}):
            with pytest.raises(ImportError, match="pyautogen"):
                routesmith_autogen_agents()

    def test_returns_tuple_of_two_agents(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            result = routesmith_autogen_agents()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_assistant_agent_has_correct_name(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            assistant, _ = routesmith_autogen_agents(assistant_name="MyBot")
        assert assistant.name == "MyBot"

    def test_user_proxy_has_correct_name(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            _, user = routesmith_autogen_agents(user_name="HumanProxy")
        assert user.name == "HumanProxy"

    def test_assistant_receives_routesmith_llm_config(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            assistant, _ = routesmith_autogen_agents(host="http://localhost:9119")
        assert "config_list" in assistant.llm_config
        assert assistant.llm_config["config_list"][0]["base_url"] == "http://localhost:9119/v1"

    def test_user_proxy_llm_config_is_false(self):
        """UserProxyAgent should not have its own LLM."""
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            _, user = routesmith_autogen_agents()
        assert user.llm_config is False

    def test_human_input_mode_defaults_to_never(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            _, user = routesmith_autogen_agents()
        assert user.human_input_mode == "NEVER"

    def test_human_input_mode_configurable(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            _, user = routesmith_autogen_agents(human_input_mode="ALWAYS")
        assert user.human_input_mode == "ALWAYS"

    def test_max_consecutive_auto_reply_default(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            _, user = routesmith_autogen_agents()
        assert user.max_consecutive_auto_reply == 10

    def test_max_consecutive_auto_reply_configurable(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            _, user = routesmith_autogen_agents(max_consecutive_auto_reply=5)
        assert user.max_consecutive_auto_reply == 5

    def test_system_message_passed_to_assistant(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            assistant, _ = routesmith_autogen_agents(
                system_message="You are a concise assistant."
            )
        assert assistant.system_message == "You are a concise assistant."

    def test_custom_host_used_in_llm_config(self):
        mock_ag = _make_mock_autogen()
        with patch.dict(sys.modules, {"autogen": mock_ag}):
            assistant, _ = routesmith_autogen_agents(host="http://prod:9119")
        cfg = assistant.llm_config["config_list"][0]
        assert cfg["base_url"] == "http://prod:9119/v1"

    def test_fallback_to_autogen_agentchat(self):
        """When 'autogen' is missing, should try autogen_agentchat.agents."""
        mock_ag = _make_mock_autogen()
        mock_agents_module = MagicMock()
        mock_agents_module.AssistantAgent.side_effect = lambda **kw: _FakeAssistantAgent(**kw)
        mock_agents_module.UserProxyAgent.side_effect = lambda **kw: _FakeUserProxyAgent(**kw)

        with patch.dict(sys.modules, {
            "autogen": None,
            "autogen_agentchat": MagicMock(),
            "autogen_agentchat.agents": mock_agents_module,
        }):
            assistant, user = routesmith_autogen_agents()

        assert assistant.name == "assistant"
        assert user.name == "user"
