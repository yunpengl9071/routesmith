"""Tests for Anthropic SDK integration (RouteSmithAnthropic)."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from routesmith.integrations.anthropic import (
    RouteSmithAnthropic,
    _anthropic_to_openai_messages,
    _litellm_to_anthropic_message,
)
from routesmith import RouteSmith, RouteSmithConfig


# ---------------------------------------------------------------------------
# _anthropic_to_openai_messages
# ---------------------------------------------------------------------------


class TestAnthropicToOpenAIMessages:
    def test_plain_string_content(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = _anthropic_to_openai_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_role_preserved(self):
        msgs = [{"role": "assistant", "content": "Hi there"}]
        result = _anthropic_to_openai_messages(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_content_block_list_flattened(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
        result = _anthropic_to_openai_messages(msgs)
        assert result == [{"role": "user", "content": "Hello world"}]

    def test_content_block_with_object_text_attr(self):
        """Handles Anthropic SDK TextBlock objects (have .text attribute)."""
        block = MagicMock()
        block.text = "Block text"
        # Not a dict with 'type', but has .text
        msgs = [{"role": "user", "content": [block]}]
        result = _anthropic_to_openai_messages(msgs)
        assert result[0]["content"] == "Block text"

    def test_non_text_block_type_skipped(self):
        """Non-text block types (image, etc.) are skipped in flattening."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"url": "http://example.com/img.png"}},
                    {"type": "text", "text": "Describe this"},
                ],
            }
        ]
        result = _anthropic_to_openai_messages(msgs)
        assert result[0]["content"] == "Describe this"

    def test_non_string_non_list_content_coerced(self):
        msgs = [{"role": "user", "content": 42}]
        result = _anthropic_to_openai_messages(msgs)
        assert result == [{"role": "user", "content": "42"}]

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        result = _anthropic_to_openai_messages(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["content"] == "Follow-up"

    def test_missing_role_defaults_to_user(self):
        msgs = [{"content": "No role"}]
        result = _anthropic_to_openai_messages(msgs)
        assert result[0]["role"] == "user"


# ---------------------------------------------------------------------------
# _litellm_to_anthropic_message
# ---------------------------------------------------------------------------


def _make_litellm_response(text="Hello", finish_reason="stop", prompt_tokens=10, completion_tokens=5):
    """Build a minimal mock LiteLLM ModelResponse."""
    response = MagicMock()
    response.id = "chatcmpl-test123"
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = finish_reason
    response.choices = [choice]
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class TestLitellmToAnthropicMessage:
    def test_basic_conversion(self):
        response = _make_litellm_response("Hi!")
        msg = _litellm_to_anthropic_message(response, "anthropic/claude-3-haiku")

        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert msg.content[0].text == "Hi!"
        assert msg.model == "anthropic/claude-3-haiku"

    def test_stop_reason_mapped(self):
        response = _make_litellm_response(finish_reason="stop")
        msg = _litellm_to_anthropic_message(response, "test-model")
        assert msg.stop_reason == "end_turn"

    def test_length_finish_reason_mapped_to_max_tokens(self):
        response = _make_litellm_response(finish_reason="length")
        msg = _litellm_to_anthropic_message(response, "test-model")
        assert msg.stop_reason == "max_tokens"

    def test_unknown_finish_reason_defaults_to_end_turn(self):
        response = _make_litellm_response(finish_reason="unknown_reason")
        msg = _litellm_to_anthropic_message(response, "test-model")
        assert msg.stop_reason == "end_turn"

    def test_usage_tokens_mapped(self):
        response = _make_litellm_response(prompt_tokens=20, completion_tokens=8)
        msg = _litellm_to_anthropic_message(response, "test-model")
        assert msg.usage.input_tokens == 20
        assert msg.usage.output_tokens == 8

    def test_empty_content(self):
        response = _make_litellm_response(text="")
        msg = _litellm_to_anthropic_message(response, "test-model")
        assert msg.content[0].text == ""

    def test_response_id_preserved(self):
        response = _make_litellm_response()
        msg = _litellm_to_anthropic_message(response, "test-model")
        assert msg.id == "chatcmpl-test123"

    def test_no_usage_attribute_defaults_zero(self):
        response = _make_litellm_response()
        response.usage = None
        msg = _litellm_to_anthropic_message(response, "test-model")
        assert msg.usage.input_tokens == 0
        assert msg.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# RouteSmithAnthropic
# ---------------------------------------------------------------------------


def _make_rs_with_mock_completion(text="Mocked response"):
    """Return a RouteSmith instance whose completion() is mocked."""
    rs = RouteSmith()
    rs.register_model(
        "anthropic/claude-3-haiku",
        cost_per_1k_input=0.25,
        cost_per_1k_output=1.25,
        quality_score=0.75,
    )
    rs.register_model(
        "anthropic/claude-3-5-sonnet",
        cost_per_1k_input=3.0,
        cost_per_1k_output=15.0,
        quality_score=0.92,
    )

    mock_response = _make_litellm_response(text)
    rs.completion = MagicMock(return_value=mock_response)
    return rs


class TestRouteSmithAnthropic:
    def test_init_with_existing_routesmith(self):
        rs = RouteSmith()
        client = RouteSmithAnthropic(routesmith=rs)
        assert client._rs is rs

    def test_init_creates_routesmith_if_none(self):
        client = RouteSmithAnthropic()
        assert isinstance(client._rs, RouteSmith)

    def test_register_model_delegates(self):
        client = RouteSmithAnthropic()
        client.register_model(
            "anthropic/claude-3-haiku",
            cost_per_1k_input=0.25,
            cost_per_1k_output=1.25,
            quality_score=0.75,
        )
        assert len(client._rs.registry) == 1

    def test_messages_create_basic(self):
        rs = _make_rs_with_mock_completion("Hello!")
        client = RouteSmithAnthropic(routesmith=rs)

        msg = client.messages.create(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=512,
        )

        assert msg.role == "assistant"
        assert msg.content[0].text == "Hello!"

    def test_messages_create_passes_max_tokens(self):
        rs = _make_rs_with_mock_completion()
        client = RouteSmithAnthropic(routesmith=rs)

        client.messages.create(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=256,
        )

        call_kwargs = rs.completion.call_args[1]
        assert call_kwargs["max_tokens"] == 256

    def test_messages_create_system_prepended(self):
        rs = _make_rs_with_mock_completion()
        client = RouteSmithAnthropic(routesmith=rs)

        client.messages.create(
            messages=[{"role": "user", "content": "Hi"}],
            system="You are a helpful assistant.",
            max_tokens=512,
        )

        call_kwargs = rs.completion.call_args[1]
        msgs = call_kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert msgs[1] == {"role": "user", "content": "Hi"}

    def test_messages_create_no_system_when_none(self):
        rs = _make_rs_with_mock_completion()
        client = RouteSmithAnthropic(routesmith=rs)

        client.messages.create(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=512,
        )

        call_kwargs = rs.completion.call_args[1]
        msgs = call_kwargs["messages"]
        assert all(m["role"] != "system" for m in msgs)

    def test_messages_create_temperature_forwarded(self):
        rs = _make_rs_with_mock_completion()
        client = RouteSmithAnthropic(routesmith=rs)

        client.messages.create(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=512,
            temperature=0.2,
        )

        call_kwargs = rs.completion.call_args[1]
        assert call_kwargs["temperature"] == 0.2

    def test_messages_create_no_temperature_when_none(self):
        rs = _make_rs_with_mock_completion()
        client = RouteSmithAnthropic(routesmith=rs)

        client.messages.create(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=512,
        )

        call_kwargs = rs.completion.call_args[1]
        assert "temperature" not in call_kwargs

    def test_stats_property_delegates(self):
        rs = RouteSmith()
        client = RouteSmithAnthropic(routesmith=rs)
        stats = client.stats
        assert "request_count" in stats

    def test_messages_resource_on_client(self):
        client = RouteSmithAnthropic()
        assert hasattr(client, "messages")
        assert hasattr(client.messages, "create")

    def test_content_block_list_converted(self):
        """Anthropic-style content block list is handled correctly."""
        rs = _make_rs_with_mock_completion("OK")
        client = RouteSmithAnthropic(routesmith=rs)

        client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is"},
                        {"type": "text", "text": " 2+2?"},
                    ],
                }
            ],
            max_tokens=64,
        )

        call_kwargs = rs.completion.call_args[1]
        assert call_kwargs["messages"][0]["content"] == "What is  2+2?"


# ---------------------------------------------------------------------------
# routesmith openclaw-config CLI
# ---------------------------------------------------------------------------


class TestOpenClawConfig:
    def test_stdout_output_valid_json(self, capsys):
        from routesmith.cli.openclaw import run_openclaw_config

        args = Namespace(host="http://localhost:9119", output=None)
        result = run_openclaw_config(args)

        assert result == 0
        captured = capsys.readouterr()
        config = json.loads(captured.out)
        assert "models" in config
        assert "routesmith" in config["models"]["providers"]

    def test_base_url_in_output(self, capsys):
        from routesmith.cli.openclaw import run_openclaw_config

        args = Namespace(host="http://localhost:9119", output=None)
        run_openclaw_config(args)

        captured = capsys.readouterr()
        config = json.loads(captured.out)
        provider = config["models"]["providers"]["routesmith"]
        assert provider["baseUrl"] == "http://localhost:9119/v1"

    def test_custom_host(self, capsys):
        from routesmith.cli.openclaw import run_openclaw_config

        args = Namespace(host="http://myserver:8080", output=None)
        run_openclaw_config(args)

        captured = capsys.readouterr()
        config = json.loads(captured.out)
        provider = config["models"]["providers"]["routesmith"]
        assert provider["baseUrl"] == "http://myserver:8080/v1"

    def test_trailing_slash_stripped(self, capsys):
        from routesmith.cli.openclaw import run_openclaw_config

        args = Namespace(host="http://localhost:9119/", output=None)
        run_openclaw_config(args)

        captured = capsys.readouterr()
        config = json.loads(captured.out)
        provider = config["models"]["providers"]["routesmith"]
        assert not provider["baseUrl"].endswith("//v1")
        assert provider["baseUrl"] == "http://localhost:9119/v1"

    def test_auto_model_present(self, capsys):
        from routesmith.cli.openclaw import run_openclaw_config

        args = Namespace(host="http://localhost:9119", output=None)
        run_openclaw_config(args)

        captured = capsys.readouterr()
        config = json.loads(captured.out)
        models = config["models"]["providers"]["routesmith"]["models"]
        assert any(m["id"] == "auto" for m in models)

    def test_agents_section_present(self, capsys):
        from routesmith.cli.openclaw import run_openclaw_config

        args = Namespace(host="http://localhost:9119", output=None)
        run_openclaw_config(args)

        captured = capsys.readouterr()
        config = json.loads(captured.out)
        assert "agents" in config
        assert "routesmith/auto" in config["agents"]["defaults"]["models"]

    def test_file_output(self, tmp_path, capsys):
        from routesmith.cli.openclaw import run_openclaw_config

        out_file = tmp_path / "openclaw.json"
        args = Namespace(host="http://localhost:9119", output=str(out_file))
        result = run_openclaw_config(args)

        assert result == 0
        assert out_file.exists()
        config = json.loads(out_file.read_text())
        assert "routesmith" in config["models"]["providers"]

        captured = capsys.readouterr()
        assert "Wrote OpenClaw config" in captured.out

    def test_cli_openclaw_config_command(self, capsys):
        """Test openclaw-config is reachable via the main CLI."""
        from routesmith.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["openclaw-config", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--host" in captured.out
