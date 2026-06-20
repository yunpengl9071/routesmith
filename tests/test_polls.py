"""Tests for quality poll generation and sampling."""

from unittest.mock import patch

import pytest

from routesmith.feedback.polls import (
    PollSampler,
    PollSignalMapper,
    generate_poll,
)


class TestPollGeneration:
    def test_generates_correct_structure(self):
        """generate_poll produces correctly structured Poll."""
        poll = generate_poll(
            request_id="abc123",
            model_id="gpt-4o-mini",
            cost_usd=0.0012,
        )

        assert poll.id == "abc123"
        assert poll.type == "numbered"
        assert poll.question == "How was this response?"
        assert len(poll.options) == 5
        assert poll.context["model"] == "gpt-4o-mini"
        assert poll.context["cost"] == "$0.0012"
        assert poll.context["request_id"] == "abc123"

    def test_options_have_correct_labels(self):
        """Poll options match the design spec."""
        poll = generate_poll("abc123", "gpt-4o-mini", 0.0012)
        labels = [o.label for o in poll.options]

        assert labels[0] == "Good"
        assert "wrong output" in labels[1]
        assert "too slow" in labels[2]
        assert "too expensive" in labels[3]
        assert "too verbose" in labels[4]

    def test_to_dict_serializable(self):
        """Poll.to_dict() produces JSON-compatible output."""
        poll = generate_poll("abc123", "gpt-4o-mini", 0.0012)
        d = poll.to_dict()

        assert isinstance(d, dict)
        assert d["id"] == "abc123"
        assert d["type"] == "numbered"
        assert isinstance(d["options"], list)
        assert d["options"][0]["id"] == 1


class TestPollSampler:
    def test_new_agent_sampled_more(self):
        """New agents (convergence=0) get sampled at base rate."""
        sampler = PollSampler(base_rate=1.0)  # 100% rate for testing

        # With convergence=0, should always sample at base_rate=1.0
        assert sampler.should_sample(agent_id="new-agent", convergence=0.0) is True

    def test_converged_agent_sampled_less(self):
        """Converged agents (convergence=1) get base_rate * 0 = 0%."""
        sampler = PollSampler(base_rate=0.1)

        # With convergence=1, effective_rate = 0.1 * (1-1) = 0
        assert sampler.should_sample(agent_id="converged", convergence=1.0) is False

    def test_uses_random_sample(self):
        """Sampling uses random.random() for decisions."""
        sampler = PollSampler(base_rate=0.5)

        with patch("random.random", return_value=0.3):
            assert sampler.should_sample(agent_id="a", convergence=0.0) is True

        with patch("random.random", return_value=0.7):
            assert sampler.should_sample(agent_id="a", convergence=0.0) is False


class TestPollSignalMapper:
    def test_good_answer_maps_to_high_quality(self):
        """Option 1 (Good) maps to quality=1.0."""
        signal = PollSignalMapper.map(option=1)
        assert signal["quality"] == 1.0
        assert signal["reason"] == "quality_good"

    def test_wrong_output_maps_to_low_quality(self):
        """Option 2 (wrong output) maps to quality=0.0."""
        signal = PollSignalMapper.map(option=2)
        assert signal["quality"] == 0.0
        assert signal["reason"] == "quality_low"

    def test_too_slow_maps_to_moderate_quality(self):
        """Option 3 (too slow) maps to quality=0.5."""
        signal = PollSignalMapper.map(option=3)
        assert signal["quality"] == pytest.approx(0.5)

    def test_too_expensive_maps_to_low_quality(self):
        """Option 4 (too expensive) maps to quality < 0.5."""
        signal = PollSignalMapper.map(option=4)
        assert signal["quality"] < 0.5

    def test_too_verbose_maps_to_moderate_quality(self):
        """Option 5 (too verbose) maps to quality < 0.5."""
        signal = PollSignalMapper.map(option=5)
        assert signal["quality"] < 0.5

    def test_unknown_option_returns_none(self):
        """Unknown option ID returns None."""
        signal = PollSignalMapper.map(option=99)
        assert signal is None
