"""Tests for LLM-as-judge evaluator."""
from unittest.mock import patch

from routesmith import RouteSmith, RouteSmithConfig
from routesmith.config import JudgeConfig
from routesmith.feedback.judge import LLMJudge
from tests.helpers import fake_response


class TestParse:
    def test_parse_clean_json(self):
        assert LLMJudge._parse('{"score": 0.85, "reason": "good"}') == 0.85

    def test_parse_json_in_prose(self):
        val = LLMJudge._parse(
            'Result: {"score": 0.75, "reason": "mostly correct"}'
        )
        assert val == 0.75

    def test_parse_garbage_returns_none(self):
        assert LLMJudge._parse("not json at all") is None

    def test_parse_clamps_range(self):
        assert LLMJudge._parse('{"score": 1.7}') == 1.0
        assert LLMJudge._parse('{"score": -0.5}') == 0.0


class TestJudgeNeverRaises:
    def test_judge_never_raises(self):
        judge = LLMJudge(model="test-model")
        with patch("litellm.completion", side_effect=Exception("API error")):
            result = judge.score(
                [{"role": "user", "content": "hello"}], "some response"
            )
        assert result is None


class TestJudgeWiring:
    def test_judge_updates_predictor_when_sampled(self):
        config = RouteSmithConfig(
            predictor_type="adaptive",
            judge=JudgeConfig(
                enabled=True, sample_rate=1.0, judge_model="gpt-4o-mini"
            ),
        )
        rs = RouteSmith(config=config)
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

        with patch("routesmith.client.litellm.completion") as mock_llm:
            mock_llm.return_value = fake_response(content="a helpful answer")
            with patch.object(rs._judge, "score", return_value=0.9):
                with patch.object(rs, "record_outcome") as mock_record:
                    rs.completion(
                        messages=[{"role": "user", "content": "hello"}]
                    )
                    mock_record.assert_called_once()
                    assert mock_record.call_args[1]["score"] == 0.9

    def test_judge_not_called_below_sample(self):
        config = RouteSmithConfig(
            predictor_type="adaptive",
            judge=JudgeConfig(
                enabled=True, sample_rate=0.0, judge_model="gpt-4o-mini"
            ),
        )
        rs = RouteSmith(config=config)
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

        with patch("routesmith.client.litellm.completion") as mock_llm:
            mock_llm.return_value = fake_response(content="hello")
            with patch.object(rs._judge, "score") as mock_score:
                rs.completion(
                    messages=[{"role": "user", "content": "hello"}]
                )
                mock_score.assert_not_called()

    def test_judge_disabled_by_default(self):
        assert RouteSmithConfig().judge.enabled is False
