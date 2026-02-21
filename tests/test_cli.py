"""Tests for CLI commands."""

import pytest
from unittest.mock import patch, MagicMock

from routesmith.cli.main import main
from routesmith.cli.stats import print_stats_table


class TestCLIMain:
    """Tests for main CLI entry point."""

    def test_help_flag(self, capsys):
        """Test --help shows usage."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "RouteSmith" in captured.out
        assert "serve" in captured.out
        assert "stats" in captured.out

    def test_version_flag(self, capsys):
        """Test --version shows version."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "0.1.0" in captured.out

    def test_serve_help(self, capsys):
        """Test serve --help shows usage."""
        with pytest.raises(SystemExit) as exc_info:
            main(["serve", "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--port" in captured.out
        assert "--config" in captured.out

    def test_stats_help(self, capsys):
        """Test stats --help shows usage."""
        with pytest.raises(SystemExit) as exc_info:
            main(["stats", "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--server" in captured.out
        assert "--json" in captured.out

    def test_no_command_shows_help(self, capsys):
        """Test no command shows help."""
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "routesmith" in captured.out.lower()


class TestStatsCommand:
    """Tests for stats command."""

    @patch("routesmith.cli.stats.httpx")
    def test_stats_json_output(self, mock_httpx, capsys):
        """Test stats with --json flag."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "request_count": 100,
            "total_cost_usd": 1.5,
            "estimated_without_routing": 5.0,
            "cost_savings_usd": 3.5,
            "savings_percent": 70.0,
            "registered_models": 3,
            "feedback_samples": 10,
        }
        mock_httpx.get.return_value = mock_response

        from routesmith.cli.stats import run_stats
        from argparse import Namespace

        args = Namespace(server="http://localhost:9119", json=True)
        result = run_stats(args)

        assert result == 0
        captured = capsys.readouterr()
        assert '"request_count": 100' in captured.out

    @patch("routesmith.cli.stats.httpx")
    def test_stats_table_output(self, mock_httpx, capsys):
        """Test stats with table output."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "request_count": 100,
            "total_cost_usd": 1.5,
            "estimated_without_routing": 5.0,
            "cost_savings_usd": 3.5,
            "savings_percent": 70.0,
            "registered_models": 3,
            "feedback_samples": 10,
        }
        mock_httpx.get.return_value = mock_response

        from routesmith.cli.stats import run_stats
        from argparse import Namespace

        args = Namespace(server="http://localhost:9119", json=False)
        result = run_stats(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "RouteSmith Cost Report" in captured.out
        assert "100" in captured.out
        assert "70.0%" in captured.out

    @patch("routesmith.cli.stats.httpx")
    def test_stats_connection_error(self, mock_httpx, capsys):
        """Test stats handles connection error."""
        import httpx
        mock_httpx.get.side_effect = httpx.ConnectError("Connection refused")
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPError = httpx.HTTPError

        from routesmith.cli.stats import run_stats
        from argparse import Namespace

        args = Namespace(server="http://localhost:9119", json=False)
        result = run_stats(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Could not connect" in captured.err


class TestPrintStatsTable:
    """Tests for stats table formatting."""

    def test_print_stats_table_basic(self, capsys):
        """Test basic stats table output."""
        stats = {
            "request_count": 50,
            "total_cost_usd": 0.5,
            "estimated_without_routing": 2.0,
            "cost_savings_usd": 1.5,
            "savings_percent": 75.0,
            "registered_models": 4,
            "feedback_samples": 5,
        }

        print_stats_table(stats)
        captured = capsys.readouterr()

        assert "50" in captured.out
        assert "0.5" in captured.out
        assert "75.0%" in captured.out
        assert "RouteSmith Cost Report" in captured.out

    def test_print_stats_table_with_last_routing(self, capsys):
        """Test stats table with last routing info."""
        stats = {
            "request_count": 1,
            "total_cost_usd": 0.001,
            "estimated_without_routing": 0.01,
            "cost_savings_usd": 0.009,
            "savings_percent": 90.0,
            "registered_models": 2,
            "feedback_samples": 1,
            "last_routing": {
                "model_selected": "gpt-4o-mini",
                "routing_reason": "cheapest model",
                "estimated_cost_usd": 0.001,
                "cost_savings_usd": 0.009,
            },
        }

        print_stats_table(stats)
        captured = capsys.readouterr()

        assert "Last routing decision" in captured.out
        assert "gpt-4o-mini" in captured.out


class TestYamlLoader:
    """Tests for YAML config loader."""

    def test_load_config_empty_file(self, tmp_path):
        """Test loading empty config file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        from routesmith.cli.yaml_loader import load_config_file

        config, models = load_config_file(config_file)

        assert config is not None
        assert models == []

    def test_load_config_with_models(self, tmp_path):
        """Test loading config with models."""
        config_content = """
models:
  - id: gpt-4o
    cost_per_1k_input: 0.005
    cost_per_1k_output: 0.015
    quality_score: 0.95
  - id: gpt-4o-mini
    cost_per_1k_input: 0.00015
    cost_per_1k_output: 0.0006
    quality_score: 0.85
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        from routesmith.cli.yaml_loader import load_config_file

        config, models = load_config_file(config_file)

        assert len(models) == 2
        assert models[0]["model_id"] == "gpt-4o"
        assert models[0]["cost_per_1k_input"] == 0.005
        assert models[1]["model_id"] == "gpt-4o-mini"
        assert models[1]["quality_score"] == 0.85

    def test_load_config_with_routing(self, tmp_path):
        """Test loading config with routing settings."""
        config_content = """
routing:
  strategy: cascade
  fallback_model: gpt-4o-mini

budget:
  quality_threshold: 0.9
  max_cost_per_request: 0.05

models:
  - id: test-model
    cost_per_1k_input: 0.001
    cost_per_1k_output: 0.002
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        from routesmith.cli.yaml_loader import load_config_file
        from routesmith.config import RoutingStrategy

        config, models = load_config_file(config_file)

        assert config.default_strategy == RoutingStrategy.CASCADE
        assert config.fallback_model == "gpt-4o-mini"
        assert config.budget.quality_threshold == 0.9
        assert config.budget.max_cost_per_request == 0.05

    def test_load_config_all_strategies(self, tmp_path):
        """Test loading all routing strategy values."""
        from routesmith.cli.yaml_loader import load_config_file
        from routesmith.config import RoutingStrategy

        for strategy_name, expected in [
            ("direct", RoutingStrategy.DIRECT),
            ("cascade", RoutingStrategy.CASCADE),
            ("parallel", RoutingStrategy.PARALLEL),
            ("speculative", RoutingStrategy.SPECULATIVE),
            ("DIRECT", RoutingStrategy.DIRECT),  # Case insensitive
        ]:
            config_content = f"""
routing:
  strategy: {strategy_name}
models:
  - id: test
    cost_per_1k_input: 0.001
    cost_per_1k_output: 0.002
"""
            config_file = tmp_path / f"config_{strategy_name}.yaml"
            config_file.write_text(config_content)
            config, _ = load_config_file(config_file)
            assert config.default_strategy == expected

    def test_load_config_unknown_strategy_defaults_to_direct(self, tmp_path):
        """Test that unknown strategy defaults to DIRECT."""
        config_content = """
routing:
  strategy: unknown_strategy
models:
  - id: test
    cost_per_1k_input: 0.001
    cost_per_1k_output: 0.002
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        from routesmith.cli.yaml_loader import load_config_file
        from routesmith.config import RoutingStrategy

        config, _ = load_config_file(config_file)
        assert config.default_strategy == RoutingStrategy.DIRECT

    def test_load_config_with_optional_model_fields(self, tmp_path):
        """Test loading config with all optional model fields."""
        config_content = """
models:
  - id: full-model
    cost_per_1k_input: 0.005
    cost_per_1k_output: 0.015
    quality_score: 0.95
    latency_p50_ms: 500
    latency_p99_ms: 2000
    context_window: 128000
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        from routesmith.cli.yaml_loader import load_config_file

        _, models = load_config_file(config_file)

        assert len(models) == 1
        assert models[0]["latency_p50_ms"] == 500
        assert models[0]["latency_p99_ms"] == 2000
        assert models[0]["context_window"] == 128000

    def test_load_config_with_cache_settings(self, tmp_path):
        """Test loading config with cache settings."""
        config_content = """
cache:
  enabled: true
  similarity_threshold: 0.92
  ttl_seconds: 7200

models:
  - id: test
    cost_per_1k_input: 0.001
    cost_per_1k_output: 0.002
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        from routesmith.cli.yaml_loader import load_config_file

        config, _ = load_config_file(config_file)

        assert config.cache.enabled is True
        assert config.cache.similarity_threshold == 0.92
        assert config.cache.ttl_seconds == 7200

    def test_load_config_minimal_model(self, tmp_path):
        """Test loading config with minimal required model fields."""
        config_content = """
models:
  - id: minimal
    cost_per_1k_input: 0.001
    cost_per_1k_output: 0.002
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        from routesmith.cli.yaml_loader import load_config_file

        _, models = load_config_file(config_file)

        assert models[0]["model_id"] == "minimal"
        assert models[0]["quality_score"] == 0.8  # Default

    def test_load_config_file_not_found(self, tmp_path):
        """Test error when config file doesn't exist."""
        from routesmith.cli.yaml_loader import load_config_file

        with pytest.raises(FileNotFoundError):
            load_config_file(tmp_path / "nonexistent.yaml")


class TestCLIEdgeCases:
    """Edge case tests for CLI."""

    def test_unknown_command(self, capsys):
        """Test unknown command is handled."""
        # main() should return 0 and print help for no/unknown command
        result = main([])
        assert result == 0

    def test_serve_default_values(self, capsys):
        """Verify default argument values for serve command."""
        with pytest.raises(SystemExit):
            main(["serve", "--help"])

        captured = capsys.readouterr()
        assert "9119" in captured.out  # Default port
        assert "127.0.0.1" in captured.out  # Default host
        assert "routesmith.yaml" in captured.out  # Default config
        assert "INFO" in captured.out  # Default log level


class TestStatsFormatting:
    """Tests for stats output formatting edge cases."""

    def test_print_stats_zero_values(self, capsys):
        """Test stats table with all zero values."""
        stats = {
            "request_count": 0,
            "total_cost_usd": 0.0,
            "estimated_without_routing": 0.0,
            "cost_savings_usd": 0.0,
            "savings_percent": 0.0,
            "registered_models": 0,
            "feedback_samples": 0,
        }

        print_stats_table(stats)
        captured = capsys.readouterr()
        assert "0" in captured.out
        assert "0.0%" in captured.out

    def test_print_stats_large_values(self, capsys):
        """Test stats table with large values."""
        stats = {
            "request_count": 1000000,
            "total_cost_usd": 12345.6789,
            "estimated_without_routing": 50000.0,
            "cost_savings_usd": 37654.3211,
            "savings_percent": 75.3,
            "registered_models": 100,
            "feedback_samples": 100000,
        }

        print_stats_table(stats)
        captured = capsys.readouterr()
        assert "1,000,000" in captured.out
        assert "12345.6789" in captured.out or "12,345" in captured.out

    def test_print_stats_missing_keys(self, capsys):
        """Test stats table handles missing keys gracefully."""
        stats = {}  # Empty stats

        print_stats_table(stats)
        captured = capsys.readouterr()
        # Should not crash, should show 0 for missing values
        assert "RouteSmith Cost Report" in captured.out
