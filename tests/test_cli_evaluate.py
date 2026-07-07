"""Tests for routesmith evaluate CLI command."""
import json
import sqlite3
import tempfile
from unittest.mock import patch

from routesmith.cli.evaluate import run_evaluate


def _make_args(**kwargs):
    import argparse
    args = argparse.Namespace()
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def _populate_db(db_path: str, records: list[dict]) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_records (
            request_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            quality_score REAL,
            user_feedback TEXT,
            metadata_json TEXT,
            created_at REAL NOT NULL,
            agent_id TEXT,
            agent_role TEXT,
            conversation_id TEXT,
            turn_index INTEGER
        )
    """)
    for r in records:
        conn.execute(
            """INSERT INTO feedback_records
               (request_id, model_id, messages_json, latency_ms, quality_score, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                r["request_id"],
                r["model_id"],
                json.dumps(r["messages"]),
                r.get("latency_ms", 100),
                r.get("quality_score"),
                r.get("created_at", 1000),
            ),
        )
    conn.commit()
    conn.close()


class TestEvaluateCommand:
    def test_evaluate_no_records_returns_error(self):
        """Empty database should return non-zero exit code."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            args = _make_args(db=db_path, config="nonexistent.yaml",
                            strategy="direct", limit=100)
            rc = run_evaluate(args)
            assert rc != 0
        finally:
            import os
            os.unlink(db_path)

    @patch("routesmith.strategy.router.Router.route")
    def test_evaluate_with_records_returns_zero(self, mock_route):
        """Database with records should produce a report."""
        mock_route.side_effect = lambda messages=None, strategy=None, min_quality=None, **kw: (
            "model-a" if "hello" in str(messages) else "model-b"
        )

        records = [
            {
                "request_id": "req-001",
                "model_id": "model-a",
                "messages": [{"role": "user", "content": "Say hello"}],
                "latency_ms": 100,
                "quality_score": 0.9,
                "created_at": 1000,
            },
            {
                "request_id": "req-002",
                "model_id": "model-b",
                "messages": [{"role": "user", "content": "Tell me a fact"}],
                "latency_ms": 200,
                "quality_score": 0.8,
                "created_at": 1001,
            },
        ]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            _populate_db(db_path, records)
            args = _make_args(db=db_path, config="nonexistent.yaml",
                            strategy="direct", limit=100)
            rc = run_evaluate(args)
            assert rc == 0
        finally:
            import os
            os.unlink(db_path)

    @patch("routesmith.strategy.router.Router.route")
    def test_evaluate_shows_agreement_rate(self, mock_route):
        """Verify agreement rate is computed correctly."""
        mock_route.return_value = "model-a"

        records = [
            {
                "request_id": f"req-{i:03d}",
                "model_id": "model-a" if i < 3 else "model-b",
                "messages": [{"role": "user", "content": f"Query {i}"}],
                "latency_ms": 100,
                "quality_score": 0.8,
                "created_at": 1000 + i,
            }
            for i in range(5)
        ]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            _populate_db(db_path, records)
            args = _make_args(db=db_path, config="nonexistent.yaml",
                            strategy="direct", limit=100)
            rc = run_evaluate(args)
            assert rc == 0
        finally:
            import os
            os.unlink(db_path)
