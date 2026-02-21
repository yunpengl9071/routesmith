"""SQLite persistence for feedback records and quality signals."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any


class FeedbackStorage:
    """
    SQLite-backed storage for feedback records and outcome signals.

    Uses WAL mode for concurrent read performance. Lazily connects
    on first operation.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """
        Initialize feedback storage.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory.
        """
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        """Lazy connection with WAL mode and schema initialization."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            if self._db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_tables()
        return self._conn

    def _create_tables(self) -> None:
        conn = self._get_conn() if self._conn else self._conn
        # _conn is guaranteed set by caller (_get_conn sets it before calling us)
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS feedback_records (
                request_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                quality_score REAL,
                user_feedback TEXT,
                metadata_json TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS outcome_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                signal_value REAL NOT NULL,
                raw_value_json TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (request_id) REFERENCES feedback_records(request_id)
            );

            CREATE INDEX IF NOT EXISTS idx_signals_request
                ON outcome_signals(request_id);
            CREATE INDEX IF NOT EXISTS idx_records_model
                ON feedback_records(model_id);
        """)

    def store_record(
        self,
        request_id: str,
        model_id: str,
        messages: list[dict[str, str]],
        latency_ms: float,
        quality_score: float | None = None,
        user_feedback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a feedback record."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO feedback_records
               (request_id, model_id, messages_json, latency_ms,
                quality_score, user_feedback, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                model_id,
                json.dumps(messages),
                latency_ms,
                quality_score,
                user_feedback,
                json.dumps(metadata) if metadata else None,
                time.time(),
            ),
        )
        conn.commit()

    def store_signal(
        self,
        request_id: str,
        signal_type: str,
        signal_name: str,
        signal_value: float,
        raw_value: Any = None,
    ) -> None:
        """Store an outcome signal for a request."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO outcome_signals
               (request_id, signal_type, signal_name, signal_value,
                raw_value_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                signal_type,
                signal_name,
                signal_value,
                json.dumps(raw_value) if raw_value is not None else None,
                time.time(),
            ),
        )
        conn.commit()

    def get_record(self, request_id: str) -> dict[str, Any] | None:
        """Get a feedback record by request_id."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM feedback_records WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def update_record(
        self,
        request_id: str,
        quality_score: float | None = None,
        user_feedback: str | None = None,
    ) -> bool:
        """Update an existing record with outcome data. Returns True if found."""
        conn = self._get_conn()
        updates = []
        params: list[Any] = []
        if quality_score is not None:
            updates.append("quality_score = ?")
            params.append(quality_score)
        if user_feedback is not None:
            updates.append("user_feedback = ?")
            params.append(user_feedback)
        if not updates:
            return False
        params.append(request_id)
        cursor = conn.execute(
            f"UPDATE feedback_records SET {', '.join(updates)} WHERE request_id = ?",
            params,
        )
        conn.commit()
        return cursor.rowcount > 0

    def get_training_data(
        self,
        min_quality: float | None = None,
        model_id: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Get records with quality scores for predictor training."""
        conn = self._get_conn()
        query = "SELECT * FROM feedback_records WHERE quality_score IS NOT NULL"
        params: list[Any] = []
        if min_quality is not None:
            query += " AND quality_score >= ?"
            params.append(min_quality)
        if model_id is not None:
            query += " AND model_id = ?"
            params.append(model_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_model_stats(self) -> dict[str, dict[str, Any]]:
        """Get aggregated statistics per model from stored records."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT
                model_id,
                COUNT(*) as count,
                AVG(latency_ms) as avg_latency_ms,
                AVG(quality_score) as avg_quality,
                COUNT(quality_score) as quality_samples
            FROM feedback_records
            GROUP BY model_id
        """).fetchall()
        return {
            row["model_id"]: {
                "count": row["count"],
                "avg_latency_ms": row["avg_latency_ms"],
                "avg_quality": row["avg_quality"],
                "quality_samples": row["quality_samples"],
            }
            for row in rows
        }

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a Row to a plain dict, deserializing JSON fields."""
        d = dict(row)
        if "messages_json" in d and d["messages_json"]:
            d["messages"] = json.loads(d["messages_json"])
            del d["messages_json"]
        if "metadata_json" in d:
            d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
            del d["metadata_json"]
        return d

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
