"""Audit log storage for routing decisions."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any


class AuditStorage:
    """Stores structured audit log entries for routing decisions.

    Each entry captures the full decision context including candidate scores,
    strategy used, selected model, and request metadata.
    """

    def __init__(self, db_path: str = "routesmith_feedback.db") -> None:
        self._db_path = db_path
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            c = sqlite3.connect(self._db_path)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
            self._init_db()
        conn: sqlite3.Connection = self._local.conn
        return conn

    def _init_db(self) -> None:
        self._local.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                project_id TEXT,
                timestamp TEXT NOT NULL,
                model_selected TEXT NOT NULL,
                routing_strategy TEXT NOT NULL,
                routing_reason TEXT,
                routing_latency_ms REAL,
                estimated_cost_usd REAL,
                counterfactual_cost_usd REAL,
                cost_savings_usd REAL,
                models_considered TEXT,
                cache_hit INTEGER DEFAULT 0,
                fallback_from TEXT,
                candidate_scores TEXT,
                context_metadata TEXT,
                agent_role TEXT,
                conversation_id TEXT
            )
        """)
        self._local.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_request_id
            ON audit_logs(request_id)
        """)
        self._local.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_project_id
            ON audit_logs(project_id)
        """)
        self._local.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp
            ON audit_logs(timestamp)
        """)

    def record(
        self,
        request_id: str,
        model_selected: str,
        routing_strategy: str,
        routing_reason: str,
        routing_latency_ms: float,
        estimated_cost_usd: float,
        counterfactual_cost_usd: float,
        cost_savings_usd: float,
        models_considered: list[str],
        cache_hit: bool = False,
        fallback_from: str | None = None,
        candidate_scores: dict | None = None,
        context_metadata: dict | None = None,
        agent_role: str | None = None,
        conversation_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """Record an audit log entry for a routing decision."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO audit_logs (
                request_id, project_id, timestamp, model_selected,
                routing_strategy, routing_reason, routing_latency_ms,
                estimated_cost_usd, counterfactual_cost_usd, cost_savings_usd,
                models_considered, cache_hit, fallback_from,
                candidate_scores, context_metadata, agent_role, conversation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            project_id,
            datetime.now(timezone.utc).isoformat(),
            model_selected,
            routing_strategy,
            routing_reason,
            routing_latency_ms,
            estimated_cost_usd,
            counterfactual_cost_usd,
            cost_savings_usd,
            json.dumps(models_considered),
            1 if cache_hit else 0,
            fallback_from,
            json.dumps(candidate_scores) if candidate_scores else None,
            json.dumps(context_metadata) if context_metadata else None,
            agent_role,
            conversation_id,
        ))
        conn.commit()

    def get_records(
        self,
        limit: int = 50,
        project_id: str | None = None,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve audit log entries with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []

        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if model_id:
            conditions.append("model_selected = ?")
            params.append(model_id)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        rows = conn.execute(
            f"SELECT * FROM audit_logs {where} ORDER BY timestamp DESC LIMIT ?",
            [*params, limit],
        ).fetchall()

        results = []
        for r in rows:
            entry = dict(r)
            if entry["models_considered"]:
                entry["models_considered"] = json.loads(entry["models_considered"])
            if entry["candidate_scores"]:
                entry["candidate_scores"] = json.loads(entry["candidate_scores"])
            if entry["context_metadata"]:
                entry["context_metadata"] = json.loads(entry["context_metadata"])
            entry["cache_hit"] = bool(entry["cache_hit"])
            results.append(entry)

        return results

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


class NullAuditStorage:
    """No-op audit storage for when audit logging is disabled."""

    def record(self, **kwargs: Any) -> None:
        pass

    def get_records(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        pass
