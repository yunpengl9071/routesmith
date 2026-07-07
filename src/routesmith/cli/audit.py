"""CLI audit command - view routing decision audit log."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from datetime import datetime


def run_audit(args: Namespace) -> int:
    """Run the audit command to view routing decision logs."""
    from routesmith.feedback.audit import AuditStorage

    db_path = args.db or "routesmith_feedback.db"

    try:
        storage = AuditStorage(db_path)
    except Exception as e:
        print(f"Error: Could not open audit storage at {db_path}: {e}", file=sys.stderr)
        return 1

    project = args.project or None
    model = args.model or None

    records = storage.get_records(
        limit=args.limit,
        project_id=project if project else None,
        model_id=model if model else None,
    )

    if args.json:
        print(json.dumps(records, indent=2, default=str))
        return 0

    if not records:
        print("No audit records found.")
        return 0

    header = "Routing Decision Audit Log"
    if project:
        header += f" (project: {project})"
    if model:
        header += f" (model: {model})"
    print(f"\n{header}")
    print("=" * len(header))
    print()

    for i, r in enumerate(records, 1):
        ts = r.get("timestamp", "?")
        try:
            dt = datetime.fromisoformat(ts)
            ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            pass

        proj = r.get("project_id") or "(default)"
        role = r.get("agent_role") or "(none)"
        conv = r.get("conversation_id") or ""

        print(f"  #{i}  [{ts}]  proj={proj}")
        print(f"      model={r['model_selected']}  strategy={r['routing_strategy']}  cache_hit={'yes' if r.get('cache_hit') else 'no'}")
        print(f"      reason={r.get('routing_reason', '')}")
        cost = r.get("estimated_cost_usd", 0)
        saved = r.get("cost_savings_usd", 0)
        print(f"      cost=${cost:.6f}  saved=${saved:.6f}  latency={r.get('routing_latency_ms', 0):.1f}ms")
        models = r.get("models_considered", [])
        if models:
            print(f"      candidates={', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
        if role != "(none)":
            print(f"      role={role}")
        if conv:
            print(f"      conversation={conv}")
        cs = r.get("candidate_scores")
        if cs:
            scores_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(cs.items())[:6])
            print(f"      scores={scores_str}{'...' if len(cs) > 6 else ''}")
        print()

    return 0
