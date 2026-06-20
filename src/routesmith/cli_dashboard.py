"""Interactive TUI dashboard for RouteSmith.

Provides a live-refreshing terminal dashboard showing cost savings,
model usage, and verification stats. Requires the 'textual' library.

To install: pip install routesmith[tui]
To run: routesmith dashboard
"""


def run_dashboard(db_path: str = "routesmith_feedback.db") -> int:
    """Launch the Textual TUI dashboard.

    Falls back to a simple stats print if textual is not installed.

    Args:
        db_path: Path to the SQLite feedback database.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    try:
        from textual.app import App
        from textual.containers import Container
        from textual.widgets import Footer, Header, Static
    except ImportError:
        print("textual not installed. Install with: pip install routesmith[tui]")
        print()
        print("Falling back to stats --local:")
        from argparse import Namespace

        from routesmith.cli.stats import run_stats as _run_stats
        return _run_stats(Namespace(local=True, json=False, watch=True, db=db_path))

    from routesmith.feedback.storage import FeedbackStorage

    storage = FeedbackStorage(db_path)

    class RouteSmithDashboard(App):
        """Live RouteSmith monitoring dashboard."""

        CSS = """
        Screen { align: center middle; }
        #stats { width: 60; }
        """

        def compose(self):
            yield Header()
            with Container(id="stats"):
                yield Static("Loading...")
            yield Footer()

        def on_mount(self):
            self.set_interval(2.0, self.refresh_stats)

        def refresh_stats(self):
            try:
                records = storage.get_all_records(limit=10000)
                total = len(records)
                cost = sum(float(r.get("estimated_cost_usd", 0) or 0) for r in records)
                by_model: dict[str, int] = {}
                for r in records:
                    model = r.get("model_id", "unknown")
                    by_model[model] = by_model.get(model, 0) + 1

                top_models = sorted(by_model.items(), key=lambda x: -x[1])[:5]
                model_lines = "\n".join(
                    f"  {m}: {c} requests" for m, c in top_models
                )

                stats_widget = self.query_one("#stats", Static)
                stats_widget.update(
                    f"\n  [bold]RouteSmith Dashboard[/bold]\n"
                    f"  {'─' * 40}\n"
                    f"  Total Requests: {total}\n"
                    f"  Total Cost: ${cost:.4f}\n"
                    f"  {'─' * 40}\n"
                    f"  Top Models:\n{model_lines or '  No data'}\n"
                )
            except Exception as e:
                stats_widget = self.query_one("#stats", Static)
                stats_widget.update(f"Error: {e}")

    try:
        app = RouteSmithDashboard()
        app.run()
        return 0
    except Exception as e:
        print(f"Dashboard error: {e}", file=sys.stderr)
        return 1


import sys  # noqa: E402  # placed here for use in run_dashboard
