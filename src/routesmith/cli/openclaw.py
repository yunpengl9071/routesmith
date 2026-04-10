"""CLI openclaw-config command — generate OpenClaw provider config."""

from __future__ import annotations

import json
import sys
from argparse import Namespace


def run_openclaw_config(args: Namespace) -> int:
    """Output an OpenClaw provider config that points at the RouteSmith proxy."""
    host = args.host.rstrip("/")

    config = {
        "models": {
            "mode": "merge",
            "providers": {
                "routesmith": {
                    "baseUrl": f"{host}/v1",
                    "apiKey": "dummy",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": "auto",
                            "name": "RouteSmith Auto Router",
                            "contextWindow": 128000,
                            "maxTokens": 8192,
                        }
                    ],
                }
            },
        },
        "agents": {
            "defaults": {
                "models": {
                    "routesmith/auto": {
                        "alias": "routesmith"
                    }
                }
            }
        },
    }

    output = json.dumps(config, indent=2)

    if args.output:
        from pathlib import Path
        Path(args.output).write_text(output + "\n")
        print(f"Wrote OpenClaw config to {args.output}")
        print()
        print("Add to your OpenClaw settings.json, or pass with --config.")
        print(f"Make sure RouteSmith is running: routesmith serve")
    else:
        print(output)

    return 0
