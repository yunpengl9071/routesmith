"""CLI serve command implementation."""

from __future__ import annotations

import asyncio
import logging
import sys
from argparse import Namespace
from pathlib import Path

from routesmith import RouteSmith
from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig


def run_serve(args: Namespace) -> int:
    """
    Run the serve command.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code.
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("routesmith.serve")

    # Load configuration
    config_path = Path(args.config)
    routesmith_config = None
    models: list[dict] = []

    if not config_path.exists():
        print(f"No config file found at '{config_path}'.")
        print()
        print("To get started, run:")
        print(f"  routesmith init --output {config_path}")
        print()
        print("This will fetch the OpenRouter model catalog and let you pick")
        print("3–10 models interactively, then write the config file for you.")
        print()
        print("Alternatively, create the file manually — see the docs for format.")
        return 1

    logger.info(f"Loading config from {config_path}")
    try:
        from routesmith.cli.yaml_loader import load_config_file
        routesmith_config, models = load_config_file(config_path)
    except ImportError:
        print("PyYAML not installed. Run: pip install 'routesmith[proxy]'")
        return 1
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return 1

    # Initialize RouteSmith
    rs = RouteSmith(config=routesmith_config)

    # Register models from config
    for model in models:
        model_id = model.pop("model_id")
        rs.register_model(model_id, **model)
        logger.info(f"Registered model: {model_id}")

    if len(rs.registry) == 0:
        print(f"No models registered from '{config_path}'.")
        print()
        print("Add a 'models:' or 'openrouter_models:' section, or re-run:")
        print(f"  routesmith init --output {config_path} --force")
        return 1

    # Create server
    server_config = ServerConfig(
        host=args.host,
        port=args.port,
    )
    server = RouteSmithProxyServer(rs, server_config)

    # Print startup message
    print(f"RouteSmith proxy server starting on http://{args.host}:{args.port}")
    print(f"Registered models: {len(rs.registry)}")
    print()
    print("Endpoints:")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  GET  http://{args.host}:{args.port}/v1/stats")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print()
    print("Press Ctrl+C to stop")
    print()

    # Run server
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        pass

    # Print final stats
    stats = rs.stats
    print()
    print("Session statistics:")
    print(f"  Requests: {stats['request_count']}")
    print(f"  Total cost: ${stats['total_cost_usd']:.6f}")
    print(f"  Savings: ${stats['cost_savings_usd']:.6f} ({stats['savings_percent']:.1f}%)")

    return 0
