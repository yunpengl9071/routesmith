"""CLI roles command - manage per-role routing policies."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace

import yaml  # noqa: I001

ROLES_SECTION = "role_policies"


def _get_config_path(args: Namespace) -> str:
    return args.config or "routesmith.yaml"


def _load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save_config(path: str, config: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_roles(args: Namespace) -> int:
    """Run the roles command to manage per-role routing policies."""
    config_path = _get_config_path(args)
    config = _load_config(config_path)

    role = args.role or ""

    if args.action == "list":
        return _list_roles(config, args)
    elif args.action == "set":
        return _set_role(config, config_path, role, args)
    elif args.action == "unset":
        return _unset_role(config, config_path, role, args)
    else:
        print(f"Unknown action: {args.action}", file=sys.stderr)
        return 1


def _list_roles(config: dict, args: Namespace) -> int:
    policies = config.get(ROLES_SECTION, {})

    if args.json:
        print(json.dumps(policies, indent=2))
        return 0

    if not policies:
        print("No role policies configured.")
        return 0

    print("\nPer-Role Routing Policies")
    print("=" * 40)
    for role_name, policy in sorted(policies.items()):
        print(f"\n  Role: {role_name}")
        model_pool = policy.get("model_pool", [])
        rewards = policy.get("reward_fns", [])

        if model_pool:
            print(f"    Model pool: {', '.join(model_pool)}")
        else:
            print("    Model pool: (inherits global pool)")

        if rewards:
            print(f"    Reward fns: {', '.join(rewards)}")
        else:
            print("    Reward fns: (inherits global reward fns)")

    print()
    return 0


def _set_role(config: dict, config_path: str, role: str, args: Namespace) -> int:
    if not role:
        print("Error: --role is required for set action", file=sys.stderr)
        return 1

    if ROLES_SECTION not in config:
        config[ROLES_SECTION] = {}

    policy = config[ROLES_SECTION].get(role, {})
    if args.model_pool:
        policy["model_pool"] = args.model_pool
    if args.reward:
        policy["reward_fns"] = args.reward

    config[ROLES_SECTION][role] = policy
    _save_config(config_path, config)

    print(f"Updated policy for role '{role}' in {config_path}")
    return 0


def _unset_role(config: dict, config_path: str, role: str, args: Namespace) -> int:
    if not role:
        print("Error: --role is required for unset action", file=sys.stderr)
        return 1

    if ROLES_SECTION not in config or role not in config[ROLES_SECTION]:
        print(f"No policy found for role '{role}'", file=sys.stderr)
        return 1

    del config[ROLES_SECTION][role]
    if not config[ROLES_SECTION]:
        del config[ROLES_SECTION]
    _save_config(config_path, config)

    print(f"Removed policy for role '{role}' from {config_path}")
    return 0
