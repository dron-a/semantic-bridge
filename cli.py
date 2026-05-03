#!/usr/bin/env python3
"""
semantic-bridge CLI
====================
Reference command line interface for the semantic bridge.

Usage:
    # Ask a question
    python cli.py ask "What was our revenue last month?" \\
        --project ./examples/jaffle_shop \\
        --user analyst_1 \\
        --roles analyst

    # List available metrics for a user
    python cli.py metrics \\
        --project ./examples/jaffle_shop \\
        --user analyst_1 \\
        --roles analyst

    # With access control
    python cli.py ask "What is our gross profit?" \\
        --project ./examples/jaffle_shop \\
        --access-config ./examples/jaffle_shop/access_control.yml \\
        --user analyst_1 \\
        --roles analyst

    # With API key via environment variable
    export ANTHROPIC_API_KEY=sk-ant-...
    python cli.py ask "What is our churn rate?" --project ./examples/jaffle_shop
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from protocol import UserContext
from connectors.dbt_core import DbtCoreConnector
from adapters.claude import ClaudeAdapter
from access.passthrough import PassthroughAccessController
from access.static_yaml import StaticYamlAccessController
from bridge.engine import BridgeEngine, BridgeConfig


def build_engine(args: argparse.Namespace) -> BridgeEngine:
    """Assemble the bridge engine from CLI arguments."""

    # Connector
    connector = DbtCoreConnector(args.project)

    # Access controller
    if hasattr(args, "access_config") and args.access_config:
        access_controller = StaticYamlAccessController(args.access_config)
        print(f"  Access control: static_yaml ({args.access_config})")
    else:
        from access.static_yaml import StaticYamlAccessController
        access_controller = StaticYamlAccessController.from_env_or_default() or PassthroughAccessController()
        print("  Access control: passthrough (dev mode — all metrics visible)")
        

    # LLM adapter
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠ ANTHROPIC_API_KEY not set.")
        print("  Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    adapter = ClaudeAdapter(api_key=api_key)

    config = BridgeConfig(require_health_check=True, log_questions=False)

    return BridgeEngine(
        connector=connector,
        adapter=adapter,
        access_controller=access_controller,
        config=config,
    )


def build_user_context(args: argparse.Namespace) -> UserContext:
    user_id = getattr(args, "user", "cli_user")
    roles = getattr(args, "roles", "").split(",") if getattr(args, "roles", "") else []
    roles = [r.strip() for r in roles if r.strip()]
    return UserContext(user_id=user_id, roles=roles)


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question through the bridge."""
    print("\n── semantic-bridge ──────────────────────────────")
    print(f"  Connector:  dbt_core ({args.project})")
    print(f"  User:       {args.user} | roles: {args.roles or 'none'}")

    engine = build_engine(args)
    user_context = build_user_context(args)

    print(f"\n  Question: {args.question}")
    print("─" * 50)

    answer = engine.ask(args.question, user_context)

    print(f"\n{answer.answer}")
    print("\n── Provenance ───────────────────────────────────")
    print(answer.citation_summary())
    print(f"\n  Resolution confidence: {answer.resolution_confidence:.0%}")
    if answer.out_of_scope_flags:
        print(f"  ⚠ Out of scope: {answer.out_of_scope_flags}")
    print("─" * 50)


def cmd_metrics(args: argparse.Namespace) -> None:
    """List available metrics for a user."""
    print("\n── semantic-bridge: available metrics ───────────")
    print(f"  Project: {args.project}")
    print(f"  User:    {args.user} | roles: {args.roles or 'none'}\n")

    connector = DbtCoreConnector(args.project)

    if hasattr(args, "access_config") and args.access_config:
        access_controller = StaticYamlAccessController(args.access_config)
    else:
        access_controller = PassthroughAccessController()

    user_context = build_user_context(args)
    all_metrics = connector.list_metrics(user_context)
    visible = access_controller.filter_metrics(user_context, all_metrics)
    visible_names = {m.name for m in visible}

    print(f"  {len(visible)}/{len(all_metrics)} metrics accessible\n")

    for m in all_metrics:
        accessible = m.name in visible_names
        status = "✓" if accessible else "✗"
        dims = ", ".join(d.name for d in m.dimensions[:4])
        if len(m.dimensions) > 4:
            dims += f" +{len(m.dimensions) - 4} more"
        print(f"  {status} {m.name} ({m.type})")
        if accessible:
            print(f"      {m.description or 'No description'}")
            print(f"      Dimensions: {dims}")
        else:
            print(f"      [access denied]")
        print()

    print(f"  Definition version: {connector.get_definition_version()}")
    print("─" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="semantic-bridge",
        description="AI-to-semantic-layer bridge — governed answers from your metrics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--project",
        default="./examples/jaffle_shop",
        help="Path to dbt project (default: ./examples/jaffle_shop)",
    )
    shared.add_argument("--user", default="cli_user", help="User ID")
    shared.add_argument(
        "--roles",
        default="",
        help="Comma-separated roles e.g. analyst,finance",
    )
    shared.add_argument(
        "--access-config",
        default=None,
        help="Path to access_control.yml (omit for passthrough)",
    )

    # ask command
    ask_parser = subparsers.add_parser("ask", parents=[shared], help="Ask a question")
    ask_parser.add_argument("question", help="Natural language business question")

    # metrics command
    subparsers.add_parser(
        "metrics", parents=[shared], help="List available metrics for a user"
    )

    args = parser.parse_args()

    if args.command == "ask":
        cmd_ask(args)
    elif args.command == "metrics":
        cmd_metrics(args)


if __name__ == "__main__":
    main()
