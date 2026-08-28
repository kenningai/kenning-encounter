from . import server
import asyncio
import argparse
import logging

from .utils import process_config

logger = logging.getLogger("kenning_encounter")
logger.setLevel(logging.INFO)


def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(
        description="Kenning Encounter (Kenning Encounter) MCP Server"
    )
    parser.add_argument("--db-url", default=None, help="Neo4j connection URL")
    parser.add_argument("--username", default=None, help="Neo4j username")
    parser.add_argument("--password", default=None, help="Neo4j password")
    parser.add_argument("--database", default=None, help="Neo4j database name")
    parser.add_argument("--namespace", default=None, help="Tool namespace prefix")
    parser.add_argument(
        "--transport", default=None, help="Transport type (stdio, sse, streamable-http)"
    )
    parser.add_argument("--server-host", default=None, help="HTTP host (default: 127.0.0.1)")
    parser.add_argument(
        "--server-port", type=int, default=None, help="HTTP port (default: 8003)"
    )
    parser.add_argument("--server-path", default=None, help="HTTP path (default: /mcp/)")
    parser.add_argument(
        "--allow-origins", default=None,
        help="Comma-separated list of allowed CORS origins",
    )
    parser.add_argument(
        "--allowed-hosts", default=None,
        help="Comma-separated list of allowed hosts for DNS rebinding protection",
    )
    parser.add_argument(
        "--read-timeout", type=int, default=None,
        help="Read query timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--infuse-frontier-bias", type=float, default=None,
        help=(
            "Frontier seed bias for the governed infusion blend, 0.0-1.0 "
            "(default: 0.3)"
        ),
    )
    parser.add_argument(
        "--infuse-refresh-turns", type=int, default=None,
        help=(
            "Renewal refresh horizon: turns before a standing body "
            "re-delivers (default: 10; a tunable whose correct value is "
                "an empirical question, not a preference)"
        ),
    )
    parser.add_argument(
        "--matcher-endpoint", default=None,
        help=(
            "Base URL for the meaning matcher's Gemini API (default: "
            "https://generativelanguage.googleapis.com/v1beta). The key "
            "comes from GEMINI_API_KEY, env only; no key -> infuse falls "
            "back to the lexical Extract->Match path"
        ),
    )
    parser.add_argument(
        "--matcher-model", default=None,
        help="Matcher model name (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--matcher-timeout-ms", type=int, default=None,
        help=(
            "Hard matcher timeout in ms (default: 5000, sized to the "
            "measured reasoning tail; fallback on expiry)"
        ),
    )
    parser.add_argument(
        "--matcher-sidecar", default=None,
        help=(
            "Path to the compressed-meanings sidecar (default: "
            "models/meaning_sidecar.json; missing -> lexical fallback)"
        ),
    )

    args = parser.parse_args()
    config = process_config(args)
    asyncio.run(server.main(**config))


__all__ = ["main", "server"]
