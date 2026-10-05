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
        "--server-stateless", action="store_true", default=None,
        help=(
            "Streamable-HTTP without sessions: a restart leaves nothing stale "
            "for a client to hold; every locus-scoped call must pass session_id"
        ),
    )
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
        "--matcher-top-n", type=int, default=None,
        help=(
            "Selections per matcher call (default: 12, the seed limit; "
            "lower cuts decode time, the dominant cost on a local backend)"
        ),
    )
    parser.add_argument(
        "--matcher-sidecar", default=None,
        help=(
            "Path to the compressed-meanings sidecar (default: "
            "models/meaning_sidecar.json; missing -> lexical fallback)"
        ),
    )

    parser.add_argument(
        "--matcher-lead", default=None,
        choices=["gemini", "openai", "alternate"],
        help=(
            "Which matcher voice leads (default: gemini). 'alternate' picks "
            "the lead per locus from the session id. The other configured "
            "voice is the fallback"
        ),
    )
    parser.add_argument(
        "--matcher-openai-model", default=None,
        help=(
            "OpenAI matcher voice model (default: gpt-6-luna). The key "
            "comes from OPENAI_API_KEY, env only; no key -> no OpenAI voice"
        ),
    )
    parser.add_argument(
        "--matcher-openai-endpoint", default=None,
        help="OpenAI API base URL (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--matcher-thinking", default=None,
        help=(
            "Gemini voice thinkingLevel: minimal | low | medium | high, or "
            "empty to send none (NEO4J_MATCHER_THINKING; default empty)"
        ),
    )
    parser.add_argument(
        "--matcher-hedge-ms", type=int, default=None,
        help=(
            "Start the next matcher voice alongside a lead still silent "
            "this many ms after it began; 0 = sequential (default: 3700)"
        ),
    )
    parser.add_argument(
        "--matcher-cooldown-s", type=int, default=None,
        help=(
            "Seconds a failed matcher voice is tried after the healthy ones "
            "(default: 300)"
        ),
    )
    parser.add_argument(
        "--matcher-shadow", action="store_true", default=None,
        help=(
            "Also match every prompt with the non-delivering voice, in the "
            "background, and log the pair (NEO4J_MATCHER_SHADOW)"
        ),
    )
    parser.add_argument(
        "--matcher-shadow-log", default=None,
        help=(
            "Shadow log path (default: matcher-shadow.jsonl beside the "
            "sidecar)"
        ),
    )

    parser.add_argument(
        "--server-profile", default=None, choices=["full", "reader", "author"],
        help=(
            "full (default) or reader: a read-only allowlist of tools for "
            "every harness but the trajectory's own; author is the v0.18 "
            "name for reader (NEO4J_MCP_SERVER_PROFILE)"
        ),
    )
    parser.add_argument(
        "--reader-hide-loci", "--author-hide-loci", dest="reader_hide_loci",
        default=None,
        help=(
            "Comma-separated session ids or 'Locus ...' names whose nodes the "
            "reader profile withholds, for an experiment; default none "
            "(NEO4J_READER_HIDE_LOCI)"
        ),
    )

    args = parser.parse_args()
    config = process_config(args)
    asyncio.run(server.main(**config))


__all__ = ["main", "server"]
