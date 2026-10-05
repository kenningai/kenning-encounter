import argparse
import os
import logging
from pathlib import Path
from typing import Any, LiteralString, cast

from neo4j import AsyncDriver, Query

logger = logging.getLogger("kenning_encounter")
logger.setLevel(logging.INFO)

# -- Cypher Query Loading -----------------------------------------------------

_CYPHER_DIR = Path(__file__).parent / "_cypher"
_CYPHER_CACHE: dict[str, str] = {}


def lit(query: str) -> LiteralString:
    """Mark a runtime-built query/text string as acceptable to driver APIs
    typed `LiteralString`. Use ONLY for trusted, structurally-built strings —
    never for user values, which always pass through $params.

    The neo4j stubs type query text as LiteralString to discourage dynamic
    SQL; our templating legitimately produces str. This centralizes the cast.
    """
    return cast(LiteralString, query)


def load_cypher(name: str, **replacements: str) -> LiteralString:
    """Load a Cypher query from _cypher/{name}.cypher.

    Caches file contents on first load. Template placeholders use
    __name__ convention (e.g. __label__, __rel_type__) to avoid
    collision with Cypher's {prop: $param} syntax.

    Structural fragments only (labels, relationship types, dynamic SET
    clauses) go through __name__ placeholders. User values ALWAYS go
    through $params — never substitute user input here.

    Args:
        name: Query filename without .cypher extension
        **replacements: Template substitutions (key -> __key__ in file)
    """
    if name not in _CYPHER_CACHE:
        path = _CYPHER_DIR / f"{name}.cypher"
        _CYPHER_CACHE[name] = path.read_text()
    query = _CYPHER_CACHE[name]
    for key, value in replacements.items():
        query = query.replace(f"__{key}__", value)
    return cast(LiteralString, query)


# -- Utilities ----------------------------------------------------------------

def format_namespace(namespace: str) -> str:
    """Format namespace by ensuring it ends with a hyphen if not empty."""
    if namespace:
        return namespace if namespace.endswith("-") else namespace + "-"
    return ""


async def _is_write_query(query: str, driver: AsyncDriver, database: str | None = None) -> bool:
    """Detect write intent via Neo4j's planner instead of regex.

    EXPLAIN compiles the query without executing it; the summary's
    query_type is one of 'r', 'w', 'rw', 's' (schema-write). Anything
    containing 'w' is a write. This avoids the false positives that
    string-matching produces (CREATE inside string literals, identifiers
    like creation_date, comments, etc.).
    """
    _, summary, _ = await driver.execute_query(
        Query(lit("EXPLAIN " + query)),
        database_=database,
    )
    return "w" in (summary.query_type or "")


def _value_sanitize(d: Any, list_limit: int = 128) -> Any:
    """Sanitize input by removing embedding-like values and oversized lists.

    Strips properties that would occupy significant context space without
    contributing to LLM reasoning (e.g. vector embeddings). Downstream graph
    analytics traverse topology, not stored vectors — embedding-shaped values
    have no business in this substrate, and the sanitizer enforces that.
    """
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                sanitized_value = _value_sanitize(value)
                if sanitized_value is not None:
                    new_dict[key] = sanitized_value
            elif isinstance(value, list):
                if len(value) < list_limit:
                    sanitized_value = _value_sanitize(value)
                    if sanitized_value is not None:
                        new_dict[key] = sanitized_value
            else:
                new_dict[key] = value
        return new_dict
    elif isinstance(d, list):
        if len(d) < list_limit:
            return [
                _value_sanitize(item)
                for item in d
                if _value_sanitize(item) is not None
            ]
        else:
            return None
    else:
        return d


def process_config(args: argparse.Namespace) -> dict[str, Any]:
    """Process CLI arguments and environment variables into a config dict.

    Resolution order: CLI args -> environment variables -> defaults.
    """
    config: dict[str, Any] = {}

    # Neo4j connection
    config["neo4j_uri"] = (
        args.db_url
        or os.getenv("NEO4J_URL")
        or os.getenv("NEO4J_URI")
        or _default("neo4j_uri", "bolt://localhost:7687")
    )
    config["neo4j_user"] = (
        args.username
        or os.getenv("NEO4J_USERNAME")
        or _default("neo4j_user", "neo4j")
    )
    config["neo4j_password"] = (
        args.password
        or os.getenv("NEO4J_PASSWORD")
        or _default("neo4j_password", "password")
    )
    config["neo4j_database"] = (
        args.database
        or os.getenv("NEO4J_DATABASE")
        or _default("neo4j_database", "kenning-encounter")
    )

    # Transport
    config["transport"] = (
        args.transport
        or os.getenv("NEO4J_TRANSPORT")
        or _default("transport", "stdio")
    )

    # Server host/port/path
    config["host"] = (
        args.server_host
        or os.getenv("NEO4J_MCP_SERVER_HOST")
        or (None if config["transport"] == "stdio" else "127.0.0.1")
    )
    config["port"] = (
        args.server_port
        or _env_int("NEO4J_MCP_SERVER_PORT")
        or (None if config["transport"] == "stdio" else 8003)
    )
    config["path"] = (
        args.server_path
        or os.getenv("NEO4J_MCP_SERVER_PATH")
        or (None if config["transport"] == "stdio" else "/mcp/")
    )

    # CORS and host security
    config["allow_origins"] = _parse_csv(
        args.allow_origins, "NEO4J_MCP_SERVER_ALLOW_ORIGINS", []
    )
    config["allowed_hosts"] = _parse_csv(
        args.allowed_hosts,
        "NEO4J_MCP_SERVER_ALLOWED_HOSTS",
        ["localhost", "127.0.0.1"],
    )

    # Namespace (tool-name prefixing across co-loaded MCPs — NOT frame sharing)
    config["namespace"] = (
        args.namespace or os.getenv("NEO4J_NAMESPACE") or ""
    )

    # Read timeout
    config["read_timeout"] = (
        args.read_timeout
        or _env_int("NEO4J_READ_TIMEOUT")
        or 30
    )

    # Infusion governor: the frontier seed bias in the governed blend
    # (focal @1.0 + frontier @bias). A tunable whose correct value is an
    # empirical question — pre-registered as such; 0.3 is the design default.
    bias = args.infuse_frontier_bias
    if bias is None:
        bias = _env_float("NEO4J_INFUSE_FRONTIER_BIAS")
    config["infuse_frontier_bias"] = 0.3 if bias is None else max(0.0, min(1.0, bias))

    # Infusion governor: the renewal refresh horizon — turns before a
    # standing body re-delivers (redundancy re-injection against
    # standing-picture attenuation). Hand-set 10 until measurement places
    # it from the attenuation curve; do not tune it by feel.
    rt = args.infuse_refresh_turns
    if rt is None:
        rt = _env_int("NEO4J_INFUSE_REFRESH_TURNS")
    config["infuse_refresh_turns"] = 10 if rt is None else max(1, min(100, rt))

    # The meaning matcher (v0.9.0) — infuse's full-mode seed source: one
    # model compresses every node's meaning offline into the sidecar (built
    # by scripts/build_meaning_sidecar.py — never node properties), and the
    # SAME model matches trajectory ↔ meanings per prompt (single-voice
    # authorship is what closes the two-idiolect gap; measured T0 recall 7/8
    # vs the 2/8 lexical baseline). The key comes from GEMINI_API_KEY (env
    # only, never a CLI flag — argv is visible in process listings); without
    # it, infuse falls back to the lexical Extract->Match path and says so.
    # Timeout default 5000 ms — a deliberate, measured renegotiation: the
    # arc-dependent cases that justify the mechanism take ~3.5s of
    # reasoning, and a cap that cuts the tail kills exactly the calls
    # buying the most. gemini-3.5-flash-lite is the most performant capable
    # model reachable without hardware/self-hosting; residual network TTFT
    # is the price of that portability.
    config["matcher_api_key"] = os.getenv("GEMINI_API_KEY") or ""
    config["matcher_endpoint"] = (
        args.matcher_endpoint
        or os.getenv("NEO4J_MATCHER_ENDPOINT")
        or "https://generativelanguage.googleapis.com/v1beta"
    )
    config["matcher_model"] = (
        args.matcher_model
        or os.getenv("NEO4J_MATCHER_MODEL")
        or "gemini-3.5-flash-lite"
    )
    # Gemini voice's thinkingLevel. "" sends none, the measured
    # flash-lite behaviour. Thinking-only models (3.8 Flash cannot turn it
    # off, and refuses "minimal") need "low" to stay inside the budget. An
    # unknown value is refused here, at startup, not by a 400 per prompt.
    th = getattr(args, "matcher_thinking", None)
    if th is None:
        th = os.getenv("NEO4J_MATCHER_THINKING", "")
    th = th.strip().lower()
    if th not in ("", "minimal", "low", "medium", "high"):
        raise ValueError(
            f"NEO4J_MATCHER_THINKING={th!r}: use minimal, low, medium, high or empty"
        )
    config["matcher_thinking"] = th
    mt = args.matcher_timeout_ms
    if mt is None:
        mt = _env_int("NEO4J_MATCHER_TIMEOUT_MS")
    config["matcher_timeout_ms"] = 8500 if mt is None else max(100, min(10_000, mt))
    # Hedge point: a lead voice still silent this long after it
    # began starts the next voice alongside it; first valid answer wins.
    # 0 = sequential.
    hm = getattr(args, "matcher_hedge_ms", None)
    if hm is None:
        hm = _env_int("NEO4J_MATCHER_HEDGE_MS")
    config["matcher_hedge_ms"] = 3700 if hm is None else max(0, min(10_000, hm))
    # Selections per matcher call. Output tokens dominate a warm call on a
    # local backend (a field deployment measured decode at 82% of it), so
    # this is the matcher's main latency lever. Ceiling = the seed limit:
    # infuse keeps at most that many seeds, so more selections buy nothing.
    tn = args.matcher_top_n
    if tn is None:
        tn = _env_int("NEO4J_MATCHER_TOP_N")
    config["matcher_top_n"] = 12 if tn is None else max(1, min(12, tn))
    config["matcher_sidecar"] = (
        args.matcher_sidecar
        or os.getenv("NEO4J_MATCHER_SIDECAR")
        or "models/meaning_sidecar.json"
    )

    # The second voice (v0.17.0): an OpenAI model that compresses its OWN
    # sidecar and matches against it. The NEO4J_MATCHER_MODEL/_ENDPOINT
    # family above stays the Gemini voice's, so an existing deployment's
    # settings keep their meaning. Key from OPENAI_API_KEY, env only; with
    # no key the voice is simply absent and nothing changes.
    config["matcher_openai_api_key"] = os.getenv("OPENAI_API_KEY") or ""
    config["matcher_openai_model"] = (
        getattr(args, "matcher_openai_model", None)
        or os.getenv("NEO4J_MATCHER_OPENAI_MODEL")
        or "gpt-6-luna"
    )
    config["matcher_openai_endpoint"] = (
        getattr(args, "matcher_openai_endpoint", None)
        or os.getenv("NEO4J_MATCHER_OPENAI_ENDPOINT")
        or "https://api.openai.com/v1"
    )
    # Which voice leads: gemini | openai | alternate (chosen per locus from
    # the session id, so a comparison between voices can be made in the
    # window). Unknown values fall back to gemini, the pre-v0.17.0 voice,
    # rather than refusing to start over a typo in an infusion setting.
    lead = (
        getattr(args, "matcher_lead", None)
        or os.getenv("NEO4J_MATCHER_LEAD")
        or "gemini"
    ).strip().lower()
    if lead not in ("gemini", "openai", "alternate"):
        logger.warning(
            f"NEO4J_MATCHER_LEAD={lead!r} is not gemini|openai|alternate; using gemini"
        )
        lead = "gemini"
    config["matcher_lead"] = lead
    # Seconds a failed voice is tried after the healthy ones instead of first.
    cd = getattr(args, "matcher_cooldown_s", None)
    if cd is None:
        cd = _env_int("NEO4J_MATCHER_COOLDOWN_S")
    config["matcher_cooldown_s"] = 300 if cd is None else max(0, min(3600, cd))
    # Shadow matching: every other configured voice also matches each
    # prompt, in the background, and the pair is logged — never delivered.
    # Off by default: it costs a second provider call per prompt.
    sh = getattr(args, "matcher_shadow", None)
    if sh is None:
        sh = (os.getenv("NEO4J_MATCHER_SHADOW") or "").strip().lower() in (
            "1", "true", "yes", "on",
        )
    config["matcher_shadow"] = bool(sh)
    config["matcher_shadow_log"] = (
        getattr(args, "matcher_shadow_log", None)
        or os.getenv("NEO4J_MATCHER_SHADOW_LOG")
        or str(Path(config["matcher_sidecar"]).with_name("matcher-shadow.jsonl"))
    )

    # Server profile: "full" (default) or "reader", the read-only face for
    # every harness but the trajectory's own (reader.py). "author" is the
    # v0.18 name, kept so older operator instructions still work.
    # Withholding is opt-in: a reader sees the whole trajectory unless an
    # experiment names loci to withhold, because a view with pieces of the
    # past removed is not a view of the trajectory.
    prof = (
        getattr(args, "server_profile", None)
        or os.getenv("NEO4J_MCP_SERVER_PROFILE")
        or "full"
    ).strip().lower()
    if prof == "author":
        prof = "reader"
    if prof not in ("full", "reader"):
        raise SystemExit(f"NEO4J_MCP_SERVER_PROFILE={prof!r} is not full|reader")
    config["server_profile"] = prof
    from .reader import locus_names
    hide = (getattr(args, "reader_hide_loci", None)
            or os.getenv("NEO4J_READER_HIDE_LOCI")
            or os.getenv("NEO4J_AUTHOR_HIDE_LOCI") or "").strip()
    config["reader_hide_loci"] = [] if hide.lower() in ("", "none") else locus_names(hide)

    # Streamable-HTTP sessions: stateful by default, stateless on opt-in.
    # v0.3.0 made them always stateful because the transport session WAS the
    # locus key. Since v0.12.0 the locus identity is the harness session id,
    # held in the graph; the transport session keys only a cache. A stateful
    # server loses its sessions on restart, and a client or bridge that does
    # not re-initialize on the spec's 404 then fails every call until someone
    # reconnects it by hand. Stateless issues no session, so there is nothing
    # to go stale; every locus-scoped call must then pass session_id.
    st = getattr(args, "server_stateless", None)
    if st is None:
        st = (os.getenv("NEO4J_MCP_SERVER_STATELESS") or "").strip().lower() in (
            "1", "true", "yes", "on")
    config["stateless_http"] = bool(st)

    return config


def _default(key: str, value: str) -> str:
    logger.warning(f"No {key} provided. Using default: {value}")
    return value


def _env_int(var: str) -> int | None:
    val = os.getenv(var)
    return int(val) if val is not None else None


def _env_float(var: str) -> float | None:
    val = os.getenv(var)
    return float(val) if val is not None else None


def _parse_csv(
    cli_val: str | None, env_var: str, default: list[str]
) -> list[str]:
    if cli_val is not None:
        return [s.strip() for s in cli_val.split(",") if s.strip()]
    env_val = os.getenv(env_var)
    if env_val is not None:
        return [s.strip() for s in env_val.split(",") if s.strip()]
    return default
