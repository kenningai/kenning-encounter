#!/usr/bin/env python3
"""Build the meaning sidecar — compressed node meanings for the matcher.

Offline batch (latency irrelevant): reads every semantic + reference node
(name + description, never the process layer) from Neo4j read-only, compresses each
through one VOICE — the same model that will match against it, which is
load-bearing: single-voice authorship is what kills the two-idiolect
problem — and writes that voice's stamped sidecar beside --out
(models/meaning_sidecar.<provider>-<model>.json). Gemini by default;
--provider openai builds the OpenAI voice's.

Idempotent by content hash: unchanged nodes keep their meaning, new or
edited nodes recompute, departed nodes drop. Nothing is ever written to
Neo4j, and the sidecar never enters node properties — the graph stays
authored by the trajectory alone (the settled constraint).

NOTE (v0.9.0): in a deployed stack this script is a DEV/OFFLINE tool, not
an operational step. The sidecar lives in the stack's named volume and the
server maintains it itself — the on-write trigger meaning-makes new nodes,
and the startup reconcile sweep (hash- and PROMPT_VERSION-aware) rebuilds
any gap from the graph, including a first boot from empty. This script
writes a HOST-side file the container does not read; use it for local
development against a bind mount, or to pre-generate a sidecar you then
copy into the volume (docker cp <file> kenning_encounter-mcp:/app/models/).

Usage (plain argv, any shell):
    uv run python scripts/build_meaning_sidecar.py
    uv run python scripts/build_meaning_sidecar.py --out models/meaning_sidecar.json --concurrency 8
    uv run python scripts/build_meaning_sidecar.py --provider openai
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from kenning_encounter.meaning import (  # noqa: E402
    DEFAULT_MODELS,
    KEY_ENV,
    PROVIDERS,
    MeaningIndex,
    MeaningUnavailable,
    Voice,
    compress_meaning,
    content_hash,
    sidecar_path,
)


def _env_from_dotenv() -> None:
    env_path = REPO / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def _fetch_nodes(uri: str, user: str, password: str, database: str):
    from neo4j import AsyncGraphDatabase, RoutingControl

    from kenning_encounter.kenning_encounter import not_process_node

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password), database=database)
    try:
        res = await driver.execute_query(
            # The server's reconcile query, so a bulk build and a startup
            # sweep agree on the corpus (no process layer: Locus as well as
            # Encounter) and on its order (creation order numbers a voice's
            # sidecar the way the server would have built it from empty).
            f"MATCH (n) WHERE {not_process_node('n')} AND n.name IS NOT NULL "
            "RETURN n.name AS name, labels(n)[0] AS type, "
            "       coalesce(n.description, '') AS description "
            "ORDER BY n.t_created, n.name",
            routing_=RoutingControl.READ,
        )
        return [dict(r) for r in res.records]
    finally:
        await driver.close()


async def _compress(sem, voice: Voice, node: dict) -> tuple[str, str]:
    """The server's own compression call, with retries for a batch run."""
    async with sem:
        for attempt in range(3):
            try:
                return node["name"], await compress_meaning(node, voice)
            except MeaningUnavailable as e:
                if attempt == 2:
                    raise RuntimeError(f"{node['name']}: {e}") from e
                await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{node['name']}: no output after retries")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models/meaning_sidecar.json")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--provider", choices=PROVIDERS, default="gemini")
    parser.add_argument(
        "--model", default=None,
        help="Model for the voice (default: the provider's matcher default)",
    )
    parser.add_argument(
        "--thinking", default=None,
        help=(
            "Gemini thinkingLevel (minimal|low|medium|high); default "
            "NEO4J_MATCHER_THINKING, else none. Use the level the server runs."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute every node, ignoring content hashes",
    )
    args = parser.parse_args()

    _env_from_dotenv()
    key_env = KEY_ENV[args.provider]
    api_key = os.environ.get(key_env, "")
    if not api_key:
        print(f"{key_env} not set (env or .env)", file=sys.stderr)
        return 1
    thinking = (
        args.thinking if args.thinking is not None
        else os.environ.get("NEO4J_MATCHER_THINKING", "")
    ).strip().lower() if args.provider == "gemini" else ""
    voice = Voice(args.provider, args.model or DEFAULT_MODELS[args.provider],
                  api_key, thinking=thinking)

    uri = args.db_url or os.environ.get("NEO4J_URL") or "bolt://localhost:7687"
    password = (
        os.environ.get("NEO4J_PASSWORD")
        or os.environ.get("NEO4J_KENNING_ENCOUNTER_PASSWORD") or ""
    )
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    database = os.environ.get("NEO4J_DATABASE", "kenning_encounter")

    nodes = await _fetch_nodes(uri, user, password, database)
    print(f"corpus: {len(nodes)} nodes (semantic + reference, no process layer)")

    out_path = sidecar_path(args.out, voice)
    index = MeaningIndex(out_path, voice.id)
    # Only this voice's own earlier meanings are reused.
    existing: dict = {} if args.force else index.known_nodes()

    result: dict[str, dict] = {}
    todo: list[dict] = []
    for n in nodes:
        h = content_hash(n["name"], n["description"])
        prev = existing.get(n["name"])
        if prev and prev.get("hash") == h and prev.get("meaning"):
            result[n["name"]] = {"type": n["type"], "hash": h,
                                 "meaning": prev["meaning"]}
        else:
            n["hash"] = h
            todo.append(n)
    print(f"unchanged: {len(result)} | to compress: {len(todo)}")

    if todo:
        sem = asyncio.Semaphore(args.concurrency)
        done = 0
        for coro in asyncio.as_completed([_compress(sem, voice, n) for n in todo]):
            name, meaning = await coro
            node = next(n for n in todo if n["name"] == name)
            result[name] = {"type": node["type"], "hash": node["hash"],
                            "meaning": meaning}
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"  compressed {done}/{len(todo)}")

    index._write(result)
    total_chars = sum(len(v["meaning"]) for v in result.values())
    print(f"wrote {out_path} ({voice.id}) — {len(result)} meanings, ~{total_chars:,} chars "
          f"(~{total_chars // 4:,} tokens as the matcher prefix)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
