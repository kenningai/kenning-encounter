#!/usr/bin/env python3
"""Build the meaning sidecar — compressed node meanings for the matcher.

Offline batch (latency irrelevant): reads every semantic + reference node
(name + description, NOT Encounters) from Neo4j read-only, compresses each
through gemini-3.5-flash-lite — THE SAME MODEL that matches, which is
load-bearing: single-voice authorship is what kills the two-idiolect
problem — and writes models/meaning_sidecar.json.

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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from kenning_encounter.meaning import MEANING_PROMPT, content_hash  # noqa: E402

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
_MODEL = "gemini-3.5-flash-lite"
_DESC_CAP = 4000
_MEANING_MAX_TOKENS = 256


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

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password), database=database)
    try:
        res = await driver.execute_query(
            "MATCH (n) WHERE NOT n:Encounter AND n.name IS NOT NULL "
            "RETURN n.name AS name, labels(n)[0] AS type, "
            "       coalesce(n.description, '') AS description ORDER BY name",
            routing_=RoutingControl.READ,
        )
        return [dict(r) for r in res.records]
    finally:
        await driver.close()


async def _compress(client, sem, api_key: str, node: dict) -> tuple[str, str]:
    prompt = MEANING_PROMPT.format(
        type=node["type"], name=node["name"],
        description=node["description"][:_DESC_CAP],
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0, "maxOutputTokens": _MEANING_MAX_TOKENS,
        },
    }
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{_ENDPOINT}/models/{_MODEL}:generateContent",
                    json=payload, headers={"x-goog-api-key": api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                parts = data["candidates"][0]["content"].get("parts", [])
                text = " ".join(
                    "".join(p.get("text", "") for p in parts).split()
                )
                if text:
                    return node["name"], text
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"{node['name']}: {e}") from e
                await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{node['name']}: empty output after retries")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models/meaning_sidecar.json")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute every node, ignoring content hashes",
    )
    args = parser.parse_args()

    _env_from_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set (env or .env)", file=sys.stderr)
        return 1

    uri = args.db_url or os.environ.get("NEO4J_URL") or "bolt://localhost:7687"
    password = (
        os.environ.get("NEO4J_PASSWORD")
        or os.environ.get("NEO4J_KENNING_ENCOUNTER_PASSWORD") or ""
    )
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    database = os.environ.get("NEO4J_DATABASE", "kenning_encounter")

    nodes = await _fetch_nodes(uri, user, password, database)
    print(f"corpus: {len(nodes)} nodes (semantic + reference, no Encounters)")

    out_path = Path(args.out)
    existing: dict = {}
    if out_path.is_file() and not args.force:
        try:
            existing = json.loads(out_path.read_text()).get("nodes", {})
        except Exception:
            existing = {}

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
        import httpx

        sem = asyncio.Semaphore(args.concurrency)
        async with httpx.AsyncClient(timeout=30.0) as client:
            done = 0
            for coro in asyncio.as_completed(
                [_compress(client, sem, api_key, n) for n in todo]
            ):
                name, meaning = await coro
                node = next(n for n in todo if n["name"] == name)
                result[name] = {"type": node["type"], "hash": node["hash"],
                                "meaning": meaning}
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(f"  compressed {done}/{len(todo)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"model": _MODEL, "prompt_version": "v2", "nodes": result},
        indent=1, ensure_ascii=False,
    ))
    total_chars = sum(len(v["meaning"]) for v in result.values())
    print(f"wrote {out_path} — {len(result)} meanings, ~{total_chars:,} chars "
          f"(~{total_chars // 4:,} tokens as the matcher prefix)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
