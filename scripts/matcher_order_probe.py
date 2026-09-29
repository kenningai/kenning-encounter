#!/usr/bin/env python3
"""How does the matcher's prefix ORDER change which nodes it selects?

The matcher now numbers its prefix in sidecar insertion order instead of by
name, so a write no longer renumbers the block and voids the prefix cache.
The numbering never leaves the call, but the model reads the list in a
different order, with the newest nodes last. This DESCRIBES how selection
shifts under that change. It does not and cannot say whether an infusion
carried more or less meaning: meaning is not binary (two different selections
can each bear on a trajectory, differently), and the only frame that can
judge an infusion's efficacy is the human frame its trajectory represents.

For each trajectory, three calls against the same sidecar:
  A1  name-ordered prefix (the previous behaviour)
  A2  name-ordered prefix again       -> run-to-run variation (A1 vs A2)
  B   insertion-ordered prefix        -> variation with the change (A1 vs B)
and, per call, the mean insertion-order position of the selected nodes
(0 = oldest, 1 = newest). A shift under B is only a shift if it exceeds the
A1-vs-A2 variation.

Trajectories here are drawn from a transcript OUTSIDE the infusion window.
That makes this a mechanical description, not an evaluation. The observe log,
recorded inside the window across real trajectories, remains the instrument
for how the matcher behaves in use.

Output: docs/research/matcher-order-probe-<UTC>.json (local-only by the
repo's ignore rule; it names nodes) plus a printed summary of counts only.

Usage (plain argv, any shell):
    docker cp kenning_encounter-mcp:/app/models/meaning_sidecar.json /tmp/sidecar.json
    uv run python scripts/matcher_order_probe.py --sidecar /tmp/sidecar.json \\
        --transcript ~/.claude/projects/<proj>/<session>.jsonl --trajectories 8
GEMINI_API_KEY is read from the environment, else from ./.env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from kenning_encounter.meaning import (
    MeaningUnavailable,
    assemble_trajectory,
    build_prefix,
    match_meanings,
    user_turns_from_transcript,
)

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEY":
                return value.strip().strip('"')
    return ""


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def _recency(selected: list[str], position: dict[str, int], n: int) -> float:
    if not selected or n < 2:
        return 0.0
    return statistics.fmean(position[s] / (n - 1) for s in selected)


async def _select(prefix, ordered, traj_text, top_n, key, model, timeout_ms):
    out = await match_meanings(
        prefix, ordered, traj_text, top_n, key, model, ENDPOINT,
        timeout_ms=timeout_ms,
    )
    return [s["name"] for s in out["selections"]]


async def run(args: argparse.Namespace) -> int:
    key = _api_key()
    if not key:
        print("GEMINI_API_KEY not set (env or ./.env) — nothing measured.")
        return 2
    nodes = json.loads(Path(args.sidecar).read_text())["nodes"]
    insertion = list(nodes)
    position = {name: i for i, name in enumerate(insertion)}
    n = len(insertion)
    name_prefix, name_order = build_prefix(dict(sorted(nodes.items())))
    ins_prefix, ins_order = build_prefix(nodes)

    turns = user_turns_from_transcript(args.transcript, 10_000)
    if len(turns) < 2:
        print(f"only {len(turns)} user turns in the transcript — nothing measured.")
        return 2
    step = max(1, len(turns) // args.trajectories)
    cuts = list(range(step, len(turns) + 1, step))[: args.trajectories]

    rows = []
    for cut in cuts:
        window = turns[max(0, cut - args.window): cut]
        traj_text, _ = assemble_trajectory(window[:-1], window[-1])
        try:
            a1 = await _select(name_prefix, name_order, traj_text, args.top_n,
                               key, args.model, args.timeout_ms)
            a2 = await _select(name_prefix, name_order, traj_text, args.top_n,
                               key, args.model, args.timeout_ms)
            b = await _select(ins_prefix, ins_order, traj_text, args.top_n,
                              key, args.model, args.timeout_ms)
        except MeaningUnavailable as exc:
            print(f"matcher unavailable at turn {cut}: {exc} — stopping; "
                  "a partial run is reported as partial, never as a result.")
            break
        rows.append({
            "turn": cut,
            "a1": a1, "a2": a2, "b": b,
            "rerun_jaccard": _jaccard(a1, a2),
            "order_jaccard": _jaccard(a1, b),
            "recency_a1": _recency(a1, position, n),
            "recency_a2": _recency(a2, position, n),
            "recency_b": _recency(b, position, n),
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("docs/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"matcher-order-probe-{stamp}.json"
    summary = {
        "sidecar_nodes": n,
        "trajectories_planned": len(cuts),
        "trajectories_measured": len(rows),
        "complete": len(rows) == len(cuts),
        "top_n": args.top_n,
        "model": args.model,
    }
    if rows:
        summary.update({
            "rerun_jaccard_mean": statistics.fmean(r["rerun_jaccard"] for r in rows),
            "order_jaccard_mean": statistics.fmean(r["order_jaccard"] for r in rows),
            "recency_shift_rerun": statistics.fmean(
                r["recency_a2"] - r["recency_a1"] for r in rows),
            "recency_shift_order": statistics.fmean(
                r["recency_b"] - r["recency_a1"] for r in rows),
        })
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1,
                              ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")
    return 0 if summary["complete"] else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--transcript", required=True,
                    help="a harness session transcript (.jsonl) to draw trajectories from")
    ap.add_argument("--trajectories", type=int, default=8)
    ap.add_argument("--window", type=int, default=7, help="user turns per trajectory")
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--timeout-ms", type=int, default=10_000)
    raise SystemExit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
