# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Measure how much of each voice's sidecar is copied from the nodes.

A sidecar meaning is supposed to be a compression: one sentence, in the
voice's own words, of what a node means. A voice whose sentences are lifted
from the node's own description has not compressed anything. Matching
against them is closer to matching against the author's wording, and the
nearer a voice's model is to the model that authored the graph, the more
this is worth measuring before the voice is compared with the others.

For every node in every sidecar, against that node's name and description
(the description cut where the compressor cut it):

  - longest run: the longest sequence of consecutive words in the meaning
    that appears, in order, in the source;
  - 5-gram share: the fraction of the meaning's five-word sequences that
    appear in the source.

MECHANICS ONLY. A high figure says a voice quotes; it does not say the
voice matches worse or better. The report carries counts and percentiles,
never node text.

Inputs: a node dump (a JSON list of {"name", "description"}) and the
directory holding the stamped sidecars.

    uv run scripts/matcher_copy_rate.py --nodes nodes.json --sidecar-dir models/

Deposits docs/research/<UTC>-matcher-copy-rate.{md,json} and prints the report.
Exit 0 written, 2 refused (no sidecar, or a sidecar with no voice stamp).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "docs" / "research"
DESC_CAP = 4000          # the compressor's own cut (meaning._COMPRESS_DESC_CAP)
N = 5
LONG_RUN = 8             # a run this long is a quoted clause, not a shared phrase
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def longest_run(meaning: list[str], source: list[str]) -> int:
    """Longest run of consecutive meaning words found in order in source."""
    best = 0
    prev = [0] * (len(source) + 1)
    for m in meaning:
        cur = [0] * (len(source) + 1)
        for j, s in enumerate(source, 1):
            if m == s:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def ngram_share(meaning: list[str], source: list[str], n: int = N) -> float | None:
    """Share of the meaning's n-grams present in source. None when the
    meaning is shorter than n words."""
    grams = [tuple(meaning[i:i + n]) for i in range(len(meaning) - n + 1)]
    if not grams:
        return None
    have = {tuple(source[i:i + n]) for i in range(len(source) - n + 1)}
    return sum(g in have for g in grams) / len(grams)


def measure(sidecar: dict, nodes: dict[str, str]) -> dict:
    runs, shares, lengths, unmatched = [], [], [], 0
    for name, entry in sidecar.items():
        if name not in nodes:
            unmatched += 1
            continue
        m = words(entry.get("meaning", ""))
        src = words(name + " " + nodes[name][:DESC_CAP])
        if not m:
            continue
        runs.append(longest_run(m, src))
        lengths.append(len(m))
        share = ngram_share(m, src)
        if share is not None:
            shares.append(share)

    def pct(xs: list[float], q: float) -> float:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0

    return {
        "nodes": len(runs), "not_in_dump": unmatched,
        "meaning_words_p50": statistics.median(lengths) if lengths else 0,
        "longest_run_p50": statistics.median(runs) if runs else 0,
        "longest_run_p90": pct(runs, 0.9), "longest_run_max": max(runs, default=0),
        f"runs_of_{LONG_RUN}_or_more": sum(r >= LONG_RUN for r in runs),
        "share5_p50": round(statistics.median(shares), 3) if shares else 0.0,
        "share5_p90": round(pct(shares, 0.9), 3),
        "share5_half_or_more": sum(s >= 0.5 for s in shares),
    }


def render(results: dict[str, dict], now: datetime, dump_size: int) -> str:
    L = [f"# Matcher copy rate — {now:%Y-%m-%d %H:%M:%S}Z", "",
         f"Node dump: {dump_size} nodes. Per voice: how much of each sidecar meaning is "
         "lifted word for word from its node's name and description. Mechanics only.", "",
         "| voice | nodes | meaning words p50 | longest run p50 | p90 | max | "
         f"runs ≥ {LONG_RUN} | 5-gram share p50 | p90 | meanings ≥ half copied |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for voice, r in sorted(results.items()):
        L.append(f"| {voice} | {r['nodes']} | {r['meaning_words_p50']:.0f} | "
                 f"{r['longest_run_p50']:.0f} | {r['longest_run_p90']:.0f} | {r['longest_run_max']} | "
                 f"{r[f'runs_of_{LONG_RUN}_or_more']} | {r['share5_p50']:.3f} | {r['share5_p90']:.3f} | "
                 f"{r['share5_half_or_more']} |")
    skipped = {v: r["not_in_dump"] for v, r in results.items() if r["not_in_dump"]}
    if skipped:
        L += ["", "Sidecar entries with no node in the dump (not measured): "
              + "; ".join(f"{v} {n}" for v, n in sorted(skipped.items())) + "."]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True, help='JSON list of {"name", "description"}')
    ap.add_argument("--sidecar-dir", required=True)
    a = ap.parse_args()

    nodes = {n["name"]: n.get("description") or "" for n in json.loads(Path(a.nodes).read_text())}
    results: dict[str, dict] = {}
    for f in sorted(Path(a.sidecar_dir).glob("meaning_sidecar.*.json")):
        doc = json.loads(f.read_text())
        voice = doc.get("voice")
        if not voice:
            print(f"REFUSED — {f.name} carries no voice stamp", file=sys.stderr)
            return 2
        results[voice] = measure(doc["nodes"], nodes)
    if not results:
        print(f"REFUSED — no stamped sidecar in {a.sidecar_dir}", file=sys.stderr)
        return 2

    now = datetime.now(timezone.utc)
    report = render(results, now, len(nodes))
    print(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{now:%Y%m%dT%H%M%SZ}-matcher-copy-rate"
    stem.with_suffix(".md").write_text(report + "\n")
    stem.with_suffix(".json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nreport: {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
