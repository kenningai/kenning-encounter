# /// script
# requires-python = ">=3.12"
# dependencies = ["neo4j>=5.20", "kenning-encounter"]
#
# [tool.uv.sources]
# kenning-encounter = { path = "..", editable = true }
# ///
"""Describe how the matcher voices behave, from the server's shadow log.

With NEO4J_MATCHER_SHADOW on, every infusion records what the delivering
voice selected and what another voice selected for the same trajectory
(one JSON line per prompt, joined to the observe log by match_id). This
reads that log and reports MECHANICS only, per voice and per pair:

  - calls, failures by reason, latency p50/p95, prompt and cached tokens
  - how far each voice's selections spread across the graph's Leiden
    chapters (the same coherence-only projection orient uses) — a
    mechanical proxy for breadth, not a measure of meaning
  - overlap between the two voices' selections on the same prompt
  - all of it bucketed by sidecar size, so a change as the graph grows
    shows as a trend rather than a single reading

It never grades meaning. Two different selections can each bear on a
moment, differently. Whether an infusion carried meaning is judged in the
window, by the human frame the trajectory represents, in the observe log —
which this script does not read.

    uv run scripts/matcher_voices_report.py                    # log from the container
    uv run scripts/matcher_voices_report.py --log path.jsonl   # a local copy
    uv run scripts/matcher_voices_report.py --no-leiden        # skip the graph read

Deposits docs/research/<UTC>-matcher-voices.{md,json} and prints the report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "docs" / "research"

_SECRET_KEY = re.compile(r"(?i)\b([A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|API_KEY))\s*[=:]\s*\S+")
_URL_CRED = re.compile(r"([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@")
_KEY_SHAPE = re.compile(r"\b(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+|\b(AIza[0-9A-Za-z_-]{6})[0-9A-Za-z_-]+")


def redact(text: str) -> str:
    """Both rule classes on the one output path: key-name and value-shape."""
    text = _SECRET_KEY.sub(lambda m: f"{m.group(1)}=[redacted]", text)
    text = _URL_CRED.sub(lambda m: f"{m.group(1)}[redacted]@", text)
    return _KEY_SHAPE.sub(lambda m: f"{m.group(1) or m.group(2)}…[redacted]", text)


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


def load_log(args) -> list[dict]:
    if args.log:
        text = Path(args.log).read_text()
    else:
        r = subprocess.run(
            ["docker", "exec", args.container, "cat", args.container_log],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise SystemExit(
                f"could not read {args.container_log} in {args.container}: "
                f"{r.stderr.strip() or 'exit ' + str(r.returncode)} — is "
                "NEO4J_MATCHER_SHADOW on, and has a prompt run since?"
            )
        text = r.stdout
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def leiden_chapters() -> dict[str, int]:
    """node name -> Leiden community, over orient's coherence projection."""
    from neo4j import GraphDatabase, NotificationDisabledClassification

    from kenning_encounter.kenning_encounter import coherence_projection_parts

    _env_from_dotenv()
    uri = os.environ.get("NEO4J_URL") or "bolt://localhost:7687"
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD") or os.environ.get("NEO4J_KENNING_ENCOUNTER_PASSWORD") or ""
    database = os.environ.get("NEO4J_DATABASE", "kenning_encounter")
    rel_filter, src_labels, tgt_labels = coherence_projection_parts()
    proj = "matcher-voices-report"
    with GraphDatabase.driver(
        uri, auth=(user, password),
        # A coherence edge type with no instances yet is not news.
        notifications_disabled_classifications=[NotificationDisabledClassification.UNRECOGNIZED],
    ) as driver:
        with driver.session(database=database) as s:
            s.run("CALL gds.graph.drop($p, false) YIELD graphName RETURN graphName", p=proj).consume()
            try:
                s.run(
                    f"MATCH (source)-[r:{rel_filter}]->(target) "
                    f"WHERE ({src_labels}) AND ({tgt_labels}) "
                    "RETURN gds.graph.project($p, source, target, {}, "
                    "{undirectedRelationshipTypes: ['*']}) AS g",
                    p=proj,
                ).consume()
                res = s.run(
                    "CALL gds.leiden.stream($p) YIELD nodeId, communityId "
                    "RETURN gds.util.asNode(nodeId).name AS name, communityId",
                    p=proj,
                )
                return {r["name"]: r["communityId"] for r in res}
            finally:
                s.run("CALL gds.graph.drop($p, false) YIELD graphName RETURN graphName", p=proj).consume()


def pct(xs: list[float], p: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def bucket(size: int | None, width: int) -> str:
    if not size:
        return "?"
    lo = (size // width) * width
    return f"{lo}-{lo + width - 1}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help="local shadow log (default: read it from the container)")
    ap.add_argument("--container", default="kenning_encounter-mcp")
    ap.add_argument("--container-log", default="/app/models/matcher-shadow.jsonl")
    ap.add_argument("--no-leiden", action="store_true", help="skip the Leiden chapter reading")
    ap.add_argument("--bucket", type=int, default=100, help="sidecar-size bucket width (default 100)")
    a = ap.parse_args()

    rows = load_log(a)
    if not rows:
        raise SystemExit("shadow log is empty")
    chapters: dict[str, int] = {}
    leiden_note = "skipped (--no-leiden)"
    if not a.no_leiden:
        try:
            chapters = leiden_chapters()
            leiden_note = f"{len(set(chapters.values()))} chapters over {len(chapters)} coherence nodes, read now"
        except Exception as e:  # the report stands without it, and says so
            leiden_note = f"NOT READ — {type(e).__name__}: {e}"

    per_voice: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "failures": defaultdict(int), "ms": [], "prompt_tokens": [],
        "cached_share": [], "chapters": [], "outside": [], "delivered": 0, "shadowed": 0,
    })
    pairs: list[dict] = []
    by_bucket: dict[str, dict] = defaultdict(lambda: {"overlap": [], "ms": defaultdict(list), "tokens": defaultdict(list)})

    for r in rows:
        sides = [("delivered", r.get("delivered") or {}), ("shadow", r.get("shadow") or {})]
        sel = {}
        for role, v in sides:
            vid = v.get("voice")
            if not vid:
                continue
            pv = per_voice[vid]
            pv["calls"] += 1
            pv["delivered" if role == "delivered" else "shadowed"] += 1
            b = bucket(v.get("sidecar_size"), a.bucket)
            if "error" in v:
                reason = re.sub(r"\d{3,}", "N", str(v["error"]))[:120]
                pv["failures"][reason] += 1
                continue
            names = v.get("selections") or []
            sel[vid] = names
            pv["ms"].append(v.get("ms") or 0)
            pt = v.get("prompt_tokens") or 0
            pv["prompt_tokens"].append(pt)
            if pt:
                pv["cached_share"].append((v.get("cached_tokens") or 0) / pt)
            by_bucket[b]["ms"][vid].append(v.get("ms") or 0)
            by_bucket[b]["tokens"][vid].append(pt)
            if chapters:
                pv["chapters"].append(len({chapters[n] for n in names if n in chapters}))
                pv["outside"].append(sum(1 for n in names if n not in chapters))
        if len(sel) == 2:
            (va, A), (vb, B) = sel.items()
            sa, sb = set(A), set(B)
            union = sa | sb
            pair = {
                "match_id": r.get("match_id"), "voices": [va, vb],
                "overlap": len(sa & sb), "jaccard": len(sa & sb) / len(union) if union else 0.0,
                "top3_overlap": len(set(A[:3]) & set(B[:3])),
                "sidecar_bucket": bucket((r.get("delivered") or {}).get("sidecar_size"), a.bucket),
            }
            pairs.append(pair)
            by_bucket[pair["sidecar_bucket"]]["overlap"].append(pair["jaccard"])

    now = datetime.now(timezone.utc)
    first, last = rows[0].get("ts", "?"), rows[-1].get("ts", "?")
    L = [f"# Matcher voices — {now:%Y-%m-%d %H:%M:%S}Z", "",
         f"Shadow log: {len(rows)} prompts, {first} → {last}. Leiden: {leiden_note}.", "",
         "Mechanics only. Nothing here says which voice carried more meaning; that is "
         "judged in the window, in the observe log, joined by match_id.", "",
         "## Per voice", "",
         "| voice | calls (delivered / shadow) | failed | p50 ms | p95 ms | prompt tok p50 | cached share p50 | chapters spanned (mean) | outside coherence graph (mean) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for vid, pv in sorted(per_voice.items()):
        nf = sum(pv["failures"].values())
        mean = lambda xs: f"{statistics.mean(xs):.2f}" if xs else "—"  # noqa: E731
        cs = pct(pv["cached_share"], .5)
        L.append(
            f"| {vid} | {pv['calls']} ({pv['delivered']} / {pv['shadowed']}) | {nf} | "
            f"{pct(pv['ms'], .5) or '—'} | {pct(pv['ms'], .95) or '—'} | {pct(pv['prompt_tokens'], .5) or '—'} | "
            f"{f'{cs:.2f}' if cs is not None else '—'} | {mean(pv['chapters'])} | {mean(pv['outside'])} |"
        )
    L += ["", "Failures, by reason (digits folded):", ""]
    any_fail = False
    for vid, pv in sorted(per_voice.items()):
        for reason, n in sorted(pv["failures"].items(), key=lambda kv: -kv[1]):
            any_fail = True
            L.append(f"- {vid}: {n}× {reason}")
    if not any_fail:
        L.append("- none")
    L += ["", "## Between voices, same prompt", ""]
    if pairs:
        js = [p["jaccard"] for p in pairs]
        L.append(f"{len(pairs)} prompts where both voices selected. Jaccard p50 {pct(js, .5):.2f} "
                 f"(min {min(js):.2f}, max {max(js):.2f}); shared selections mean "
                 f"{statistics.mean(p['overlap'] for p in pairs):.1f}; top-3 shared mean "
                 f"{statistics.mean(p['top3_overlap'] for p in pairs):.2f}.")
        L.append("")
        L.append("A low overlap is not a disagreement about correctness: each set may bear "
                 "on the moment, differently.")
    else:
        L.append("No prompt yet where both voices returned a selection.")
    L += ["", "## By sidecar size", "",
          "| sidecar size | prompts paired | Jaccard p50 | " + " | ".join(f"{v} p50 ms / prompt tok" for v in sorted(per_voice)) + " |",
          "|---|---|---|" + "---|" * len(per_voice)]
    for b in sorted(by_bucket, key=lambda k: (k == "?", int(k.split("-")[0]) if k != "?" else 0)):
        d = by_bucket[b]
        jp = pct(d["overlap"], .5)
        cells = [f"{pct(d['ms'][v], .5) or '—'} / {pct(d['tokens'][v], .5) or '—'}" for v in sorted(per_voice)]
        L.append(f"| {b} | {len(d['overlap'])} | {f'{jp:.2f}' if jp is not None else '—'} | " + " | ".join(cells) + " |")

    report = redact("\n".join(L))
    print(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{now:%Y%m%dT%H%M%SZ}-matcher-voices"
    stem.with_suffix(".md").write_text(report + "\n")
    stem.with_suffix(".json").write_text(redact(json.dumps({
        "generated": now.isoformat(), "prompts": len(rows), "leiden": leiden_note,
        "per_voice": {k: {**v, "failures": dict(v["failures"])} for k, v in per_voice.items()},
        "pairs": pairs,
    }, indent=2)) + "\n")
    print(f"\nreport: {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
