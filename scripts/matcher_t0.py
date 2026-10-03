# /// script
# requires-python = ">=3.12"
# dependencies = ["kenning-encounter"]
#
# [tool.uv.sources]
# kenning-encounter = { path = "..", editable = true }
# ///
"""Run the T0 recall gate per matcher arm on one frozen corpus.

T0 is how the meaning matcher earned its place (recall 7/8 against 2/8
lexical, the trajectory adding 2). This runs the same gate for each ARM —
a voice's model in one request shape — on the same cases, so a difference
between models can be told apart from a difference between request shapes.

It COMPARES METHODS on a constructed stress set. It does not score the
substrate and does not grade the meaning of a live infusion.

Held equal between arms, or measured where it cannot be held:
  - the node set and the ORDER of the meanings list: one shared order, the
    intersection of every voice's sidecar (each voice keeps its own
    meanings; only the line each node sits on is shared);
  - the corpus, trajectory assembly, k, timeout, temperature and the
    request text;
  - time: every call is scheduled in one seeded random order, interleaving
    the arms, so provider load cannot line up with an arm;
  - the graph: the scored set runs against sidecar SNAPSHOTS taken when the
    run starts, and a resumed run uses the same snapshots;
  - reasoning: reasoning and output tokens are recorded per call.

Fails closed before any call: a malformed corpus, an arc containing a
content word of its target's name, or a target/decoy absent from any
voice's sidecar (that case leaves every arm).

    uv run scripts/matcher_t0.py corpus.json --check-only
    uv run scripts/matcher_t0.py corpus.json --set dev
    uv run scripts/matcher_t0.py corpus.json

The held-out run keeps its state in lab/runs/t0-voices/<corpus sha>/. If it
stops (a rate limit that outlasts the retries, a closed laptop), the same
command resumes it where it stopped. Once it is complete, the same command only re-renders the
report: the held-out set is scored once. Dev runs are never kept.

Deposits docs/research/<UTC>-matcher-t0-<set>.{md,json} and prints the report.
Exit 0 complete, 2 refused, 3 incomplete (run again to resume).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "docs" / "research"
STATE_DIR = REPO / "lab" / "runs" / "t0-voices"

STRATA = ("unnamed-constraint", "arc-dependent", "negation", "recent", "old")
SETS = ("dev", "heldout")
MODES = ("arc", "prompt_alone")
# arm -> (provider, request shape). Each model in both shapes, so model and
# shape can be told apart. Which arm was production is recorded in each
# run's state (meta.native_shapes) at the moment the run starts: Gemini's
# production shape changed from flat to split in v0.18.1.
ARMS: dict[str, tuple[str, str]] = {
    "gemini-flat": ("gemini", "flat"),
    "gemini-split": ("gemini", "split"),
    "openai-split": ("openai", "split"),
    "openai-flat": ("openai", "flat"),
}
K = 30
PRODUCTION_K = 12          # what infusion keeps
# The reading rule, fixed before any held-out call.
TIE_BAND = {"recall@30": 2.0, "recall@12": 2.0, "mrr": 0.10}
ALPHA = 0.05
FITNESS_FLOOR = 0.75
PERMUTATIONS = 20_000

# Function words: modals and auxiliaries, pronouns, determiners,
# prepositions and conjunctions of four letters or more. A shared function
# word is not the target being named.
_STOP = {
    "will", "would", "shall", "should", "could", "might", "must", "been",
    "being", "have", "having", "does", "doing", "done", "were", "isn't",
    "wasn't", "aren't", "don't", "doesn't", "didn't", "won't", "can't",
    "that", "this", "these", "those", "their", "theirs", "them", "they",
    "there", "here", "your", "yours", "itself", "what", "which", "whom",
    "whose", "each", "every", "some", "such", "both", "either", "neither",
    "many", "much", "more", "most", "other", "another", "same", "only",
    "about", "above", "after", "again", "against", "along", "also", "among",
    "around", "because", "before", "behind", "below", "beneath", "beside",
    "between", "beyond", "during", "even", "ever", "from", "into", "just",
    "like", "near", "never", "once", "onto", "over", "rather", "since",
    "than", "then", "though", "through", "toward", "towards", "under",
    "unless", "until", "upon", "very", "when", "whenever", "where",
    "whether", "while", "with", "within", "without", "yet",
}

_KEY_SHAPE = re.compile(r"\b(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+|\b(AIza[0-9A-Za-z_-]{6})[0-9A-Za-z_-]+")
_URL_CRED = re.compile(r"([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@")
_SECRET_KEY = re.compile(r"(?i)\b([A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|API_KEY))\s*[=:]\s*\S+")


def redact(text: str) -> str:
    """Both rule classes on the one output path: key-name and value-shape."""
    text = _SECRET_KEY.sub(lambda m: f"{m.group(1)}=[redacted]", text)
    text = _URL_CRED.sub(lambda m: f"{m.group(1)}[redacted]@", text)
    return _KEY_SHAPE.sub(lambda m: f"{m.group(1) or m.group(2)}…[redacted]", text)


# -- Corpus checks (pure) -----------------------------------------------------


def validate_corpus(data: Any) -> list[str]:
    """Every structural problem in the corpus; empty means well-formed."""
    problems: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        return ["corpus must be an object with a 'cases' list"]
    meta = data.get("corpus") or {}
    for f in ("author", "written"):
        if not meta.get(f):
            problems.append(f"corpus.{f} missing")
    seen: set[str] = set()
    for i, c in enumerate(data["cases"]):
        cid = c.get("id") or f"#{i}"
        if not c.get("id"):
            problems.append(f"{cid}: id missing")
        elif cid in seen:
            problems.append(f"{cid}: duplicate id")
        seen.add(cid)
        if c.get("set") not in SETS:
            problems.append(f"{cid}: set must be one of {SETS}")
        if c.get("stratum") not in STRATA:
            problems.append(f"{cid}: stratum must be one of {STRATA}")
        for f in ("target", "prompt", "rationale"):
            if not isinstance(c.get(f), str) or not c[f].strip():
                problems.append(f"{cid}: {f} missing")
        traj = c.get("trajectory")
        if not isinstance(traj, list) or not 3 <= len(traj) <= 6 or not all(
            isinstance(t, str) and t.strip() for t in traj
        ):
            problems.append(f"{cid}: trajectory must be 3-6 non-empty turns")
        if c.get("decoy") and c.get("decoy") == c.get("target"):
            problems.append(f"{cid}: decoy is the target")
    return problems


def ineligible_uses(cases: list[dict], names: set[str]) -> list[str]:
    """Cases whose target or decoy is on the ineligible list (A5: nodes the
    corpus author itself authored). Empty means none."""
    out = []
    for c in cases:
        for role in ("target", "decoy"):
            if c.get(role) and c[role] in names:
                out.append(f"{c.get('id')}: {role} is ineligible ({c[role][:80]})")
    return out


def content_words(name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9-]{3,}", name.lower()) if w not in _STOP}


def leaks(case: dict) -> list[str]:
    """Content words of the target's NAME that appear as words in the arc or
    prompt — the cheapest way 'never named' breaks, and the one keyword
    matching would exploit."""
    text = " ".join([*case.get("trajectory", []), case.get("prompt", "")]).lower()
    words = set(re.findall(r"[a-z][a-z0-9-]{3,}", text))
    return sorted(content_words(case.get("target", "")) & words)


def transient(error: str) -> bool:
    """A provider-side failure worth retrying: rate limit, 5xx, timeout."""
    return bool(re.search(r"\b(429|5\d\d)\b|Timeout", error))


def summarize(ranks: list[int | None], k: int) -> float:
    """Fraction of runs in which the target was in the top k."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def shared_order(sidecars: dict[str, dict[str, Any]], order_from: str) -> tuple[list[str], dict[str, list[str]]]:
    """The one node order every arm reads: `order_from`'s insertion order,
    restricted to the nodes EVERY sidecar holds. Returns (order, dropped),
    where dropped[voice] lists that voice's nodes outside the intersection."""
    common = set.intersection(*(set(n) for n in sidecars.values()))
    order = [name for name in sidecars[order_from] if name in common]
    dropped = {v: sorted(set(n) - common) for v, n in sidecars.items()}
    return order, dropped


def schedule(case_ids: list[str], arms: list[str], repeats: int, seed: int) -> list[tuple[str, str, str, int]]:
    """Every (arm, case, mode, repeat) call, in one seeded random order."""
    tasks = [(a, c, m, r) for a in arms for c in case_ids for m in MODES for r in range(repeats)]
    random.Random(seed).shuffle(tasks)
    return tasks


def task_key(arm: str, cid: str, mode: str, rep: int) -> str:
    return f"{arm}|{cid}|{mode}|{rep}"


def case_scores(runs: list[dict]) -> dict[str, float] | None:
    """Per-case means over the runs that returned (an error is neither hit
    nor miss). None when no run returned."""
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return None
    ranks = [r.get("rank") for r in ok]
    return {
        "recall@30": summarize(ranks, K),
        "recall@12": summarize(ranks, PRODUCTION_K),
        "mrr": sum(1.0 / x for x in ranks if x) / len(ranks),
    }


def sign_flip_p(diffs: list[float], seed: int = 0, n: int = PERMUTATIONS) -> float:
    """Two-sided paired sign-flip permutation p for mean(diffs) != 0. Exact
    when there are few non-zero differences, otherwise `n` seeded flips.
    Cases are the unit; repeats are already averaged inside each case."""
    d = [x for x in diffs if x != 0]
    if not d:
        return 1.0
    observed = abs(sum(d))
    if len(d) <= 16:
        hits = 0
        for mask in range(1 << len(d)):
            s = sum(x if mask >> i & 1 else -x for i, x in enumerate(d))
            hits += abs(s) >= observed - 1e-12
        return hits / (1 << len(d))
    rng = random.Random(seed)
    hits = sum(
        abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= observed - 1e-12
        for _ in range(n)
    )
    return (hits + 1) / (n + 1)


def contrast(a: dict[str, dict], b: dict[str, dict], metric: str) -> dict[str, Any]:
    """Paired difference a − b on one metric over the cases both scored.
    Recall differences are in CASES (a sum); MRR is a mean."""
    ids = sorted(set(a) & set(b))
    diffs = [a[c][metric] - b[c][metric] for c in ids]
    total = (statistics.mean(diffs) if diffs else 0.0) if metric == "mrr" else sum(diffs)
    p = sign_flip_p(diffs, seed=len(ids))
    return {"n": len(ids), "diff": total, "p": p,
            "claim": abs(total) > TIE_BAND[metric] and p < ALPHA}


# -- Running ------------------------------------------------------------------


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


def fetch_sidecar(voice, args, dst_dir: Path) -> Path:
    from kenning_encounter.meaning import sidecar_path

    src = sidecar_path("/app/models/meaning_sidecar.json", voice)
    dst = dst_dir / src.name
    if args.sidecar_dir:
        shutil.copy(sidecar_path(Path(args.sidecar_dir) / "meaning_sidecar.json", voice), dst)
        return dst
    r = subprocess.run(["docker", "cp", f"{args.container}:{src}", str(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"could not copy {src} from {args.container}: {r.stderr.strip()}")
    return dst


async def _call(prefix, names, text, voice, shape, timeout_ms):
    """One call, retried when the PROVIDER failed rather than the arm (a
    429, a 5xx, a timeout). A malformed answer or another 4xx is the arm's
    own result and is never retried."""
    from kenning_encounter.meaning import MeaningUnavailable, match_meanings

    for attempt in range(4):
        try:
            return await match_meanings(prefix, names, text, K, voice,
                                        timeout_ms=timeout_ms, shape=shape)
        except MeaningUnavailable as e:
            if not transient(str(e)) or attempt == 3:
                raise
            await asyncio.sleep(20 * (attempt + 1) if "429" in str(e) else 5 * (attempt + 1))
    raise AssertionError("unreachable")


async def run_tasks(state: dict, tasks, arms_cfg, cases_by_id, timeout_ms: int, save) -> dict[str, str]:
    """Run every task not already in state['runs']. A rate limit that
    survives the retries pauses that arm for this invocation; its calls stay
    unrecorded and the next invocation resumes them."""
    from kenning_encounter.meaning import MeaningUnavailable, assemble_trajectory

    paused: dict[str, str] = {}
    warmed: set[str] = set()
    done = 0
    for arm, cid, mode, rep in tasks:
        key = task_key(arm, cid, mode, rep)
        if key in state["runs"] or arm in paused:
            continue
        voice, shape, prefix, names = arms_cfg[arm]
        if arm not in warmed:  # unscored: the first call cannot hit the cache
            try:
                await _call(prefix, names, "TRAJECTORY:\n[current prompt] warm-up", voice, shape, timeout_ms)
            except MeaningUnavailable:
                pass
            warmed.add(arm)
        c = cases_by_id[cid]
        text, _ = assemble_trajectory(c["trajectory"] if mode == "arc" else [], c["prompt"])
        try:
            m = await _call(prefix, names, text, voice, shape, timeout_ms)
        except MeaningUnavailable as e:
            if "429" in str(e):
                paused[arm] = str(e)[:160]
                continue
            state["runs"][key] = {"error": str(e)[:200]}
            save()
            continue
        sel = [s["name"] for s in m["selections"]]
        state["runs"][key] = {
            "rank": sel.index(c["target"]) + 1 if c["target"] in sel else None,
            "decoy_rank": sel.index(c["decoy"]) + 1 if c.get("decoy") and c["decoy"] in sel else None,
            "ms": m["ms"], "prompt_tokens": m["prompt_tokens"], "cached_tokens": m["cached_tokens"],
            "output_tokens": m["output_tokens"], "reasoning_tokens": m["reasoning_tokens"],
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save()
        done += 1
        if done % 20 == 0:
            print(f"  {done} calls made", file=sys.stderr)
    return paused


# -- Report -------------------------------------------------------------------


def _fmt(runs: list[dict], field: str = "rank") -> str:
    return ",".join("ERR" if "error" in r else ("—" if r.get(field) is None else str(r[field])) for r in runs)


def render(state: dict, cases: list[dict], arms: list[str], incomplete: int, paused: dict[str, str]) -> str:
    runs = state["runs"]
    reps = state["repeats"]
    meta = state["meta"]
    L = [f"# Matcher T0 by arm — {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z", "",
         f"Corpus `{meta['corpus']}` sha256 `{meta['corpus_sha256'][:16]}`; set **{meta['set']}**; "
         f"{len(cases)} cases; {reps} repeats per case per mode; k={K} (and {PRODUCTION_K}); "
         f"schedule seed {meta['seed']}.",
         f"Meanings list: {meta['order_size']} nodes in one shared order (from {meta['order_from']}); "
         + "; ".join(f"{v} sidecar sha256 `{h[:16]}` ({meta['sidecar_sizes'][v]} nodes, "
                     f"{len(meta['dropped'][v])} outside the shared set)"
                     for v, h in meta["sidecar_sha256"].items()) + ".",
         "A method comparison on a constructed stress set. Not a score for the substrate; "
         "not a grade of meaning.", ""]
    if meta.get("excluded"):
        L.append("Excluded from every arm:")
        L += [f"- {cid}: {why}" for cid, why in sorted(meta["excluded"].items())]
        L.append("")
    if incomplete:
        L += [f"**INCOMPLETE: {incomplete} calls not yet made.** "
              + ("Paused: " + "; ".join(f"{a} ({why})" for a, why in paused.items()) + ". " if paused else "")
              + "Run the same command again to resume. Nothing below is a result until this line is gone.", ""]

    scores: dict[str, dict[str, dict]] = {}
    for arm in arms:
        provider, shape = ARMS[arm]
        L += [f"## {arm} — {meta['models'][provider]}, {shape} request", "",
              "| case | stratum | arc ranks | prompt-alone ranks | decoy ranks (arc) |",
              "|---|---|---|---|---|"]
        per_case: dict[str, dict] = {}
        alone_case: dict[str, dict] = {}
        all_runs: list[dict] = []
        unstable = 0
        strata: dict[str, list[float]] = {}
        for c in cases:
            arc = [runs[k] for r in range(reps) if (k := task_key(arm, c["id"], "arc", r)) in runs]
            alone = [runs[k] for r in range(reps) if (k := task_key(arm, c["id"], "prompt_alone", r)) in runs]
            all_runs += arc + alone
            L.append(f"| {c['id']} | {c['stratum']} | {_fmt(arc)} | {_fmt(alone)} | {_fmt(arc, 'decoy_rank')} |")
            sc = case_scores(arc)
            if sc is not None:
                per_case[c["id"]] = sc
                strata.setdefault(c["stratum"], []).append(sc["recall@30"])
                unstable += len({r.get("rank") for r in arc if "error" not in r}) > 1
            sa = case_scores(alone)
            if sa is not None:
                alone_case[c["id"]] = sa
        scores[arm] = per_case
        n = len(per_case)
        ok = [r for r in all_runs if "error" not in r]
        if n and ok:
            r30 = sum(s["recall@30"] for s in per_case.values())
            r12 = sum(s["recall@12"] for s in per_case.values())
            mrr = statistics.mean(s["mrr"] for s in per_case.values())
            alone30 = sum(s["recall@30"] for s in alone_case.values())
            decoy_over = sum(
                1 for c in cases for r in range(reps)
                if (x := runs.get(task_key(arm, c["id"], "arc", r))) and "error" not in x
                and x.get("decoy_rank") and (x.get("rank") is None or x["decoy_rank"] < x["rank"])
            )
            cached = [r["cached_tokens"] / r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
            L += ["",
                  f"- **recall@{K}: {r30:.2f}/{n}** · **recall@{PRODUCTION_K}: {r12:.2f}/{n}** · "
                  f"**MRR: {mrr:.3f}** — fitness floor {FITNESS_FLOOR:.0%} at k={K}: "
                  f"{'met' if r30 / n >= FITNESS_FLOOR else 'NOT met'}",
                  f"- prompt alone recall@{K}: {alone30:.2f}/{len(alone_case)}; trajectory delta {r30 - alone30:+.2f}",
                  f"- cases whose arc rank changed between repeats: {unstable}/{n}",
                  f"- arc runs where the decoy outranked the target: {decoy_over}",
                  f"- calls errored after retries: {len(all_runs) - len(ok)} (shown as ERR; neither hit nor miss)",
                  f"- reasoning tokens per call: mean {statistics.mean(r['reasoning_tokens'] for r in ok):.1f}, "
                  f"max {max(r['reasoning_tokens'] for r in ok)} · output tokens mean "
                  f"{statistics.mean(r['output_tokens'] for r in ok):.1f}",
                  f"- latency p50 {statistics.median(r['ms'] for r in ok):.0f} ms · cached share p50 "
                  f"{statistics.median(cached) if cached else 0:.2f}",
                  f"- by stratum (recall@{K}): " + "; ".join(
                      f"{s} {sum(v):.2f}/{len(v)}" for s, v in sorted(strata.items())), ""]

    from kenning_encounter.meaning import NATIVE_SHAPE_BEFORE_0_18_1

    native = meta.get("native_shapes") or NATIVE_SHAPE_BEFORE_0_18_1
    deployed = (f"gemini-{native['gemini']}", f"openai-{native['openai']}")
    pairs = [
        (*deployed, "AS DEPLOYED (each model in its production shape when this run started)"),
        ("gemini-split", "openai-split", "MODEL, split shape held equal"),
        ("gemini-flat", "openai-flat", "MODEL, flat shape held equal"),
        ("gemini-split", "gemini-flat", "SHAPE, within Gemini"),
        ("openai-split", "openai-flat", "SHAPE, within OpenAI"),
    ]
    L += ["## Contrasts", "",
          f"Paired over cases. A difference is CLAIMED only when it is outside the tie band "
          f"(recall ±{TIE_BAND['recall@30']:.0f} cases, MRR ±{TIE_BAND['mrr']:.2f}) AND the sign-flip "
          f"p < {ALPHA}. A MODEL difference is claimed only when it is claimed in BOTH shape-held "
          "contrasts, in the same direction.", "",
          "| contrast | what it isolates | n | recall@30 Δ (p) | recall@12 Δ (p) | MRR Δ (p) |",
          "|---|---|---|---|---|---|"]
    claims: dict[tuple[str, str], dict] = {}
    for a, b, what in pairs:
        if a not in scores or b not in scores:
            continue
        cells, n = [], 0
        for metric in ("recall@30", "recall@12", "mrr"):
            r = contrast(scores[a], scores[b], metric)
            claims[(f"{a}|{b}", metric)] = r
            fmtd = f"{r['diff']:+.3f}" if metric == "mrr" else f"{r['diff']:+.2f}"
            cells.append(f"{fmtd} (p={r['p']:.3f})" + (" **claimed**" if r["claim"] else ""))
            n = r["n"]
        L.append(f"| {a} − {b} | {what} | {n} | " + " | ".join(cells) + " |")
    L.append("")
    for metric in ("recall@30", "recall@12", "mrr"):
        s = claims.get(("gemini-split|openai-split", metric))
        f = claims.get(("gemini-flat|openai-flat", metric))
        if s and f:
            same_dir = (s["diff"] > 0) == (f["diff"] > 0)
            verdict = ("a MODEL difference is claimed" if s["claim"] and f["claim"] and same_dir
                       else "no model difference is claimed")
            L.append(f"- {metric}: {verdict} (split {'claimed' if s['claim'] else 'not claimed'}, "
                     f"flat {'claimed' if f['claim'] else 'not claimed'}).")
    L += ["", "Which voice leads is not decided by this table alone.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="the frozen corpus JSON")
    ap.add_argument("--set", default="heldout", choices=SETS,
                    help="dev checks the apparatus; heldout is scored ONCE")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--repeats", type=int, default=5,
                    help="runs per case per mode (A2: 5, from the measured spread of identical calls)")
    ap.add_argument("--seed", type=int, default=20261002, help="schedule seed (recorded)")
    ap.add_argument("--order-from", default="gemini", choices=["gemini", "openai"],
                    help="whose sidecar insertion order is the shared order")
    ap.add_argument("--timeout-ms", type=int, default=15_000,
                    help="per call; generous, because a gate measures the arm, not the hook budget")
    ap.add_argument("--container", default="kenning_encounter-mcp")
    ap.add_argument("--sidecar-dir", help="read sidecars here instead of the container")
    ap.add_argument("--state-dir", help="held-out state directory (default lab/runs/t0-voices/<sha>)")
    ap.add_argument("--check-only", action="store_true",
                    help="run the corpus gates and the sidecar check, make no match calls")
    ap.add_argument("--ineligible", help="file of node names no target or decoy may use, one per line")
    ap.add_argument("--allow-leaks", action="store_true",
                    help="run even when an arc names its target's words (the report lists them)")
    a = ap.parse_args()

    raw = Path(a.corpus).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    problems = validate_corpus(data)
    if problems:
        print("REFUSED — the corpus is malformed:", *[f"  - {p}" for p in problems], sep="\n")
        return 2
    if a.ineligible:
        names = {ln.strip() for ln in Path(a.ineligible).read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")}
        bad = ineligible_uses(data["cases"], names)
        if bad:
            print("REFUSED — cases use ineligible nodes:", *[f"  - {b}" for b in bad], sep="\n")
            return 2
    cases = [c for c in data["cases"] if c["set"] == a.set]
    leaked = {c["id"]: leaks(c) for c in cases if leaks(c)}
    if leaked:
        print("LEAKS — arcs containing a content word of the target's name:",
              *[f"  - {cid}: {', '.join(w)}" for cid, w in leaked.items()], sep="\n")
        if not a.allow_leaks:
            print("REFUSED — fix the cases, or pass --allow-leaks to run and report them.")
            return 2

    from kenning_encounter.meaning import (
        DEFAULT_MODELS, KEY_ENV, NATIVE_SHAPE, Voice, build_prefix, sidecar_path,
    )

    _env_from_dotenv()
    providers = sorted({ARMS[arm][0] for arm in a.arms} | {a.order_from})
    voices = {}
    for p in providers:
        key = os.environ.get(KEY_ENV[p], "")
        if not key:
            raise SystemExit(f"{KEY_ENV[p]} not set: an arm with a missing key is not run")
        model = os.environ.get("NEO4J_MATCHER_MODEL" if p == "gemini" else "NEO4J_MATCHER_OPENAI_MODEL") or DEFAULT_MODELS[p]
        voices[p] = Voice(p, model, key)

    # Held-out state persists (resume; scored once). Dev never persists.
    persist = a.set == "heldout" and not a.check_only
    state_dir = Path(a.state_dir) if a.state_dir else STATE_DIR / digest[:16]
    state_file = state_dir / "state.json"
    tmp = None
    state = None
    if persist and state_file.exists():
        state = json.loads(state_file.read_text())
        if state["meta"]["corpus_sha256"] != digest:
            print("REFUSED — the state belongs to a different corpus")
            return 2
        snap_dir = state_dir
    else:
        if persist:
            state_dir.mkdir(parents=True, exist_ok=True)
            snap_dir = state_dir
        else:
            tmp = tempfile.TemporaryDirectory()
            snap_dir = Path(tmp.name)
        for v in voices.values():
            fetch_sidecar(v, a, snap_dir)

    sidecars, shas = {}, {}
    for p, v in voices.items():
        f = sidecar_path(snap_dir / "meaning_sidecar.json", v)
        blob = f.read_bytes()
        shas[p] = hashlib.sha256(blob).hexdigest()
        doc = json.loads(blob)
        if doc.get("voice") != v.id:
            print(f"REFUSED — {f.name} is stamped {doc.get('voice')!r}, not {v.id}")
            return 2
        sidecars[p] = doc["nodes"]
    if state is not None and shas != state["meta"]["sidecar_sha256"]:
        print("REFUSED — the snapshot sidecars no longer match the run being resumed")
        return 2
    order, dropped = shared_order(sidecars, a.order_from)
    in_order = set(order)
    excluded: dict[str, str] = {}
    for c in cases:
        for role in ("target", "decoy"):
            if c.get(role) and c[role] not in in_order:
                excluded.setdefault(c["id"], f"{role} not in every voice's sidecar")
    scored = [c for c in cases if c["id"] not in excluded]

    if a.check_only:
        print(f"CHECK PASSED — corpus sha256 {digest[:16]}; set {a.set}: {len(cases)} cases, "
              f"{len(scored)} scorable in every arm, {len(excluded)} excluded; shared order "
              f"{len(order)} nodes" + "".join(f"; {p} has {len(d)} outside it" for p, d in dropped.items())
              + "".join(f"\n  - {cid}: {why}" for cid, why in sorted(excluded.items()))
              + "\nNo match calls made.")
        if tmp:
            tmp.cleanup()
        return 0

    if state is None:
        state = {"meta": {
            "corpus": a.corpus, "corpus_sha256": digest, "set": a.set, "seed": a.seed,
            "arms": a.arms, "order_from": a.order_from, "order_size": len(order),
            "sidecar_sha256": shas, "sidecar_sizes": {p: len(n) for p, n in sidecars.items()},
            "dropped": dropped, "excluded": excluded,
            "models": {p: v.model for p, v in voices.items()},
            "native_shapes": dict(NATIVE_SHAPE),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, "repeats": a.repeats, "runs": {}}
    elif (state["meta"]["arms"], state["repeats"], state["meta"]["seed"]) != (a.arms, a.repeats, a.seed):
        print("REFUSED — arms, repeats and seed must match the run being resumed")
        return 2

    def save():
        if persist:
            tmpf = state_file.with_suffix(".tmp")
            tmpf.write_text(json.dumps(state, indent=1))
            tmpf.replace(state_file)

    arms_cfg = {}
    for arm in a.arms:
        p, shape = ARMS[arm]
        prefix, names = build_prefix({n: sidecars[p][n] for n in order})
        arms_cfg[arm] = (voices[p], shape, prefix, names)
    tasks = schedule([c["id"] for c in scored], a.arms, a.repeats, a.seed)
    pending = sum(1 for t in tasks if task_key(*t) not in state["runs"])
    paused: dict[str, str] = {}
    if pending:
        print(f"{'resuming' if state['runs'] else 'starting'}: {pending} of {len(tasks)} calls to make",
              file=sys.stderr)
        paused = asyncio.run(run_tasks(state, tasks, arms_cfg, {c["id"]: c for c in scored}, a.timeout_ms, save))
    save()
    incomplete = sum(1 for t in tasks if task_key(*t) not in state["runs"])

    report = render(state, scored, a.arms, incomplete, paused)
    if leaked:
        report += "\n\nLEAKS (run with --allow-leaks): " + "; ".join(
            f"{cid}: {', '.join(w)}" for cid, w in leaked.items())
    report = redact(report)
    print(report)
    now = datetime.now(timezone.utc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{now:%Y%m%dT%H%M%SZ}-matcher-t0-{a.set}"
    stem.with_suffix(".md").write_text(report + "\n")
    stem.with_suffix(".json").write_text(redact(json.dumps(state, indent=1)) + "\n")
    print(f"\nreport: {stem.with_suffix('.md')}" + (f"\nstate:  {state_file}" if persist else ""))
    if tmp:
        tmp.cleanup()
    return 3 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
