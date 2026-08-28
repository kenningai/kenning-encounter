"""The meaning matcher — Gemini as the matcher (T0, temporary).

The convergence of three measured negatives (lexical 2/8, names-only MaxSim
2/8 on negation, query lifting 4/8 with blending actively harmful): every
intermediate representation between the prompt and the corpus discarded
meaning at a comparison function. This design stops building intermediates
and lets the meaning-maker be the matcher: offline, one model compresses
every node's name+description into a ~20-30-token meaning (the SIDECAR);
per prompt, the SAME model reads all ~519 compressed meanings as a static
prefix plus the session TRAJECTORY (not a snapshot — meaning is temporal,
and every prior attempt selected against a tenseless point) and returns
the nodes that bear on the current moment, unnamed constraints included.

What the design resolves by construction: two idiolects (one model authors
both sides), negation (a model understands "not"), the level gap (it reads
abstraction, not tokens), unnamed constraints and snapshot selection (the
arc is the input). At 519 nodes the whole collection fits in context —
similarity search exists for corpora this design does not have.

STATUS AND BOUNDARIES (shipped in v0.9.0 after the T0 gates passed — recall
7/8 vs the 2/8 lexical baseline, trajectory +2):
  - The matcher is infuse's full-mode SEED SOURCE; everything downstream
    (frontier bias, B2 expansion, biased rank, triage, assembly, renewal)
    is unchanged, and infuse_meaning remains as the diagnostic surface.
  - SIDECAR, NEVER NODE PROPERTIES — settled, self-constitutively: the
    Kenning Encounter's contents are authored by the agent, and machine-generated text
    stored inside its nodes would be another voice in its own. The sidecar
    serves the identical retrieval function and keeps the graph authored.
    It lives in the stack's named volume and MAINTAINS ITSELF: the
    on-write trigger meaning-makes new/edited nodes, the startup reconcile
    sweep (hash- and PROMPT_VERSION-aware) covers the tail, deletions drop
    entries.
  - Nothing is written to Neo4j; the graph is read-only throughout.
  - Fallback guaranteed: on missing sidecar, missing key, timeout,
    transport error, or malformed output the caller falls back to the
    lexical Extract path and reports it — the matcher can never block a
    selection.
  - The per-node one-line reason is a DEBUGGING ARTIFACT: generated
    alongside the selection, not read off the mechanism. It is labeled as
    such in the report and must never be recorded as evidence about why
    selection worked.

Prefix caching is the latency story: the meanings block is byte-identical
between calls and sits FIRST in the request; only the trajectory tail
changes. The result carries usageMetadata's cached-token count so Gate 3
can verify caching actually engaged before latency is scored.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

# The sidecar compression prompt (work order Phase 1 — iterate on the dev
# set, never the held-out set).
MEANING_PROMPT = (
    "Read this node from an agent's own knowledge substrate. In one "
    "sentence of AT MOST 30 words, state what it means and when it would "
    "bear on a decision — including situations it does not name. Write "
    "plainly. No preamble — never begin with 'This node', 'This concept', "
    "or 'This means'; start directly with the content.\n\n"
    "TYPE: {type}\nNAME: {name}\nDESCRIPTION: {description}"
)

# The static matcher header. DELIBERATELY VARIABLE-FREE: everything that
# changes per call (trajectory, top_n) rides AFTER the meanings block, so
# the header + meanings form one byte-identical cacheable prefix.
MATCHER_HEADER = (
    "You are the matcher for an agent's own knowledge substrate. Below is "
    "the complete collection of its node meanings, numbered. After the "
    "collection comes the recent trajectory of a working session, most "
    "recent last, followed by the selection request. Select the nodes "
    "whose meaning most bears on the current moment of the trajectory — "
    "including nodes whose relevance is an unnamed constraint on the "
    "situation rather than its topic. Output one line per selection, best "
    "first: the node number, then a space-dash-space, then a one-line "
    "reason. Numbers must come from the collection. No other text.\n\n"
    "NODE MEANINGS:\n"
)

_MAX_TURN_CHARS = 2000        # per-turn cap before budgeting
_DIGEST_SNIP_CHARS = 200      # older-half turns snip to this in the digest
_MATCHER_MAX_OUTPUT = 1024    # ~30 lines of number+reason; flash-lite thinks 0

_LINE_RE = re.compile(r"^\s*(\d{1,4})\b[\s.—:–-]*(.*)$")


class MeaningUnavailable(Exception):
    """The matcher could not run — the caller falls back to Extract seeds."""


# The compression prompt's version. Bumping it makes the reconcile sweep
# recompress EVERY node on next startup — the in-code form of the builder's
# --force, which matters now that the sidecar lives in a named Docker
# volume no host-side tool writes to.
PROMPT_VERSION = "v2"


def content_hash(name: str, description: str) -> str:
    """The sidecar's change key: sha1 over name + description. One
    definition, shared by the offline builder and the server's on-write
    trigger — two implementations of this hash would mean two opinions
    about what 'unchanged' means."""
    import hashlib

    return hashlib.sha1(f"{name}\x00{description}".encode()).hexdigest()[:16]


def sidecar_diff(
    graph_nodes: list[dict[str, Any]],
    sidecar_nodes: dict[str, dict[str, Any]],
    sidecar_version: str = PROMPT_VERSION,
) -> tuple[list[dict[str, Any]], list[str]]:
    """(to_compress, to_remove) between the graph and the sidecar.

    graph_nodes: [{name, type, description}] — the current corpus.
    A node compresses when absent or when its content hash changed; a
    sidecar entry is removed when its node left the graph. A sidecar
    written under a DIFFERENT compression-prompt version recompresses
    wholesale — same content, different meaning-making — so a prompt
    change ships as a code change and takes effect at the next startup.
    Pure — this is the reconcile sweep's judgment, testable without a DB
    or an API."""
    force = sidecar_version != PROMPT_VERSION
    to_compress: list[dict[str, Any]] = []
    graph_names: set[str] = set()
    for n in graph_nodes:
        graph_names.add(n["name"])
        h = content_hash(n["name"], n.get("description") or "")
        prev = sidecar_nodes.get(n["name"])
        if force or not prev or prev.get("hash") != h or not prev.get("meaning"):
            to_compress.append({**n, "hash": h})
    to_remove = [name for name in sidecar_nodes if name not in graph_names]
    return to_compress, to_remove


# -- Sidecar ------------------------------------------------------------------


class MeaningIndex:
    """The loaded sidecar: name -> {meaning, type, hash}, with the static
    matcher prefix prebuilt. Reloads when the file's mtime changes, and
    persists its own updates atomically — the server both reads and
    maintains this file (it lives in the stack's named volume)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mtime: float | None = None
        self._nodes: dict[str, dict[str, Any]] = {}
        self._version: str = PROMPT_VERSION
        self._prefix: str = ""
        self._ordered_names: list[str] = []

    def _load_if_changed(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError as exc:
            raise MeaningUnavailable(
                f"meaning sidecar missing at {self.path} "
                "(run scripts/build_meaning_sidecar.py)"
            ) from exc
        if mtime == self._mtime:
            return
        try:
            data = json.loads(self.path.read_text())
            nodes = data["nodes"]
        except Exception as exc:
            raise MeaningUnavailable(f"meaning sidecar unreadable: {exc}") from exc
        if not nodes:
            raise MeaningUnavailable("meaning sidecar is empty")
        self._nodes = nodes
        self._version = str(data.get("prompt_version", ""))
        self._prefix, self._ordered_names = build_prefix(nodes)
        self._mtime = mtime

    @property
    def size(self) -> int:
        try:
            self._load_if_changed()
        except MeaningUnavailable:
            pass
        return len(self._nodes)

    def prefix(self) -> tuple[str, list[str]]:
        self._load_if_changed()
        return self._prefix, self._ordered_names

    def node_type(self, name: str) -> str:
        return self._nodes.get(name, {}).get("type", "?")

    def upsert(self, entries: dict[str, dict[str, Any]]) -> None:
        """Merge entries ({name: {type, hash, meaning}}) into the sidecar
        and persist ATOMICALLY (tmp + rename): the file is also read by
        mtime-reload, and a torn read must be impossible. Loads the current
        file first so a concurrent offline build is merged over, not
        clobbered — both writers converge because both key on the same
        content hash. Creates the sidecar if it does not exist yet."""
        try:
            current = json.loads(self.path.read_text()).get("nodes", {})
        except Exception:
            current = {}
        current.update(entries)
        payload = {"prompt_version": PROMPT_VERSION, "nodes": current}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        tmp.replace(self.path)
        self._mtime = None  # next prefix() reloads the merged file

    def remove(self, names: list[str]) -> None:
        """Drop departed nodes from the sidecar, atomically."""
        try:
            current = json.loads(self.path.read_text()).get("nodes", {})
        except Exception:
            return
        changed = False
        for name in names:
            if name in current:
                del current[name]
                changed = True
        if not changed:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"prompt_version": PROMPT_VERSION, "nodes": current},
            indent=1, ensure_ascii=False,
        ))
        tmp.replace(self.path)
        self._mtime = None

    def file_version(self) -> str:
        """The compression-prompt version the sidecar was written under
        (empty when the file is absent/unreadable — which reads as a
        version mismatch and forces a full recompress, correctly)."""
        try:
            self._load_if_changed()
        except MeaningUnavailable:
            return ""
        return self._version

    def known_nodes(self) -> dict[str, dict[str, Any]]:
        """The sidecar's current entries (loading if needed) — the
        reconcile sweep's view of what has already been meaning-made."""
        try:
            self._load_if_changed()
        except MeaningUnavailable:
            return {}
        return dict(self._nodes)


def build_prefix(nodes: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """The static meanings block. Sorted by name so numbering — and
    therefore the cacheable prefix — is deterministic for a given sidecar;
    a sidecar refresh renumbers, costing exactly one cache miss."""
    ordered = sorted(nodes)
    lines = [MATCHER_HEADER]
    for i, name in enumerate(ordered, 1):
        meaning = " ".join(str(nodes[name].get("meaning", "")).split())
        lines.append(f"{i}. {name} :: {meaning}")
    return "\n".join(lines) + "\n", ordered


# -- Trajectory ---------------------------------------------------------------


def assemble_trajectory(
    prior_turns: list[str], current: str, max_chars: int = 12_000
) -> tuple[str, dict[str, int]]:
    """Fixed-budget trajectory text, current prompt last.

    Recent turns survive verbatim, newest backwards, until the budget; the
    older remainder folds into a story-so-far digest (each turn snipped) —
    the work order's compress-the-older-half, implemented deterministically
    so no second generation call rides inside the latency budget."""
    current = current.strip()[: _MAX_TURN_CHARS * 2]
    turns = [t.strip()[:_MAX_TURN_CHARS] for t in prior_turns if t and t.strip()]

    budget = max_chars - len(current)
    verbatim: list[str] = []
    digest: list[str] = []
    for t in reversed(turns):
        if budget - len(t) > 0 and not digest:
            verbatim.insert(0, t)
            budget -= len(t)
        else:
            snip = t[:_DIGEST_SNIP_CHARS]
            if budget - len(snip) > 0:
                digest.insert(0, snip + ("…" if len(t) > len(snip) else ""))
                budget -= len(snip)
            # else: dropped entirely — the budget is fixed, not growing.

    parts: list[str] = ["TRAJECTORY (oldest first, most recent last):"]
    if digest:
        parts.append("story so far (older turns, digested):")
        parts.extend(f"  · {d}" for d in digest)
    for i, t in enumerate(verbatim, 1):
        parts.append(f"[turn -{len(verbatim) - i + 1}] {t}")
    parts.append(f"[current prompt] {current}")
    meta = {
        "verbatim_turns": len(verbatim),
        "digested_turns": len(digest),
        "dropped_turns": len(turns) - len(verbatim) - len(digest),
    }
    return "\n".join(parts), meta


def user_turns_from_transcript(path: str | Path, n: int) -> list[str]:
    """The last n real user turns from a Claude Code transcript (JSONL).

    Tolerant by design: unparseable lines are skipped, tool_result-only
    entries are not user turns, and hook-injected <system-reminder> blocks
    are stripped — the trajectory is what the person said, not what the
    harness wrapped around it. Any failure returns [] and the caller
    proceeds prompt-only (reported, never fatal)."""
    turns: list[str] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if entry.get("type") != "user" or entry.get("isMeta"):
                    continue
                message = entry.get("message") or {}
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                texts: list[str] = []
                if isinstance(content, str):
                    texts = [content]
                elif isinstance(content, list):
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                cleaned = [
                    t.strip() for t in texts
                    if t.strip() and not t.lstrip().startswith("<system-reminder")
                ]
                if cleaned:
                    turns.append("\n".join(cleaned))
    except OSError:
        return []
    return turns[-n:]


# -- Matcher output parsing ---------------------------------------------------


def parse_matcher_output(
    raw: str, ordered_names: list[str], top_n: int
) -> list[dict[str, Any]]:
    """Numbered lines -> [{name, reason}], best first, deduped, capped.

    Numbers outside the collection are dropped, not guessed at. An empty
    result means MALFORMED and the caller falls back."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        if not 1 <= idx <= len(ordered_names):
            continue
        name = ordered_names[idx - 1]
        if name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "reason": m.group(2).strip()})
        if len(out) >= top_n:
            break
    return out


# -- The generation calls -----------------------------------------------------

_COMPRESS_DESC_CAP = 4000
_COMPRESS_MAX_TOKENS = 256


async def compress_meaning(
    node: dict[str, Any],
    api_key: str,
    model: str,
    endpoint: str,
    timeout_s: float = 30.0,
) -> str:
    """Compress one node's name+description into its sidecar meaning —
    the server-side twin of the offline builder's per-node call, used by
    the on-write trigger and the startup reconcile sweep. Raises
    MeaningUnavailable on any failure; callers log and move on (the node
    stays visible to the lexical path and the next sweep retries it)."""
    if not api_key or not model:
        raise MeaningUnavailable("compression not configured (GEMINI_API_KEY)")
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx ships with fastmcp
        raise MeaningUnavailable(f"httpx missing: {exc}") from exc

    prompt = MEANING_PROMPT.format(
        type=node.get("type", "?"),
        name=node["name"],
        description=(node.get("description") or "")[:_COMPRESS_DESC_CAP],
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": _COMPRESS_MAX_TOKENS,
        },
    }
    url = f"{endpoint.rstrip('/')}/models/{model}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                url, json=payload, headers={"x-goog-api-key": api_key}
            )
            resp.raise_for_status()
            data = resp.json()
        parts = data["candidates"][0]["content"].get("parts", [])
        meaning = " ".join("".join(p.get("text", "") for p in parts).split())
    except Exception as exc:
        raise MeaningUnavailable(
            f"compress failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not meaning:
        raise MeaningUnavailable("compress returned empty output")
    return meaning


async def match_meanings(
    prefix: str,
    ordered_names: list[str],
    trajectory_text: str,
    top_n: int,
    api_key: str,
    model: str,
    endpoint: str,
    timeout_ms: int = 2500,
    include_reasons: bool = False,
) -> dict[str, Any]:
    """One generateContent call: cached prefix + trajectory tail -> top_n
    selections. Raises MeaningUnavailable on every failure mode; the
    caller's fallback is the only handler.

    include_reasons defaults OFF, and the default is measured, not
    stylistic: at top_n=30 the reasons cost 554-762 output tokens and
    3.5-4.2s of generation (a guaranteed ReadTimeout at the 2500ms hard
    cap — Desktop's first run hit exactly this), while bare numbers cost
    ~115 tokens and ~1.2s. Reasons are debugging artifacts by the work
    order's own definition; request them only at small top_n."""
    if not api_key or not model:
        raise MeaningUnavailable(
            "matcher not configured (GEMINI_API_KEY / NEO4J_LIFTER_MODEL)"
        )
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx ships with fastmcp
        raise MeaningUnavailable(f"httpx missing: {exc}") from exc

    if include_reasons:
        request = (
            f"Select the {top_n} nodes that most bear on the current "
            "moment, best first. One line each: number - reason. No other "
            "text."
        )
    else:
        request = (
            f"Select the {top_n} nodes that most bear on the current "
            "moment, best first. Output ONLY the node numbers, one per "
            "line. No other text."
        )
    tail = f"\n{trajectory_text}\n\n{request}"
    url = f"{endpoint.rstrip('/')}/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prefix + tail}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": _MATCHER_MAX_OUTPUT,
        },
    }
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000.0) as client:
            resp = await client.post(
                url, json=payload, headers={"x-goog-api-key": api_key}
            )
            resp.raise_for_status()
            data = resp.json()
        parts = data["candidates"][0]["content"].get("parts", [])
        raw = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
    except Exception as exc:
        raise MeaningUnavailable(
            f"match failed: {type(exc).__name__}: {exc}"
        ) from exc
    ms = round((time.perf_counter() - t0) * 1000, 1)
    selections = parse_matcher_output(raw, ordered_names, top_n)
    if not selections:
        raise MeaningUnavailable(
            f"malformed matcher output ({len(raw)} chars, 0 selections)"
        )
    return {
        "selections": selections,
        "raw": raw,
        "ms": ms,
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "cached_tokens": usage.get("cachedContentTokenCount", 0),
    }


# -- Report formatting (pure, unit-tested) ------------------------------------


def format_meaning_report(
    selections: list[dict[str, Any]],
    pipeline_rows: list[dict[str, Any]],
    trajectory_meta: dict[str, int],
    match_ms: float | None,
    pipeline_ms: float,
    cached_tokens: int | None = None,
    prompt_tokens: int | None = None,
    sidecar_size: int | None = None,
    fallback: bool = False,
    fallback_reason: str | None = None,
) -> str:
    """The meaning arm's report: matcher selection first (with the
    debugging-artifact reasons, labeled as such), then the resolved
    pipeline candidates, then timing with the cache evidence Gate 3 needs."""
    lines = ["ARM M — MEANING MATCHER (trajectory ↔ compressed meanings)"]
    if fallback:
        lines.append(
            f"  FALLBACK — matcher unavailable ({fallback_reason or 'unknown'}); "
            "ran current Extract behaviour instead"
        )
    else:
        vt = trajectory_meta.get("verbatim_turns", 0)
        dg = trajectory_meta.get("digested_turns", 0)
        dr = trajectory_meta.get("dropped_turns", 0)
        traj = f"{vt} verbatim turn{'s' if vt != 1 else ''}"
        if dg:
            traj += f" + {dg} digested"
        if dr:
            traj += f" + {dr} dropped"
        lines.append(f"  TRAJECTORY: {traj}; current prompt last")

    if selections:
        lines.append("")
        lines.append(
            "MATCHER SELECTION (best first; reasons are a debugging "
            "artifact, not evidence):"
        )
        for i, s in enumerate(selections, 1):
            lines.append(f"  {i:2d}. {s['name']}")
            if s.get("reason"):
                lines.append(f"      ↳ {s['reason']}")

    lines.append("")
    lines.append("PIPELINE CANDIDATES (B2 expansion + rank over matcher seeds):")
    if pipeline_rows:
        for i, row in enumerate(pipeline_rows, 1):
            chans = ",".join(row.get("channels", []) or ["?"])
            lines.append(f"  {i:2d}. {row['name']}   {row['score']:.6f}  [{chans}]")
    else:
        lines.append("  (no candidates)")

    lines.append("")
    match_part = f"match {match_ms:.1f} ms" if match_ms is not None else "match —"
    tail = [match_part, f"pipeline {pipeline_ms:.1f} ms"]
    if prompt_tokens is not None:
        cache_note = (
            f"cached {cached_tokens}/{prompt_tokens} prompt tokens"
            if cached_tokens is not None
            else f"prompt {prompt_tokens} tokens"
        )
        tail.append(cache_note)
    if sidecar_size is not None:
        tail.append(f"sidecar {sidecar_size} meanings")
    lines.append("TIMING:  " + " | ".join(tail))
    return "\n".join(lines)
