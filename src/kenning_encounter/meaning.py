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
    Kenning Encounter's contents are authored by the trajectory, and machine-generated text
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

VOICES (v0.17.0). A voice is one model that BOTH compresses a sidecar and
matches against it — single-voice authorship stated as a type. More than one
voice may be configured (Gemini and OpenAI today); each keeps its OWN sidecar,
stamped with the voice that wrote it, and the index refuses a sidecar stamped
by any other voice. One voice leads and the others are fallbacks; which one
leads is configuration, not a judgment written into the code, because which
voice brings more meaning to an infusion has not been measured and is not
ours to assume.

Prefix caching is the latency story: the meanings block is byte-identical
between calls and sits FIRST in the request; only the trajectory tail
changes. The result carries usageMetadata's cached-token count so Gate 3
can verify caching actually engaged before latency is scored.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    """The matcher could not run — the caller falls back to Extract seeds.

    `attempts` is the per-voice account when a whole chain failed: every
    voice tried, skipped or refused, each with its reason."""

    def __init__(self, message: str, attempts: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


# -- Voices -------------------------------------------------------------------

PROVIDERS = ("gemini", "openai")
DEFAULT_ENDPOINTS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openai": "https://api.openai.com/v1",
}
DEFAULT_MODELS = {"gemini": "gemini-3.5-flash-lite", "openai": "gpt-6-luna"}
KEY_ENV = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}


@dataclass(frozen=True)
class Voice:
    """One meaning-maker: the model that compresses a sidecar AND matches
    against it. The pairing is the point. The T0 matcher passed because one
    model wrote both sides; a matcher reading another model's meanings is
    the two-idiolect gap that sank query lifting, reopened silently.

    The key is excluded from repr so a voice can be logged."""

    provider: str
    model: str
    api_key: str = field(default="", repr=False)
    endpoint: str = ""

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    @property
    def base(self) -> str:
        return (self.endpoint or DEFAULT_ENDPOINTS[self.provider]).rstrip("/")


def sidecar_path(base: str | Path, voice: Voice) -> Path:
    """Where this voice's sidecar lives: beside `base`, named for the voice.

    models/meaning_sidecar.json -> models/meaning_sidecar.gemini-gemini-3.5-flash-lite.json.
    The un-suffixed base file is never read: it predates the voice stamp, so
    nothing in it says which model wrote it."""
    p = Path(base)
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", voice.id.replace(":", "-"))
    return p.with_name(f"{p.stem}.{tag}{p.suffix}")


def voice_order(voices: list[Voice], lead: str, session_id: str | None) -> list[Voice]:
    """The voices in the order this call tries them.

    `lead` names a provider, or 'alternate': the lead is then chosen per
    LOCUS from a hash of the harness session id, so one session hears one
    voice throughout (its prefix cache stays warm, and its infusions are
    comparable with each other) while loci divide between voices without
    either being favoured by when it ran. Stateless: the same session id
    always lands on the same voice, across restarts."""
    if not voices:
        return []
    if lead == "alternate":
        i = 0
        if session_id:
            i = int(hashlib.sha1(session_id.encode()).hexdigest(), 16) % len(voices)
        return voices[i:] + voices[:i]
    first = [v for v in voices if v.provider == lead]
    return first + [v for v in voices if v.provider != lead]


class VoiceHealth:
    """A failed voice cools down: for `cooldown_s` after its last failure
    it is tried AFTER the healthy voices instead of first.

    Without this a slow outage costs every prompt its whole budget. On
    2026-10-01 Gemini's successes took 13-18 s; a lead that times out at
    5 s leaves the fallback nothing, so each prompt would fall to lexical
    seeds while a healthy second voice sat unused. A cooling voice is
    reordered, never removed: if every voice is cooling, they are all
    still tried."""

    def __init__(
        self, cooldown_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failed: dict[str, float] = {}

    def failed(self, voice_id: str) -> None:
        self._failed[voice_id] = self._clock()

    def ok(self, voice_id: str) -> None:
        self._failed.pop(voice_id, None)

    def cooling(self, voice_id: str) -> bool:
        t = self._failed.get(voice_id)
        return t is not None and self._clock() - t < self.cooldown_s


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


# -- Deferral reporting (never the node) --------------------------------------

# Cap on one reason string. The exception text is our own or httpx's today
# and carries no node content, but a reason is the one field that could
# reintroduce the leak this exists to close if a future exception ever
# embedded what it was compressing.
_DEFERRAL_REASON_CAP = 200


def record_deferral(counts: dict[str, int], exc: BaseException) -> None:
    """Tally one compression failure by REASON — the node is not a parameter.

    The signature is the guard. The sidecar sweep used to log one line per
    failed node, naming it; keyless, compress_meaning raises before any
    network call, so that branch fires once per node and a keyless start
    wrote the trajectory's entire corpus of node names into `docker logs`,
    readable by anything on the host. Node names in a Kenning Encounter are not
    identifiers — they are the substrate's content. Leaking a name cannot be
    a call-site mistake here because a name cannot be passed in.
    """
    reason = f"{type(exc).__name__}: {exc}"[:_DEFERRAL_REASON_CAP]
    counts[reason] = counts.get(reason, 0) + 1


def deferral_summary(counts: dict[str, int]) -> str:
    """One line: the total, how many distinct reasons, and each with its count.

    Reports the count AND the reasons rather than either alone — a bare total
    hides which failure it was, and a bare reason list hides how much of the
    corpus is missing. Ordered by frequency so the dominant cause reads first,
    then by reason for a stable line across runs.
    """
    if not counts:
        return ""
    detail = "; ".join(
        f"{n}x {reason}"
        for reason, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return (
        f"{sum(counts.values())} deferred "
        f"({len(counts)} distinct reason{'' if len(counts) == 1 else 's'}) — {detail}"
    )


# -- Sidecar ------------------------------------------------------------------


class MeaningIndex:
    """The loaded sidecar: name -> {meaning, type, hash}, with the static
    matcher prefix prebuilt. Reloads when the file's mtime changes, and
    persists its own updates atomically — the server both reads and
    maintains this file (it lives in the stack's named volume).

    Bound to ONE voice (v0.17.0). The file records the voice that wrote it,
    and a file stamped by another voice — or by none — is refused for
    matching and never merged into on write: the first upsert under this
    voice starts the file over. Before the stamp, swapping
    NEO4J_MATCHER_MODEL made the new model read the old model's meanings
    with nothing saying so."""

    def __init__(self, path: str | Path, voice_id: str = ""):
        self.path = Path(path)
        self.voice_id = voice_id
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
        written_by = str(data.get("voice") or "")
        if self.voice_id and written_by != self.voice_id:
            raise MeaningUnavailable(
                f"meaning sidecar at {self.path} was written by "
                f"{written_by or 'an unrecorded voice'}, not {self.voice_id} "
                "— refusing to match one model against another's meanings"
            )
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
        current = self._own_nodes()
        current.update(entries)
        self._write(current)

    def remove(self, names: list[str]) -> None:
        """Drop departed nodes from the sidecar, atomically."""
        current = self._own_nodes()
        changed = False
        for name in names:
            if name in current:
                del current[name]
                changed = True
        if changed:
            self._write(current)

    def _own_nodes(self) -> dict[str, dict[str, Any]]:
        """The file's entries IF this voice wrote them, else nothing — so a
        write never carries another voice's meanings forward under this
        voice's stamp."""
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return {}
        if self.voice_id and str(data.get("voice") or "") != self.voice_id:
            return {}
        return dict(data.get("nodes", {}))

    def _write(self, nodes: dict[str, dict[str, Any]]) -> None:
        payload: dict[str, Any] = {"prompt_version": PROMPT_VERSION}
        if self.voice_id:
            payload["voice"] = self.voice_id
        payload["nodes"] = nodes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        tmp.replace(self.path)
        self._mtime = None  # next prefix() reloads the written file

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
    """The static meanings block, numbered in the sidecar's INSERTION order.

    Insertion order, not name order, because the prefix is a cache key.
    `upsert` appends a new node and edits an existing one in place, so a
    write leaves every line before it byte-identical: appending costs only
    the delta, and an edit invalidates from that node's position onward.
    Sorting by name renumbered nearly the whole block on any write whose
    name sorted early — a total cache miss per write, which on a local
    backend is a full cold prefill (a field deployment measured 634s at
    ~760 nodes). Deterministic for a given sidecar file either way; the
    numbering never leaves the call (parse_matcher_output maps it back
    through ordered_names)."""
    ordered = list(nodes)
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

# OpenAI's per-request cache routing hint. Constant, so every call from this
# server lands where the meanings prefix is already cached.
_OPENAI_CACHE_KEY = "kenning_encounter-matcher"


def _openai_base(voice: Voice, max_tokens: int) -> dict[str, Any]:
    """The settings measured for gpt-6-luna on the matcher's shape
    (llm-triage profile_openai.py, 2026-10-01): no reasoning (0.9 s p50 vs
    2.9 s at the default effort, every answer parsed), low verbosity, and
    nothing stored at the provider. temperature is accepted only at effort
    none; 0 matches the Gemini voice."""
    return {
        "model": voice.model,
        "store": False,
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
        "temperature": 0,
        "max_output_tokens": max_tokens,
    }


def compress_request(voice: Voice, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    """(url, headers, body) for one compression call. Pure."""
    if voice.provider == "openai":
        body = _openai_base(voice, _COMPRESS_MAX_TOKENS)
        body["input"] = prompt
        return (f"{voice.base}/responses",
                {"Authorization": f"Bearer {voice.api_key}"}, body)
    return (
        f"{voice.base}/models/{voice.model}:generateContent",
        {"x-goog-api-key": voice.api_key},
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": _COMPRESS_MAX_TOKENS,
            },
        },
    )


# The request shape each voice is deployed with. "split" puts the meanings
# prefix in its own instruction slot (OpenAI developer message, Gemini
# systemInstruction) and the trajectory in the user turn; "flat" sends
# prefix + trajectory as one user text. A voice is model AND shape; the
# other shape exists so a comparison can separate the two.
# Gemini moved from flat to split in v0.18.1: within Gemini, split ranked
# relevant nodes higher in comparative runs, and its prefix cache held.
NATIVE_SHAPE = {"gemini": "split", "openai": "split"}
# What production sent before v0.18.1, for reading runs made under it.
NATIVE_SHAPE_BEFORE_0_18_1 = {"gemini": "flat", "openai": "split"}
SHAPES = ("flat", "split")


def match_request(
    voice: Voice, prefix: str, tail: str, shape: str | None = None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """(url, headers, body) for one match call. Pure.

    The prefix (header + meanings) is byte-identical between calls and goes
    first either way. By default each voice gets its native shape, both
    now "split": the prefix in the instruction slot, the trajectory in the
    user turn."""
    shape = shape or NATIVE_SHAPE[voice.provider]
    if shape not in SHAPES:
        raise ValueError(f"unknown request shape {shape!r}")
    if voice.provider == "openai" and shape == "flat":
        body = _openai_base(voice, _MATCHER_MAX_OUTPUT)
        body["prompt_cache_key"] = _OPENAI_CACHE_KEY
        body["input"] = [
            {"role": "user",
             "content": [{"type": "input_text", "text": prefix + tail}]},
        ]
        return (f"{voice.base}/responses",
                {"Authorization": f"Bearer {voice.api_key}"}, body)
    if voice.provider == "gemini" and shape == "split":
        return (
            f"{voice.base}/models/{voice.model}:generateContent",
            {"x-goog-api-key": voice.api_key},
            {
                "systemInstruction": {"parts": [{"text": prefix}]},
                "contents": [{"role": "user", "parts": [{"text": tail.lstrip()}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": _MATCHER_MAX_OUTPUT,
                },
            },
        )
    if voice.provider == "openai":
        body = _openai_base(voice, _MATCHER_MAX_OUTPUT)
        body["prompt_cache_key"] = _OPENAI_CACHE_KEY
        body["input"] = [
            {"role": "developer",
             "content": [{"type": "input_text", "text": prefix}]},
            {"role": "user",
             "content": [{"type": "input_text", "text": tail.lstrip()}]},
        ]
        return (f"{voice.base}/responses",
                {"Authorization": f"Bearer {voice.api_key}"}, body)
    return (
        f"{voice.base}/models/{voice.model}:generateContent",
        {"x-goog-api-key": voice.api_key},
        {
            "contents": [{"parts": [{"text": prefix + tail}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": _MATCHER_MAX_OUTPUT,
            },
        },
    )


def response_text(voice: Voice, data: dict[str, Any]) -> str:
    """The generated text out of either provider's response. Pure."""
    if voice.provider == "openai":
        return "".join(
            c.get("text", "")
            for o in data.get("output", []) if o.get("type") == "message"
            for c in o.get("content", []) if c.get("type") == "output_text"
        )
    parts = data["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def response_generation(voice: Voice, data: dict[str, Any]) -> tuple[int, int]:
    """(output_tokens, reasoning_tokens) — whether a call reasoned is
    measured per call, never assumed from the settings."""
    if voice.provider == "openai":
        u = data.get("usage") or {}
        return (int(u.get("output_tokens") or 0),
                int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0))
    u = data.get("usageMetadata") or {}
    return (int(u.get("candidatesTokenCount") or 0),
            int(u.get("thoughtsTokenCount") or 0))


def response_usage(voice: Voice, data: dict[str, Any]) -> tuple[int, int]:
    """(prompt_tokens, cached_tokens) — the cache evidence, either provider."""
    if voice.provider == "openai":
        u = data.get("usage") or {}
        return (int(u.get("input_tokens") or 0),
                int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0))
    u = data.get("usageMetadata") or {}
    return (int(u.get("promptTokenCount") or 0),
            int(u.get("cachedContentTokenCount") or 0))


async def _post(url: str, headers: dict[str, str], body: dict[str, Any],
                timeout_s: float) -> dict[str, Any]:
    """POST and return JSON, raising on any HTTP failure. The error names
    the status and URL only — never the response body, which at OpenAI can
    echo part of a rejected key."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _not_configured(voice: Voice) -> str:
    return f"{voice.id} not configured ({KEY_ENV.get(voice.provider, 'key')})"


async def compress_meaning(
    node: dict[str, Any],
    voice: Voice,
    timeout_s: float = 30.0,
) -> str:
    """Compress one node's name+description into its sidecar meaning —
    the server-side twin of the offline builder's per-node call, used by
    the on-write trigger and the startup reconcile sweep. Raises
    MeaningUnavailable on any failure; callers log and move on (the node
    stays visible to the lexical path and the next sweep retries it)."""
    if not voice.configured:
        raise MeaningUnavailable(f"compression: {_not_configured(voice)}")
    prompt = MEANING_PROMPT.format(
        type=node.get("type", "?"),
        name=node["name"],
        description=(node.get("description") or "")[:_COMPRESS_DESC_CAP],
    )
    url, headers, body = compress_request(voice, prompt)
    try:
        data = await _post(url, headers, body, timeout_s)
        meaning = " ".join(response_text(voice, data).split())
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
    voice: Voice,
    timeout_ms: float = 2500,
    include_reasons: bool = False,
    shape: str | None = None,
) -> dict[str, Any]:
    """One match call: cached prefix + trajectory tail -> top_n
    selections. Raises MeaningUnavailable on every failure mode; the
    caller's fallback is the only handler.

    include_reasons defaults OFF, and the default is measured, not
    stylistic: at top_n=30 the reasons cost 554-762 output tokens and
    3.5-4.2s of generation (a guaranteed ReadTimeout at the 2500ms hard
    cap — Desktop's first run hit exactly this), while bare numbers cost
    ~115 tokens and ~1.2s. Reasons are debugging artifacts by the work
    order's own definition; request them only at small top_n."""
    if not voice.configured:
        raise MeaningUnavailable(f"matcher: {_not_configured(voice)}")

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
    url, headers, body = match_request(voice, prefix, tail, shape)
    t0 = time.perf_counter()
    try:
        data = await _post(url, headers, body, timeout_ms / 1000.0)
        raw = response_text(voice, data)
        prompt_tokens, cached_tokens = response_usage(voice, data)
        output_tokens, reasoning_tokens = response_generation(voice, data)
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
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


# A fallback attempt gets the budget the earlier ones left. Below this it is
# skipped: a cached match has never been measured under ~1.2 s, so a smaller
# remainder buys a timeout, not a selection.
MIN_ATTEMPT_MS = 1500


async def match_chain(
    candidates: list[tuple[Voice, MeaningIndex]],
    trajectory_text: str,
    top_n: int,
    health: VoiceHealth,
    timeout_ms: float,
    include_reasons: bool = False,
    clock: Callable[[], float] = time.perf_counter,
    matcher: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Try the voices in order until one selects. One shared budget.

    `candidates` arrive in lead order (voice_order). Cooling voices move
    behind healthy ones. A voice runs only against ITS OWN sidecar, so a
    fallback never mixes idiolects. Returns the winning match plus `voice`,
    `index` and `attempts` — the per-voice account of everything tried
    before it. Raises MeaningUnavailable carrying the same account when
    every voice fails.

    The budget is shared rather than per voice because the hook that waits
    on this has one timeout (10 s by default) and the payload still has to
    be assembled after the match."""
    run = matcher or match_meanings
    ordered = (
        [c for c in candidates if not health.cooling(c[0].id)]
        + [c for c in candidates if health.cooling(c[0].id)]
    )
    attempts: list[dict[str, Any]] = []
    t0 = clock()
    for voice, index in ordered:
        if not voice.configured:
            attempts.append({"voice": voice.id, "error": _not_configured(voice)})
            continue
        remaining = timeout_ms - (clock() - t0) * 1000
        if attempts and remaining < MIN_ATTEMPT_MS:
            attempts.append({
                "voice": voice.id,
                "error": f"skipped: {remaining:.0f} ms of budget left",
            })
            continue
        try:
            prefix, names = index.prefix()
            matched = await run(
                prefix, names, trajectory_text, top_n, voice,
                timeout_ms=remaining, include_reasons=include_reasons,
            )
        except MeaningUnavailable as e:
            health.failed(voice.id)
            attempts.append({"voice": voice.id, "error": str(e)})
            continue
        health.ok(voice.id)
        attempts.append({"voice": voice.id, "ms": matched["ms"]})
        return {**matched, "voice": voice, "index": index, "attempts": attempts}
    reason = "; ".join(f"{a['voice']}: {a['error']}" for a in attempts) or "no voice configured"
    raise MeaningUnavailable(reason, attempts)


def new_match_id() -> str:
    """Joins one infusion's observe-log line to its shadow-log line."""
    return uuid.uuid4().hex[:12]


def shadow_record(
    match_id: str,
    session_id: str | None,
    lead: str,
    delivered: dict[str, Any],
    shadow: dict[str, Any],
) -> dict[str, Any]:
    """One line of the shadow log: what the delivering voice selected and
    what another voice selected for the same trajectory.

    Mechanics only — selections, timings, token counts, sidecar size. It
    never holds the trajectory text, and it carries no verdict: two
    different selections can each bear on a moment. Whether either carried
    meaning is judged in the window, by the human frame, in the observe
    log; match_id is the join."""
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "match_id": match_id,
        "session_id": session_id or "",
        "lead": lead,
        "delivered": delivered,
        "shadow": shadow,
    }


def voice_summary(
    voice: Voice, matched: dict[str, Any] | None, index: MeaningIndex,
    error: str | None = None,
) -> dict[str, Any]:
    """The per-voice half of a shadow record."""
    out: dict[str, Any] = {"voice": voice.id, "sidecar_size": index.size}
    if matched is None:
        out["error"] = error or "no result"
        return out
    out.update({
        "selections": [s["name"] for s in matched["selections"]],
        "ms": matched["ms"],
        "prompt_tokens": matched.get("prompt_tokens", 0),
        "cached_tokens": matched.get("cached_tokens", 0),
    })
    return out


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
    voice: str | None = None,
) -> str:
    """The meaning arm's report: matcher selection first (with the
    debugging-artifact reasons, labeled as such), then the resolved
    pipeline candidates, then timing with the cache evidence Gate 3 needs."""
    lines = ["ARM M — MEANING MATCHER (trajectory ↔ compressed meanings)"]
    if voice:
        lines.append(f"  VOICE: {voice} (its own sidecar)")
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


# -- Adopting a pre-voice sidecar ----------------------------------------------


def adopt_legacy_sidecar(base: str | Path, voice: Voice) -> Path:
    """Stamp a pre-v0.17.0 sidecar as written by `voice`, copying it to
    that voice's path. Returns the new path.

    A sidecar from before the stamp says nothing about which model wrote
    it, so the server never reads it; left alone, the reconcile sweep
    rebuilds the voice's sidecar from the graph, one compression per node.
    Adoption skips that rebuild, and it is an ASSERTION: the operator who
    runs it is stating which model wrote the file — knowledge the file
    cannot supply. So it does only the narrow thing that assertion
    licenses: an unstamped source, an explicitly named voice, no
    overwrite, insertion order kept (the prefix cache survives). The
    source is left in place."""
    src = Path(base)
    dst = sidecar_path(src, voice)
    try:
        data = json.loads(src.read_text())
    except Exception as exc:
        raise MeaningUnavailable(f"cannot read {src}: {exc}") from exc
    if data.get("voice"):
        raise MeaningUnavailable(
            f"{src} is already stamped by {data['voice']}; nothing to adopt"
        )
    if dst.exists():
        raise MeaningUnavailable(f"{dst} already exists; refusing to overwrite it")
    nodes = data.get("nodes") or {}
    if not nodes:
        raise MeaningUnavailable(f"{src} holds no meanings")
    index = MeaningIndex(dst, voice.id)
    index._write(dict(nodes))
    return dst


def _main(argv: list[str] | None = None) -> int:
    """python -m kenning_encounter.adopt_sidecar --adopt PROVIDER:MODEL [--sidecar PATH]"""
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m kenning_encounter.adopt_sidecar",
        description=(
            "Adopt a pre-v0.17.0 meaning sidecar for the voice that wrote it. "
            "You are asserting which model wrote the file; the file cannot "
            "say. Without adoption the server rebuilds the sidecar itself."
        ),
    )
    ap.add_argument("--adopt", required=True, metavar="PROVIDER:MODEL",
                    help="e.g. gemini:gemini-3.5-flash-lite")
    ap.add_argument("--sidecar", default="/app/models/meaning_sidecar.json",
                    help="the unstamped sidecar (default: %(default)s)")
    a = ap.parse_args(argv)
    provider, _, model = a.adopt.partition(":")
    if provider not in PROVIDERS or not model:
        ap.error(f"--adopt must be PROVIDER:MODEL with PROVIDER one of {PROVIDERS}")
    try:
        dst = adopt_legacy_sidecar(a.sidecar, Voice(provider, model))
    except MeaningUnavailable as e:
        print(f"refused: {e}")
        return 1
    print(f"adopted {a.sidecar} as {provider}:{model} -> {dst}")
    return 0

