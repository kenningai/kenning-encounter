"""The governed infusion pipeline — pure logic (DB-free, unit-tested).

Substrate weight is dispositional, not memory: it must shape perception at
every decision point, and attention attenuation dissolves a once-per-waking
briefing by turn seven. The remedy is to perform the passive synthesis
mechanically — re-awaken a topologically-relevant projection of the substrate
against each arriving present — and to GOVERN it, because in a self-authoring
substrate the reading instrument is a constitutive input to the next state:
mass-only re-injection converges every bound agent on entrenchment (the
gravity well, per-turn). The governed form blends focal seeds with the
standing frontier in one biased rank, foregrounds conflict, flags divergence,
signs its own provenance, and holds silence when the substrate has nothing
to say.

This module holds the judgment-bearing arithmetic of that pipeline —
Extract (focal signals), the Lucene query construction for Match, the
conflict triage (core vs parked, by constitutive proximity), and Format
(the signed payload) — kept pure so it is testable without a driver. The
DB legs (fulltext match, the GDS rank streams, neighborhood/trajectory
reads) live on Neo4jAgentMemory.infuse.

Scope claim, stated honestly: this pipeline does not create the passive
synthesis a biological substrate performs on its own — it compensates,
mechanically, for a substrate that does not yet stay awake between
injections. A prosthesis, not the limb.
"""

import hashlib
import json
import re
from typing import Any

# -- Extract ------------------------------------------------------------------

# Words that carry no focal signal. Deliberately small: over-filtering costs
# recall, and the fulltext scorer already down-weights ubiquitous terms. The
# second block is question-stem vocabulary — the verbs and fillers a prompt is
# asked *with* ("what do we KNOW about…"), never the thing it is about; domain
# nouns stay out of this list on purpose (BM25 IDF and the match-node cap
# bound their reach, and stopping them would cost real recall).
_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has
    have how i if in into is it its just me my no not of on or our so than that
    the their them then there these they this to und up us was we were what
    when where which who why will with would you your about after all also any
    before being between both each get here more most now only other out over
    same should some such
    know knows known knowing think thinks thought want wants wanted need needs
    needed tell tells told say says said see seen seeing look looks looking
    make makes made making going goes went gets got give gives given take takes
    taken let lets help helps try trying tried still thing things like really
    actually maybe please thanks okay yes
    """.split()
)

# Hard focal signals — always extracted at full weight regardless of position:
# IPv4 addresses (optionally CIDR) and dotted names (hostnames, FQDNs,
# index/table names). The kind of term an infrastructure encounter pivots on.
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")
_DOTTED_RE = re.compile(r"\b[A-Za-z][\w-]*(?:\.[A-Za-z][\w-]*)+\b")
# Identifier-shaped tokens: snake_case, kebab-case, CamelCase — the names of
# things (tools, repos, node names) rather than prose words.
_IDENT_RE = re.compile(
    r"\b(?:[A-Za-z][\w]*(?:[_-][\w]+)+|[A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"
)
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]{2,}\b")

# Split-and-weight heuristic for long pasted documents: the question usually
# lives at the edges. Head/tail token counts and the triple weight follow the
# spec; the middle is sampled by term frequency rather than true TF-IDF (no
# corpus exists to compute IDF against — honesty over ceremony).
_LONG_DOC_TOKENS = 400
_HEAD_TOKENS = 200
_TAIL_TOKENS = 100
_EDGE_WEIGHT = 3.0
_HARD_SIGNAL_WEIGHT = 3.0


def extract_focal_signals(text: str, max_terms: int = 24) -> list[dict[str, Any]]:
    """Reduce a prompt or tool-result batch to weighted focal signals.

    Returns [{"term": str, "weight": float}] sorted by weight descending,
    at most max_terms. Hard signals (IPs, dotted names, identifiers) carry
    full weight wherever they appear; prose terms carry positional weight
    (split-and-weight on long documents) accumulated by frequency.
    """
    if not text or not text.strip():
        return []

    scores: dict[str, float] = {}
    display: dict[str, str] = {}

    def _add(term: str, weight: float) -> None:
        key = term.lower()
        if key in _STOPWORDS or len(key) < 3:
            return
        scores[key] = scores.get(key, 0.0) + weight
        display.setdefault(key, term)

    for rx in (_IP_RE, _DOTTED_RE, _IDENT_RE):
        for m in rx.findall(text):
            _add(m, _HARD_SIGNAL_WEIGHT)

    words = _WORD_RE.findall(text)
    if len(words) <= _LONG_DOC_TOKENS:
        for w in words:
            _add(w, 1.0)
    else:
        for w in words[:_HEAD_TOKENS]:
            _add(w, _EDGE_WEIGHT)
        for w in words[-_TAIL_TOKENS:]:
            _add(w, _EDGE_WEIGHT)
        # Middle sample: frequency-scored, budgeted to half the term cap so
        # the edges (where the question lives) keep the majority voice.
        middle: dict[str, int] = {}
        middle_display: dict[str, str] = {}
        for w in words[_HEAD_TOKENS:-_TAIL_TOKENS]:
            key = w.lower()
            if key in _STOPWORDS or len(key) < 3:
                continue
            middle[key] = middle.get(key, 0) + 1
            middle_display.setdefault(key, w)
        top_middle = sorted(middle.items(), key=lambda kv: kv[1], reverse=True)
        for key, count in top_middle[: max_terms // 2]:
            _add(middle_display[key], float(count))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"term": display[key], "weight": round(weight, 2)}
        for key, weight in ranked[:max_terms]
    ]


# -- Match query construction ---------------------------------------------------

_LUCENE_SPECIALS = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


def _escape_lucene(term: str) -> str:
    return _LUCENE_SPECIALS.sub(r"\\\1", term)


def lucene_query(signals: list[dict[str, Any]]) -> str:
    """Build the fulltext Match query from extracted signals.

    Terms are OR-ed; boost is the signal weight clamped to [1, 3] so the
    extraction's positional judgment carries into Lucene scoring without
    letting a frequency runaway dominate the disjunction.
    """
    parts = []
    for s in signals:
        term = _escape_lucene(str(s["term"]))
        boost = max(1, min(3, round(float(s["weight"]))))
        parts.append(f"{term}^{boost}" if boost > 1 else term)
    return " OR ".join(parts)


# -- Conflict triage --------------------------------------------------------------

# A conflict's severity is its constitutive proximity to the focal point —
# and the biased rank already measures exactly that (personalized rank seeded
# from the focal matches IS proximity-to-what-you-are-looking-at). Core
# conflicts lead the payload at full amplitude; peripheral ones are carried in
# a parked-tensions register at the end — noted, unresolved, non-blocking —
# the way a date discrepancy in a reference list is held through an hour-long
# examination and resolved in cleanup. The tension is never dropped (dropping
# it would be dishonest); it is never allowed to seize a frame it doesn't own.
_CORE_SEVERITY = 0.25


def triage_conflicts(
    conflicts: list[dict[str, Any]],
    rank_scores: dict[str, float],
    core_severity: float = _CORE_SEVERITY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split conflicts into (core, parked) by constitutive proximity.

    severity = the max biased-rank score among the conflict's endpoints,
    normalized against the top score in the ranked subgraph. Conflicts at or
    above core_severity lead; the rest park. Each returned row carries its
    computed severity.
    """
    top = max(rank_scores.values(), default=0.0)
    core: list[dict[str, Any]] = []
    parked: list[dict[str, Any]] = []
    for c in conflicts:
        endpoints = [c.get("from_name"), c.get("to_name")]
        raw = max((rank_scores.get(e, 0.0) for e in endpoints if e), default=0.0)
        severity = round(raw / top, 4) if top > 0 else 0.0
        row = {**c, "severity": severity}
        (core if severity >= core_severity else parked).append(row)
    core.sort(key=lambda r: r["severity"], reverse=True)
    parked.sort(key=lambda r: r["severity"], reverse=True)
    return core, parked


# -- Delta novelty ------------------------------------------------------------------

# The debounce derives time from the substrate, not the scheduler: an
# announcement is indexed to the first arrival of a FACT-STATE against this
# locus, not to a batch boundary (which is the tool queue exhaling — nothing
# about the understanding changes there). Announce at first sight, suppress
# the echo: recognition is an event, not a drone, and a recognition
# re-announced is not a new impression. A batch-level trigger alone would still
# repeat identical recognitions across successive batches in one long turn;
# this suppresses across the session, with the per-prompt full infusion
# re-carrying the standing picture, so nothing is lost — only the echo. It
# also makes the cadence harness-portable: wired to a batch event
# (PostToolBatch) or a per-call event (PostToolUse, on harnesses/SDKs that
# expose nothing finer), the at-most-once-per-fact-state semantics hold.
#
# The key is a CONTENT fingerprint, not an identity pair: a conflict whose
# state changes mid-session (an edge gained, a prop revised, a dissonance
# detail shifted) is a NEW first sight — first sight of the new state — and
# re-announces. Identity-only keying would suppress the change forever.


def content_key(prefix: str, row: dict[str, Any]) -> str:
    canon = json.dumps(row, sort_keys=True, default=str)
    return f"{prefix}|{hashlib.sha1(canon.encode()).hexdigest()[:16]}"


# Kept as the delta-gate's internal spelling; one function, one fingerprint.
_content_key = content_key


# -- Renewal (the delivery ledger) --------------------------------------------------

# The v0.5.x full mode re-delivered the whole standing picture every prompt,
# and the experiment measured the cost: 19 payloads at saturation, 20-45%
# consecutive overlap, a 7-line permanent core, and trajectory yielding on
# every single turn. The effective content of a repeated block is its diff,
# and a verbatim repeat has no diff — by the fifth arrival a block reads as
# texture, not claim (habituation), while its budget cost stays full price.
#
# The renewal economy replaces redelivery: a node's FULL body is delivered
# at first sight, on state change (content fingerprint — same discipline as
# the delta gate), or when its last full delivery has gone stale; otherwise
# it re-pins as a one-line HANDLE. The handle works because node names are
# long, unique, semantically dense strings — an exact-match bridge back to
# the full body earlier in context. Tension sections are exempt: conflict
# lines are already handle-sized and never decay.

_RENEWAL_REFRESH_TURNS = 10


def renewal_partition(
    nodes: list[dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
    turn: int,
    refresh_turns: int = _RENEWAL_REFRESH_TURNS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Split ranked neighborhood nodes into (fresh, standing, fingerprints)
    by delivery history. `ledger` maps node name -> {"fp": fingerprint,
    "last_full": turn}. A node goes fresh (full body) on first sight,
    changed fingerprint, or staleness; otherwise standing (handle).

    STRICTLY PURE — the ledger is READ, never written. Selection is not
    delivery: the assembler downstream may drop lines under budget, and a
    ledger stamped here would record a body that never reached context.
    That divergence corrupts turns-since-body, which is EXPERIMENT-BARLOW
    B1's independent variable — a phantom stamp resets a node's clock on a
    delivery that did not happen, and the resulting flat curve is
    indistinguishable from B1's registered falsifier. Callers stamp via
    `commit_delivery` with what the payload ACTUALLY carried; the returned
    fingerprint map exists so they can do so without recomputing.
    (v0.7.1 — measured: 31 phantom delivery records across 7 turns.)"""
    fresh: list[dict[str, Any]] = []
    standing: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for n in nodes:
        fp = content_key(
            "node",
            {"name": n["name"], "type": n.get("type"),
             "description": n.get("description")},
        )
        fingerprints[n["name"]] = fp
        entry = ledger.get(n["name"])
        if (
            entry
            and entry["fp"] == fp
            and (turn - entry["last_full"]) < refresh_turns
        ):
            standing.append({"name": n["name"], "type": n.get("type")})
        else:
            fresh.append(n)
    return fresh, standing, fingerprints


def commit_delivery(
    ledger: dict[str, dict[str, Any]],
    fingerprints: dict[str, str],
    delivered_bodies: list[str],
    turn: int,
) -> int:
    """Stamp `last_full` for the bodies that ACTUALLY reached the payload.

    The delivery-gated half of the renewal ledger. Only names in
    `delivered_bodies` are recorded, so a node whose body was selected but
    squeezed out by the budget keeps its previous clock and re-delivers
    next turn — the honest behaviour, and the one B1's x-axis requires.
    Returns the number of stamps written."""
    written = 0
    for name in delivered_bodies:
        fp = fingerprints.get(name)
        if fp is None:
            continue
        ledger[name] = {"fp": fp, "last_full": turn}
        written += 1
    return written


def renewal_filter_edges(
    edges: list[dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
    turn: int,
    refresh_turns: int = _RENEWAL_REFRESH_TURNS,
) -> list[dict[str, Any]]:
    """The ledger applied to neighborhood edge lines. An edge has no handle
    form — its whole content IS one line — so renewal here is suppression:
    print at first sight, on change, or when stale; omit otherwise. The
    STANDING register's nodes imply their previously-shown relations; the
    replay benchmark measured unledgered edges reabsorbing every character
    the body ledger freed, keeping trajectory squeezed out."""
    kept: list[dict[str, Any]] = []
    for e in edges:
        key = f"edge|{e['from_name']}|{e['rel']}|{e['to_name']}"
        entry = ledger.get(key)
        if (
            entry
            and (turn - entry["last_full"]) < refresh_turns
        ):
            continue
        ledger[key] = {"fp": key, "last_full": turn}
        kept.append(e)
    return kept


def delta_novelty(
    recognitions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    seen: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Filter a delta to fact-states this locus has not yet been told.

    Returns (fresh_recognitions, fresh_conflicts, new_keys) — the caller adds
    new_keys to its per-locus seen set after the payload is delivered. Keys
    fingerprint the row's full content, so a changed state re-announces while
    a verbatim repeat stays suppressed. Pure: the session state lives with
    the orchestrator, the judgment lives here.
    """
    fresh_r = [
        r for r in recognitions if _content_key("recognition", r) not in seen
    ]
    fresh_c = [c for c in conflicts if _content_key("conflict", c) not in seen]
    new_keys = {_content_key("recognition", r) for r in fresh_r} | {
        _content_key("conflict", c) for c in fresh_c
    }
    return fresh_r, fresh_c, new_keys


# -- Format -----------------------------------------------------------------------

_MAX_PAYLOAD_CHARS = 10_000


def _snip(text: str | None, limit: int = 180) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _conflict_line(c: dict[str, Any]) -> str:
    kind = c.get("kind", "challenge")
    if kind == "challenge":
        line = f"{c['from_name']} CHALLENGES {c['to_name']}"
    elif kind == "divergence":
        line = (
            f"{c['from_name']} — mass rank {c.get('mass_rank')} vs focal rank "
            f"{c.get('frontier_rank')}: big because it is big"
        )
    else:  # dissonance
        line = f"{c['from_name']} — {c.get('detail', 'confidence outruns evidence')}"
    return line


def format_payload(
    seed_terms: list[str],
    seed_mode: str,
    core_conflicts: list[dict[str, Any]],
    parked_conflicts: list[dict[str, Any]],
    neighborhood_nodes: list[dict[str, Any]],
    neighborhood_edges: list[dict[str, Any]],
    open_threads: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    max_chars: int = _MAX_PAYLOAD_CHARS,
    standing_nodes: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, list[str]]]:
    """Assemble the signed governed payload, coherence tension first.

    Returns (payload, delivered) where `delivered` names the node bodies
    and standing handles that SURVIVED the budget squeeze. Selection is not
    delivery; the caller stamps the renewal ledger from this, never from
    what it handed in (v0.7.1).

    Order is the governance: signature (a proposal from the sediment, not a
    conclusion), then whatever contests the focal understanding, then the
    gravitational neighborhood (FRESH bodies only under the renewal
    ledger), then the standing handles (delivered earlier, re-pinned at one
    line each), then the open threads the frontier bias pulled into range,
    then the temporal trajectory, and last the parked tensions — carried at
    one line each. Budgeting protects the ends: the signature, core
    tension, and parked register are laid down first and NEVER yield. The
    middle degrades in a deliberate order distinct from its display order:
    trajectory yields first, standing handles second, neighborhood third,
    open threads last — the frontier's voice is the governance mechanism
    and survives the squeeze.
    """
    header = (
        f"[substrate proposal — awakened from: {', '.join(seed_terms) or '(frontier only)'}"
        f" | seed mode: {seed_mode} | a proposal from the sediment, not a conclusion]"
    )

    core_lines: list[str] = []
    if core_conflicts:
        core_lines.append("CORE TENSION — contests what you hold; weigh before proceeding:")
        for c in core_conflicts:
            line = f"  • {_conflict_line(c)} (severity {c['severity']})"
            why = (c.get("props") or {}).get("revision_why")
            if why:
                line += f" — {_snip(why, 120)}"
            core_lines.append(line)

    parked_lines: list[str] = []
    if parked_conflicts:
        parked_lines.append(
            "PARKED TENSIONS — noted, unresolved, non-blocking; resolve in cleanup:"
        )
        for c in parked_conflicts:
            parked_lines.append(f"  • {_conflict_line(c)}")

    # Name maps run parallel to the line lists (None on headers and edge
    # lines) so the assembler can report what it ACTUALLY delivered after
    # the budget squeeze, rather than what it was handed. The ledger and
    # the observe log are both stamped from that report — see
    # `commit_delivery`.
    nbhd_lines: list[str] = []
    nbhd_names: list[str | None] = []
    if neighborhood_nodes:
        nbhd_lines.append("NEIGHBORHOOD — what you already hold about this:")
        nbhd_names.append(None)
        for n in neighborhood_nodes:
            desc = _snip(n.get("description"))
            suffix = f" — {desc}" if desc else ""
            nbhd_lines.append(f"  • {n['name']} ({n['type']}){suffix}")
            nbhd_names.append(n["name"])
        for e in neighborhood_edges:
            nbhd_lines.append(f"    {e['from_name']} {e['rel']} {e['to_name']}")
            nbhd_names.append(None)

    standing_lines: list[str] = []
    standing_names: list[str | None] = []
    if standing_nodes:
        standing_lines.append(
            "STANDING — delivered earlier this waking, still in force (handles):"
        )
        standing_names.append(None)
        for s in standing_nodes:
            standing_lines.append(f"  • {s['name']} ({s.get('type', '?')})")
            standing_names.append(s["name"])

    thread_lines: list[str] = []
    if open_threads:
        thread_lines.append("OPEN THREADS — unresolved, pulled into range by the frontier bias:")
        for t in open_threads:
            thread_lines.append(f"  • [{t['type']}] {t['name']}")

    traj_lines: list[str] = []
    if trajectory:
        traj_lines.append("TRAJECTORY — how this understanding was constituted, in order:")
        for step in trajectory:
            traj_lines.append(f"  • {step['encounter']}: {step['name']}")

    # Fixed ends first; the middle yields line-by-line to the budget in
    # survival-priority order (threads > standing handles > neighborhood >
    # trajectory), then is emitted in display order (neighborhood, standing,
    # threads, trajectory).
    #
    # v0.7.1 — STANDING NOW OUTRANKS NEIGHBORHOOD. The prior order had it
    # backwards on both cost and consequence. Cost: a handle runs ~110
    # chars against a body's ~280, so the assembler was spending the budget
    # on the expensive content and then dropping the cheap content whole.
    # Consequence, which matters more: the losses are asymmetric. A fresh
    # body that does not fit simply arrives next turn. A standing handle
    # that does not fit SILENTLY WITHDRAWS an already-established node from
    # the standing picture — reduction without re-injection, which is
    # precisely the half of Barlow's duality this economy exists to honour.
    # Measured before the fix: turns carrying 9, 11 and 12 standing nodes
    # emitted 0, 0 and 4 of them, with the section header absent entirely.
    fixed = [header] + core_lines
    fixed_len = sum(len(x) + 1 for x in fixed) + sum(len(x) + 1 for x in parked_lines)
    budget = max_chars - fixed_len
    sections = {
        "nbhd": nbhd_lines,
        "standing": standing_lines,
        "threads": thread_lines,
        "traj": traj_lines,
    }
    kept: dict[str, list[str]] = {k: [] for k in sections}
    exhausted = False
    for key in ("threads", "standing", "nbhd", "traj"):
        if exhausted:
            break
        for line in sections[key]:
            cost = len(line) + 1
            if budget - cost < 0:
                # Hierarchical yield: once a section truncates, everything
                # below it in priority is dropped whole — a squeeze deep
                # enough to cut the neighborhood never shows a trajectory.
                exhausted = True
                break
            kept[key].append(line)
            budget -= cost
        if len(kept[key]) == 1 and len(sections[key]) > 1:
            # A bare section header with no content is noise; reclaim it.
            budget += len(kept[key][0]) + 1
            kept[key] = []
    middle = kept["nbhd"] + kept["standing"] + kept["threads"] + kept["traj"]

    # What actually survived the squeeze — the only honest basis for
    # stamping the ledger and the observe log.
    delivered = {
        "bodies": [n for n in nbhd_names[: len(kept["nbhd"])] if n],
        "handles": [n for n in standing_names[: len(kept["standing"])] if n],
    }

    payload = "\n".join(fixed + middle + parked_lines)
    if len(payload) > max_chars:
        # Last-resort guard, failing honest: a tension cut mid-line can
        # assert something neither node claims, so whole lines yield —
        # the parked register collapses to a count, then core lines drop
        # lowest-severity-first (they are sorted descending), each
        # withholding named as such. Only the signature may ever be cut.
        # (When this fires, the middle is already empty: fixed overflowed.)
        parked_kept = (
            [f"PARKED TENSIONS — {len(parked_conflicts)} withheld under budget; carried, not dropped"]
            if parked_conflicts
            else []
        )
        core_kept = list(core_lines)
        withheld = 0
        while core_kept and (
            len("\n".join([header] + core_kept + parked_kept)) > max_chars - 80
        ):
            core_kept.pop()
            withheld += 1
        if withheld:
            core_kept.append(
                f"  (+{withheld} core tensions withheld — substrate in high-tension state)"
            )
        payload = "\n".join([header] + core_kept + parked_kept)
        if len(payload) > max_chars:
            payload = payload[: max_chars - 1] + "…"
        # The middle is gone entirely in this branch; nothing was delivered.
        delivered = {"bodies": [], "handles": []}
    return payload, delivered


def format_delta(
    recognitions: list[dict[str, Any]],
    core_conflicts: list[dict[str, Any]],
    max_chars: int = _MAX_PAYLOAD_CHARS,
) -> str:
    """The tool-batch delta: recognition or conflict — otherwise silence.

    Attenuation is one failure of the living present (the sediment too faint
    to shape perception); continuous maximal injection is the mirror failure
    (the sediment so loud the arriving present cannot land). When the batch
    contains nothing the substrate holds and contests nothing it believes,
    the correct payload is the empty string, and the hook stays silent —
    fulfillment and disappointment both require that the world get a turn
    to speak.
    """
    if not recognitions and not core_conflicts:
        return ""
    lines = ["[substrate delta — recognition/conflict only; signed as sediment]"]
    for r in recognitions:
        encounters = ", ".join(r.get("encounters", []))
        suffix = f" (constituted in: {encounters})" if encounters else ""
        lines.append(f"  • already held: {r['name']} ({r['type']}){suffix}")
    for c in core_conflicts:
        lines.append(f"  • CONTESTS what you hold: {_conflict_line(c)}")
    payload = "\n".join(lines)
    if len(payload) > max_chars:
        payload = payload[: max_chars - 1] + "…"
    return payload
