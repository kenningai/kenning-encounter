"""The governed infusion pipeline — pure logic (DB-free, unit-tested).

Substrate weight is dispositional, not memory: it must shape perception at
every decision point, and attention attenuation dissolves a once-per-waking
briefing by turn seven. The remedy is to perform the passive synthesis
mechanically — re-awaken a topologically-relevant projection of the substrate
against each arriving present — and to GOVERN it, because in a self-authoring
substrate the reading instrument is a constitutive input to the next state:
mass-only re-injection converges every bound trajectory on entrenchment (the
gravity well, per-turn). The governed form blends focal seeds with the
standing frontier in one biased rank, foregrounds conflict, flags divergence,
signs its own provenance, and holds silence when the substrate has nothing
to say.

This module holds the judgment-bearing arithmetic of that pipeline —
Extract (focal signals), the Lucene query construction for Match, the
conflict triage (core vs parked, by constitutive proximity), and Format
(the signed payload) — kept pure so it is testable without a driver. The
DB legs (fulltext match, the GDS rank streams, neighborhood/trajectory
reads) live on Neo4jKenningEncounter.infuse.

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

# Harness-envelope noise — vocabulary the transport wraps around the content,
# never the content itself. Measured live on a real payload's own transport
# headers: task-notification plumbing consumed entire seed budgets, and
# because these tokens are identifier-shaped they rode the hard-signal weight.
# Three classes, each named in the B1 follow-up work order:
#   (1) harness-minted identifiers — mcp__<server>__<tool> names and toolu_*
#       tool-use ids;
#   (2) the envelope's own structural vocabulary — the fixed field names of the
#       hook payload / tool_response wrapper (snake and kebab forms), plus the
#       bare content-block keys "type"/"text" that recur once per block;
#   (3) unicode-escape fragments — a serialized em-dash reaches the word regex
#       as the literal text u2014.
# Deliberately NOT filtered: bare common words that double as envelope keys
# (status, output, result, summary…) — they are also real prose, and the
# over-filtering-costs-recall rule above outranks envelope hygiene for them.
_ENVELOPE_NOISE_RE = re.compile(r"^(?:mcp__|toolu_)|^u[0-9a-f]{4}$", re.IGNORECASE)
_ENVELOPE_KEYS = frozenset(
    """type text
    tool_use_id tool_name tool_input tool_response tool_result
    session_id transcript_path hook_event_name
    task-notification task-id tool-use-id output-file
    """.split()
)


def _is_envelope_noise(key: str) -> bool:
    return key in _ENVELOPE_KEYS or _ENVELOPE_NOISE_RE.match(key) is not None


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
        if key in _STOPWORDS or len(key) < 3 or _is_envelope_noise(key):
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
            if key in _STOPWORDS or len(key) < 3 or _is_envelope_noise(key):
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


# Internal spelling of the same fingerprint; one function, one fingerprint.
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
# at first sight, on state change (content fingerprint), or when its last
# full delivery has gone stale; otherwise
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
    That divergence corrupts turns-since-body, the independent variable any
    measurement of the refresh horizon rests on — a phantom stamp resets a
    node's clock on a delivery that did not happen, and the resulting flat
    curve is indistinguishable from a real null. Callers stamp via
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
# -- Progression assembly (v0.8.0, the progression reform) --------------------------

# The knowledge-vs-data repair: data means
# nothing until connected in time to other data, and an LLM handed two
# contradictory observations about one subject WITHOUT their temporal ordering
# must fabricate the connection — confabulation as the necessary consequence
# of tenseless delivery, not a handling defect. Selection stays atemporal (it
# answers "what is relevant"); DELIVERY becomes temporal (it answers "in what
# order did this become what it is"). The graph already holds the manifold:
# every multi-observation Component is a temporal progression assemblable by
# a two-hop fan-out (obs -> ABOUT -> Component -> siblings, with RECORDED
# provenance). The server surfaces the ordering; the trajectory performs the
# crossing-out. Nothing here writes: the graph stays pure WAS.

_PROG_CAP_CHARS = 1200          # per-progression budget (design of record §2.2)
_PROG_MAX_BODIES = 2            # terminus + seed when distinct
_PROG_STEP_SNIP = 90
_PROG_TERMINUS_SNIP = 300
# Types that carry state. Questions/Hypotheses in a progression are the
# frontier's voice arriving in-topic — rendered, but never the terminus:
# an open question noticed last is not the subject's current state.
_PROG_STATE_TYPES = frozenset({"Observation", "Note"})


def _date_of(step: dict[str, Any]) -> str:
    key = step.get("sort_key") or ""
    return str(key)[:10] if key else "undated"


def group_progressions(
    rows: list[dict[str, Any]],
    selected_names: list[str],
    min_steps: int = 2,
) -> list[dict[str, Any]]:
    """Group two-hop fan-out rows into per-Component progressions.

    rows: {component, name, type, description, sort_key, encounter, enc_t} —
    already sorted by (component, sort_key, enc_t) by the query. The primary
    order is noticing-time (coalesce of t_observed/t_raised/t_proposed/
    t_created); the secondary key is the anchoring encounter's t_exist, which
    is legitimate as a STRUCTURAL tiebreak because every encounter is created
    by one serialized server transaction — its order is the creation order of
    the spine made scalar, not a wall-clock reconstruction (design of record
    §6; applies post-v0.3.0-seam, and pre-seam steps simply keep noticing-time
    order, which is all the un-rethreaded braid supports).

    Dedup is by Component, not by seed: two selected observations about one
    Component assemble ONE progression. A single-step "progression" is just
    the node — filtered out (min_steps), it flows through the neighborhood
    path instead. Returns [{"component", "steps", "seed_names"}], most
    selected members first (most-relevant topic leads).
    """
    by_comp: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        comp = r.get("component")
        name = r.get("name")
        if not comp or not name:
            continue
        by_comp.setdefault(comp, {})
        # First occurrence wins; rows arrive pre-sorted so this keeps order.
        by_comp[comp].setdefault(name, r)
    selected = set(selected_names)
    out: list[dict[str, Any]] = []
    for comp, steps_by_name in by_comp.items():
        steps = list(steps_by_name.values())
        steps.sort(key=lambda s: (str(s.get("sort_key") or ""), str(s.get("enc_t") or "")))
        if len(steps) < min_steps:
            continue
        seed_names = [s["name"] for s in steps if s["name"] in selected]
        if not seed_names:
            continue
        out.append({"component": comp, "steps": steps, "seed_names": seed_names})
    out.sort(key=lambda p: len(p["seed_names"]), reverse=True)
    return out


def progression_terminus(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The last STATE step — the subject's current constituted state."""
    for s in reversed(steps):
        if s.get("type") in _PROG_STATE_TYPES:
            return s
    return None


def progression_renewal(
    component: str,
    steps: list[dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
    turn: int,
    refresh_turns: int = _RENEWAL_REFRESH_TURNS,
) -> tuple[str, str, list[str]]:
    """The progression as the unit of renewal (design of record §3).

    Returns (form, fp, new_step_names). Forms: "full" (first sight, or stale),
    "standing" (unchanged and fresh — one line re-pins the whole topic),
    "advance" (a new step arrived — terminus plus the new steps, on
    announce-on-state-change semantics).
    STRICTLY PURE: the ledger is read, never written — selection is not
    delivery (v0.7.1); callers stamp via `commit_progression` with what the
    payload actually carried.
    """
    terminus = progression_terminus(steps)
    fp = content_key(
        "prog",
        {
            "component": component,
            "steps": [s["name"] for s in steps],
            "terminus": (terminus or {}).get("description"),
        },
    )
    entry = ledger.get(f"prog|{component}")
    if entry is None:
        return "full", fp, []
    if entry.get("fp") == fp:
        if (turn - entry.get("last_full", 0)) < refresh_turns:
            return "standing", fp, []
        return "full", fp, []
    known = set(entry.get("steps") or ())
    new = [s["name"] for s in steps if s["name"] not in known]
    if not new:
        # Content changed without new steps (a step's body was revised, or
        # the terminus description shifted) — a new first sight of the new
        # state; the full form re-delivers it.
        return "full", fp, []
    return "advance", fp, new


def commit_progression(
    ledger: dict[str, dict[str, Any]],
    component: str,
    fp: str,
    step_names: list[str],
    turn: int,
) -> None:
    """Stamp a progression that ACTUALLY reached the payload."""
    ledger[f"prog|{component}"] = {
        "fp": fp,
        "last_full": turn,
        "steps": list(step_names),
    }


def format_progression(
    component: str,
    steps: list[dict[str, Any]],
    seed_names: list[str],
    protected_names: set[str] | None = None,
    form: str = "full",
    new_step_names: list[str] | None = None,
    grounding: list[str] | None = None,
    cap_chars: int = _PROG_CAP_CHARS,
) -> tuple[list[str], dict[str, list[str]]]:
    """Render one topic block. Returns (lines, delivered) with delivered =
    {"bodies": [...], "steps": [...]}.

    The block is ATOMIC — the payload assembler keeps or drops it whole. A
    progression squeezed mid-block could deliver the stale step and cut its
    resolution below it, asserting the very superposition this form exists
    to prevent.

    Role-first survival within the cap (design of record §2.2): terminus
    (currency, full body with grounding), seed step (independently ranked,
    full body when distinct), protected steps (previously delivered
    standalone — the implementable form of the crossing-out-pair rule: a
    step the bound locus has seen as a standalone claim must arrive tensed,
    never fold), origin (stories need beginnings), recent intermediates,
    then middles folded into a count line. Steps render in temporal order;
    only the SELECTION of survivors is role-first.
    """
    protected = protected_names or set()
    delivered: dict[str, list[str]] = {"bodies": [], "steps": []}
    terminus = progression_terminus(steps)

    if form == "standing":
        t_name = _snip((terminus or {}).get("name"), 70) if terminus else "?"
        line = (
            f"  ◦ {component} — progression standing "
            f"({len(steps)} steps; terminus: {t_name})"
        )
        delivered["steps"] = [s["name"] for s in steps]
        return [line], delivered

    state_steps = [s for s in steps if s.get("type") in _PROG_STATE_TYPES]
    frontier_steps = [s for s in steps if s.get("type") not in _PROG_STATE_TYPES]
    new_names = set(new_step_names or ())

    # Body roles: terminus always; seed when distinct (last seed that isn't
    # the terminus — it earned independent relevance from the rank).
    body_names: list[str] = []
    if terminus is not None:
        body_names.append(terminus["name"])
    seed_bodies = [
        n for n in reversed(seed_names)
        if terminus is None or n != terminus["name"]
    ]
    if seed_bodies and len(body_names) < _PROG_MAX_BODIES:
        body_names.append(seed_bodies[0])

    candidates = [s for s in state_steps if s["name"] not in body_names]
    seed_set = set(seed_names)

    if form == "advance":
        header = (
            f"{component} — progression advanced (+{len(new_names)} step"
            f"{'s' if len(new_names) != 1 else ''}):"
        )
        keep = {
            s["name"] for s in candidates
            if s["name"] in new_names
        }
        optional: list[str] = []
    else:
        header = (
            f"{component} — {len(steps)} steps, "
            f"{_date_of(steps[0])} → {_date_of(steps[-1])}:"
        )
        # Role-first survivor floor: origin, protected (previously delivered
        # standalone — must arrive tensed, never fold), and seeds. Recent
        # intermediates are the optional fill, oldest yielding first under
        # the cap. Render order stays temporal; only SELECTION is role-first.
        keep = set()
        if candidates:
            keep.add(candidates[0]["name"])  # origin
        for s in candidates:
            if s["name"] in protected or s["name"] in seed_set:
                keep.add(s["name"])
        optional = [
            s["name"] for s in reversed(candidates)
            if s["name"] not in keep
        ][:6]  # recent-first fill; the cap walk below trims from the tail
        keep |= set(optional)

    def _render(keep_now: set[str]) -> tuple[list[str], dict[str, list[str]]]:
        """One full render for a given survivor set. Every step renders IN
        ITS TEMPORAL PLACE — bodies included: a body hoisted out of sequence
        would put a stale claim after its resolution, and the ordering IS
        the knowledge. The fold line carries the TRUE folded count for
        exactly this render — the block never lies about what folded."""
        out: list[str] = [header]
        dlv: dict[str, list[str]] = {"bodies": [], "steps": []}
        folded_now = sum(
            1 for s in candidates if s["name"] not in keep_now
        )
        fold_line = (
            f"  → … {folded_now} more steps folded" if folded_now > 0 else None
        )
        emitted_fold = False
        for s in state_steps:
            name = s["name"]
            if terminus is not None and name == terminus["name"]:
                continue  # the terminus renders last, as NOW
            if name in body_names:
                desc = _snip(s.get("description"), _PROG_TERMINUS_SNIP)
                suffix = f" — {desc}" if desc else ""
                out.append(f"  SEED ({_date_of(s)}): {name}{suffix}")
                dlv["bodies"].append(name)
                dlv["steps"].append(name)
                continue
            if name in keep_now:
                marker = " [seed]" if name in seed_set else ""
                note = " (Note)" if s.get("type") == "Note" else ""
                out.append(
                    f"  → {_date_of(s)}: {_snip(name, _PROG_STEP_SNIP)}{note}{marker}"
                )
                dlv["steps"].append(name)
            elif fold_line is not None and not emitted_fold:
                out.append(fold_line)
                emitted_fold = True
        if terminus is not None:
            desc = _snip(terminus.get("description"), _PROG_TERMINUS_SNIP)
            suffix = f" — {desc}" if desc else ""
            out.append(f"  NOW ({_date_of(terminus)}): {terminus['name']}{suffix}")
            if grounding:
                out.append(f"      [{'; '.join(grounding[:2])}]")
            dlv["bodies"].append(terminus["name"])
            dlv["steps"].append(terminus["name"])
        for s in frontier_steps:
            out.append(
                f"  ? open ({s.get('type', '?')}): {_snip(s['name'], _PROG_STEP_SNIP)}"
            )
            dlv["steps"].append(s["name"])
        return out, dlv

    def _total(ls: list[str]) -> int:
        return sum(len(x) + 1 for x in ls)

    # Iterative cap walk: the optional fill yields oldest-first (middles
    # fold, recency survives), the fold count stays true on every pass, and
    # the floor — origin, protected, seeds, bodies, frontier — never yields
    # to the cap. The floor plus two bodies is the block's honest minimum.
    lines, delivered = _render(keep)
    yieldable = [n for n in optional]  # recent-first; pop() yields oldest
    while _total(lines) > cap_chars and yieldable:
        keep.discard(yieldable.pop())
        lines, delivered = _render(keep)

    return lines, delivered


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


_FALLBACK_REASON_CHARS = 160


def matcher_fallback_note(selection_meta: dict[str, Any] | None) -> str | None:
    """The line that tells the TRAJECTORY its seeds were not meaning-matched.

    selection_meta alone reached no reader: it rides the tool result into an
    observe log that is usually unset. A deployment with a valid endpoint, a
    valid model and a populated sidecar can then run on the lexical path for
    weeks with nothing saying so. A prompt larger than the endpoint's context
    window does it, and so does an exhausted prepaid balance. The header is
    what the trajectory reads before anything else, so the reason goes there.
    Reason text is exception shape (status, URL, counts), never node content.
    """
    if not selection_meta:
        return None
    if selection_meta.get("channel") == "meaning":
        # v0.17.0: still meaning-matched, but not by the lead voice. Said
        # without naming either voice: which voice is delivering is the
        # variable a comparison between voices measures, and the header is
        # read by the subject of that comparison. The failure is named,
        # because a lead that has gone dark is the operator's to know.
        # Under hedging the lead can deliver after a fallback
        # failed or was overtaken, so an error in attempts no longer means
        # the lead did not deliver. Only who delivered says that.
        lead, voice = selection_meta.get("lead"), selection_meta.get("voice")
        failed = [a for a in selection_meta.get("attempts") or [] if "error" in a]
        if lead and voice and lead == voice:
            return None
        lead_failed = [a for a in failed if a.get("voice") == lead]
        if lead_failed or failed:
            reason = _clip_reason(str((lead_failed or failed)[0].get("error", "")))
        elif lead and voice:
            # The lead was not even tried: it failed recently and is cooling.
            reason = "the lead failed recently and is cooling down"
        else:
            return None
        return (
            "[matcher lead unavailable — seeds below are meaning-matched by "
            f"the fallback voice: {reason or 'no reason given'}]"
        )
    if selection_meta.get("channel") != "lexical_fallback":
        return None
    reason = _clip_reason(str(selection_meta.get("fallback_reason", "")))
    return (
        "[matcher unavailable — seeds below are LEXICAL keyword matches, not "
        f"meaning-matched: {reason or 'no reason given'}]"
    )


def _clip_reason(reason: str) -> str:
    reason = " ".join(reason.split())
    if len(reason) > _FALLBACK_REASON_CHARS:
        reason = reason[: _FALLBACK_REASON_CHARS - 1] + "…"
    return reason


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
    progressions: list[dict[str, Any]] | None = None,
    selection_note: str | None = None,
    meaning_matched: bool = False,
) -> tuple[str, dict[str, list[str]]]:
    """Assemble the signed governed payload, coherence tension first.

    Returns (payload, delivered) where `delivered` names the node bodies
    and standing handles that SURVIVED the budget squeeze. Selection is not
    delivery; the caller stamps the renewal ledger from this, never from
    what it handed in (v0.7.1).

    Order is the governance: signature (a proposal from the sediment, not a
    conclusion), then whatever contests the focal understanding, then the
    PROGRESSIONS (v0.8.0 — topic blocks of knowledge connected in time,
    pre-rendered by `format_progression` and kept or dropped ATOMICALLY: a
    mid-block cut could strand a stale step without its resolution), then
    the gravitational neighborhood (FRESH bodies only under the renewal
    ledger, each tensed with its as-of date), then the standing handles
    (delivered earlier, re-pinned at one line each), then the open threads
    the frontier bias pulled into range, then the temporal trajectory, and
    last the parked tensions — carried at one line each. Budgeting protects
    the ends: the signature, core tension, and parked register are laid down
    first and NEVER yield. The middle degrades in a deliberate order
    distinct from its display order: trajectory yields first, standing
    handles second, neighborhood third, progressions fourth, open threads
    last — the frontier's voice is the governance mechanism and the
    progressions are the knowledge delivery; both survive the squeeze
    longest.

    `progressions` rows: {"component", "lines", "bodies", "steps"} from
    `format_progression`.
    """
    # The first thing the trajectory reads names the seed channel. Until
    # v0.16.3 a meaning-matched payload still opened "awakened from:
    # <keywords>" — the lexical Extract, computed but not used to seed — so
    # a healthy header and a degraded one looked alike.
    seeds = (
        "seeds: meaning-matched" if meaning_matched
        else f"awakened from: {', '.join(seed_terms) or '(frontier only)'}"
    )
    header = (
        f"[substrate proposal — {seeds}"
        f" | seed mode: {seed_mode} | a proposal from the sediment, not a conclusion]"
    )
    # Part of the signature: laid down first, never yields to the budget.
    if selection_note:
        header = f"{header}\n{selection_note}"

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
    prog_blocks: list[dict[str, Any]] = list(progressions or [])

    nbhd_lines: list[str] = []
    nbhd_names: list[str | None] = []
    if neighborhood_nodes:
        nbhd_lines.append("NEIGHBORHOOD — what you already hold about this:")
        nbhd_names.append(None)
        for n in neighborhood_nodes:
            desc = _snip(n.get("description"))
            suffix = f" — {desc}" if desc else ""
            # The minimum tense marker (design of record §1.3): every body
            # carries its as-of date, so no claim arrives as an unmodalized
            # present-tense assertion. A timestamp is knowledge; a
            # disclaimer would be noise.
            as_of = n.get("as_of")
            tense = f" [as of {as_of}]" if as_of else ""
            nbhd_lines.append(f"  • {n['name']} ({n['type']}){tense}{suffix}")
            nbhd_names.append(n["name"])
            conv = n.get("convergence")
            if conv:
                # Convergence annotation (§1.2), hard-capped to one line —
                # uncapped convergence is the gravity well given a megaphone.
                nbhd_lines.append(f"    ↳ {conv}")
                nbhd_names.append(None)
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
    kept_prog: list[dict[str, Any]] = []
    prog_banner = "PROGRESSIONS — what you hold about these, connected in time:"
    exhausted = False
    for key in ("threads", "prog", "standing", "nbhd", "traj"):
        if exhausted:
            break
        if key == "prog":
            # Atomic blocks: a progression is kept whole or dropped whole —
            # a mid-block cut could deliver the stale step and cut its
            # resolution, asserting the superposition this form prevents.
            if prog_blocks:
                banner_cost = len(prog_banner) + 1
                if budget - banner_cost < 0:
                    exhausted = True
                    continue
                budget -= banner_cost
                any_kept = False
                for block in prog_blocks:
                    cost = sum(len(x) + 1 for x in block["lines"])
                    if budget - cost < 0:
                        exhausted = True
                        break
                    kept_prog.append(block)
                    budget -= cost
                    any_kept = True
                if not any_kept:
                    budget += banner_cost
            continue
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
    prog_lines: list[str] = (
        [prog_banner] + [ln for b in kept_prog for ln in b["lines"]]
        if kept_prog
        else []
    )
    middle = (
        prog_lines + kept["nbhd"] + kept["standing"] + kept["threads"] + kept["traj"]
    )

    # What actually survived the squeeze — the only honest basis for
    # stamping the ledger and the observe log.
    delivered = {
        "bodies": [n for n in nbhd_names[: len(kept["nbhd"])] if n],
        "handles": [n for n in standing_names[: len(kept["standing"])] if n],
        "progressions": [b["component"] for b in kept_prog],
        "prog_bodies": [n for b in kept_prog for n in b.get("bodies", [])],
        "prog_steps": [n for b in kept_prog for n in b.get("steps", [])],
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
        delivered = {
            "bodies": [], "handles": [],
            "progressions": [], "prog_bodies": [], "prog_steps": [],
        }
    return payload, delivered
