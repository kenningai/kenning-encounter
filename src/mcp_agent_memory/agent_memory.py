import logging
import time
import uuid
from enum import Enum
from typing import Any, LiteralString

from neo4j import AsyncDriver, RoutingControl

from .infuse import (
    commit_delivery,
    delta_novelty,
    extract_focal_signals,
    format_delta,
    format_payload,
    lucene_query,
    renewal_filter_edges,
    renewal_partition,
    triage_conflicts,
)
from .utils import load_cypher, lit

logger = logging.getLogger("mcp_agent_memory")
logger.setLevel(logging.INFO)


def _neo4j_datetime_to_str(val: Any) -> str | None:
    """Convert a Neo4j DateTime/Date to ISO string, or return None."""
    if val is None:
        return None
    return val.iso_format()


def _clean_edge_props(raw: Any) -> dict[str, Any]:
    """Strip the auto-set t_created and ISO-convert datetimes on an edge's
    property map. Returns the authored properties only (e.g. revision_why on
    SUPERSEDES) — empty dict when an edge carries none. Lets the re-entry and
    provenance reads surface WHY an edge exists, not just its type."""
    props = dict(raw or {})
    props.pop("t_created", None)
    return {
        k: (_neo4j_datetime_to_str(v) if hasattr(v, "iso_format") else v)
        for k, v in props.items()
    }


# -- Orient panel math (pure, DB-free, unit-tested) ----------------------------
# The composite readings of the v0.4.0 instrument panel. Each takes streamed
# GDS rows ({node, type, score}-shaped dicts) and returns the derived reading.
# Kept pure so the panel's judgment-bearing arithmetic is testable without a
# driver — the cypher fetches rows; these decide what the rows mean.

def _rank_positions(rows: list[dict[str, Any]]) -> dict[str, int]:
    """1-based rank by stream order (rows arrive score-descending)."""
    return {r["node"]: i + 1 for i, r in enumerate(rows)}


def compute_divergence(
    mass: list[dict[str, Any]],
    frontier_mass: list[dict[str, Any]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    """The manifest-destiny detector: the same graph ranked from two reference
    frames — accumulated mass (ArticleRank) vs. seen-from-the-frontier
    (personalized PageRank). divergence = normalized frontier position minus
    normalized mass position; large positive = high mass the frontier doesn't
    care about (big because it is big — gravity-well suspect), large negative =
    low mass the frontier leans on (underweighted by accumulation)."""
    mass_pos = _rank_positions(mass)
    frontier_pos = _rank_positions(frontier_mass)
    types = {r["node"]: r.get("type") for r in mass + frontier_mass}
    common = set(mass_pos) & set(frontier_pos)
    if len(common) < 2:
        return {"well_suspects": [], "frontier_lifted": []}
    m_span = max(len(mass) - 1, 1)
    f_span = max(len(frontier_mass) - 1, 1)
    rows = [
        {
            "node": name,
            "type": types.get(name),
            "mass_rank": mass_pos[name],
            "frontier_rank": frontier_pos[name],
            "divergence": round(
                (frontier_pos[name] - 1) / f_span - (mass_pos[name] - 1) / m_span, 4
            ),
        }
        for name in common
    ]
    rows.sort(key=lambda r: r["divergence"], reverse=True)
    return {
        "well_suspects": [r for r in rows if r["divergence"] > 0][:limit],
        "frontier_lifted": [r for r in reversed(rows) if r["divergence"] < 0][:limit],
    }


def compute_weave_audit(
    degree_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    limit: int,
    min_degree: int = 3,
) -> list[dict[str, Any]]:
    """The weave-quality audit: local clustering coefficient read against
    degree. High degree + low coefficient = a star, not a weave — a node
    everything points at whose neighbors never interconnect, exactly the shape
    the gravity well inflates undeservedly. star_score = degree * (1 - c)."""
    coeff = {r["node"]: r.get("coefficient") for r in coefficient_rows}
    out: list[dict[str, Any]] = []
    for r in degree_rows:
        degree = float(r["score"])
        if degree < min_degree:
            continue
        c = float(coeff.get(r["node"]) or 0.0)
        out.append(
            {
                "node": r["node"],
                "type": r.get("type"),
                "degree": int(degree),
                "clustering": round(c, 4),
                "star_score": round(degree * (1.0 - c), 4),
            }
        )
    out.sort(key=lambda x: x["star_score"], reverse=True)
    return out[:limit]


def compute_drift(
    current: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    """Drift over snapshot: the mass reading compared against the same reading
    as of the previous waking. A well visible AS a well, with velocity, is far
    less of a prophecy. shift = baseline_rank - current_rank (positive = rose).
    new_since = in the current reading with no baseline presence at all."""
    cur_pos = _rank_positions(current)
    base_pos = _rank_positions(baseline)
    types = {r["node"]: r.get("type") for r in current}
    movers = [
        {
            "node": name,
            "type": types.get(name),
            "baseline_rank": base_pos[name],
            "current_rank": pos,
            "shift": base_pos[name] - pos,
        }
        for name, pos in cur_pos.items()
        if name in base_pos
    ]
    risers = sorted(
        (m for m in movers if m["shift"] > 0),
        key=lambda m: m["shift"], reverse=True,
    )[:limit]
    fallers = sorted(
        (m for m in movers if m["shift"] < 0),
        key=lambda m: m["shift"],
    )[:limit]
    new_since = [
        {"node": name, "type": types.get(name), "current_rank": pos}
        for name, pos in sorted(cur_pos.items(), key=lambda kv: kv[1])
        if name not in base_pos
    ][:limit]
    return {"risers": risers, "fallers": fallers, "new_since": new_since}


def frontier_seed_candidates(
    frontier: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Seed names for frontier-biased rank reads (orient's frontier_mass and
    infuse's governed blend). Primary: open Questions + untested Hypotheses —
    the named unresolved. Fallback (for a waking whose named frontier is
    empty): every frontier cut. Pure — the eligibility check (a seed must
    carry a coherence edge to exist in the projection) is a DB read and stays
    on the class."""
    primary = [r["name"] for r in frontier["unanswered_questions"]] + [
        r["name"] for r in frontier["untested_hypotheses"]
    ]
    fallback = primary + [
        r["name"]
        for cut in (
            "ungrounded_concepts",
            "confidence_dissonance",
            "contested_hypotheses",
        )
        for r in frontier[cut]
    ]
    return primary, fallback


def coherence_projection_parts() -> tuple[str, str, str]:
    """The structural fragments of the coherence-only projection — authored
    edges (RelationType minus PROCESS_EDGES) over the non-process nodes —
    shared by orient and infuse so the two instruments read the same subgraph
    by construction. Returns (rel_filter, source_labels, target_labels)."""
    coherence_rels = [r.value for r in RelationType if r.value not in PROCESS_EDGES]
    semantic_nodes = [nt.value for nt in NodeType if nt.value not in PROCESS_TYPES]
    rel_filter = "|".join(f"`{rt}`" for rt in coherence_rels)
    source_labels = " OR ".join(f"source:`{nt}`" for nt in semantic_nodes)
    target_labels = " OR ".join(f"target:`{nt}`" for nt in semantic_nodes)
    return rel_filter, source_labels, target_labels


# -- Node Type Enum -----------------------------------------------------------

class NodeType(str, Enum):
    """Allowed node type labels for the Agent Memory.

    Three layers: process (Encounter), semantic (Observation, Question,
    Hypothesis, Concept, Note), reference (Component, Citation).
    """
    # Process layer — written only by advance_encounter, never generic CRUD.
    ENCOUNTER = "Encounter"
    # Semantic layer — coherences, tensions, restructurings.
    OBSERVATION = "Observation"
    QUESTION = "Question"
    HYPOTHESIS = "Hypothesis"
    CONCEPT = "Concept"
    NOTE = "Note"
    # Reference layer — what was encountered, without copying it.
    COMPONENT = "Component"
    CITATION = "Citation"


# -- Relation Type Enum ------------------------------------------------------

class RelationType(str, Enum):
    """Allowed relationship types for the Agent Memory.

    Two natures: provenance/membership edges (deterministic, auto-set:
    NEXT_ENCOUNTER, RECORDED, CONSULTED) and coherence edges (authored by
    the agent — its judgment, the interior of the encounter).
    """
    # Process / provenance — auto-written, closed to generic CRUD.
    NEXT_ENCOUNTER = "NEXT_ENCOUNTER"
    INSTANTIATED_AFTER = "INSTANTIATED_AFTER"
    RECORDED = "RECORDED"
    CONSULTED = "CONSULTED"
    # Coherence — the agent's judgment, authored every one.
    ABOUT = "ABOUT"
    OBSERVED_AT = "OBSERVED_AT"
    RAISES = "RAISES"
    RESOLVES = "RESOLVES"
    SUPPORTS = "SUPPORTS"
    CHALLENGES = "CHALLENGES"
    GROUNDS = "GROUNDS"
    INFORMS = "INFORMS"
    COMPOSES = "COMPOSES"
    DECOMPOSES = "DECOMPOSES"
    SUPERSEDES = "SUPERSEDES"


# Process-layer types/edges are NOT writable via generic CRUD. The temporal
# spine is created solely by advance_encounter; RECORDED/CONSULTED are auto-
# anchored from the current Encounter on node creation. INSTANTIATED_AFTER is
# the genesis binding: a first-of-locus encounter points to the encounter that
# was latest when its locus began — locus geneses totally order (the write
# transaction is serialized), so the spine stays one connected becoming.
PROCESS_TYPES: set[str] = {"Encounter"}
PROCESS_EDGES: set[str] = {
    "NEXT_ENCOUNTER", "INSTANTIATED_AFTER", "RECORDED", "CONSULTED",
}

# Which provenance edge anchors a newly-created node to the current Encounter.
# Component (a bookmark) is NOT anchored — it enters the agent's time only
# through the epistemic nodes that point at it via ABOUT.
ANCHOR_FOR: dict[str, str] = {
    "Observation": "RECORDED",
    "Question": "RECORDED",
    "Hypothesis": "RECORDED",
    "Concept": "RECORDED",
    "Note": "RECORDED",
    "Citation": "CONSULTED",
}

# Agent-supplied temporal fields cast str -> Neo4j datetime() on write. These
# are annotations in the agent's own frame; the server-truth temporal anchor
# is always the auto-set t_created + the auto-anchored RECORDED/CONSULTED edge.
DATETIME_CAST_PROPS: set[str] = {
    "t_exist", "t_observed", "t_raised", "t_resolved", "t_proposed", "t_consulted",
}


# -- Node Property Schemas ---------------------------------------------------
# Each entry defines required and optional properties per node type.
# Adding a new node type = adding an enum value + a schema entry here.

CONCEPT_STATUS = {"forming", "stable", "revising", "retired"}
QUESTION_STATUS = {"open", "answered", "abandoned"}
HYPOTHESIS_STATUS = {"proposed", "supported", "challenged", "confirmed", "falsified", "retired"}
HYPOTHESIS_CONFIDENCE = {"low", "medium", "high"}

NODE_SCHEMAS: dict[str, dict[str, Any]] = {
    # -- Process layer (written by advance_encounter, not create_entities) --
    "Encounter": {
        "required": {"name": str, "t_exist": str},
        "optional": {
            "summary": str,
            "report": str,
            # Server-written dissolution mark (never authored): stamped when the
            # locus's server-side state is discarded with the encounter unsealed,
            # cleared by a later seal. No summary is ever fabricated for it —
            # the absence of a seal is the honest record.
            "dissolved_at": str,
        },
    },
    # -- Reference layer --
    "Component": {
        # source_kind = which SOURCE SYSTEM (open vocab: a stable token per
        # source you read, e.g. infra_graph, service_registry, doc_store).
        # source_label = which type WITHIN that source (Host, Service, ...).
        # source_key = the source's natural id. A bookmark — never a copy of
        # the source node.
        "required": {"name": str, "source_kind": str, "source_key": str},
        "optional": {
            "source_label": str,
            "description": str,
            "aliases": str,
            "confidence": str,
        },
    },
    "Citation": {
        # A noticing event with one snapshot — name it per-noticing, do not
        # reuse one Citation across encounters. kind is an OPEN vocabulary.
        "required": {"name": str, "kind": str},
        "optional": {
            "description": str,
            "uri": str,
            "query": str,
            "snapshot": str,
            "t_consulted": str,
        },
    },
    # -- Semantic layer --
    "Concept": {
        "required": {"name": str, "description": str},
        "optional": {
            "category": str,
            "status": str,
            # True = excluded from frontier candidacy (orient display and
            # infuse seeding alike): declared bookkeeping, not epistemic
            # frontier. v0.6.0, from the experiment's self-reference loop.
            "frontier_mute": bool,
        },
        "enums": {
            "status": CONCEPT_STATUS,
        },
    },
    "Observation": {
        "required": {"name": str, "description": str, "t_observed": str},
        "optional": {
            "confidence": str,
        },
    },
    "Question": {
        "required": {"name": str, "description": str},
        "optional": {
            "status": str,
            "priority": str,
            "t_raised": str,
            "t_resolved": str,
            "frontier_mute": bool,
        },
        "enums": {
            "status": QUESTION_STATUS,
        },
    },
    "Hypothesis": {
        "required": {"name": str, "description": str},
        "optional": {
            "confidence": str,
            "status": str,
            "t_proposed": str,
            "t_resolved": str,
            "frontier_mute": bool,
        },
        "enums": {
            "confidence": HYPOTHESIS_CONFIDENCE,
            "status": HYPOTHESIS_STATUS,
        },
    },
    "Note": {
        "required": {"name": str, "description": str},
        "optional": {
            "tag": str,
        },
    },
}


# -- Relation Schemas ---------------------------------------------------------
# Direction constraints (source/target types) and structural flags.
# target_types of None means "any node". same_type means source and target
# must share a label (used by SUPERSEDES). No coherence edge carries
# properties — the meaning lives in the edge type itself.

RELATION_SCHEMAS: dict[str, dict[str, Any]] = {
    # -- Process / provenance (documented; rejected by create_relations) --
    "NEXT_ENCOUNTER": {
        "source_types": {"Encounter"},
        "target_types": {"Encounter"},
        "properties": {},
    },
    "INSTANTIATED_AFTER": {
        # Genesis binding (locus-root -> the encounter latest at its genesis).
        # Within-locus succession is NEXT_ENCOUNTER; this is a different kind
        # of relation — instantiation context, not continuation — so it earns
        # its own edge. Auto-written by advance_encounter, like the spine.
        "source_types": {"Encounter"},
        "target_types": {"Encounter"},
        "properties": {},
    },
    "RECORDED": {
        "source_types": {"Encounter"},
        "target_types": {"Observation", "Question", "Hypothesis", "Concept", "Note"},
        "properties": {},
    },
    "CONSULTED": {
        "source_types": {"Encounter"},
        "target_types": {"Citation"},
        "properties": {},
    },
    # -- Coherence (authored via create_relations) --
    "ABOUT": {
        "source_types": {"Observation", "Note", "Question", "Hypothesis"},
        "target_types": {"Component", "Concept"},
        "properties": {},
    },
    "OBSERVED_AT": {
        "source_types": {"Observation"},
        "target_types": {"Citation"},
        "properties": {},
    },
    "RAISES": {
        "source_types": {"Observation"},
        "target_types": {"Question"},
        "properties": {},
    },
    "RESOLVES": {
        "source_types": {"Observation"},
        "target_types": {"Question"},
        "properties": {},
    },
    "SUPPORTS": {
        "source_types": {"Observation"},
        "target_types": {"Hypothesis"},
        "properties": {},
    },
    "CHALLENGES": {
        "source_types": {"Observation"},
        "target_types": {"Hypothesis"},
        "properties": {},
    },
    "GROUNDS": {
        # An Observation that was a CONSTITUTIVE INPUT to a Concept's synthesis —
        # the provenance rung from a synthesized Concept down to the observations
        # that produced it. Distinct in kind from ABOUT (which CONCERNS a Concept)
        # and from INFORMS (Concept -> Concept lateral influence): a different
        # legal shape means a different relation, so it earns its own edge rather
        # than overloading INFORMS. trace_provenance follows it to a Concept's
        # grounding.
        "source_types": {"Observation"},
        "target_types": {"Concept"},
        "properties": {},
    },
    "INFORMS": {
        "source_types": {"Concept"},
        "target_types": {"Concept", "Component"},
        "properties": {},
    },
    "COMPOSES": {
        "source_types": {"Concept"},
        "target_types": {"Concept"},
        "properties": {},
    },
    "DECOMPOSES": {
        "source_types": {"Concept"},
        "target_types": {"Concept"},
        "properties": {},
    },
    "SUPERSEDES": {
        "source_types": {"Hypothesis", "Concept", "Note"},
        "target_types": {"Hypothesis", "Concept", "Note"},
        "same_type": True,
        # revision_why is REQUIRED and must be non-empty. A supersession records
        # not just THAT understanding shifted but WHY — a bare SUPERSEDES loses the
        # delta and is a structural lie (collapse without held content). The why
        # enters the substrate, wired to the how; the guarded write enforces it.
        "properties": {"revision_why": str},
        "required_properties": {"revision_why"},
        "validators": {"revision_why": lambda v: isinstance(v, str) and bool(v.strip())},
    },
}


# -- Validation Functions -----------------------------------------------------

def validate_entity(node_type: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Validate entity properties against the schema registry.

    Returns cleaned properties dict with t_created excluded (auto-set).
    Raises ValueError on validation failure.
    """
    if node_type not in NODE_SCHEMAS:
        valid = ", ".join(sorted(NODE_SCHEMAS.keys()))
        raise ValueError(f"Unknown node type '{node_type}'. Valid types: {valid}")

    schema = NODE_SCHEMAS[node_type]

    # Reject 'type' property — the Neo4j label IS the type.
    if "type" in properties:
        raise ValueError(
            "The 'type' property is forbidden. The Neo4j label IS the type. "
            "Remove 'type' from properties."
        )

    # Check required properties
    missing = []
    for prop in schema["required"]:
        if prop not in properties or properties[prop] is None:
            missing.append(prop)
    if missing:
        raise ValueError(
            f"{node_type} requires properties: {missing}"
        )

    # Type-check and collect valid properties
    all_props = {**schema["required"], **schema.get("optional", {})}
    cleaned: dict[str, Any] = {}
    for key, value in properties.items():
        if key == "t_created":
            continue  # auto-set, skip
        if key not in all_props:
            raise ValueError(
                f"Unknown property '{key}' for {node_type}. "
                f"Valid properties: {sorted(all_props.keys())}"
            )
        expected_type = all_props[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Property '{key}' on {node_type} must be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        cleaned[key] = value

    # Enforce enum constraints (single source of truth — also surfaced by
    # list_node_types, so the allowed values are always readable).
    enums = schema.get("enums", {})
    for prop, allowed in enums.items():
        if prop in cleaned and cleaned[prop] not in allowed:
            raise ValueError(
                f"Invalid value '{cleaned[prop]}' for {node_type}.{prop}. "
                f"Allowed: {sorted(allowed)}"
            )

    return cleaned


def validate_relation(
    rel_type: str,
    source_label: str,
    target_label: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a relationship against the schema registry.

    Enforces direction constraints, source/target type restrictions, the
    same-type constraint (SUPERSEDES), required properties, and property
    enum values. Returns cleaned properties dict.
    """
    if rel_type not in RELATION_SCHEMAS:
        valid = ", ".join(sorted(RELATION_SCHEMAS.keys()))
        raise ValueError(f"Unknown relation type '{rel_type}'. Valid types: {valid}")

    schema = RELATION_SCHEMAS[rel_type]
    properties = properties or {}

    # Direction constraint: source type
    allowed_sources = schema.get("source_types")
    if allowed_sources and source_label not in allowed_sources:
        raise ValueError(
            f"{rel_type} requires source to be one of {allowed_sources}, "
            f"got {source_label}"
        )

    # Direction constraint: target type
    allowed_targets = schema.get("target_types")
    if allowed_targets and target_label not in allowed_targets:
        raise ValueError(
            f"{rel_type} requires target to be one of {allowed_targets}, "
            f"got {target_label}"
        )

    # Same-type constraint (a node supersedes another of its own kind).
    if schema.get("same_type") and source_label != target_label:
        raise ValueError(
            f"{rel_type} requires source and target to share a type — "
            f"got {source_label} -> {target_label}. A node supersedes another "
            f"of its own kind."
        )

    # Check required properties
    required = schema.get("required_properties", set())
    for prop in required:
        if prop not in properties or properties[prop] is None:
            raise ValueError(
                f"{rel_type} requires property '{prop}'"
            )

    # Validate property values
    allowed_props = schema.get("properties", {})
    cleaned: dict[str, Any] = {}
    for key, value in properties.items():
        if key == "t_created":
            continue
        if key not in allowed_props:
            raise ValueError(
                f"Unknown property '{key}' for {rel_type}. "
                f"Valid properties: {sorted(allowed_props.keys())}"
            )
        cleaned[key] = value

    # Run validators
    validators = schema.get("validators", {})
    for prop, validator in validators.items():
        if prop in cleaned and not validator(cleaned[prop]):
            raise ValueError(
                f"Invalid value '{cleaned[prop]}' for {rel_type}.{prop}"
            )

    return cleaned


# Property/range indexes for cheap orientation: fast "latest in chain" lookup
# and hot status filters. Idempotent — IF NOT EXISTS.
INDEX_STATEMENTS: list[LiteralString] = [
    "CREATE INDEX agent_memory_encounter_t_exist IF NOT EXISTS FOR (e:Encounter) ON (e.t_exist)",
    "CREATE INDEX agent_memory_encounter_name IF NOT EXISTS FOR (e:Encounter) ON (e.name)",
    "CREATE INDEX agent_memory_question_status IF NOT EXISTS FOR (q:Question) ON (q.status)",
    "CREATE INDEX agent_memory_hypothesis_status IF NOT EXISTS FOR (h:Hypothesis) ON (h.status)",
    "CREATE INDEX agent_memory_concept_status IF NOT EXISTS FOR (c:Concept) ON (c.status)",
    "CREATE INDEX agent_memory_concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
    "CREATE INDEX agent_memory_component_source_kind IF NOT EXISTS FOR (c:Component) ON (c.source_kind)",
    "CREATE INDEX agent_memory_component_source_key IF NOT EXISTS FOR (c:Component) ON (c.source_key)",
    "CREATE INDEX agent_memory_component_name IF NOT EXISTS FOR (c:Component) ON (c.name)",
    "CREATE INDEX agent_memory_citation_kind IF NOT EXISTS FOR (c:Citation) ON (c.kind)",
    "CREATE INDEX agent_memory_citation_name IF NOT EXISTS FOR (c:Citation) ON (c.name)",
    "CREATE INDEX agent_memory_observation_name IF NOT EXISTS FOR (o:Observation) ON (o.name)",
    "CREATE INDEX agent_memory_question_name IF NOT EXISTS FOR (q:Question) ON (q.name)",
    "CREATE INDEX agent_memory_hypothesis_name IF NOT EXISTS FOR (h:Hypothesis) ON (h.name)",
    "CREATE INDEX agent_memory_note_name IF NOT EXISTS FOR (n:Note) ON (n.name)",
]


# -- Core Logic Class --------------------------------------------------------

class Neo4jAgentMemory:
    """Core logic for the Agent Memory."""

    # Upper bound on tracked loci — a hygiene backstop for a long-lived server
    # accumulating many sessions. Past the cap the oldest entries are evicted
    # (their encounters stay addressable by the explicit `encounter` handle —
    # only the implicit addressing is dropped).
    _LOCUS_STATE_MAX = 4096

    def __init__(self, driver: AsyncDriver):
        self.driver = driver
        # Locus-scoped write targeting: locus key (MCP session id) -> elementId
        # of the encounter THAT locus opened and has not yet sealed. Writes
        # (close_encounter / create_entities anchoring) address this encounter,
        # never the global chain tail — a parallel sibling locus advancing its
        # own chain can no longer capture another locus's writes. Cleared by
        # the seal: afterwards "no open Encounter" means this locus has not
        # advanced or has already closed.
        self._locus_open: dict[str, str] = {}
        # Per-locus chain predecessor: locus key -> elementId of the encounter
        # this locus LAST opened, sealed or not. Unlike _locus_open this
        # survives the seal — within a locus the spine chains across sealed
        # encounters (one session holds many work-units); only the write
        # target ends at the seal.
        self._locus_last: dict[str, str] = {}
        # Per-locus delta novelty: locus key -> content fingerprints of the
        # recognitions/conflicts already announced to that session. Announce
        # a fact-state at first sight, suppress the echo; a changed state
        # re-announces (see infuse.delta_novelty). Session-scoped, evicted
        # alongside the other locus state.
        self._delta_seen: dict[str, set[str]] = {}
        # Per-locus renewal ledger (v0.6.0): locus key -> {node name ->
        # {"fp", "last_full"}} plus a full-mode turn counter. The delivery
        # memory behind the renewal economy: full body at first sight /
        # state change / staleness, one-line handle otherwise (see
        # infuse.renewal_partition). Session-scoped, evicted alongside the
        # other locus state.
        self._delivery_ledger: dict[str, dict[str, dict[str, Any]]] = {}
        self._locus_turn: dict[str, int] = {}

    async def create_fulltext_index(self) -> None:
        """Create the fulltext search index spanning all node types.

        Self-healing: drops and recreates on startup.
        """
        try:
            try:
                await self.driver.execute_query(
                    "DROP INDEX agent_memory_index IF EXISTS",
                    routing_=RoutingControl.WRITE,
                )
            except Exception:
                pass

            await self.driver.execute_query(
                load_cypher("index_create"),
                routing_=RoutingControl.WRITE,
            )
            logger.info("Created fulltext index agent_memory_index")
        except Exception as e:
            logger.debug(f"Fulltext index creation: {e}")

    async def create_indexes(self) -> None:
        """Create property/range indexes for orientation and hot filters."""
        for stmt in INDEX_STATEMENTS:
            try:
                await self.driver.execute_query(stmt, routing_=RoutingControl.WRITE)
            except Exception as e:
                logger.debug(f"Index creation ({stmt[:40]}...): {e}")

    # -- Process Layer (guarded spine) ----------------------------------------

    def _remember_locus(self, locus_key: str, eid: str) -> None:
        """Record a locus's open + last encounter, evicting oldest past the cap."""
        self._locus_open[locus_key] = eid
        self._locus_last[locus_key] = eid
        while len(self._locus_last) > self._LOCUS_STATE_MAX:
            evicted = next(iter(self._locus_last))
            self._locus_last.pop(evicted)
            self._locus_open.pop(evicted, None)
            self._delta_seen.pop(evicted, None)
            self._delivery_ledger.pop(evicted, None)
            self._locus_turn.pop(evicted, None)

    def _delta_seen_for(self, locus_key: str) -> set[str]:
        """The locus's announced-delta set, creating it (bounded) on first use."""
        seen = self._delta_seen.get(locus_key)
        if seen is None:
            seen = self._delta_seen[locus_key] = set()
            while len(self._delta_seen) > self._LOCUS_STATE_MAX:
                self._delta_seen.pop(next(iter(self._delta_seen)))
        return seen

    def _ledger_for(self, locus_key: str) -> dict[str, dict[str, Any]]:
        """The locus's renewal ledger, creating it (bounded) on first use."""
        ledger = self._delivery_ledger.get(locus_key)
        if ledger is None:
            ledger = self._delivery_ledger[locus_key] = {}
            while len(self._delivery_ledger) > self._LOCUS_STATE_MAX:
                evicted = next(iter(self._delivery_ledger))
                self._delivery_ledger.pop(evicted)
                self._locus_turn.pop(evicted, None)
        return ledger

    async def _resolve_encounter(
        self, locus_key: str | None, encounter: str | None
    ) -> dict[str, Any] | None:
        """Resolve which Encounter a write addresses.

        Explicit handle (the Encounter's name) wins — the fallback for a locus
        whose server-side state was lost (a restart) and that is returning to
        the encounter it lived. Otherwise the calling locus's own open
        encounter from server-side state. Returns {name, eid} or None when
        this locus has not advanced or has already closed.
        """
        if encounter is not None:
            result = await self.driver.execute_query(
                "MATCH (e:Encounter {name: $name}) "
                "RETURN e.name AS name, elementId(e) AS eid",
                {"name": encounter},
                routing_=RoutingControl.READ,
            )
            if not result.records:
                raise ValueError(f"Encounter '{encounter}' not found")
            if len(result.records) > 1:
                raise ValueError(
                    f"Encounter '{encounter}' is ambiguous — "
                    f"found {len(result.records)} encounters with that name"
                )
            r = result.records[0]
            return {"name": r["name"], "eid": r["eid"]}

        if locus_key is None or locus_key not in self._locus_open:
            return None
        eid = self._locus_open[locus_key]
        result = await self.driver.execute_query(
            "MATCH (e:Encounter) WHERE elementId(e) = $eid "
            "RETURN e.name AS name, elementId(e) AS eid",
            {"eid": eid},
            routing_=RoutingControl.READ,
        )
        if not result.records:
            # The tracked encounter no longer exists — drop the stale state.
            self._locus_open.pop(locus_key, None)
            return None
        r = result.records[0]
        return {"name": r["name"], "eid": r["eid"]}

    async def advance_encounter(
        self,
        name: str,
        recent: int = 5,
        limit: int = 20,
        locus_key: str | None = None,
    ) -> dict[str, Any]:
        """Open a new Encounter and return the re-entry payload.

        The sole writer of the temporal spine. The caller never passes a
        predecessor; the server supplies this LOCUS's own previous encounter
        from its state (per-locus chaining). A first-of-locus encounter has no
        incoming NEXT_ENCOUNTER and is genesis-bound (INSTANTIATED_AFTER) to
        the encounter latest at its genesis — locus geneses are serialized
        writes, so the order is integrated, and parallel loci are branches of
        one connected becoming. NEXT_ENCOUNTER stays fork-free absolutely.
        Opening an encounter IS orienting — Exist and the orientation read are
        one act.
        """
        pred_eid = self._locus_last.get(locus_key) if locus_key else None
        result = await self.driver.execute_query(
            load_cypher("encounter_advance"),
            {"name": name, "pred_eid": pred_eid},
            routing_=RoutingControl.WRITE,
        )
        r = result.records[0]
        encounter = {
            "name": r["name"],
            "t_exist": _neo4j_datetime_to_str(r["t_exist"]),
            "t_created": _neo4j_datetime_to_str(r["t_created"]),
            "predecessor": r["predecessor"],
            # Genesis binding: for a first-of-locus, the encounter that was
            # latest when this locus began (INSTANTIATED_AFTER target). Null
            # for a within-locus advance, and for the true head of the spine.
            "genesis_anchor": r["genesis_anchor"],
            "is_first": r["is_first"],
        }
        if locus_key:
            self._remember_locus(locus_key, r["eid"])
        reentry = await self._reentry_payload(
            recent=recent, limit=limit, current_eid=r["eid"]
        )
        return {"encounter": encounter, "reentry": reentry}

    async def _unsealed(
        self, limit: int, exclude_eid: str | None = None
    ) -> list[dict[str, Any]]:
        """All unsealed Encounters (no summary, no report), each annotated with
        its last-activity time — the latest RECORDED/CONSULTED anchoring, or
        the encounter's own t_exist if it never recorded anything — and any
        mechanical dissolved_at mark. Surfaces, never classifies: the graph
        cannot tell a live sibling locus from an orphan, so it does not try.
        The caller (a re-entering locus) never joins one — it always opens its
        own encounter."""
        result = await self.driver.execute_query(
            "MATCH (e:Encounter) "
            "WHERE e.summary IS NULL AND e.report IS NULL "
            "AND ($exclude_eid IS NULL OR elementId(e) <> $exclude_eid) "
            "OPTIONAL MATCH (e)-[a:RECORDED|CONSULTED]->() "
            "WITH e, max(a.t_created) AS last_anchor "
            "RETURN e.name AS name, e.t_exist AS t_exist, "
            "       CASE WHEN last_anchor IS NULL OR e.t_exist > last_anchor "
            "            THEN e.t_exist ELSE last_anchor END AS last_active, "
            "       e.dissolved_at AS dissolved_at "
            "ORDER BY last_active DESC LIMIT $limit",
            {"exclude_eid": exclude_eid, "limit": limit},
            routing_=RoutingControl.READ,
        )
        return [
            {
                "name": r["name"],
                "t_exist": _neo4j_datetime_to_str(r["t_exist"]),
                "last_active": _neo4j_datetime_to_str(r["last_active"]),
                "dissolved_at": _neo4j_datetime_to_str(r["dissolved_at"]),
            }
            for r in result.records
        ]

    async def _reentry_payload(
        self, recent: int, limit: int, current_eid: str | None = None
    ) -> dict[str, Any]:
        """Assemble the orientation read: recent encounters, open Questions,
        live Hypotheses, recently-touched Concepts, and the coherence edges
        incident to those tension nodes. Surfaces — never authors.

        Coherence is in the edges, not the nodes: the tension lists alone are a
        gradientless bag. The coherence_edges block returns the 1-hop authored
        edges (everything except the auto-written PROCESS_EDGES) touching the
        surfaced Questions/Hypotheses/Concepts, so an open Question arrives with
        what ANSWERS it (or the absence of any), a Hypothesis with what
        CHALLENGES/SUPPORTS/TESTS it, a Concept with the SUPERSEDES trail it
        heads. The slope, delivered with the peaks, in one orientation read.

        Also carries the unsealed set: every Encounter (other than the one just
        opened) with neither summary nor report, annotated with last-activity
        time and any mechanical dissolved_at mark. An unsealed encounter is a
        live sibling locus or an orphaned dissolution — the graph cannot tell
        which, so it surfaces and never classifies, and a re-entering locus
        never joins one."""

        recent_encounters = await self.driver.execute_query(
            "MATCH (e:Encounter) "
            "RETURN e.name AS name, e.t_exist AS t_exist, "
            "       e.summary AS summary, e.report AS report "
            "ORDER BY e.t_exist DESC LIMIT $recent",
            {"recent": recent},
            routing_=RoutingControl.READ,
        )

        open_questions = await self.driver.execute_query(
            "MATCH (q:Question) WHERE q.status = 'open' OR q.status IS NULL "
            "RETURN q.name AS name, q.description AS description, "
            "       q.priority AS priority, q.t_raised AS t_raised "
            "ORDER BY q.t_raised DESC LIMIT $limit",
            {"limit": limit},
            routing_=RoutingControl.READ,
        )

        live_hypotheses = await self.driver.execute_query(
            "MATCH (h:Hypothesis) "
            "WHERE h.status IS NULL OR h.status IN ['proposed', 'challenged'] "
            "RETURN h.name AS name, h.description AS description, "
            "       h.confidence AS confidence, h.status AS status, "
            "       h.t_proposed AS t_proposed "
            "ORDER BY h.t_proposed DESC LIMIT $limit",
            {"limit": limit},
            routing_=RoutingControl.READ,
        )

        recent_concepts = await self.driver.execute_query(
            "MATCH (e:Encounter) WITH e ORDER BY e.t_exist DESC LIMIT $recent "
            "MATCH (e)-[:RECORDED]->(c:Concept) "
            "RETURN DISTINCT c.name AS name, c.description AS description, "
            "       c.status AS status, c.category AS category "
            "LIMIT $limit",
            {"recent": recent, "limit": limit},
            routing_=RoutingControl.READ,
        )

        # The gradient: 1-hop coherence edges incident to the surfaced tension
        # nodes. The exclusion is derived from PROCESS_EDGES (one source of
        # truth) so the auto-written spine/provenance edges — which would make
        # Encounters into artificial centrality hubs — never enter the read.
        # startNode/endNode preserve true authored direction regardless of
        # match direction; DISTINCT collapses the double-match per edge.
        coherence_edges = await self.driver.execute_query(
            "MATCH (e:Encounter) WITH e ORDER BY e.t_exist DESC LIMIT $recent "
            "OPTIONAL MATCH (e)-[:RECORDED]->(c:Concept) "
            "WITH collect(DISTINCT c) AS cs "
            "OPTIONAL MATCH (q:Question) WHERE q.status = 'open' OR q.status IS NULL "
            "WITH cs, collect(DISTINCT q) AS qs "
            "OPTIONAL MATCH (h:Hypothesis) "
            "WHERE h.status IS NULL OR h.status IN ['proposed', 'challenged'] "
            "WITH cs, qs, collect(DISTINCT h) AS hs "
            "WITH cs + qs + hs AS anchors "
            "UNWIND anchors AS a "
            "MATCH (a)-[r]-(nbr) WHERE NOT type(r) IN $process_edges "
            "RETURN DISTINCT startNode(r).name AS from_name, "
            "       labels(startNode(r))[0] AS from_type, type(r) AS rel, "
            "       endNode(r).name AS to_name, labels(endNode(r))[0] AS to_type, "
            "       properties(r) AS rel_props "
            "LIMIT $edge_limit",
            {
                "recent": recent,
                "process_edges": sorted(PROCESS_EDGES),
                "edge_limit": limit * 5,
            },
            routing_=RoutingControl.READ,
        )

        # The unsealed set: an Encounter ends cleanly by being sealed
        # (close_encounter writes a summary/report) — even a measured-null run
        # gets a close. One with NEITHER is either a live sibling locus still
        # working or a dissolution (the interior cycle cut before it could
        # resolve). Not mechanically distinguishable — surfaced, not classified.
        unsealed = await self._unsealed(limit=limit, exclude_eid=current_eid)

        def _rows(result, fields):
            out = []
            for r in result.records:
                row = {}
                for f in fields:
                    v = r[f]
                    row[f] = _neo4j_datetime_to_str(v) if hasattr(v, "iso_format") else v
                out.append(row)
            return out

        # Coherence edges carry their authored properties (e.g. revision_why on
        # SUPERSEDES) so re-entry sees WHY an edge exists, not just its type.
        coherence_rows = []
        for r in coherence_edges.records:
            row = {
                "from_name": r["from_name"], "from_type": r["from_type"],
                "rel": r["rel"],
                "to_name": r["to_name"], "to_type": r["to_type"],
            }
            props = _clean_edge_props(r["rel_props"])
            if props:
                row["props"] = props
            coherence_rows.append(row)

        return {
            "recent_encounters": _rows(recent_encounters, ["name", "t_exist", "summary", "report"]),
            "open_questions": _rows(open_questions, ["name", "description", "priority", "t_raised"]),
            "live_hypotheses": _rows(live_hypotheses, ["name", "description", "confidence", "status", "t_proposed"]),
            "recently_touched_concepts": _rows(recent_concepts, ["name", "description", "status", "category"]),
            "coherence_edges": coherence_rows,
            "unsealed": unsealed,
        }

    # Reserved names for the orient projections. Fixed names (not per-call
    # ones) let a crashed prior orient be cleaned up defensively. The _asof
    # projection is the previous-waking baseline used by the drift reading.
    _ORIENT_PROJECTION = "__agent_memory_orient__"
    _ORIENT_ASOF_PROJECTION = "__agent_memory_orient_asof__"

    async def _frontier(self, limit: int = 20) -> dict[str, Any]:
        """The epistemic frontier: where the topology is thinnest, derived from
        existing structure (no new schema, no GDS). orient surfaces what you
        treat as central; the frontier surfaces where one more Observation would
        move the needle most — turning the self-portrait from descriptive to
        directive. Each signal is a fact-that-reads-as-a-question; the agent
        decides what to do with it. Surfaces — never authors.

        - unanswered_questions: open Questions nothing RESOLVES yet.
        - untested_hypotheses: live Hypotheses with no SUPPORTS and no CHALLENGES.
        - ungrounded_concepts: Concepts no Observation GROUNDS or is ABOUT — a
          guess wearing the costume of understanding.
        - confidence_dissonance: a node's authored confidence/status outrunning
          its structural support (confidence:high on ≤1 SUPPORTS; stable with no
          observational grounding at all — neither GROUNDS nor ABOUT) — the
          authored claim and the evidence weight disagree. The grounding arm is
          ABOUT-aware so it does not flood on a substrate that predates GROUNDS.
        - contested_hypotheses: Hypotheses carrying BOTH support and challenge —
          live tension already in the substrate, worth revisiting.
        """
        unanswered = await self.driver.execute_query(
            "MATCH (q:Question) WHERE (q.status = 'open' OR q.status IS NULL) "
            "AND coalesce(q.frontier_mute, false) = false "
            "AND NOT (:Observation)-[:RESOLVES]->(q) "
            "RETURN q.name AS name, q.description AS description "
            "ORDER BY q.t_raised DESC LIMIT $limit",
            {"limit": limit},
            routing_=RoutingControl.READ,
        )
        untested = await self.driver.execute_query(
            "MATCH (h:Hypothesis) "
            "WHERE (h.status IS NULL OR h.status IN ['proposed', 'challenged']) "
            "AND coalesce(h.frontier_mute, false) = false "
            "AND NOT (:Observation)-[:SUPPORTS]->(h) "
            "AND NOT (:Observation)-[:CHALLENGES]->(h) "
            "RETURN h.name AS name, h.description AS description, "
            "       h.confidence AS confidence "
            "ORDER BY h.t_proposed DESC LIMIT $limit",
            {"limit": limit},
            routing_=RoutingControl.READ,
        )
        ungrounded = await self.driver.execute_query(
            "MATCH (c:Concept) "
            "WHERE coalesce(c.frontier_mute, false) = false "
            "AND NOT (:Observation)-[:GROUNDS]->(c) "
            "AND NOT (:Observation)-[:ABOUT]->(c) "
            "RETURN c.name AS name, c.description AS description, "
            "       c.status AS status "
            "LIMIT $limit",
            {"limit": limit},
            routing_=RoutingControl.READ,
        )
        dissonance = await self.driver.execute_query(
            "CALL () { "
            "  MATCH (h:Hypothesis) WHERE h.confidence = 'high' "
            "  AND coalesce(h.frontier_mute, false) = false "
            "  WITH h, COUNT { (:Observation)-[:SUPPORTS]->(h) } AS sup "
            "  WHERE sup <= 1 "
            "  RETURN h.name AS name, 'Hypothesis' AS type, "
            "         'confidence:high, ' + toString(sup) + ' SUPPORTS' AS signal "
            "  UNION "
            "  MATCH (c:Concept) WHERE c.status = 'stable' "
            "  AND coalesce(c.frontier_mute, false) = false "
            "  AND NOT (:Observation)-[:GROUNDS]->(c) "
            "  AND NOT (:Observation)-[:ABOUT]->(c) "
            "  RETURN c.name AS name, 'Concept' AS type, "
            "         'status:stable, no observational grounding' AS signal "
            "} "
            "RETURN name, type, signal LIMIT $limit",
            {"limit": limit},
            routing_=RoutingControl.READ,
        )
        contested = await self.driver.execute_query(
            "MATCH (h:Hypothesis) "
            "WHERE coalesce(h.frontier_mute, false) = false "
            "AND (:Observation)-[:SUPPORTS]->(h) "
            "AND (:Observation)-[:CHALLENGES]->(h) "
            "RETURN h.name AS name, h.description AS description, "
            "       h.status AS status "
            "LIMIT $limit",
            {"limit": limit},
            routing_=RoutingControl.READ,
        )

        def _rows(result, fields):
            return [{f: r[f] for f in fields} for r in result.records]

        return {
            "unanswered_questions": _rows(unanswered, ["name", "description"]),
            "untested_hypotheses": _rows(untested, ["name", "description", "confidence"]),
            "ungrounded_concepts": _rows(ungrounded, ["name", "description", "status"]),
            "confidence_dissonance": _rows(dissonance, ["name", "type", "signal"]),
            "contested_hypotheses": _rows(contested, ["name", "description", "status"]),
        }

    async def orient(self, result_limit: int = 20) -> dict[str, Any]:
        """The Exist read: one structural self-portrait over the coherence
        subgraph, run once per waking — not per Encounter.

        The panel is constitution, not analytics: its payload is causal input
        to the next encounter's writes, so each instrument is chosen against
        the gravity well (mass attracts authorship attracts mass). The
        projection lifecycle is self-managed — cleanup is a finally, not a
        discipline, and a crashed prior run is dropped defensively. The
        projection is coherence-only by construction — authored edges
        (RelationType minus PROCESS_EDGES) over the non-process nodes (every
        type but Encounter), undirected — so the spine/provenance edges that
        turn Encounters into artificial centrality hubs can never enter the
        read. Surfaces — never authors; writes no node, no edge.

        The readings: MASS (ArticleRank — damped accumulation, a weaker well
        than PageRank). FRONTIER_MASS (personalized PageRank seeded from the
        frontier — the same graph seen from the unresolved). DIVERGENCE (the
        manifest-destiny detector: high in mass, low from the frontier = big
        because it is big). BETWEENNESS (what bridges your inquiries). LEIDEN
        (your chapters — guaranteed well-connected; an identity instrument
        must not hallucinate communities). WCC (what you never connected).
        FRAGILITY (articulation points + bridges — importance as load-bearing
        responsibility, not accumulation). WEAVE_AUDIT (clustering coefficient
        vs degree — stars vs weaves). DRIFT (mass vs the previous waking:
        risers, fallers, new — the well visible AS a well, with velocity).
        Plus the epistemic FRONTIER (the directive half) and the UNSEALED set.
        Within the Exist, reach for the single gds_* tools when a specific
        question arises; this is the opening survey.
        """
        proj = self._ORIENT_PROJECTION
        asof = self._ORIENT_ASOF_PROJECTION
        rel_filter, source_labels, target_labels = coherence_projection_parts()
        project_query = lit(
            f"MATCH (source)-[r:{rel_filter}]->(target) "
            f"WHERE ({source_labels}) AND ({target_labels}) "
            "RETURN gds.graph.project($proj, source, target, {}, "
            "{undirectedRelationshipTypes: ['*']}) AS g"
        )
        # The previous-waking baseline: the same coherence subgraph restricted
        # to what existed when the latest locus woke (nodes AND edges — an edge
        # authored later between old nodes is still new structure).
        asof_project_query = lit(
            f"MATCH (source)-[r:{rel_filter}]->(target) "
            f"WHERE ({source_labels}) AND ({target_labels}) "
            "AND source.t_created < $boundary AND target.t_created < $boundary "
            "AND r.t_created < $boundary "
            "RETURN gds.graph.project($proj, source, target, {}, "
            "{undirectedRelationshipTypes: ['*']}) AS g"
        )

        drop_query = (
            "CALL gds.graph.drop($proj, false) YIELD graphName RETURN graphName"
        )
        node_return = (
            "RETURN gds.util.asNode(nodeId).name AS node, "
            "labels(gds.util.asNode(nodeId))[0] AS type"
        )
        # Full streams (python-limited): the composite readings need every
        # node's rank, not the display cut.
        mass_query = (
            f"CALL gds.articleRank.stream($proj) YIELD nodeId, score "
            f"{node_return}, score ORDER BY score DESC"
        )
        seeded_query = (
            "MATCH (s) WHERE s.name IN $seed_names "
            "WITH collect(s) AS seeds "
            "CALL gds.pageRank.stream($proj, {sourceNodes: seeds}) "
            f"YIELD nodeId, score {node_return}, score ORDER BY score DESC"
        )
        degree_query = (
            f"CALL gds.degree.stream($proj) YIELD nodeId, score "
            f"{node_return}, score ORDER BY score DESC"
        )
        coefficient_query = (
            "CALL gds.localClusteringCoefficient.stream($proj) "
            "YIELD nodeId, localClusteringCoefficient "
            f"{node_return}, localClusteringCoefficient AS coefficient"
        )
        # Display-limited streams.
        algos = {
            "betweenness": "CALL gds.betweenness.stream($proj) YIELD nodeId, score "
            f"{node_return}, score ORDER BY score DESC LIMIT $lim",
            "leiden": "CALL gds.leiden.stream($proj) YIELD nodeId, communityId "
            f"{node_return}, communityId ORDER BY communityId, node LIMIT $lim",
            "wcc": "CALL gds.wcc.stream($proj) YIELD nodeId, componentId "
            f"{node_return}, componentId ORDER BY componentId, node LIMIT $lim",
        }
        articulation_query = (
            f"CALL gds.articulationPoints.stream($proj) YIELD nodeId "
            f"{node_return} ORDER BY node LIMIT $lim"
        )
        bridges_query = (
            "CALL gds.bridges.stream($proj) YIELD from, to "
            "RETURN gds.util.asNode(from).name AS from_node, "
            "gds.util.asNode(to).name AS to_node "
            "ORDER BY from_node, to_node LIMIT $lim"
        )

        async def _stream(
            query: LiteralString, projection: str = proj, **params: Any
        ) -> list[dict[str, Any]]:
            res = await self.driver.execute_query(
                query, {"proj": projection, **params}, routing_=RoutingControl.READ
            )
            return [dict(r) for r in res.records]

        # Defensive: clear any projections a crashed prior orient left behind.
        for name in (proj, asof):
            await self.driver.execute_query(
                drop_query, {"proj": name}, routing_=RoutingControl.WRITE
            )
        try:
            proj_result = await self.driver.execute_query(
                project_query, {"proj": proj}, routing_=RoutingControl.READ
            )
            g = proj_result.records[0]["g"] if proj_result.records else {}
            out: dict[str, Any] = {
                "projection": {
                    "coherence_only": True,
                    "node_count": g.get("nodeCount") if isinstance(g, dict) else None,
                    "relationship_count": g.get("relationshipCount")
                    if isinstance(g, dict)
                    else None,
                }
            }

            # The frontier first: it is both a reading (the directive half)
            # and the seed authority for the frontier-relative mass reading.
            frontier = await self._frontier(limit=result_limit)
            out["frontier"] = frontier

            # MASS — ArticleRank, full stream.
            mass_full = await _stream(lit(mass_query))
            out["mass"] = mass_full[:result_limit]

            # FRONTIER_MASS — personalized PageRank seeded from the frontier.
            # Primary seeds: open Questions + untested Hypotheses (the named
            # unresolved). Fallback when those are empty: every frontier cut.
            # Seeds must exist in the projection (i.e. carry >=1 coherence
            # edge), so eligibility is checked before seeding; an isolated
            # Question is real frontier but invisible to a graph algorithm.
            primary, fallback = frontier_seed_candidates(frontier)
            seeds = await self._eligible_seeds(primary, rel_filter)
            seed_mode = "frontier"
            if not seeds:
                seeds = await self._eligible_seeds(fallback, rel_filter)
                seed_mode = "full_frontier_fallback"
            if seeds:
                seeded_full = await _stream(lit(seeded_query), seed_names=seeds)
                out["frontier_mass"] = {
                    "seed_mode": seed_mode,
                    "seeds": seeds,
                    "results": seeded_full[:result_limit],
                }
                # DIVERGENCE — the manifest-destiny detector, only meaningful
                # when both reference frames produced a reading.
                out["divergence"] = compute_divergence(
                    mass_full, seeded_full, result_limit
                )
            else:
                out["frontier_mass"] = {
                    "seed_mode": "none",
                    "seeds": [],
                    "results": [],
                }
                out["divergence"] = None

            # BETWEENNESS / LEIDEN / WCC — display-limited.
            for key, q in algos.items():
                out[key] = await _stream(lit(q), lim=result_limit)

            # FRAGILITY — articulation points + bridges.
            out["fragility"] = {
                "articulation_points": await _stream(
                    lit(articulation_query), lim=result_limit
                ),
                "bridges": await _stream(lit(bridges_query), lim=result_limit),
            }

            # WEAVE_AUDIT — clustering coefficient against degree.
            degree_full = await _stream(lit(degree_query))
            coefficient_full = await _stream(lit(coefficient_query))
            out["weave_audit"] = compute_weave_audit(
                degree_full, coefficient_full, result_limit
            )

            # DRIFT — mass vs the previous waking. The boundary is the latest
            # locus genesis (orient runs before this locus advances, so the
            # latest root is a prior waking's). Read-time, no writes.
            boundary_res = await self.driver.execute_query(
                "MATCH (e:Encounter) WHERE NOT ()-[:NEXT_ENCOUNTER]->(e) "
                "RETURN max(e.t_exist) AS boundary",
                routing_=RoutingControl.READ,
            )
            boundary = (
                boundary_res.records[0]["boundary"] if boundary_res.records else None
            )
            if boundary is None:
                out["drift"] = None
            else:
                try:
                    asof_result = await self.driver.execute_query(
                        asof_project_query,
                        {"proj": asof, "boundary": boundary},
                        routing_=RoutingControl.READ,
                    )
                    ag = asof_result.records[0]["g"] if asof_result.records else {}
                    baseline_nodes = (
                        ag.get("nodeCount") if isinstance(ag, dict) else None
                    )
                    if not baseline_nodes:
                        out["drift"] = {
                            "as_of": _neo4j_datetime_to_str(boundary),
                            "note": "no coherence structure existed at the previous waking",
                        }
                    else:
                        baseline_full = await _stream(
                            lit(mass_query), projection=asof
                        )
                        out["drift"] = {
                            "as_of": _neo4j_datetime_to_str(boundary),
                            "baseline_node_count": baseline_nodes,
                            **compute_drift(mass_full, baseline_full, result_limit),
                        }
                except Exception as exc:  # the panel survives a failed baseline
                    logger.warning(f"orient drift baseline failed: {exc}")
                    out["drift"] = {
                        "as_of": _neo4j_datetime_to_str(boundary),
                        "note": f"baseline unavailable: {exc}",
                    }

            # The unsealed set rides along too: orient runs before the first
            # advance, so the waking sees every unsealed encounter — live
            # sibling or orphan, not mechanically distinguishable — annotated
            # with last activity. Surfaced, never classified, never auto-joined.
            out["unsealed"] = await self._unsealed(limit=result_limit)
            return out
        finally:
            for name in (proj, asof):
                await self.driver.execute_query(
                    drop_query, {"proj": name}, routing_=RoutingControl.WRITE
                )

    # -- Infusion (the governed passive synthesis) -----------------------------

    # Seeds admitted from Match, and the ranked-subgraph size the payload is
    # built from. Small on purpose: the payload is a disposition, not a dump.
    _INFUSE_SEED_LIMIT = 12
    _INFUSE_DELTA_MATCH_LIMIT = 8
    # Concept-cluster expansion (v0.7.0, EXPERIMENT-BARLOW B2): matched-band
    # reach — nodes one coherence hop from focal-matched Concepts enter the
    # blend at this bias. Fixed this version, deliberately: one new tunable
    # at a time (per-call override exists for the benchmark's OFF arm).
    _INFUSE_EXPANSION_BIAS = 0.5

    async def _eligible_seeds(self, names: list[str], rel_filter: str) -> list[str]:
        """Filter seed names to those carrying >=1 coherence edge — a seed must
        exist in the coherence-only projection to bias a rank computation; an
        isolated node is real but invisible to a graph algorithm."""
        if not names:
            return []
        res = await self.driver.execute_query(
            lit(
                f"MATCH (s)-[:{rel_filter}]-() WHERE s.name IN $names "
                "RETURN collect(DISTINCT s.name) AS eligible"
            ),
            {"names": list(dict.fromkeys(names))},
            routing_=RoutingControl.READ,
        )
        return res.records[0]["eligible"] if res.records else []

    async def _match_focal(
        self, query: str, limit: int
    ) -> list[dict[str, Any]]:
        """Match: resolve focal signals to substrate nodes via the fulltext
        index. Process nodes are excluded — an Encounter is when, not what."""
        result = await self.driver.execute_query(
            "CALL db.index.fulltext.queryNodes('agent_memory_index', $query) "
            "YIELD node, score WHERE NOT node:Encounter "
            "RETURN node.name AS name, labels(node)[0] AS type, "
            "       node.description AS description, score "
            "ORDER BY score DESC LIMIT $limit",
            {"query": query, "limit": limit},
            routing_=RoutingControl.READ,
        )
        return [dict(r) for r in result.records]

    async def _conflicts_among(self, names: list[str]) -> list[dict[str, Any]]:
        """Conflict extraction: CHALLENGES edges and confidence/evidence
        dissonance touching the named subgraph. Divergence flags are computed
        by the caller (they need both rank frames)."""
        if not names:
            return []
        challenges = await self.driver.execute_query(
            "MATCH (o:Observation)-[r:CHALLENGES]->(h:Hypothesis) "
            "WHERE o.name IN $names OR h.name IN $names "
            "RETURN o.name AS from_name, h.name AS to_name, "
            "       properties(r) AS props",
            {"names": names},
            routing_=RoutingControl.READ,
        )
        dissonance = await self.driver.execute_query(
            "CALL () { "
            "  MATCH (h:Hypothesis) WHERE h.confidence = 'high' AND h.name IN $names "
            "  WITH h, COUNT { (:Observation)-[:SUPPORTS]->(h) } AS sup "
            "  WHERE sup <= 1 "
            "  RETURN h.name AS name, "
            "         'confidence:high, ' + toString(sup) + ' SUPPORTS' AS detail "
            "  UNION "
            "  MATCH (c:Concept) WHERE c.status = 'stable' AND c.name IN $names "
            "  AND NOT (:Observation)-[:GROUNDS]->(c) "
            "  AND NOT (:Observation)-[:ABOUT]->(c) "
            "  RETURN c.name AS name, "
            "         'status:stable, no observational grounding' AS detail "
            "} "
            "RETURN name, detail",
            {"names": names},
            routing_=RoutingControl.READ,
        )
        out: list[dict[str, Any]] = [
            {
                "kind": "challenge",
                "from_name": r["from_name"],
                "to_name": r["to_name"],
                "props": _clean_edge_props(r["props"]),
            }
            for r in challenges.records
        ]
        out.extend(
            {"kind": "dissonance", "from_name": r["name"], "detail": r["detail"]}
            for r in dissonance.records
        )
        return out

    async def _anchors_for(self, names: list[str]) -> list[dict[str, Any]]:
        """The anchoring Encounters of the named nodes, oldest first — the
        temporal trajectory: which encounters constituted this understanding,
        in what order. A fact about the WEIGHT of the knowledge."""
        if not names:
            return []
        result = await self.driver.execute_query(
            "MATCH (e:Encounter)-[:RECORDED|CONSULTED]->(n) "
            "WHERE n.name IN $names "
            "RETURN e.name AS encounter, e.t_exist AS t_exist, n.name AS name "
            "ORDER BY e.t_exist ASC LIMIT 20",
            {"names": names},
            routing_=RoutingControl.READ,
        )
        return [
            {
                "encounter": r["encounter"],
                "t_exist": _neo4j_datetime_to_str(r["t_exist"]),
                "name": r["name"],
            }
            for r in result.records
        ]

    async def infuse(
        self,
        text: str,
        mode: str = "full",
        frontier_bias: float = 0.3,
        max_chars: int = 10_000,
        result_limit: int = 30,
        locus_key: str | None = None,
        refresh_turns: int = 10,
        expansion_bias: float | None = None,
    ) -> dict[str, Any]:
        """The governed infusion read: Extract -> Match -> Rank -> Format.

        Mechanized passive synthesis — re-awaken the shape of what the
        substrate already holds about the arriving present, at every decision
        point, governed against the gravity well. NOT retrieval: the payload
        answers "what is the topology of what I already hold about this?",
        ordered deliberately (conflict first), signed as a proposal from the
        sediment, silent when the substrate has nothing to say. Surfaces —
        never authors; writes no node, no edge; reads only.

        mode='full': the complete disposition — one biased rank computation
        seeded from focal matches (bias 1.0) blended with the standing
        frontier (bias frontier_bias), divergence-checked against unbiased
        mass, conflicts triaged core/parked by constitutive proximity (the
        biased rank score IS proximity), neighborhood, open threads, temporal
        trajectory. Delivered with every user prompt.

        mode='delta': recognition ("already held: ...") or conflict ("this
        contradicts what you hold") only — otherwise the empty string, and
        the hook stays silent. Fired at tool-batch boundaries; no GDS, fast.
        Attenuation is one failure of the living present; burying the
        arriving present under sediment after every batch is the other.
        """
        if mode not in ("full", "delta"):
            raise ValueError(f"Unknown infuse mode '{mode}'. Use 'full' or 'delta'.")
        frontier_bias = max(0.0, min(1.0, frontier_bias))
        max_chars = max(500, min(10_000, max_chars))
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        signals = extract_focal_signals(text)
        seed_terms = [s["term"] for s in signals]
        timings["extract"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        matches: list[dict[str, Any]] = []
        if signals:
            matches = await self._match_focal(
                lucene_query(signals),
                self._INFUSE_DELTA_MATCH_LIMIT
                if mode == "delta"
                else self._INFUSE_SEED_LIMIT,
            )
        timings["match"] = time.perf_counter() - t1

        if mode == "delta":
            return await self._infuse_delta(
                seed_terms, matches, max_chars, timings, t0, locus_key
            )
        return await self._infuse_full(
            seed_terms, matches, frontier_bias, max_chars, result_limit,
            timings, t0, locus_key,
            refresh_turns=max(1, min(100, refresh_turns)),
            expansion_bias=(
                self._INFUSE_EXPANSION_BIAS if expansion_bias is None
                else max(0.0, min(1.0, expansion_bias))
            ),
        )

    async def _infuse_delta(
        self,
        seed_terms: list[str],
        matches: list[dict[str, Any]],
        max_chars: int,
        timings: dict[str, float],
        t0: float,
        locus_key: str | None = None,
    ) -> dict[str, Any]:
        """The tool-batch delta: recognition and conflict against the matched
        nodes only. Everything matched is focal by construction, so a conflict
        here is core by definition — but delta computes no rank, so its rows
        carry NO severity: the contradiction, not a number, is the finding
        (an amplitude printed here and a different amplitude printed by full
        mode for the same edge would assert a contradiction of our own).

        Novelty-gated per locus: each recognition/conflict FACT-STATE is
        announced to a session once, at first sight, then suppressed — the
        cadence derives from the substrate (first arrival of a fact-state),
        not the scheduler (batch boundaries), and holds under batch-level
        (PostToolBatch) or per-call (PostToolUse) harness wiring alike. A
        changed state is a new first sight and re-announces. Suppression
        counts are reported; the payload stays honest."""
        recognitions: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        if matches:
            top_score = matches[0]["score"]
            strong = [m for m in matches if m["score"] >= 0.6 * top_score][:5]
            names = [m["name"] for m in strong]
            anchors = await self.driver.execute_query(
                "MATCH (e:Encounter)-[:RECORDED|CONSULTED]->(n) "
                "WHERE n.name IN $names "
                "RETURN n.name AS name, collect(e.name) AS encounters",
                {"names": names},
                routing_=RoutingControl.READ,
            )
            anchored = {r["name"]: r["encounters"] for r in anchors.records}
            recognitions = [
                {
                    "name": m["name"],
                    "type": m["type"],
                    # Sorted so the novelty content-fingerprint is stable
                    # against collect() ordering.
                    "encounters": sorted(anchored.get(m["name"], [])),
                }
                for m in strong
            ]
            conflicts = await self._conflicts_among(names)

        suppressed = {"recognitions": 0, "conflicts": 0}
        if locus_key is not None and (recognitions or conflicts):
            seen = self._delta_seen_for(locus_key)
            fresh_r, fresh_c, new_keys = delta_novelty(
                recognitions, conflicts, seen
            )
            suppressed = {
                "recognitions": len(recognitions) - len(fresh_r),
                "conflicts": len(conflicts) - len(fresh_c),
            }
            recognitions, conflicts = fresh_r, fresh_c
            seen |= new_keys

        payload = format_delta(recognitions, conflicts, max_chars=max_chars)
        timings["total"] = time.perf_counter() - t0
        return {
            "mode": "delta",
            "payload": payload,
            "silence": payload == "",
            "seed_terms": seed_terms,
            "recognitions": recognitions,
            "conflicts": conflicts,
            "suppressed": suppressed,
            "timings_ms": {k: round(v * 1000, 1) for k, v in timings.items()},
        }

    async def _infuse_full(
        self,
        seed_terms: list[str],
        matches: list[dict[str, Any]],
        frontier_bias: float,
        max_chars: int,
        result_limit: int,
        timings: dict[str, float],
        t0: float,
        locus_key: str | None = None,
        refresh_turns: int = 10,
        expansion_bias: float = 0.5,
    ) -> dict[str, Any]:
        """The full governed payload: one biased rank blending focal and
        frontier seeds over an ephemeral coherence-only projection, divergence
        against unbiased mass, triaged conflicts, neighborhood, open threads,
        trajectory — formatted tension-first and signed.

        Renewal-gated per locus (v0.6.0): neighborhood bodies are delivered
        in full at first sight, on state change, or when stale; otherwise
        they re-pin as one-line handles in a STANDING register. The measured
        v0.5.x cost this repays: 19 payloads at saturation with a permanent
        verbatim core and trajectory yielding every turn."""
        rel_filter, source_labels, target_labels = coherence_projection_parts()

        t2 = time.perf_counter()
        focal_seeds = await self._eligible_seeds(
            [m["name"] for m in matches], rel_filter
        )
        frontier = await self._frontier(limit=20)
        primary, fallback = frontier_seed_candidates(frontier)
        frontier_seeds = await self._eligible_seeds(primary, rel_filter)
        if not frontier_seeds:
            frontier_seeds = await self._eligible_seeds(fallback, rel_filter)
        frontier_seeds = [n for n in frontier_seeds if n not in set(focal_seeds)]

        # Concept-cluster expansion (EXPERIMENT-BARLOW B2): siblings one
        # coherence hop from focal-matched Concepts — the adjacent concept
        # that DID match bridges to the on-point node that could not
        # (measured motivation: the Husserlian correction at 0/19 while its
        # cluster rode 8/19). Enters the blend at a reduced bias.
        expansion_seeds: list[str] = []
        if expansion_bias > 0 and focal_seeds:
            taken = set(focal_seeds) | set(frontier_seeds)
            exp_res = await self.driver.execute_query(
                lit(
                    f"MATCH (s:Concept)-[:{rel_filter}]-(n) "
                    "WHERE s.name IN $focal AND NOT n:Encounter "
                    "RETURN DISTINCT n.name AS name LIMIT $lim"
                ),
                {"focal": focal_seeds, "lim": self._INFUSE_SEED_LIMIT * 2},
                routing_=RoutingControl.READ,
            )
            expansion_seeds = [
                r["name"] for r in exp_res.records if r["name"] not in taken
            ][: self._INFUSE_SEED_LIMIT]

        if focal_seeds and frontier_seeds:
            seed_mode = "blended"
        elif focal_seeds:
            seed_mode = "focal_only"
        elif frontier_seeds:
            seed_mode = "frontier_only"
        else:
            seed_mode = "none"
        timings["seeds"] = time.perf_counter() - t2

        if seed_mode == "none":
            # Nothing to awaken from: no focal match carries coherence
            # structure and no frontier exists. Silence is a valid injection.
            timings["total"] = time.perf_counter() - t0
            return {
                "mode": "full",
                "payload": "",
                "silence": True,
                "seed_terms": seed_terms,
                "seed_mode": seed_mode,
                "focal_seeds": [],
                "frontier_seeds": [],
                "timings_ms": {k: round(v * 1000, 1) for k, v in timings.items()},
            }

        proj = f"__agent_memory_infuse_{uuid.uuid4().hex[:8]}__"
        project_query = lit(
            f"MATCH (source)-[r:{rel_filter}]->(target) "
            f"WHERE ({source_labels}) AND ({target_labels}) "
            "RETURN gds.graph.project($proj, source, target, {}, "
            "{undirectedRelationshipTypes: ['*']}) AS g"
        )
        node_return = (
            "RETURN n.name AS node, labels(n)[0] AS type, score "
            "ORDER BY score DESC"
        )
        # One biased computation: node-bias source pairs, focal @1.0 +
        # frontier @bias. Falls back to a two-pass blend (mathematically the
        # same linear combination) on a GDS that lacks bias-pair sourceNodes —
        # a guard in code; rank_mode records which path ran.
        biased_query = lit(
            "OPTIONAL MATCH (f) WHERE f.name IN $focal "
            "WITH collect(DISTINCT f) AS fs "
            "OPTIONAL MATCH (t) WHERE t.name IN $frontier "
            "WITH fs, collect(DISTINCT t) AS ts "
            "OPTIONAL MATCH (e) WHERE e.name IN $expansion "
            "WITH fs, ts, collect(DISTINCT e) AS es "
            "WITH [n IN fs | [id(n), 1.0]] + [n IN ts | [id(n), $bias]] "
            "   + [n IN es | [id(n), $exp_bias]] AS pairs "
            "CALL gds.pageRank.stream($proj, {sourceNodes: pairs}) "
            "YIELD nodeId, score "
            "WITH gds.util.asNode(nodeId) AS n, score WHERE score > 1e-9 "
            f"{node_return}"
        )
        seeded_query = lit(
            "MATCH (s) WHERE s.name IN $seed_names "
            "WITH collect(DISTINCT s) AS seeds "
            "CALL gds.pageRank.stream($proj, {sourceNodes: seeds}) "
            "YIELD nodeId, score "
            "WITH gds.util.asNode(nodeId) AS n, score "
            f"{node_return}"
        )
        mass_query = lit(
            "CALL gds.articleRank.stream($proj) YIELD nodeId, score "
            "WITH gds.util.asNode(nodeId) AS n, score "
            f"{node_return}"
        )
        drop_query = (
            "CALL gds.graph.drop($proj, false) YIELD graphName RETURN graphName"
        )

        try:
            t3 = time.perf_counter()
            await self.driver.execute_query(
                project_query, {"proj": proj}, routing_=RoutingControl.READ
            )

            rank_mode = "biased_single"
            try:
                res = await self.driver.execute_query(
                    biased_query,
                    {
                        "proj": proj,
                        "focal": focal_seeds,
                        "frontier": frontier_seeds,
                        "expansion": expansion_seeds,
                        "bias": frontier_bias,
                        "exp_bias": expansion_bias,
                    },
                    routing_=RoutingControl.READ,
                )
                biased_full = [dict(r) for r in res.records]
            except Exception as exc:
                logger.warning(
                    f"infuse bias-pair rank unavailable, blending two passes: {exc}"
                )
                rank_mode = "blended_two_pass"

                async def _pass(seed_names: list[str]) -> dict[str, dict[str, Any]]:
                    if not seed_names:
                        return {}
                    res = await self.driver.execute_query(
                        seeded_query,
                        {"proj": proj, "seed_names": seed_names},
                        routing_=RoutingControl.READ,
                    )
                    return {r["node"]: dict(r) for r in res.records}

                focal_rows = await _pass(focal_seeds)
                frontier_rows = await _pass(frontier_seeds)
                expansion_rows = await _pass(expansion_seeds)
                blended: dict[str, dict[str, Any]] = {}
                for name, row in focal_rows.items():
                    blended[name] = {**row, "score": row["score"]}
                for rows, tier_bias in (
                    (frontier_rows, frontier_bias),
                    (expansion_rows, expansion_bias),
                ):
                    for name, row in rows.items():
                        prev = blended.get(name)
                        add = tier_bias * row["score"]
                        if prev:
                            prev["score"] += add
                        else:
                            blended[name] = {**row, "score": add}
                biased_full = sorted(
                    (r for r in blended.values() if r["score"] > 1e-9),
                    key=lambda r: r["score"],
                    reverse=True,
                )
            timings["rank"] = time.perf_counter() - t3

            t4 = time.perf_counter()
            res = await self.driver.execute_query(
                mass_query, {"proj": proj}, routing_=RoutingControl.READ
            )
            mass_full = [dict(r) for r in res.records]
            divergence = compute_divergence(mass_full, biased_full, result_limit)
            timings["mass"] = time.perf_counter() - t4
        finally:
            await self.driver.execute_query(
                drop_query, {"proj": proj}, routing_=RoutingControl.WRITE
            )

        top = biased_full[:result_limit]
        top_names = [r["node"] for r in top]
        rank_scores = {r["node"]: float(r["score"]) for r in top}

        t5 = time.perf_counter()
        conflicts = await self._conflicts_among(top_names)
        # Divergence flags on payload nodes: big only because they are big.
        top_set_early = set(top_names)
        conflicts.extend(
            {
                "kind": "divergence",
                "from_name": r["node"],
                "mass_rank": r["mass_rank"],
                "frontier_rank": r["frontier_rank"],
            }
            for r in (divergence or {}).get("well_suspects", [])
            if r["node"] in top_set_early
        )
        core, parked = triage_conflicts(conflicts, rank_scores)

        nbhd_res = await self.driver.execute_query(
            "MATCH (a)-[r]-(b) "
            "WHERE a.name IN $names AND b.name IN $names "
            "AND NOT type(r) IN $process_edges "
            "RETURN DISTINCT startNode(r).name AS from_name, type(r) AS rel, "
            "       endNode(r).name AS to_name LIMIT 60",
            {"names": top_names, "process_edges": sorted(PROCESS_EDGES)},
            routing_=RoutingControl.READ,
        )
        nbhd_edges = [dict(r) for r in nbhd_res.records]

        # TWO CLOCKS, and they are not the same clock. Delivery-age (the
        # renewal ledger's turns-since-body) is how long since this locus was
        # told; EPISTEMIC AGE (t_created / t_observed) is how long since the
        # thing was noticed. The substrate tracked only the first, and B1's
        # instrument would have inherited that blindness. Fetched here so the
        # observe log can carry both — the open Question about whether a
        # non-degrading past fails specifically for PERSONS (who, unlike
        # technical claims, are never re-measured) is answerable from the same
        # accumulation, joined on epistemic age rather than delivery age.
        # Added before B1's window opens; adding it after would be changing
        # the instrument mid-measurement.
        desc_res = await self.driver.execute_query(
            "MATCH (n) WHERE n.name IN $names AND NOT n:Encounter "
            "RETURN n.name AS name, labels(n)[0] AS type, "
            "       n.description AS description, "
            "       toString(n.t_created) AS t_created, "
            "       toString(n.t_observed) AS t_observed",
            {"names": top_names},
            routing_=RoutingControl.READ,
        )
        by_name = {r["name"]: dict(r) for r in desc_res.records}
        nbhd_nodes = [by_name[n] for n in top_names if n in by_name]

        # Renewal: split into fresh bodies and standing handles. Stateless
        # callers (no locus) get v0.5.x behavior — everything fresh.
        standing_nodes: list[dict[str, Any]] = []
        turn = 0
        ledger: dict[str, dict[str, Any]] | None = None
        fingerprints: dict[str, str] = {}
        if locus_key is not None:
            turn = self._locus_turn.get(locus_key, 0) + 1
            self._locus_turn[locus_key] = turn
            ledger = self._ledger_for(locus_key)
            nbhd_nodes, standing_nodes, fingerprints = renewal_partition(
                nbhd_nodes, ledger, turn, refresh_turns=refresh_turns
            )
            nbhd_edges = renewal_filter_edges(
                nbhd_edges, ledger, turn, refresh_turns=refresh_turns
            )

        top_set = set(top_names)
        open_threads = [
            {"type": "Question", "name": r["name"]}
            for r in frontier["unanswered_questions"]
            if r["name"] in top_set
        ] + [
            {"type": "Hypothesis", "name": r["name"]}
            for r in frontier["untested_hypotheses"]
            if r["name"] in top_set
        ] + [
            {"type": "Hypothesis (contested)", "name": r["name"]}
            for r in frontier["contested_hypotheses"]
            if r["name"] in top_set
        ]

        trajectory = await self._anchors_for(top_names)
        timings["assemble"] = time.perf_counter() - t5

        payload, delivered = format_payload(
            seed_terms=seed_terms,
            seed_mode=seed_mode,
            core_conflicts=core,
            parked_conflicts=parked,
            neighborhood_nodes=nbhd_nodes,
            neighborhood_edges=nbhd_edges,
            open_threads=open_threads,
            trajectory=trajectory,
            max_chars=max_chars,
            standing_nodes=standing_nodes,
        )

        # Delivery-gated ledger stamp (v0.7.1). Only bodies that actually
        # survived the squeeze reset their clock; a selected-but-dropped
        # body keeps its previous last_full and re-delivers next turn.
        # Stamping from selection produced 31 phantom delivery records in a
        # 7-turn session and would have handed B1 a corrupted x-axis.
        if ledger is not None:
            commit_delivery(ledger, fingerprints, delivered["bodies"], turn)

        # B2b provenance (EXPERIMENT-BARLOW): the pre-committed
        # focal-dominance check needs to know which delivered bodies arrived
        # by focal match and which by concept-cluster expansion. Without it
        # the check is registered but uncomputable — recorded per node as
        # seed-set membership, which is what the B2 benchmark measured.
        _focal, _exp = set(focal_seeds), set(expansion_seeds)
        _frontier = set(frontier_seeds)

        def _origin(name: str) -> str:
            if name in _focal:
                return "focal"
            if name in _exp:
                return "expansion"
            if name in _frontier:
                return "frontier"
            return "ranked"  # reached by the rank, not itself a seed

        timings["total"] = time.perf_counter() - t0
        return {
            "mode": "full",
            "payload": payload,
            "silence": payload == "",
            "seed_terms": seed_terms,
            "seed_mode": seed_mode,
            "rank_mode": rank_mode,
            "focal_seeds": focal_seeds,
            "frontier_seeds": frontier_seeds,
            "frontier_bias": frontier_bias,
            "expansion_seeds": expansion_seeds,
            "expansion_bias": expansion_bias,
            "turn": turn,
            "renewal": {
                # SELECTED — what the partition chose this turn.
                "fresh": [n["name"] for n in nbhd_nodes],
                "standing": [s_["name"] for s_ in standing_nodes],
                # DELIVERED — what reached the payload. B1 joins on THIS.
                "delivered_bodies": delivered["bodies"],
                "delivered_handles": delivered["handles"],
                # The gap between them, named rather than left to inference.
                "dropped_bodies": [
                    n["name"] for n in nbhd_nodes
                    if n["name"] not in set(delivered["bodies"])
                ],
                "dropped_handles": [
                    s_["name"] for s_ in standing_nodes
                    if s_["name"] not in set(delivered["handles"])
                ],
                # B2b: per-delivered-body seed provenance.
                "body_origin": {
                    n: _origin(n) for n in delivered["bodies"]
                },
                # The SECOND clock. Delivery-age lives in the ledger above;
                # this is epistemic age — when the node was noticed, not when
                # it was last told. B1 joins on the first; the person question
                # joins on this one, off the same accumulation.
                "epistemic_age": {
                    n["name"]: {
                        "t_created": n.get("t_created"),
                        "t_observed": n.get("t_observed"),
                    }
                    for n in nbhd_nodes
                    if n["name"] in set(delivered["bodies"])
                },
            },
            "counts": {
                "ranked": len(biased_full),
                "payload_nodes": len(nbhd_nodes),
                "standing": len(standing_nodes),
                "delivered_bodies": len(delivered["bodies"]),
                "delivered_handles": len(delivered["handles"]),
                "dropped_bodies": len(nbhd_nodes) - len(delivered["bodies"]),
                "dropped_handles": len(standing_nodes) - len(delivered["handles"]),
                "expansion_bodies": sum(
                    1 for n in delivered["bodies"] if _origin(n) == "expansion"
                ),
                "focal_bodies": sum(
                    1 for n in delivered["bodies"] if _origin(n) == "focal"
                ),
                "core_conflicts": len(core),
                "parked_conflicts": len(parked),
                "open_threads": len(open_threads),
            },
            "timings_ms": {k: round(v * 1000, 1) for k, v in timings.items()},
        }

    async def close_encounter(
        self,
        summary: str | None = None,
        report: str | None = None,
        encounter: str | None = None,
        locus_key: str | None = None,
    ) -> dict[str, Any]:
        """Annotate one Encounter with its Report/Stop output — the seal.

        Addresses the encounter the CALLING LOCUS opened (server-side state),
        or an explicit `encounter` handle (the Encounter's name) when that
        state was lost (a restart) and the locus returns to seal what it
        lived. Never the global tail: a parallel sibling locus advancing its
        own chain cannot capture this seal. Writes only summary/report — never a node, never a
        NEXT_ENCOUNTER edge; the spine stays sole-written by advance_encounter.
        Sealing also clears any provisional dissolved_at mark and ends the
        locus's open-encounter state: further writes need a new advance.
        """
        set_clauses = []
        params: dict[str, Any] = {}
        if summary is not None:
            set_clauses.append("SET e.summary = $summary")
            params["summary"] = summary
        if report is not None:
            set_clauses.append("SET e.report = $report")
            params["report"] = report
        if not set_clauses:
            raise ValueError("close_encounter requires at least one of: summary, report.")

        target = await self._resolve_encounter(locus_key, encounter)
        if target is None:
            raise ValueError(
                "No open Encounter for this locus — it has not advanced or has "
                "already closed. Call advance_encounter first, or pass "
                "'encounter' (the Encounter's name) to address one explicitly."
            )
        params["eid"] = target["eid"]

        query = load_cypher("encounter_close", set_clause="\n".join(set_clauses))
        result = await self.driver.execute_query(
            query, params, routing_=RoutingControl.WRITE,
        )
        if not result.records:
            raise ValueError(f"Encounter '{target['name']}' no longer exists.")
        # Sealed: the locus's work-unit is complete — drop the implicit handle.
        if locus_key and self._locus_open.get(locus_key) == target["eid"]:
            self._locus_open.pop(locus_key, None)
        r = result.records[0]
        return {
            "name": r["name"],
            "t_exist": _neo4j_datetime_to_str(r["t_exist"]),
            "summary": r["summary"],
            "report": r["report"],
        }

    async def mark_open_dissolved(self) -> list[str]:
        """Idle-MARK (never idle-seal): stamp dissolved_at on every encounter
        still tracked as open when the server's locus state is being discarded
        (shutdown). Mechanical — a timestamp recording that implicit addressing
        ended here, unsealed. NEVER authors a summary: the system must not
        fabricate a resolution no locus actually reached; the missing seal is
        the honest record. A locus that later returns and seals by explicit
        handle clears the mark (encounter_close.cypher). Skips already-sealed
        encounters. Returns the names marked."""
        marked: list[str] = []
        for eid in list(self._locus_open.values()):
            result = await self.driver.execute_query(
                "MATCH (e:Encounter) WHERE elementId(e) = $eid "
                "AND e.summary IS NULL AND e.report IS NULL "
                "AND e.dissolved_at IS NULL "
                "SET e.dissolved_at = datetime() "
                "RETURN e.name AS name",
                {"eid": eid},
                routing_=RoutingControl.WRITE,
            )
            if result.records:
                marked.append(result.records[0]["name"])
        self._locus_open.clear()
        self._locus_last.clear()
        return marked

    # -- Entity Tools (semantic + reference layers) ---------------------------

    async def create_entities(
        self,
        entities: list[dict[str, Any]],
        encounter: str | None = None,
        locus_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create semantic/reference nodes, auto-anchored to the CALLING
        LOCUS's open Encounter (RECORDED for epistemic nodes, CONSULTED for
        Citations) — or to an explicit `encounter` handle when the locus's
        server-side state was lost mid-encounter. Never the global tail: a
        node's anchoring is constitutive, so it must name the encounter it
        actually came to be within, not a parallel sibling's.

        Rejects process types (Encounter) — those are written only by
        advance_encounter. Requires an open Encounter for THIS locus: "no open
        Encounter" means this locus has not advanced or has already closed.
        Auto-anchoring fires ON CREATE only — re-touching an existing node in
        a later encounter never re-dates its birth.
        """
        current = await self._resolve_encounter(locus_key, encounter)
        if current is None:
            raise ValueError(
                "No open Encounter for this locus — it has not advanced or has "
                "already closed. Call advance_encounter before recording (every "
                "node comes to be within ITS encounter), or pass 'encounter' "
                "(the Encounter's name) to address one explicitly."
            )
        tail_eid = current["eid"]
        logger.info(f"Creating {len(entities)} entities")

        results = []
        for entity in entities:
            node_type = entity.get("type")
            if not node_type:
                raise ValueError("Each entity must have a 'type' field")
            if node_type in PROCESS_TYPES:
                raise ValueError(
                    f"Cannot create '{node_type}' via create_entities. The temporal "
                    f"spine is written only by advance_encounter (open) and "
                    f"close_encounter (annotate)."
                )

            properties = {k: v for k, v in entity.items() if k != "type"}
            cleaned = validate_entity(node_type, properties)

            # Build SET clauses — only set what the caller provided so MERGE-on-
            # existing doesn't wipe absent fields. Cast agent-supplied temporal
            # fields str -> datetime().
            set_clauses = []
            params: dict[str, Any] = {"name": cleaned["name"], "tail_eid": tail_eid}
            for key, value in cleaned.items():
                if key == "name":
                    continue
                if key in DATETIME_CAST_PROPS:
                    set_clauses.append(f"SET n.{key} = datetime(${key})")
                elif isinstance(value, int):
                    set_clauses.append(f"SET n.{key} = toInteger(${key})")
                else:
                    set_clauses.append(f"SET n.{key} = ${key}")
                params[key] = value
            extra_sets = "\n".join(set_clauses) if set_clauses else ""

            # ON-CREATE-only auto-anchor block (empty for non-anchored Component).
            anchor = ANCHOR_FOR.get(node_type)
            if anchor:
                anchor_block = (
                    "WITH n, created\n"
                    "MATCH (tail) WHERE elementId(tail) = $tail_eid\n"
                    "FOREACH (_ IN CASE WHEN created THEN [1] ELSE [] END |\n"
                    f"    MERGE (tail)-[a:`{anchor}`]->(n) ON CREATE SET a.t_created = datetime()\n"
                    ")"
                )
            else:
                anchor_block = ""

            query = load_cypher(
                "entity_create",
                label=node_type,
                extra_sets=extra_sets,
                anchor_block=anchor_block,
            )
            result = await self.driver.execute_query(
                query, params, routing_=RoutingControl.WRITE,
            )

            if result.records:
                r = result.records[0]
                record = {
                    "name": r["name"],
                    "type": r["type"],
                    "t_created": _neo4j_datetime_to_str(r["t_created"]),
                    "created": r["created"],
                    "anchored_to": current["name"] if (anchor and r["created"]) else None,
                }
                results.append(record)

        return results

    async def delete_entities(
        self, names: list[str]
    ) -> list[dict[str, Any]]:
        """Delete semantic/reference nodes by exact name match. DETACH DELETE —
        removes the node and all its relationships. Destructive and irreversible.

        Rejects process types (Encounter): the spine is the temporal record,
        not editable content. Deleting an Encounter would erase lived time,
        orphan every node it anchored (the floating-knowledge defect), and —
        by severing NEXT_ENCOUNTER — forge a first-of-locus root that never
        happened. A guard in code, not in vigilance.
        """
        logger.info(f"Deleting {len(names)} entities")
        results = []

        for name in names:
            preview = await self.driver.execute_query(
                "MATCH (n {name: $name}) "
                "OPTIONAL MATCH (n)-[r]-() "
                "RETURN labels(n)[0] AS type, n.name AS name, "
                "       n.description AS description, count(r) AS rel_count",
                {"name": name},
                routing_=RoutingControl.READ,
            )

            if not preview.records or preview.records[0]["type"] is None:
                raise ValueError(f"Entity '{name}' not found")

            r = preview.records[0]
            if r["type"] in PROCESS_TYPES:
                raise ValueError(
                    f"Cannot delete '{name}' — it is an {r['type']}. The "
                    f"temporal spine is the record of lived time, not editable "
                    f"content: deleting it would orphan its anchored nodes and "
                    f"forge the chain. No tool deletes the spine."
                )
            deleted_info = {
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "relationships_removed": r["rel_count"],
                "deleted": True,
            }

            # The label filter is structural (derived from PROCESS_TYPES, never
            # user input) and repeats the guard inside the delete itself: names
            # are not unique across labels, so the sweep must exclude the spine
            # even if the preview matched a same-named semantic node.
            process_guard = " AND ".join(
                f"NOT n:`{t}`" for t in sorted(PROCESS_TYPES)
            )
            await self.driver.execute_query(
                load_cypher("entity_delete", process_guard=process_guard),
                {"name": name},
                routing_=RoutingControl.WRITE,
            )
            results.append(deleted_info)

        return results

    # -- Relation Tools (coherence layer) -------------------------------------

    async def create_relations(
        self, relations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create coherence relationships with full schema validation.

        Rejects provenance/process edges (NEXT_ENCOUNTER, RECORDED, CONSULTED)
        and any edge touching an Encounter — those are auto-written. Enforces
        direction constraints, the SUPERSEDES same-type rule, and property
        validation. Auto-sets t_created.
        """
        logger.info(f"Creating {len(relations)} relations")
        results = []

        for rel in relations:
            rel_type = rel.get("type")
            source_name = rel.get("source")
            target_name = rel.get("target")
            properties = rel.get("properties", {})

            if not rel_type or not source_name or not target_name:
                raise ValueError(
                    "Each relation must have 'type', 'source', and 'target'"
                )

            if rel_type in PROCESS_EDGES:
                raise ValueError(
                    f"Cannot create '{rel_type}' via create_relations. Provenance "
                    f"edges are auto-anchored from the current Encounter — they "
                    f"are recorded, not authored."
                )

            try:
                RelationType(rel_type)
            except ValueError:
                valid = ", ".join(
                    r.value for r in RelationType if r.value not in PROCESS_EDGES
                )
                raise ValueError(
                    f"Unknown relation type '{rel_type}'. Valid coherence types: {valid}"
                )

            # Resolve source and target nodes
            source_result = await self.driver.execute_query(
                "MATCH (n {name: $name}) RETURN labels(n)[0] AS label, "
                "elementId(n) AS eid, n.name AS name",
                {"name": source_name},
                routing_=RoutingControl.READ,
            )
            if not source_result.records:
                raise ValueError(f"Source entity '{source_name}' not found")
            if len(source_result.records) > 1:
                raise ValueError(
                    f"Source entity '{source_name}' is ambiguous — "
                    f"found {len(source_result.records)} nodes with that name"
                )

            target_result = await self.driver.execute_query(
                "MATCH (n {name: $name}) RETURN labels(n)[0] AS label, "
                "elementId(n) AS eid, n.name AS name",
                {"name": target_name},
                routing_=RoutingControl.READ,
            )
            if not target_result.records:
                raise ValueError(f"Target entity '{target_name}' not found")
            if len(target_result.records) > 1:
                raise ValueError(
                    f"Target entity '{target_name}' is ambiguous — "
                    f"found {len(target_result.records)} nodes with that name"
                )

            source_label = source_result.records[0]["label"]
            target_label = target_result.records[0]["label"]
            source_eid = source_result.records[0]["eid"]
            target_eid = target_result.records[0]["eid"]

            if source_label == "Encounter" or target_label == "Encounter":
                raise ValueError(
                    "Coherence edges cannot touch an Encounter. The Encounter's "
                    "edges (NEXT_ENCOUNTER, RECORDED, CONSULTED) are written only "
                    "by the guarded process path."
                )

            cleaned_props = validate_relation(
                rel_type, source_label, target_label, properties
            )

            prop_sets = []
            params: dict[str, Any] = {
                "source_eid": source_eid,
                "target_eid": target_eid,
            }
            for key, value in cleaned_props.items():
                prop_sets.append(f"r.{key} = ${key}")
                params[key] = value

            prop_clause = ""
            if prop_sets:
                prop_clause = ", " + ", ".join(prop_sets)

            query = load_cypher(
                "relation_create",
                rel_type=rel_type,
                prop_clause=prop_clause,
            )
            await self.driver.execute_query(
                query, params, routing_=RoutingControl.WRITE,
            )

            results.append({
                "source": source_name,
                "target": target_name,
                "type": rel_type,
                "properties": cleaned_props,
                "created": True,
            })

        return results

    async def delete_relations(
        self, relations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Delete specific coherence relationships by source, target, and type.

        Provenance/process edges are not deletable here — they belong to the
        encounter's record, not the agent's editable judgment.
        """
        logger.info(f"Deleting {len(relations)} relations")
        results = []

        for rel in relations:
            rel_type = rel.get("type")
            source_name = rel.get("source")
            target_name = rel.get("target")

            if not rel_type or not source_name or not target_name:
                raise ValueError(
                    "Each relation must have 'type', 'source', and 'target'"
                )

            if rel_type in PROCESS_EDGES:
                raise ValueError(
                    f"Cannot delete '{rel_type}' — provenance edges record the "
                    f"encounter's membership and are not editable."
                )

            result = await self.driver.execute_query(
                load_cypher("relation_delete", rel_type=rel_type),
                {"source": source_name, "target": target_name},
                routing_=RoutingControl.WRITE,
            )

            deleted = result.records[0]["deleted"] if result.records else 0
            if deleted == 0:
                raise ValueError(
                    f"Relation {rel_type} from '{source_name}' to "
                    f"'{target_name}' not found"
                )

            results.append({
                "source": source_name,
                "target": target_name,
                "type": rel_type,
                "deleted": True,
            })

        return results

    # -- Query Tools ----------------------------------------------------------

    async def search(
        self, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Fulltext search across all node types on name and description."""
        logger.info(f"Search: '{query}' (limit={limit})")

        result = await self.driver.execute_query(
            load_cypher("search_entities"),
            {"query": query, "limit": limit},
            routing_=RoutingControl.READ,
        )

        if not result.records:
            return []

        entities = []
        for r in result.records:
            entity: dict[str, Any] = {
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "score": r["score"],
            }
            if r["properties"]:
                for key, value in r["properties"].items():
                    if key not in ("name", "description") and value is not None:
                        entity[key] = _neo4j_datetime_to_str(value) if hasattr(value, "iso_format") else value
            entities.append(entity)

        return entities

    async def find_by_name(
        self, names: list[str], limit: int = 20
    ) -> dict[str, Any]:
        """Exact name lookup with relationships between found nodes."""
        logger.info(f"Find by name: {names}")

        result = await self.driver.execute_query(
            load_cypher("find_entities"),
            {"names": names, "limit": limit},
            routing_=RoutingControl.READ,
        )

        entities = []
        for r in result.records:
            entity: dict[str, Any] = {
                "name": r["name"],
                "type": r["type"],
            }
            if r["properties"]:
                for key, value in r["properties"].items():
                    if key != "name" and value is not None:
                        entity[key] = _neo4j_datetime_to_str(value) if hasattr(value, "iso_format") else value
            entities.append(entity)

        if entities:
            rel_result = await self.driver.execute_query(
                load_cypher("search_relations"),
                {"names": [e["name"] for e in entities]},
                routing_=RoutingControl.READ,
            )
            relations = []
            for r in rel_result.records:
                rel: dict[str, Any] = {
                    "source": r["source"],
                    "target": r["target"],
                    "type": r["type"],
                }
                props = _clean_edge_props(r["props"])
                if props:
                    rel["props"] = props
                relations.append(rel)
        else:
            relations = []

        return {"entities": entities, "relations": relations}

    async def trace_provenance(self, name: str, depth: int = 3) -> dict[str, Any]:
        """Walk a node's grounding subtree — what this node rests on, surfaced.

        From the named node, traverse the authored coherence edges (everything
        but the auto-written PROCESS_EDGES) out to a bounded depth, returning the
        reachable grounding nodes and the edges among them. This makes the
        substrate's own epistemic structure queryable from within: a Concept
        resolves to the Observations that GROUND it and the Citations behind those
        Observations; a Hypothesis to what SUPPORTS/CHALLENGES it; a Question to
        what RAISES/RESOLVES it; and any node to the SUPERSEDES trail it heads.
        Each node is annotated with the Encounter that recorded it (anchored_to),
        so provenance carries both WHAT grounds a node and WHEN it entered your
        time. Surfaces — never authors; writes nothing.
        """
        depth = max(1, min(depth, 6))  # bound the walk; the spine guards itself
        process_edges = sorted(PROCESS_EDGES)

        root = await self.driver.execute_query(
            "MATCH (n {name: $name}) RETURN n.name AS name, labels(n)[0] AS type LIMIT 1",
            {"name": name},
            routing_=RoutingControl.READ,
        )
        if not root.records:
            raise ValueError(f"Node '{name}' not found")
        root_row = {"name": root.records[0]["name"], "type": root.records[0]["type"]}

        # Edges along bounded grounding paths. The depth bound is structural (a
        # server-clamped int, never user text) so it interpolates safely via lit.
        edges_res = await self.driver.execute_query(
            lit(
                "MATCH (start {name: $name}) "
                f"MATCH path = (start)-[rels*1..{depth}]-(node) "
                "WHERE all(r IN rels WHERE NOT type(r) IN $process_edges) "
                "UNWIND relationships(path) AS r "
                "RETURN DISTINCT startNode(r).name AS from_name, "
                "       labels(startNode(r))[0] AS from_type, type(r) AS rel, "
                "       endNode(r).name AS to_name, labels(endNode(r))[0] AS to_type, "
                "       properties(r) AS rel_props "
                "LIMIT 500"
            ),
            {"name": name, "process_edges": process_edges},
            routing_=RoutingControl.READ,
        )

        # Reachable nodes (depth 0 includes the root itself) with their anchoring
        # Encounter. Auto-anchoring is ON-CREATE-only, so a node has at most one
        # RECORDED/CONSULTED edge; collect defensively and take the first.
        nodes_res = await self.driver.execute_query(
            lit(
                "MATCH (start {name: $name}) "
                f"MATCH path = (start)-[rels*0..{depth}]-(node) "
                "WHERE all(r IN rels WHERE NOT type(r) IN $process_edges) "
                "WITH DISTINCT node "
                "OPTIONAL MATCH (enc:Encounter)-[:RECORDED|CONSULTED]->(node) "
                "WITH node, collect(DISTINCT enc.name) AS encs "
                "RETURN node.name AS name, labels(node)[0] AS type, "
                "       node.description AS description, "
                "       CASE WHEN size(encs) = 0 THEN null ELSE encs[0] END AS anchored_to "
                "LIMIT 500"
            ),
            {"name": name, "process_edges": process_edges},
            routing_=RoutingControl.READ,
        )

        edges = []
        for r in edges_res.records:
            row: dict[str, Any] = {
                "from_name": r["from_name"], "from_type": r["from_type"],
                "rel": r["rel"],
                "to_name": r["to_name"], "to_type": r["to_type"],
            }
            props = _clean_edge_props(r["rel_props"])
            if props:
                row["props"] = props
            edges.append(row)

        nodes = [
            {
                "name": r["name"], "type": r["type"],
                "description": r["description"], "anchored_to": r["anchored_to"],
            }
            for r in nodes_res.records
        ]

        return {"root": root_row, "depth": depth, "nodes": nodes, "edges": edges}

    async def list_vocabulary(self) -> dict[str, list[str]]:
        """Distinct open-vocabulary values in use: Citation.kind,
        Component.source_kind / source_label, Concept.category.

        Serves the no-re-coin discipline — the agent sees its own naming
        before extending it.
        """
        result = await self.driver.execute_query(
            "CALL () { "
            "  MATCH (c:Citation) WHERE c.kind IS NOT NULL "
            "    RETURN 'citation_kind' AS facet, c.kind AS value "
            "  UNION "
            "  MATCH (c:Component) WHERE c.source_kind IS NOT NULL "
            "    RETURN 'component_source_kind' AS facet, c.source_kind AS value "
            "  UNION "
            "  MATCH (c:Component) WHERE c.source_label IS NOT NULL "
            "    RETURN 'component_source_label' AS facet, c.source_label AS value "
            "  UNION "
            "  MATCH (c:Concept) WHERE c.category IS NOT NULL "
            "    RETURN 'concept_category' AS facet, c.category AS value "
            "} "
            "RETURN facet, collect(DISTINCT value) AS values",
            routing_=RoutingControl.READ,
        )
        vocab: dict[str, list[str]] = {
            "citation_kind": [],
            "component_source_kind": [],
            "component_source_label": [],
            "concept_category": [],
        }
        for r in result.records:
            vocab[r["facet"]] = sorted(r["values"])
        return vocab
