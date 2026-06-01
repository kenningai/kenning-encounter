import logging
from enum import Enum
from typing import Any, LiteralString

from neo4j import AsyncDriver, RoutingControl

from .utils import load_cypher, lit

logger = logging.getLogger("mcp_agent_memory")
logger.setLevel(logging.INFO)


def _neo4j_datetime_to_str(val: Any) -> str | None:
    """Convert a Neo4j DateTime/Date to ISO string, or return None."""
    if val is None:
        return None
    return val.iso_format()


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
    RECORDED = "RECORDED"
    CONSULTED = "CONSULTED"
    # Coherence — the agent's judgment, authored every one.
    ABOUT = "ABOUT"
    OBSERVED_AT = "OBSERVED_AT"
    RAISES = "RAISES"
    RESOLVES = "RESOLVES"
    SUPPORTS = "SUPPORTS"
    CHALLENGES = "CHALLENGES"
    INFORMS = "INFORMS"
    COMPOSES = "COMPOSES"
    DECOMPOSES = "DECOMPOSES"
    SUPERSEDES = "SUPERSEDES"


# Process-layer types/edges are NOT writable via generic CRUD. The temporal
# spine is created solely by advance_encounter; RECORDED/CONSULTED are auto-
# anchored from the current Encounter on node creation.
PROCESS_TYPES: set[str] = {"Encounter"}
PROCESS_EDGES: set[str] = {"NEXT_ENCOUNTER", "RECORDED", "CONSULTED"}

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
        "properties": {},
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

    def __init__(self, driver: AsyncDriver):
        self.driver = driver

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

    async def _current_encounter(self) -> dict[str, Any] | None:
        """Return the current chain tail (the Encounter with no outgoing
        NEXT_ENCOUNTER), or None if no encounter has ever opened."""
        result = await self.driver.execute_query(
            "MATCH (e:Encounter) WHERE NOT (e)-[:NEXT_ENCOUNTER]->(:Encounter) "
            "RETURN e.name AS name, elementId(e) AS eid, e.t_exist AS t_exist "
            "ORDER BY e.t_exist DESC LIMIT 1",
            routing_=RoutingControl.READ,
        )
        if not result.records:
            return None
        r = result.records[0]
        return {"name": r["name"], "eid": r["eid"], "t_exist": r["t_exist"]}

    async def advance_encounter(
        self, name: str, recent: int = 5, limit: int = 20
    ) -> dict[str, Any]:
        """Open a new Encounter and return the re-entry payload.

        The sole writer of the temporal spine. Takes no predecessor: it finds
        the chain tail server-side and appends in one transaction. Opening an
        encounter IS orienting — Exist and the orientation read are one act.
        """
        result = await self.driver.execute_query(
            load_cypher("encounter_advance"),
            {"name": name},
            routing_=RoutingControl.WRITE,
        )
        r = result.records[0]
        encounter = {
            "name": r["name"],
            "t_exist": _neo4j_datetime_to_str(r["t_exist"]),
            "t_created": _neo4j_datetime_to_str(r["t_created"]),
            "predecessor": r["predecessor"],
            "is_first": r["is_first"],
        }
        reentry = await self._reentry_payload(recent=recent, limit=limit)
        return {"encounter": encounter, "reentry": reentry}

    async def _reentry_payload(self, recent: int, limit: int) -> dict[str, Any]:
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

        Also carries a dissolution marker: if the encounter just before the new
        tail was never sealed (no summary or report), the prior existence was cut
        before it could resolve — surfaced so the agent can treat that encounter's
        coherence as possibly incomplete. Null in the normal sealed case."""

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
            "       endNode(r).name AS to_name, labels(endNode(r))[0] AS to_type "
            "LIMIT $edge_limit",
            {
                "recent": recent,
                "process_edges": sorted(PROCESS_EDGES),
                "edge_limit": limit * 5,
            },
            routing_=RoutingControl.READ,
        )

        # Dissolution marker: an Encounter ends cleanly by being sealed
        # (close_encounter writes a summary/report) — even a measured-null run
        # gets a close. A predecessor with NEITHER summary nor report was never
        # sealed: the interior cycle was cut before it could resolve — it
        # dissolved mid-thought (context exhausted). This
        # finds the encounter just before the current tail and surfaces it iff
        # unsealed. The chain head (only one encounter) never triggers it.
        dissolution = await self.driver.execute_query(
            "MATCH (e:Encounter) WITH e ORDER BY e.t_exist DESC LIMIT 2 "
            "WITH collect(e) AS es "
            "WHERE size(es) = 2 AND es[1].summary IS NULL AND es[1].report IS NULL "
            "RETURN es[1].name AS name, es[1].t_exist AS t_exist",
            routing_=RoutingControl.READ,
        )

        def _rows(result, fields):
            out = []
            for r in result.records:
                row = {}
                for f in fields:
                    v = r[f]
                    row[f] = _neo4j_datetime_to_str(v) if hasattr(v, "iso_format") else v
                out.append(row)
            return out

        return {
            "recent_encounters": _rows(recent_encounters, ["name", "t_exist", "summary", "report"]),
            "open_questions": _rows(open_questions, ["name", "description", "priority", "t_raised"]),
            "live_hypotheses": _rows(live_hypotheses, ["name", "description", "confidence", "status", "t_proposed"]),
            "recently_touched_concepts": _rows(recent_concepts, ["name", "description", "status", "category"]),
            "coherence_edges": _rows(coherence_edges, ["from_name", "from_type", "rel", "to_name", "to_type"]),
            "dissolution": (lambda d: d[0] if d else None)(_rows(dissolution, ["name", "t_exist"])),
        }

    # Reserved name for the orient projection. A single fixed name (not a
    # per-call one) lets a crashed prior orient be cleaned up defensively.
    _ORIENT_PROJECTION = "__agent_memory_orient__"

    async def orient(self, result_limit: int = 20) -> dict[str, Any]:
        """The Exist read: one structural self-portrait over the coherence
        subgraph, run once per waking — not per Encounter.

        Bundles the whole GDS spread — projection lifecycle + PageRank,
        Betweenness, Louvain, WCC — into a single call. The agent never
        manages a projection by hand: cleanup is a finally, not a discipline,
        and a crashed prior run is dropped defensively before this one starts.
        The projection is coherence-only by construction — authored edges
        (RelationType minus PROCESS_EDGES) over the non-process nodes (every
        type but Encounter), undirected — so the spine/provenance edges that
        turn Encounters into artificial centrality hubs can never enter the
        read. Surfaces — never authors; writes no node, no edge.

        PageRank = what you have come to treat as central. Betweenness = what
        bridges your lines of inquiry. Louvain = your chapters. WCC = what you
        noticed and never connected. Within the Exist, reach for the single
        gds_* tools when a specific question arises; this is the opening survey.
        """
        proj = self._ORIENT_PROJECTION
        coherence_rels = [r.value for r in RelationType if r.value not in PROCESS_EDGES]
        semantic_nodes = [nt.value for nt in NodeType if nt.value not in PROCESS_TYPES]

        rel_filter = "|".join(f"`{rt}`" for rt in coherence_rels)
        source_labels = " OR ".join(f"source:`{nt}`" for nt in semantic_nodes)
        target_labels = " OR ".join(f"target:`{nt}`" for nt in semantic_nodes)
        project_query = lit(
            f"MATCH (source)-[r:{rel_filter}]->(target) "
            f"WHERE ({source_labels}) AND ({target_labels}) "
            "RETURN gds.graph.project($proj, source, target, {}, "
            "{undirectedRelationshipTypes: ['*']}) AS g"
        )

        drop_query = (
            "CALL gds.graph.drop($proj, false) YIELD graphName RETURN graphName"
        )
        algos = {
            "pagerank": "CALL gds.pageRank.stream($proj) YIELD nodeId, score "
            "RETURN gds.util.asNode(nodeId).name AS node, "
            "labels(gds.util.asNode(nodeId))[0] AS type, score "
            "ORDER BY score DESC LIMIT $lim",
            "betweenness": "CALL gds.betweenness.stream($proj) YIELD nodeId, score "
            "RETURN gds.util.asNode(nodeId).name AS node, "
            "labels(gds.util.asNode(nodeId))[0] AS type, score "
            "ORDER BY score DESC LIMIT $lim",
            "louvain": "CALL gds.louvain.stream($proj) YIELD nodeId, communityId "
            "RETURN gds.util.asNode(nodeId).name AS node, "
            "labels(gds.util.asNode(nodeId))[0] AS type, communityId "
            "ORDER BY communityId, node LIMIT $lim",
            "wcc": "CALL gds.wcc.stream($proj) YIELD nodeId, componentId "
            "RETURN gds.util.asNode(nodeId).name AS node, "
            "labels(gds.util.asNode(nodeId))[0] AS type, componentId "
            "ORDER BY componentId, node LIMIT $lim",
        }

        # Defensive: clear any projection a crashed prior orient left behind.
        await self.driver.execute_query(
            drop_query, {"proj": proj}, routing_=RoutingControl.WRITE
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
            for key, q in algos.items():
                res = await self.driver.execute_query(
                    lit(q),
                    {"proj": proj, "lim": result_limit},
                    routing_=RoutingControl.READ,
                )
                out[key] = [dict(r) for r in res.records]
            return out
        finally:
            await self.driver.execute_query(
                drop_query, {"proj": proj}, routing_=RoutingControl.WRITE
            )

    async def close_encounter(
        self, summary: str | None = None, report: str | None = None
    ) -> dict[str, Any]:
        """Annotate the current (tail) Encounter with its Report/Stop output.

        Writes only summary/report on the existing tail — never a node, never
        a NEXT_ENCOUNTER edge. The spine stays sole-written by advance_encounter.
        """
        set_clauses = []
        params: dict[str, Any] = {}
        if summary is not None:
            set_clauses.append("SET tail.summary = $summary")
            params["summary"] = summary
        if report is not None:
            set_clauses.append("SET tail.report = $report")
            params["report"] = report
        if not set_clauses:
            raise ValueError("close_encounter requires at least one of: summary, report.")

        query = load_cypher("encounter_close", set_clause="\n".join(set_clauses))
        result = await self.driver.execute_query(
            query, params, routing_=RoutingControl.WRITE,
        )
        if not result.records:
            raise ValueError("No open Encounter to close. Call advance_encounter first.")
        r = result.records[0]
        return {
            "name": r["name"],
            "t_exist": _neo4j_datetime_to_str(r["t_exist"]),
            "summary": r["summary"],
            "report": r["report"],
        }

    # -- Entity Tools (semantic + reference layers) ---------------------------

    async def create_entities(
        self, entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create semantic/reference nodes, auto-anchored to the current
        Encounter (RECORDED for epistemic nodes, CONSULTED for Citations).

        Rejects process types (Encounter) — those are written only by
        advance_encounter. Requires an open Encounter: a node comes to be
        WITHIN an encounter, and the graph records that constitutively.
        Auto-anchoring fires ON CREATE only — re-touching an existing node in
        a later encounter never re-dates its birth.
        """
        current = await self._current_encounter()
        if current is None:
            raise ValueError(
                "No open Encounter. Call advance_encounter before recording — "
                "every node comes to be within an encounter."
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
        """Delete nodes by exact name match. DETACH DELETE — removes the node
        and all its relationships. Destructive and irreversible."""
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
            deleted_info = {
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "relationships_removed": r["rel_count"],
                "deleted": True,
            }

            await self.driver.execute_query(
                load_cypher("entity_delete"),
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
            relations = [
                {
                    "source": r["source"],
                    "target": r["target"],
                    "type": r["type"],
                }
                for r in rel_result.records
            ]
        else:
            relations = []

        return {"entities": entities, "relations": relations}

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
