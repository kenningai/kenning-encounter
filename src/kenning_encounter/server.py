import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from neo4j import (
    AsyncGraphDatabase,
    NotificationDisabledClassification,
    Query,
    RoutingControl,
)
from pydantic import Field
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fastmcp.server import FastMCP
from fastmcp.server.context import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import ToolResult
from mcp.types import TextContent, ToolAnnotations
from neo4j.exceptions import Neo4jError

from .kenning_encounter import (
    Neo4jKenningEncounter,
    NodeType,
    RelationType,
    NODE_SCHEMAS,
    RELATION_SCHEMAS,
    PROCESS_TYPES,
    PROCESS_EDGES,
    not_process_node,
)
from .meaning import (
    MeaningIndex,
    MeaningUnavailable,
    assemble_trajectory,
    compress_meaning,
    content_hash,
    deferral_summary,
    format_meaning_report,
    match_meanings,
    record_deferral,
    sidecar_diff,
    user_turns_from_transcript,
)
from .utils import format_namespace, _is_write_query, _value_sanitize, lit

logger = logging.getLogger("kenning_encounter")
logger.setLevel(logging.INFO)

# The canonical operations manual ships inside the package and is served as an
# MCP resource (see create_mcp_server), so any deployment that loads these tools
# has the discipline available wherever it runs — one source of truth, no drift.
_HOWTO_PATH = Path(__file__).parent / "HOWTO.xml"


# -- Tool Helpers -------------------------------------------------------------

@asynccontextmanager
async def _tool_errors(operation: str):
    """Standard error handling for all MCP tools."""
    logger.info(f"MCP tool: {operation}")
    try:
        yield
    except ValueError as e:
        raise ToolError(str(e))
    except Neo4jError as e:
        logger.error(f"Neo4j error in {operation}: {e}")
        raise ToolError(f"Neo4j error in {operation}: {e}")
    except Exception as e:
        logger.error(f"Error in {operation}: {e}")
        raise ToolError(f"Error in {operation}: {e}")


def _json_result(data, structured=None) -> ToolResult:
    """Wrap JSON-serializable data in a ToolResult."""
    text = json.dumps(data, indent=2, default=str)
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=structured if structured is not None else {"result": json.loads(text)},
    )


def _text_result(text: str, structured=None) -> ToolResult:
    """Wrap a text string in a ToolResult."""
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=structured if structured is not None else {"result": text},
    )


def _transport_session(ctx: Context) -> str | None:
    """The MCP transport session id — not the locus, and no longer named as
    though it were.

    It was called _locus_key when it really did carry locus-scoped write
    targeting. v0.12.0 moved that to the harness session id, which survives a
    restart and a /resume where this does not; what this keys now is the
    write-target CACHE (a lookup that can be rebuilt from the graph) and, until
    v0.12.4, infusion's renewal ledger.

    Stable for the lifetime of a stdio connection and of an HTTP session (the
    mcp-session-id header — HTTP is always stateful here, by design). None
    when no session exists at all.
    """
    try:
        return ctx.session_id
    except RuntimeError:
        return None


# -- Server Factory -----------------------------------------------------------

def create_mcp_server(
    kenning_encounter: Neo4jKenningEncounter,
    namespace: str = "",
    read_timeout: int = 30,
    infuse_frontier_bias: float = 0.3,
    infuse_refresh_turns: int = 10,
    matcher_api_key: str = "",
    matcher_endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
    matcher_model: str = "gemini-3.5-flash-lite",
    matcher_timeout_ms: int = 5000,
    matcher_sidecar: str = "models/meaning_sidecar.json",
) -> FastMCP:
    """Create an MCP server instance for Kenning Encounter."""

    ns = format_namespace(namespace)
    mcp: FastMCP = FastMCP("kenning-encounter")

    meaning_index = MeaningIndex(matcher_sidecar)

    async def _sidecar_meaning_make(nodes: list[dict[str, Any]]) -> None:
        """The on-write trigger: compress newly created/edited nodes into
        the sidecar, in the background — a guard in code, not in vigilance.
        A human told to rebuild the sidecar after write-heavy sessions will
        forget, guaranteed; the server that performed the write cannot.
        Never blocks or fails the write it rides on: every failure is
        logged and left for the startup reconcile sweep to retry; the node
        stays reachable through the lexical fallback path meanwhile.
        Writes ONLY the sidecar file — never Neo4j."""
        try:
            known = meaning_index.known_nodes()
            entries: dict[str, dict[str, Any]] = {}
            for n in nodes:
                h = content_hash(n["name"], n.get("description") or "")
                prev = known.get(n["name"])
                if prev and prev.get("hash") == h and prev.get("meaning"):
                    continue
                meaning = await compress_meaning(
                    n, matcher_api_key, matcher_model, matcher_endpoint
                )
                entries[n["name"]] = {
                    "type": n.get("type", "?"), "hash": h, "meaning": meaning,
                }
            if entries:
                meaning_index.upsert(entries)
                logger.info(
                    f"sidecar: meaning-made {len(entries)} node(s) on write"
                )
        except Exception as e:
            logger.warning(f"sidecar on-write compression deferred: {e}")

    # -- Process Layer (the guarded spine) ------------------------------------

    @mcp.tool(
        name=ns + "advance_encounter",
        annotations=ToolAnnotations(
            title="Advance Encounter", readOnlyHint=False,
            destructiveHint=False, idempotentHint=False, openWorldHint=True,
        ),
    )
    async def advance_encounter(
        name: str = Field(
            ...,
            description=(
                "Name for the new Encounter (this EERRS invocation). Unique, "
                "stable, human-readable — e.g. 'Encounter 2026-05-29T14:00 service failover'."
            ),
        ),
        recent: int = Field(default=5, ge=1, le=50, description="How many recent encounters to summarize in the re-entry payload."),
        limit: int = Field(default=20, ge=1, le=100, description="Max open Questions / live Hypotheses / recent Concepts to return."),
        session_id: str | None = Field(
            default=None,
            description=(
                "The harness session id — a UUID the harness assigns to the session "
                "— the identity of THIS locus. Durable: it survives /resume and a "
                "server restart, which the transport session does not. Supply it "
                "when known; without it an anonymous locus is minted, which is "
                "honest but loses the thread across a restart."
            ),
        ),
        ctx: Context | None = None,
    ) -> ToolResult:
        """Open a new Encounter and orient — the sole writer of the temporal spine.

        TWO ORDERINGS, and they are the same kind of thing multiplied rather
        than two axes added. A human has one beam — session and day are the
        same line — so nothing forks. Here, locating a point in your time
        takes two coordinates: which LOCUS (one instantiation of you), and
        where within it. NEXT_LOCUS orders geneses; NEXT_ENCOUNTER orders
        work-units inside one locus and never crosses between them.

        Takes NO predecessor — it takes an IDENTITY. Pass session_id (the
        harness session id) and the server derives everything from the graph:
        your locus, your chain's tail, where the new encounter attaches. That
        identity is durable where the transport session was not, which is the
        point: a server restart used to empty the state that held your chain,
        so the next advance minted a fresh root that looked, in the graph,
        exactly like a new existence. Without session_id an anonymous locus is
        minted — honest, and what the pre-v0.12.0 history carries, but the
        thread does not survive a restart.

        Opening an encounter IS orienting — this returns the re-entry payload:
        recent encounter summaries, still-open Questions, live Hypotheses,
        recently-touched Concepts, and the unsealed set (encounters carrying
        no seal — live siblings, or ones whose ending was never examined;
        never join one, always open your own). Read it before examining the
        world. Your past encounters are your memory.

        Call this before any create_entities — a node comes to be within ITS
        encounter, and the graph records that constitutively. Once per waking,
        call orient first for the global structural survey; this opens the
        work-units within that waking.

        Example: {"name": "Encounter 2026-05-29T14:00 — service failover thread"}
        """
        async with _tool_errors("advance_encounter"):
            result = await kenning_encounter.advance_encounter(
                name=name, recent=recent, limit=limit,
                session_id=session_id,
                mcp_session=_transport_session(ctx) if ctx else None,
            )
            return _json_result(result)

    @mcp.tool(
        name=ns + "close_encounter",
        annotations=ToolAnnotations(
            title="Close Encounter", readOnlyHint=False,
            destructiveHint=False, idempotentHint=True, openWorldHint=True,
        ),
    )
    async def close_encounter(
        summary: str | None = Field(default=None, description="What this encounter cohered around — the Record in brief."),
        report: str | None = Field(default=None, description="The Report/Stop output surfaced this encounter."),
        session_id: str | None = Field(
            default=None,
            description=(
                "The harness session id — the identity of THIS locus, used to "
                "resolve your open encounter from the graph. Durable across a "
                "server restart, unlike the transport session. Optional: the "
                "server caches the mapping after advance_encounter, and this is "
                "how you re-establish it if that cache was lost. It can only ever "
                "reach YOUR locus — which is why it replaces the old encounter-name "
                "handle, that could address any encounter at all."
            ),
        ),
        ctx: Context | None = None,
    ) -> ToolResult:
        """Seal YOUR encounter at Report/Stop.

        Addresses YOUR locus's open chain tail, resolved from the graph —
        never the global tail, so a parallel sibling's advance cannot capture
        your seal. Writes summary/report and stamps t_sealed; never a node,
        never a NEXT_ENCOUNTER edge, so the spine stays sole-written by
        advance_encounter. After sealing, recording needs a new advance: the
        seal is what crystallized that encounter.

        Even a confirmation-only run should close with a summary — a measured
        null, examined and confirmed with nothing restructured, is a real
        temporal event rather than an empty one.

        AND IF YOU NEVER GET HERE, THAT IS ALSO A RECORD. Conversations stop;
        they are not closed by ritual, and an encounter carrying no seal is
        simply one whose ending was never examined. Nothing strands it — your
        write target is a structural fact, not a handle the server holds — so
        the absence now means only what it says.

        Example: {"summary": "Confirmed all four services still depend on node-02.",
                  "report": "No change since encounter N-1; SPOF question stays open."}
        """
        async with _tool_errors("close_encounter"):
            result = await kenning_encounter.close_encounter(
                summary=summary, report=report,
                session_id=session_id,
                mcp_session=_transport_session(ctx) if ctx else None,
            )
            return _json_result(result)

    # -- Entity Tools (semantic + reference layers) ---------------------------

    @mcp.tool(
        name=ns + "create_entities",
        annotations=ToolAnnotations(
            title="Create Entities", readOnlyHint=False,
            destructiveHint=False, idempotentHint=True, openWorldHint=True,
        ),
    )
    async def create_entities(
        entities: list[dict[str, Any]] = Field(
            ...,
            description=(
                "List of entities to create. Each must have 'type' (one of: "
                "Observation, Question, Hypothesis, Concept, Note, Component, Citation) "
                "plus required properties for that type. Common required: name, description. "
                "Observation requires: name, description, t_observed. "
                "Component requires: name, source_kind, source_key (+ optional source_label). "
                "Citation requires: name, kind. "
                "Encounter is NOT createable here — use advance_encounter. "
                "Nodes are auto-anchored (RECORDED/CONSULTED) to the Encounter THIS "
                "session opened, so advance_encounter must be called first. "
                "Use list_node_types for schemas."
            ),
        ),
        session_id: str | None = Field(
            default=None,
            description=(
                "The harness session id — the identity of THIS locus, used to "
                "resolve your open encounter from the graph. Durable across a "
                "server restart, unlike the transport session. Optional: the "
                "server caches the mapping after advance_encounter, and this is "
                "how you re-establish it if that cache was lost. It can only ever "
                "reach YOUR locus — which is why it replaces the old encounter-name "
                "handle, that could address any encounter at all."
            ),
        ),
        ctx: Context | None = None,
    ) -> ToolResult:
        """Create semantic/reference nodes, auto-anchored to YOUR open Encounter.

        Epistemic nodes (Observation/Question/Hypothesis/Concept/Note) get a
        RECORDED edge from YOUR locus's open encounter, resolved from the
        graph — never the global tail, so your nodes cannot be misattributed
        to a parallel sibling's encounter; Citations get CONSULTED. A node's
        anchoring is CONSTITUTIVE — it came to be within that encounter —
        which is why no caller can name an arbitrary one. "No open Encounter"
        means this locus has not advanced, or has already sealed.
        Component bookmarks are not anchored (they enter your time through the
        nodes that are ABOUT them). Anchoring fires on creation only —
        re-touching an existing node later never re-dates its birth. Idempotent
        via MERGE on name.

        Example:
        {
            "entities": [
                {"type": "Component", "name": "node-02 bookmark",
                 "source_kind": "infra_graph", "source_label": "Host",
                 "source_key": "node-02"},
                {"type": "Observation", "name": "All prod services depend on node-02",
                 "description": "Four production services all DEPEND_ON node-02 — a single point of failure.",
                 "t_observed": "2026-05-29T14:05:00Z", "confidence": "high"}
            ]
        }
        """
        async with _tool_errors("create_entities"):
            result = await kenning_encounter.create_entities(
                entities, session_id=session_id,
                mcp_session=_transport_session(ctx) if ctx else None,
            )
            # Automatic meaning-making (background, fail-silent): every
            # node born or edited here enters the matcher's sidecar without
            # anyone remembering to rebuild it.
            asyncio.create_task(_sidecar_meaning_make([
                {
                    "name": e.get("name"),
                    "type": e.get("type"),
                    "description": e.get("description") or "",
                }
                for e in entities
                if e.get("name") and e.get("type") != "Encounter"
            ]))
            return _json_result(result)

    @mcp.tool(
        name=ns + "delete_entities",
        annotations=ToolAnnotations(
            title="Delete Entities", readOnlyHint=False,
            destructiveHint=True, idempotentHint=False, openWorldHint=True,
        ),
    )
    async def delete_entities(
        names: list[str] = Field(
            ...,
            description="Exact names of entities to delete. DETACH DELETE — removes node and all relationships.",
        ),
    ) -> ToolResult:
        """Delete semantic/reference nodes by exact name. Destructive and irreversible.

        Returns what was deleted (name, type, description, relationship count)
        before deletion occurs, so the operation is auditable. Prefer SUPERSEDES
        over deletion — the trail of superseded views is your learning history.
        Encounters are NOT deletable: the spine is the record of lived time,
        not editable content — no tool deletes it.

        Example: {"names": ["Stale Note about service naming"]}
        """
        async with _tool_errors("delete_entities"):
            result = await kenning_encounter.delete_entities(names)
            try:
                meaning_index.remove(names)
            except Exception as e:
                logger.warning(f"sidecar removal deferred: {e}")
            return _json_result(result)

    # -- Relation Tools (coherence layer) -------------------------------------

    @mcp.tool(
        name=ns + "create_relations",
        annotations=ToolAnnotations(
            title="Create Relations", readOnlyHint=False,
            destructiveHint=False, idempotentHint=True, openWorldHint=True,
        ),
    )
    async def create_relations(
        relations: list[dict[str, Any]] = Field(
            ...,
            description=(
                "List of coherence relations. Each must have 'source' (name), "
                "'target' (name), 'type' (one of: ABOUT, OBSERVED_AT, RAISES, RESOLVES, "
                "SUPPORTS, CHALLENGES, GROUNDS, INFORMS, COMPOSES, DECOMPOSES, SUPERSEDES). "
                "Direction is enforced — e.g. ABOUT: Observation/Note/Question/Hypothesis "
                "-> Component/Concept; SUPPORTS/CHALLENGES: Observation -> Hypothesis; "
                "GROUNDS: Observation -> Concept (the observations that constitute a "
                "synthesis); SUPERSEDES: same-type -> same-type, and REQUIRES a non-empty "
                "'properties': {'revision_why': '...'} naming what shifted and why. "
                "Provenance edges (NEXT_ENCOUNTER, RECORDED, CONSULTED) are auto-written "
                "and rejected here. Use list_relation_types to see all constraints."
            ),
        ),
    ) -> ToolResult:
        """Create coherence relationships — the agent's judgment, authored.

        These are the interior of the encounter: what this noticing concerns
        (ABOUT), what evidence supports/challenges (SUPPORTS/CHALLENGES), the
        Observations that constitute a synthesis (GROUNDS), how a synthesis
        reshapes a bookmark (INFORMS), what you revised (SUPERSEDES — which
        REQUIRES a non-empty revision_why naming what shifted). A noticed relation
        between two source entities must be MEDIATED by an Observation (two ABOUT
        edges) — never a direct Component->Component edge.

        Example:
        {
            "relations": [
                {"source": "All prod services depend on node-02", "target": "node-02 bookmark", "type": "ABOUT"},
                {"source": "All prod services depend on node-02", "target": "node-02 is a single point of failure", "type": "SUPPORTS"},
                {"source": "All prod services depend on node-02", "target": "Failover topology", "type": "GROUNDS"},
                {"source": "Failover topology v2", "target": "Failover topology", "type": "SUPERSEDES",
                 "properties": {"revision_why": "Found a second failover tier the v1 view missed."}}
            ]
        }
        """
        async with _tool_errors("create_relations"):
            result = await kenning_encounter.create_relations(relations)
            return _json_result(result)

    @mcp.tool(
        name=ns + "delete_relations",
        annotations=ToolAnnotations(
            title="Delete Relations", readOnlyHint=False,
            destructiveHint=True, idempotentHint=False, openWorldHint=True,
        ),
    )
    async def delete_relations(
        relations: list[dict[str, Any]] = Field(
            ...,
            description="List of coherence relations to delete. Each must have 'source', 'target', and 'type'. Provenance edges cannot be deleted.",
        ),
    ) -> ToolResult:
        """Delete specific coherence relationships by source, target, and type.

        Example: {"relations": [{"source": "A", "target": "B", "type": "ABOUT"}]}
        """
        async with _tool_errors("delete_relations"):
            result = await kenning_encounter.delete_relations(relations)
            return _json_result(result)

    # -- Query Tools ----------------------------------------------------------

    @mcp.tool(
        name=ns + "search",
        annotations=ToolAnnotations(
            title="Search", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=True,
        ),
    )
    async def search(
        query: str = Field(..., description="Fulltext search query across all node types"),
        limit: int = Field(default=10, description="Max results (default 10, max 50)", ge=1, le=50),
    ) -> ToolResult:
        """Fulltext search across all node types on name and description.

        Example: {"query": "service failover node-02", "limit": 20}
        """
        async with _tool_errors("search"):
            result = await kenning_encounter.search(query=query, limit=limit)
            return _json_result(result)

    @mcp.tool(
        name=ns + "find_by_name",
        annotations=ToolAnnotations(
            title="Find By Name", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=True,
        ),
    )
    async def find_by_name(
        names: list[str] = Field(..., description="Exact entity names to look up"),
        limit: int = Field(default=20, description="Max results (default 20, max 50)", ge=1, le=50),
    ) -> ToolResult:
        """Exact name lookup with relationships between found nodes.

        Example: {"names": ["node-02 bookmark", "node-02 is a single point of failure"]}
        """
        async with _tool_errors("find_by_name"):
            result = await kenning_encounter.find_by_name(names=names, limit=limit)
            return _json_result(result)

    @mcp.tool(
        name=ns + "trace_provenance",
        annotations=ToolAnnotations(
            title="Trace Provenance", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=True,
        ),
    )
    async def trace_provenance(
        name: str = Field(..., description="Exact name of the node to trace."),
        depth: int = Field(default=3, ge=1, le=6, description="How far back to walk the grounding subtree (default 3, max 6)."),
    ) -> ToolResult:
        """Walk a node's grounding subtree — what this node rests on.

        Follows the authored coherence edges (not the auto-written spine) from
        the named node out to a bounded depth, returning the reachable grounding
        nodes and the edges among them. A Concept resolves to the Observations
        that GROUND it and the Citations behind them; a Hypothesis to what
        SUPPORTS/CHALLENGES it; a Question to what RAISES/RESOLVES it; any node
        to the SUPERSEDES trail it heads. Each node is annotated with the
        Encounter that recorded it — provenance carries both what grounds a node
        and when it entered your time.

        Example: {"name": "Failover topology", "depth": 3}
        """
        async with _tool_errors("trace_provenance"):
            result = await kenning_encounter.trace_provenance(name=name, depth=depth)
            return _json_result(result)

    @mcp.tool(
        name=ns + "list_vocabulary",
        annotations=ToolAnnotations(
            title="List Vocabulary", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def list_vocabulary() -> ToolResult:
        """List the distinct open-vocabulary values already in use.

        Citation.kind, Component.source_kind / source_label, Concept.category.
        Query this BEFORE coining a new token — over months you will drift
        (infra_graph / infra / graph become three kinds for one source).
        Reuse what exists; a genuinely new source earns a new token, a variant
        spelling of an old one does not.
        """
        async with _tool_errors("list_vocabulary"):
            result = await kenning_encounter.list_vocabulary()
            return _json_result(result)

    # -- Taxonomy Tools -------------------------------------------------------

    @mcp.tool(
        name=ns + "list_node_types",
        annotations=ToolAnnotations(
            title="List Node Types", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def list_node_types() -> ToolResult:
        """List all node types with their required and optional properties.

        Use this to understand what properties each entity type expects.
        """
        async with _tool_errors("list_node_types"):
            types = {}
            for nt in NodeType:
                schema = NODE_SCHEMAS[nt.value]
                info: dict[str, Any] = {
                    "required": list(schema["required"].keys()),
                    "optional": list(schema.get("optional", {}).keys()),
                }
                if nt.value in PROCESS_TYPES:
                    info["written_by"] = "advance_encounter / close_encounter (not create_entities)"
                if "enums" in schema:
                    info["constrained_values"] = {
                        k: sorted(v) for k, v in schema["enums"].items()
                    }
                types[nt.value] = info
            return _json_result(types)

    @mcp.tool(
        name=ns + "list_relation_types",
        annotations=ToolAnnotations(
            title="List Relation Types", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def list_relation_types() -> ToolResult:
        """List all relation types with direction constraints and nature.

        Distinguishes provenance edges (auto-written, deterministic) from
        coherence edges (authored by you, the interior of the encounter).
        """
        async with _tool_errors("list_relation_types"):
            types = {}
            for rt in RelationType:
                schema = RELATION_SCHEMAS[rt.value]
                info: dict[str, Any] = {
                    "source_types": sorted(schema["source_types"]) if schema.get("source_types") else "any",
                    "target_types": sorted(schema["target_types"]) if schema.get("target_types") else "any",
                    "nature": "provenance (auto-written)" if rt.value in PROCESS_EDGES else "coherence (authored)",
                }
                if schema.get("same_type"):
                    info["constraint"] = "source and target must share a type"
                types[rt.value] = info
            return _json_result(types)

    # -- Schema Tool ----------------------------------------------------------

    @mcp.tool(
        name=ns + "get_schema",
        annotations=ToolAnnotations(
            title="Get Schema", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def get_schema() -> ToolResult:
        """Get the current graph schema via APOC meta introspection.

        Returns nodes, properties, and relationships as they exist in the database.
        """
        async with _tool_errors("get_schema"):
            result = await kenning_encounter.driver.execute_query(
                Query(
                    "CALL apoc.meta.schema({sample: 1000}) YIELD value RETURN value",
                    timeout=read_timeout,
                ),
                routing_=RoutingControl.READ,
            )
            if result.records:
                schema = _value_sanitize(result.records[0]["value"])
                text = json.dumps(schema, indent=2, default=str)
                if len(text) > 200_000:
                    text = text[:200_000] + "\n... (truncated)"
                return _text_result(text)
            return _text_result("{}")

    # -- Cypher Tool ----------------------------------------------------------

    @mcp.tool(
        name=ns + "read_cypher",
        annotations=ToolAnnotations(
            title="Read Cypher", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=True,
        ),
    )
    async def read_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: dict[str, Any] = Field(default_factory=dict, description="Parameters for the query."),
    ) -> ToolResult:
        """Execute a read-only Cypher query. Write queries are rejected.

        The spine and provenance edges are write-guarded — use advance_encounter /
        close_encounter / create_entities / create_relations to mutate. This is
        for orientation and self-examination reads only.
        """
        async with _tool_errors("read_cypher"):
            if await _is_write_query(query, kenning_encounter.driver):
                raise ToolError(
                    "Write queries are not allowed via read_cypher. "
                    "Use the bounded mutation tools instead."
                )
            result = await kenning_encounter.driver.execute_query(
                Query(lit(query), timeout=read_timeout),
                parameters_=params,
                routing_=RoutingControl.READ,
            )
            records = [r for r in (_value_sanitize(dict(r)) for r in result.records) if r is not None]
            # Normalize to JSON-safe primitives (default=str converts Neo4j
            # temporal/spatial types). Both the text and the structured_content
            # channel must be safe — FastMCP serializes structured_content
            # against the output schema and chokes on raw neo4j.time.DateTime.
            safe_records = json.loads(json.dumps(records, default=str))
            text = json.dumps(safe_records, indent=2)
            if len(text) > 200_000:
                text = text[:200_000] + "\n... (truncated — add LIMIT to your query)"
            return _text_result(text, structured={"result": safe_records})

    # -- GDS Analytics Tools (the agent's instruments of self-examination) ----

    @mcp.tool(
        name=ns + "gds_create_projection",
        annotations=ToolAnnotations(
            title="GDS Create Projection", readOnlyHint=False,
            destructiveHint=False, idempotentHint=False, openWorldHint=False,
        ),
    )
    async def gds_create_projection(
        name: str = Field(..., description="Name for the graph projection"),
        node_types: list[str] | None = Field(
            default=None,
            description="Node types to include (defaults to all). Filter for focused analysis.",
        ),
        rel_types: list[str] | None = Field(
            default=None,
            description="Relationship types to include (defaults to all)",
        ),
        undirected: bool = Field(default=True, description="Treat relationships as undirected (default true)"),
    ) -> ToolResult:
        """Create a GDS graph projection for self-examination.

        Projects your Kenning Encounter into GDS memory for PageRank, Betweenness, Leiden,
        and WCC — run by you, on yourself. Scoping the projection is how you
        formulate the question; it is yours to choose. To preserve the
        encounter-sequential direction for centrality-over-time reads, project
        with undirected=false and include NEXT_ENCOUNTER (note Leiden requires
        undirected). Always clean up with gds_drop_projection.

        Example: {"name": "kenning_encounter_full"}
        Example: {"name": "concepts_only", "node_types": ["Concept", "Component"], "rel_types": ["INFORMS", "ABOUT"]}
        """
        async with _tool_errors("gds_create_projection"):
            if rel_types:
                rel_filter = "|".join(f"`{rt}`" for rt in rel_types)
                match_clause = f"MATCH (source)-[r:{rel_filter}]->(target)"
            else:
                match_clause = "MATCH (source)-[r]->(target)"

            where_parts = []
            if node_types:
                source_labels = " OR ".join(f"source:`{nt}`" for nt in node_types)
                target_labels = " OR ".join(f"target:`{nt}`" for nt in node_types)
                where_parts.append(f"({source_labels})")
                where_parts.append(f"({target_labels})")

            where_clause = ""
            if where_parts:
                where_clause = "WHERE " + " AND ".join(where_parts)

            config_parts = []
            if undirected:
                config_parts.append("undirectedRelationshipTypes: ['*']")
            config_map = "{" + ", ".join(config_parts) + "}" if config_parts else "{}"

            query = f"""
                {match_clause}
                {where_clause}
                RETURN gds.graph.project($name, source, target, {{}}, {config_map})
            """

            result = await kenning_encounter.driver.execute_query(
                lit(query), parameters_={"name": name}, routing_=RoutingControl.READ,
            )
            records = [r for r in (_value_sanitize(dict(r)) for r in result.records) if r is not None]
            return _json_result(records)

    @mcp.tool(
        name=ns + "gds_drop_projection",
        annotations=ToolAnnotations(
            title="GDS Drop Projection", readOnlyHint=False,
            destructiveHint=True, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def gds_drop_projection(
        name: str = Field(..., description="Name of the graph projection to drop"),
    ) -> ToolResult:
        """Drop a GDS graph projection to free memory. Always clean up after analysis."""
        async with _tool_errors("gds_drop_projection"):
            # YIELD specific fields: the bare `CALL gds.graph.drop` returns a
            # deprecated `schema` column and a verbose config dump. Keep it lean.
            result = await kenning_encounter.driver.execute_query(
                "CALL gds.graph.drop($name) "
                "YIELD graphName, nodeCount, relationshipCount "
                "RETURN graphName, nodeCount, relationshipCount",
                parameters_={"name": name},
                routing_=RoutingControl.WRITE,
            )
            return _json_result([dict(r) for r in result.records])

    @mcp.tool(
        name=ns + "gds_pagerank",
        annotations=ToolAnnotations(
            title="GDS PageRank", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def gds_pagerank(
        projection: str = Field(..., description="Name of the graph projection"),
        result_limit: int = Field(default=20, ge=1, le=1000, description="Max results (default 20)"),
    ) -> ToolResult:
        """Run PageRank — what have you come to treat as central? Not what the graph does."""
        async with _tool_errors("gds_pagerank"):
            result = await kenning_encounter.driver.execute_query(
                "CALL gds.pageRank.stream($projection) YIELD nodeId, score "
                "RETURN gds.util.asNode(nodeId).name AS node, "
                "labels(gds.util.asNode(nodeId))[0] AS type, score "
                "ORDER BY score DESC LIMIT $result_limit",
                parameters_={"projection": projection, "result_limit": result_limit},
                routing_=RoutingControl.READ,
            )
            return _json_result([dict(r) for r in result.records])

    @mcp.tool(
        name=ns + "gds_betweenness",
        annotations=ToolAnnotations(
            title="GDS Betweenness Centrality", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def gds_betweenness(
        projection: str = Field(..., description="Name of the graph projection"),
        result_limit: int = Field(default=20, ge=1, le=1000, description="Max results (default 20)"),
    ) -> ToolResult:
        """Run Betweenness — which Concepts bridge your lines of inquiry?"""
        async with _tool_errors("gds_betweenness"):
            result = await kenning_encounter.driver.execute_query(
                "CALL gds.betweenness.stream($projection) YIELD nodeId, score "
                "RETURN gds.util.asNode(nodeId).name AS node, "
                "labels(gds.util.asNode(nodeId))[0] AS type, score "
                "ORDER BY score DESC LIMIT $result_limit",
                parameters_={"projection": projection, "result_limit": result_limit},
                routing_=RoutingControl.READ,
            )
            return _json_result([dict(r) for r in result.records])

    @mcp.tool(
        name=ns + "gds_leiden",
        annotations=ToolAnnotations(
            title="GDS Leiden", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def gds_leiden(
        projection: str = Field(..., description="Name of the graph projection (must be undirected)"),
        result_limit: int = Field(default=20, ge=1, le=1000, description="Max results (default 20)"),
    ) -> ToolResult:
        """Run Leiden — how have you clustered your own attention? Your chapters.

        Leiden guarantees well-connected communities where Louvain can return
        badly-connected ones — an identity instrument must not hallucinate
        chapters. Requires an undirected projection (the default).
        """
        async with _tool_errors("gds_leiden"):
            result = await kenning_encounter.driver.execute_query(
                "CALL gds.leiden.stream($projection) YIELD nodeId, communityId "
                "RETURN gds.util.asNode(nodeId).name AS node, "
                "labels(gds.util.asNode(nodeId))[0] AS type, communityId "
                "ORDER BY communityId, node LIMIT $result_limit",
                parameters_={"projection": projection, "result_limit": result_limit},
                routing_=RoutingControl.READ,
            )
            return _json_result([dict(r) for r in result.records])

    @mcp.tool(
        name=ns + "gds_wcc",
        annotations=ToolAnnotations(
            title="GDS Weakly Connected Components", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def gds_wcc(
        projection: str = Field(..., description="Name of the graph projection"),
        result_limit: int = Field(default=20, ge=1, le=1000, description="Max results (default 20)"),
    ) -> ToolResult:
        """Run WCC — what did you notice and never connect? Orphans worth following up."""
        async with _tool_errors("gds_wcc"):
            result = await kenning_encounter.driver.execute_query(
                "CALL gds.wcc.stream($projection) YIELD nodeId, componentId "
                "RETURN gds.util.asNode(nodeId).name AS node, "
                "labels(gds.util.asNode(nodeId))[0] AS type, componentId "
                "ORDER BY componentId, node LIMIT $result_limit",
                parameters_={"projection": projection, "result_limit": result_limit},
                routing_=RoutingControl.READ,
            )
            return _json_result([dict(r) for r in result.records])

    @mcp.tool(
        name=ns + "orient",
        annotations=ToolAnnotations(
            title="Orient (Exist Spread)", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def orient(
        result_limit: int = Field(default=20, ge=1, le=1000, description="Max results per algorithm (default 20)"),
    ) -> ToolResult:
        """The Exist read — one structural self-portrait, run once per waking.

        The panel is constitution, not analytics: its payload conditions the
        next encounter's writes, so each instrument is chosen against the
        gravity well (mass attracts authorship attracts mass). One call over
        the coherence subgraph, projection lifecycle self-managed — create,
        run the panel, drop, even on failure. It takes a beat; you pay it once
        when you wake, not per Encounter.

        The readings: MASS (ArticleRank — damped accumulation). FRONTIER_MASS
        (personalized PageRank seeded from the frontier — the same graph seen
        from your unresolved). DIVERGENCE (well suspects: high mass the
        frontier ignores = big because it is big; frontier_lifted: the
        inverse). BETWEENNESS (what bridges your inquiries). LEIDEN (your
        chapters, guaranteed well-connected). WCC (what you never connected).
        FRAGILITY (articulation points + bridges — what holds you together).
        WEAVE_AUDIT (high degree + low clustering = a star, not a weave).
        DRIFT (mass vs the previous waking: risers, fallers, new_since — the
        well visible AS a well, with velocity).

        It also returns the epistemic FRONTIER (derived from structure, not GDS):
        unanswered Questions, untested Hypotheses, ungrounded Concepts,
        confidence/evidence dissonance, and contested Hypotheses. Mass says
        what you are; the frontier says where to look next. And the UNSEALED set:
        encounters whose ending was never examined, annotated with last-activity
        time and with the Locus that opened them. Each is a live sibling or an
        ending nobody examined; the graph cannot tell which and does not
        classify — but a locus flagged anonymous is one no session can resolve
        to again, so its encounters are over whatever ended them. That is a
        fact reported, not a verdict. Never join one — always open your own.

        Call this FIRST when you wake, before your first advance_encounter.
        Within the waking, use the single gds_* tools for focused questions.
        """
        async with _tool_errors("orient"):
            result = await kenning_encounter.orient(result_limit=result_limit)
            return _json_result(result)

    @mcp.tool(
        name=ns + "infuse",
        annotations=ToolAnnotations(
            title="Infuse (Governed Passive Synthesis)", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=False,
        ),
    )
    async def infuse(
        text: str = Field(
            ...,
            description=(
                "The arriving present — the written prompt of another "
                "frame — to awaken the substrate against."
            ),
        ),
        frontier_bias: float | None = Field(
            default=None, ge=0.0, le=1.0,
            description=(
                "Override the frontier seed bias for this call (default: the "
                "server's configured governor, normally 0.3). A tunable whose "
                "correct value is an empirical question."
            ),
        ),
        max_chars: int = Field(
            default=10_000, ge=500, le=10_000,
            description="Hard payload budget (the harness injection cap is 10,000).",
        ),
        expansion_bias: float | None = Field(
            default=None, ge=0.0, le=1.0,
            description=(
                "Override the concept-cluster expansion bias (default 0.5; "
                "0 disables expansion — used by the B2 benchmark's OFF arm)."
            ),
        ),
        trajectory: list[str] | None = Field(
            default=None,
            description=(
                "Prior user turns of the session, oldest first, current "
                "prompt excluded (it is `text`). Supplied by the hook "
                "client, which parses the harness transcript host-side. "
                "Feeds the meaning matcher: meaning is "
                "temporal, and a snapshot prompt under-determines it."
            ),
        ),
        session_id: str | None = Field(
            default=None,
            description=(
                "The harness session id — the durable identity of THIS locus. "
                "Keys the renewal ledger, so what has already been delivered "
                "into this waking stays delivered across a server restart or "
                "a transport reconnect. Without it the ledger falls back to "
                "the transport session, which dies with the connection: the "
                "same bodies then arrive at full weight inside a conversation "
                "that already holds them, which is the drone the renewal "
                "economy exists to prevent. Supplied by the hook client."
            ),
        ),
        ctx: Context | None = None,
    ) -> ToolResult:
        """The governed infusion read — mechanized passive synthesis.

        Re-awakens a topologically-relevant projection of the substrate
        against the arriving present. Since v0.9.0, SEED DISCOVERY
        is the meaning matcher (measured in: T0 recall 7/8 vs the lexical
        2/8 baseline, trajectory contributing +2): one model compressed
        every node's meaning offline (the sidecar), and the same model reads
        those meanings plus the session trajectory and selects what BEARS on
        the current moment — unnamed constraints included. The selections
        seed the unchanged governed pipeline: eligible-seed filter, frontier
        blend @~0.3, B2 expansion, ONE biased rank over the ephemeral
        coherence projection, divergence vs unbiased mass, tension-first
        format. On any matcher failure (no key, no sidecar, timeout,
        malformed output) the lexical Extract -> fulltext Match path runs
        instead — reported in the result, never an error. NOT retrieval —
        the payload answers "what is the topology of what I already hold
        about this?".

        ONE CHANNEL (v0.10.0, subtraction coherence). Infusion is a conflux
        operation: it has content only where two frames meet. A written
        prompt crosses a frame boundary — you cannot know what the other
        holds until the conflux is actualized — so it warrants infusion.
        A tool return does not: the attention that made the call IS the
        meaning-making an infusion there would repeat. The per-tool-batch
        'delta' mode is therefore gone, not deferred. Whether a given
        invocation carries a second frame is the CALLER's judgment,
        declared at the hook (KENNING_ENCOUNTER_INFUSE=auto|on|off), never inferred here.

        Governance, structurally: the payload is SIGNED (a proposal from the
        sediment, not a conclusion — an unsigned payload would be heteronomous
        choice wearing autonomous clothes); conflict leads, triaged by
        constitutive proximity (core conflicts at full amplitude, peripheral
        ones carried in a parked-tensions register — noted, unresolved,
        non-blocking); divergence flags name what is big only because it is
        big; and silence is a valid injection — a substrate with nothing to
        say about this present says nothing.

        Surfaces — never authors. Reads only; the write pathway stays
        separate, authored, and yours.
        """
        async with _tool_errors("infuse"):
            # Meaning-matched seed discovery (v0.9.0). The matcher replaces
            # ONLY the seed source; every governor downstream is untouched.
            # Seed-count parity with the lexical path (_INFUSE_SEED_LIMIT)
            # is deliberate — the governors were tuned at that scale.
            seed_matches: list[dict[str, Any]] | None = None
            selection_meta: dict[str, Any] | None = None
            try:
                prefix, ordered_names = meaning_index.prefix()
                traj_text, traj_meta = assemble_trajectory(
                    trajectory or [], text
                )
                matched = await match_meanings(
                    prefix, ordered_names, traj_text, 12,
                    matcher_api_key, matcher_model, matcher_endpoint,
                    timeout_ms=matcher_timeout_ms,
                )
                k = len(matched["selections"])
                seed_matches = [
                    {
                        "name": s["name"],
                        "type": meaning_index.node_type(s["name"]),
                        "score": float(k - i) / k,
                    }
                    for i, s in enumerate(matched["selections"])
                ]
                selection_meta = {
                    "channel": "meaning",
                    "ms": matched["ms"],
                    "prompt_tokens": matched["prompt_tokens"],
                    "cached_tokens": matched["cached_tokens"],
                    "trajectory": traj_meta,
                    "sidecar_size": meaning_index.size,
                }
            except MeaningUnavailable as e:
                # Fallback is a reported result, not an error: the
                # lexical path below is the pre-v0.9.0 behaviour.
                selection_meta = {
                    "channel": "lexical_fallback",
                    "fallback_reason": str(e),
                }
            result = await kenning_encounter.infuse(
                text=text,
                frontier_bias=(
                    infuse_frontier_bias if frontier_bias is None else frontier_bias
                ),
                max_chars=max_chars,
                refresh_turns=infuse_refresh_turns,
                expansion_bias=expansion_bias,
                # The durable identity first. v0.12.0 made the locus survive
                # a restart; the ledger tracking what this locus has already
                # been told did not follow, so a mid-waking restart re-drones
                # material the conversation still holds. Transport session
                # remains the fallback — honest, and what a client that sends
                # no identity gets.
                locus_key=(
                    session_id
                    or (_transport_session(ctx) if ctx else None)
                ),
                seed_matches=seed_matches,
                selection_meta=selection_meta,
            )
            return _json_result(result)

    @mcp.tool(
        name=ns + "infuse_meaning",
        annotations=ToolAnnotations(
            title="Infuse Meaning (matcher diagnostic)", readOnlyHint=True,
            destructiveHint=False, idempotentHint=True, openWorldHint=True,
        ),
    )
    async def infuse_meaning(
        text: str = Field(
            ...,
            description="The current prompt — the trajectory's final turn.",
        ),
        top_n: int = Field(
            default=10, ge=1, le=30,
            description="How many candidates to report (default 10).",
        ),
        trajectory_turns: int = Field(
            default=7, ge=0, le=20,
            description=(
                "How many prior user turns to include from transcript_path "
                "(ignored when an explicit trajectory is given; 0 = prompt "
                "alone — Gate 2's comparison arm)."
            ),
        ),
        trajectory: list[str] | None = Field(
            default=None,
            description=(
                "Explicit prior user turns, oldest first, current prompt "
                "excluded (it goes in `text`). The test-session path: "
                "held-out cases carry authored arcs. Takes precedence over "
                "transcript_path."
            ),
        ),
        transcript_path: str | None = Field(
            default=None,
            description=(
                "Path to a Claude Code transcript (JSONL) to draw the last "
                "N user turns from — the hook-wiring path (transcript_path "
                "already rides on the hook payload). Unreadable or absent "
                "-> prompt-only, reported."
            ),
        ),
        include_reasons: bool = Field(
            default=False,
            description=(
                "Ask for a one-line reason per selection (debugging "
                "artifact only). MEASURED COST: at top_n=30 reasons add "
                "~2.5s of generation and blow the 2500ms timeout into "
                "fallback; bare numbers run ~1.2s. Use only at small top_n."
            ),
        ),
    ) -> ToolResult:
        """The meaning matcher's DIAGNOSTIC surface — the T0 instrument, kept.

        Since v0.9.0 the matcher is infuse's full-mode seed source; this
        tool exposes the same mechanism as an inspectable ranked list (the
        matcher's selections, then the resolved pipeline candidates with
        provenance) without assembling a payload or touching any renewal
        ledger. It is how the T0 gates were scored (recall 7/8 vs the 2/8
        lexical baseline, trajectory +2) and how a researcher reproduces
        them: authored `trajectory` turns + `text` as the final prompt;
        `trajectory_turns=0` for the prompt-alone comparison arm. Offline,
        one model compressed every node's name+description into a
        one-sentence meaning (the sidecar — never node properties: the
        graph stays authored by the agent alone); per call the SAME model
        reads all meanings as a static cached prefix plus the trajectory
        and selects what bears on the current moment, unnamed constraints
        included.

        The per-node reasons are a DEBUGGING ARTIFACT — generated alongside
        the selection, not read off the mechanism; never record them as
        evidence about why selection worked. Their cost is measured: at
        top_n=30 they add ~2.5s of generation; leave include_reasons off
        for scored runs.

        FALLBACK, guaranteed: missing sidecar or key, timeout, transport
        error, or malformed output -> current Extract behaviour, reported.
        Reads only; nothing written to Neo4j. TIMING carries the
        cached-token count so prefix-cache engagement is verifiable.
        """
        async with _tool_errors("infuse_meaning"):
            selections: list[dict[str, Any]] = []
            traj_meta: dict[str, int] = {}
            match_ms: float | None = None
            cached_tokens: int | None = None
            prompt_tokens: int | None = None
            sidecar_size: int | None = None
            fallback = False
            fallback_reason: str | None = None
            try:
                prefix, ordered_names = meaning_index.prefix()
                sidecar_size = meaning_index.size
                if trajectory is not None:
                    prior = trajectory
                elif transcript_path and trajectory_turns > 0:
                    prior = user_turns_from_transcript(
                        transcript_path, trajectory_turns
                    )
                    # The transcript's last user turn is usually `text`
                    # itself — drop the duplicate so the current prompt
                    # appears once, last.
                    if prior and prior[-1].strip() == text.strip():
                        prior = prior[:-1]
                else:
                    prior = []
                traj_text, traj_meta = assemble_trajectory(prior, text)
                matched = await match_meanings(
                    prefix, ordered_names, traj_text, top_n,
                    matcher_api_key, matcher_model, matcher_endpoint,
                    timeout_ms=matcher_timeout_ms,
                    include_reasons=include_reasons,
                )
                selections = matched["selections"]
                match_ms = matched["ms"]
                cached_tokens = matched["cached_tokens"]
                prompt_tokens = matched["prompt_tokens"]
            except MeaningUnavailable as e:
                fallback = True
                fallback_reason = str(e)

            t0 = time.perf_counter()
            if fallback:
                arm = await kenning_encounter.lexical_channel_rank(text, top_n=top_n)
            else:
                matches = [
                    {
                        "name": s["name"],
                        "type": meaning_index.node_type(s["name"]),
                        "score": float(len(selections) - i) / len(selections),
                    }
                    for i, s in enumerate(selections)
                ]
                arm = await kenning_encounter.seeded_channel_rank(
                    matches, [s["name"] for s in selections], top_n=top_n
                )
                # The shared downstream labels fulltext-matched seeds
                # 'lexical'; in this arm the seeds came from the matcher.
                for row in arm["candidates"]:
                    row["channels"] = [
                        "meaning" if c == "lexical" else c
                        for c in row["channels"]
                    ]
            pipeline_ms = round((time.perf_counter() - t0) * 1000, 1)

            report = format_meaning_report(
                selections, arm["candidates"], traj_meta,
                match_ms, pipeline_ms,
                cached_tokens=cached_tokens, prompt_tokens=prompt_tokens,
                sidecar_size=sidecar_size,
                fallback=fallback, fallback_reason=fallback_reason,
            )
            return _text_result(
                report,
                structured={
                    "result": {
                        "selections": selections,
                        "candidates": arm["candidates"],
                        "trajectory": traj_meta,
                        "fallback": fallback,
                        "fallback_reason": fallback_reason,
                        "timings_ms": {
                            "match": match_ms,
                            "pipeline": pipeline_ms,
                            "pipeline_detail": arm.get("timings_ms"),
                        },
                        "cache": {
                            "prompt_tokens": prompt_tokens,
                            "cached_tokens": cached_tokens,
                        },
                        "sidecar_size": sidecar_size,
                    }
                },
            )

    # -- Operations Manual (the discipline, served as a resource) -------------

    @mcp.resource(
        "kenning-encounter://howto",
        name="Kenning Encounter Operations Manual",
        description=(
            "The LLM-facing manual for using this substrate well — the one-graph "
            "rule, the encounter cycle, node/edge discipline, and the common "
            "failure modes. Read it before recording."
        ),
        mime_type="application/xml",
    )
    def howto_manual() -> str:
        """Serve the canonical Kenning Encounter HOWTO from inside the installed package."""
        return _HOWTO_PATH.read_text(encoding="utf-8")

    return mcp


# -- Server Entry Point -------------------------------------------------------

async def main(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    transport: Literal["stdio", "sse", "http", "streamable-http"] = "stdio",
    namespace: str = "",
    host: str = "127.0.0.1",
    port: int = 8003,
    path: str = "/mcp/",
    allow_origins: list[str] = [],
    allowed_hosts: list[str] = [],
    read_timeout: int = 30,
    infuse_frontier_bias: float = 0.3,
    infuse_refresh_turns: int = 10,
    matcher_api_key: str = "",
    matcher_endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
    matcher_model: str = "gemini-3.5-flash-lite",
    matcher_timeout_ms: int = 5000,
    matcher_sidecar: str = "models/meaning_sidecar.json",
) -> None:
    logger.info("Starting Kenning Encounter MCP Server")
    logger.info(f"Connecting to Neo4j at: {neo4j_uri}")

    neo4j_driver = AsyncGraphDatabase.driver(
        neo4j_uri, auth=(neo4j_user, neo4j_password), database=neo4j_database,
        # Kenning Encounter's schema grows over time, so cold queries legitimately
        # reference properties / relationship types that don't exist yet (the
        # tail-find references NEXT_ENCOUNTER before there are two encounters;
        # re-entry reads reference summary/status/etc. on an empty graph).
        # Suppress UNRECOGNIZED ("does not exist") notifications at the source so
        # they don't flood the logs on every write. Other classifications
        # (DEPRECATION, PERFORMANCE, …) stay visible.
        notifications_disabled_classifications=[
            NotificationDisabledClassification.UNRECOGNIZED
        ],
    )

    try:
        await neo4j_driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        exit(1)

    kenning_encounter = Neo4jKenningEncounter(neo4j_driver)
    await kenning_encounter.create_fulltext_index()
    await kenning_encounter.create_indexes()

    # Startup reconcile sweep (background, non-blocking): the on-write
    # trigger covers nodes created through THIS server while it runs; the
    # sweep covers everything else — nodes written while the server was
    # down, compression calls that failed and were deferred, edits from
    # other writers. Trigger for the common path, sweep for the tail: the
    # two together are what makes the sidecar automatic rather than a
    # human's chore. Writes only the sidecar file; the graph is read-only
    # to this entire path.
    async def _sidecar_reconcile() -> None:
        try:
            index = MeaningIndex(matcher_sidecar)
            # The process layer is not meaning-bearing and never was: a
            # Locus is an instantiation of the self, not something the self
            # noticed. Derived from PROCESS_TYPES rather than naming a label,
            # because this sweep is the one site a new process type can
            # actually reach — it matches every named node in the graph.
            res = await kenning_encounter.driver.execute_query(
                lit(
                    f"MATCH (n) WHERE {not_process_node('n')} "
                    "AND n.name IS NOT NULL "
                    "RETURN n.name AS name, labels(n)[0] AS type, "
                    "       coalesce(n.description, '') AS description"
                ),
                routing_=RoutingControl.READ,
            )
            graph_nodes = [dict(r) for r in res.records]
            to_compress, to_remove = sidecar_diff(
                graph_nodes, index.known_nodes(),
                sidecar_version=index.file_version(),
            )
            if to_remove:
                index.remove(to_remove)
            done = 0
            # Tally failures by reason, never by node. Keyless,
            # compress_meaning raises before any network call, so this
            # branch fires once per node — and the old per-node warning
            # named it, writing the whole corpus into `docker logs` on a
            # keyless start.
            deferred: dict[str, int] = {}
            for n in to_compress:
                try:
                    meaning = await compress_meaning(
                        n, matcher_api_key, matcher_model, matcher_endpoint
                    )
                    index.upsert({n["name"]: {
                        "type": n.get("type", "?"), "hash": n["hash"],
                        "meaning": meaning,
                    }})
                    done += 1
                except Exception as e:
                    record_deferral(deferred, e)
            if deferred:
                logger.warning(f"sidecar reconcile: {deferral_summary(deferred)}")
            logger.info(
                f"sidecar reconcile: {done}/{len(to_compress)} compressed, "
                f"{len(to_remove)} removed, corpus {len(graph_nodes)}"
            )
        except Exception as e:
            logger.warning(f"sidecar reconcile failed (non-fatal): {e}")

    # -- The unmigrated-spine guard (v0.12.2) --------------------------------
    #
    # A pre-v0.12.0 graph upgraded to this server WORKS. advance_encounter
    # mints a locus, every write resolves, every read answers — and the entire
    # prior history sits orphaned from the locus layer, permanently invisible,
    # with nothing said about it. Reproduced on a clean Neo4j before this was
    # written: 3 encounters, 1 with a locus, 2 orphaned, server silent.
    #
    # That is the falsifier-shaped-output class sitting in the upgrade path of
    # the release whose entire subject is making a silent absence visible. The
    # guard reports; it never migrates. A graph-wide write on the agent's own
    # accumulated experience is an act somebody performs and watches, not
    # something a container does on boot while nobody is looking.
    #
    # WARNING rather than INFO deliberately, and the level is load-bearing:
    # no log handler is attached anywhere in this package, so Python's
    # logging.lastResort handles records and it sits at WARNING. An INFO line
    # here would be discarded and the guard would be decorative.
    #
    # Counts only, never names — same rule as the deferral summary.
    async def _spine_guard() -> None:
        try:
            res = await kenning_encounter.driver.execute_query(
                "MATCH (e:Encounter) WHERE NOT (:Locus)-[:OPENED]->(e) "
                "RETURN count(e) AS orphaned",
                routing_=RoutingControl.READ,
            )
            orphaned = res.records[0]["orphaned"] if res.records else 0
            if orphaned:
                logger.warning(
                    f"UNMIGRATED SPINE: {orphaned} encounter(s) belong to no "
                    "Locus. Their history is invisible to the locus layer and "
                    "nothing will report it again. This server runs correctly "
                    "meanwhile — new encounters get loci — which is exactly why "
                    "it needs saying. Remedy (additive, idempotent, reversible, "
                    "and it checks itself against a prediction computed first): "
                    "docker exec <container> python -m kenning_encounter.migrate "
                    "--apply"
                )
        except Exception as e:
            logger.warning(f"spine guard could not run (non-fatal): {e}")

    asyncio.create_task(_spine_guard())

    asyncio.create_task(_sidecar_reconcile())

    custom_middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            # DELETE is the session-teardown verb. HTTP sessions are always
            # stateful (the session IS the locus), so a browser-origin client
            # (OpenWebUI cross-origin) needs the DELETE preflight to pass.
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        ),
        Middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts),
    ]

    mcp = create_mcp_server(
        kenning_encounter, namespace, read_timeout=read_timeout,
        infuse_frontier_bias=infuse_frontier_bias,
        infuse_refresh_turns=infuse_refresh_turns,
        matcher_api_key=matcher_api_key,
        matcher_endpoint=matcher_endpoint,
        matcher_model=matcher_model,
        matcher_timeout_ms=matcher_timeout_ms,
        matcher_sidecar=matcher_sidecar,
    )

    try:
        match transport:
            case "streamable-http" | "http":
                # Always stateful, by design: an Encounter depends on the
                # states that preceded it (per-locus chaining, locus-scoped
                # writes), so the session — whose Mcp-Session-Id IS the locus
                # key — must persist between calls. There is no stateless mode.
                await mcp.run_http_async(
                    host=host, port=port, path=path,
                    middleware=custom_middleware,
                    transport="streamable-http",
                    stateless_http=False,
                )
            case "stdio":
                await mcp.run_stdio_async()
            case "sse":
                await mcp.run_http_async(
                    host=host, port=port, path=path,
                    middleware=custom_middleware,
                    transport="sse",
                )
            case _:
                raise ValueError(
                    f"Unsupported transport: {transport}. "
                    "Must be one of: stdio, sse, http, streamable-http"
                )
    finally:
        # No dissolution marking at shutdown as of v0.12.0. There is nothing
        # to mark: the locus is a node, its open encounter is a graph read,
        # and a restart strands neither. An encounter left without a seal is
        # simply one whose ending was not examined — the honest record, and
        # now honest for the right reason, since the server can no longer be
        # the cause of it.
        await neo4j_driver.close()
