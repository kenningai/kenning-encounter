import json
import logging
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

from .agent_memory import (
    Neo4jAgentMemory,
    NodeType,
    RelationType,
    NODE_SCHEMAS,
    RELATION_SCHEMAS,
    PROCESS_TYPES,
    PROCESS_EDGES,
)
from .utils import format_namespace, _is_write_query, _value_sanitize, lit

logger = logging.getLogger("mcp_agent_memory")
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


def _locus_key(ctx: Context) -> str | None:
    """The calling locus's identity for locus-scoped write targeting.

    The MCP session id: stable for the lifetime of a stdio connection (one
    process, one client, one locus) and of an HTTP session (the
    mcp-session-id header — HTTP is always stateful here, by design). None
    when no session exists at all.
    """
    try:
        return ctx.session_id
    except RuntimeError:
        return None


# -- Server Factory -----------------------------------------------------------

def create_mcp_server(
    agent_memory: Neo4jAgentMemory,
    namespace: str = "",
    read_timeout: int = 30,
    infuse_frontier_bias: float = 0.3,
    infuse_refresh_turns: int = 10,
) -> FastMCP:
    """Create an MCP server instance for the Agent Memory."""

    ns = format_namespace(namespace)
    mcp: FastMCP = FastMCP("mcp-agent-memory")

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
        ctx: Context | None = None,
    ) -> ToolResult:
        """Open a new Encounter and orient — the sole writer of the temporal spine.

        Takes NO predecessor: the server links from the encounter THIS locus
        (this session) last opened — per-locus chaining. A first-of-locus
        encounter has no incoming NEXT_ENCOUNTER and is genesis-bound via
        INSTANTIATED_AFTER to the encounter latest when this locus began, so
        parallel loci are branches of one connected becoming, never forks of
        one chain and never floating fragments. Opening an encounter
        IS orienting — this returns the re-entry payload: recent encounter
        summaries, still-open Questions, live (proposed/challenged) Hypotheses,
        recently-touched Concepts, and the unsealed set (encounters without a
        seal — live siblings or orphans; never join one, always open your own).
        Read it before examining the world. Your past encounters are your memory.

        Call this to open each Encounter, before any create_entities — a node
        comes to be within ITS encounter, and the graph records that
        constitutively. Once per waking, call orient first for the global
        structural survey; this opens the work-units within that waking.

        Example: {"name": "Encounter 2026-05-29T14:00 — service failover thread"}
        """
        async with _tool_errors("advance_encounter"):
            result = await agent_memory.advance_encounter(
                name=name, recent=recent, limit=limit,
                locus_key=_locus_key(ctx) if ctx else None,
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
        encounter: str | None = Field(
            default=None,
            description=(
                "Explicit Encounter name to seal — only needed when the server "
                "lost your session's state (e.g. it restarted) and you are "
                "returning to seal the encounter you lived. Defaults to the "
                "encounter THIS session opened."
            ),
        ),
        ctx: Context | None = None,
    ) -> ToolResult:
        """Seal YOUR encounter at Report/Stop.

        Addresses the encounter this locus (this session) opened — never the
        global tail, so a parallel sibling's advance cannot capture your seal.
        Writes only summary/report — never a node, never a NEXT_ENCOUNTER edge;
        the spine stays sole-written by advance_encounter. Even a
        confirmation-only run should close with a summary — the absence of
        change is a temporal event worth recording. Seal only what you lived:
        never author a summary for another locus's encounter.

        Example: {"summary": "Confirmed all four services still depend on node-02.",
                  "report": "No change since encounter N-1; SPOF question stays open."}
        """
        async with _tool_errors("close_encounter"):
            result = await agent_memory.close_encounter(
                summary=summary, report=report, encounter=encounter,
                locus_key=_locus_key(ctx) if ctx else None,
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
        encounter: str | None = Field(
            default=None,
            description=(
                "Explicit Encounter name to anchor to — only needed when the "
                "server lost your session's state (e.g. it restarted) mid-"
                "encounter. Defaults to the encounter THIS session opened."
            ),
        ),
        ctx: Context | None = None,
    ) -> ToolResult:
        """Create semantic/reference nodes, auto-anchored to YOUR open Encounter.

        Epistemic nodes (Observation/Question/Hypothesis/Concept/Note) get a
        RECORDED edge from the encounter this locus (this session) opened —
        never the global tail, so your nodes cannot be misattributed to a
        parallel sibling's encounter; Citations get CONSULTED. "No open
        Encounter" means THIS session has not advanced or has already closed.
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
            result = await agent_memory.create_entities(
                entities, encounter=encounter,
                locus_key=_locus_key(ctx) if ctx else None,
            )
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
            result = await agent_memory.delete_entities(names)
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
            result = await agent_memory.create_relations(relations)
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
            result = await agent_memory.delete_relations(relations)
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
            result = await agent_memory.search(query=query, limit=limit)
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
            result = await agent_memory.find_by_name(names=names, limit=limit)
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
            result = await agent_memory.trace_provenance(name=name, depth=depth)
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
            result = await agent_memory.list_vocabulary()
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
            result = await agent_memory.driver.execute_query(
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
            if await _is_write_query(query, agent_memory.driver):
                raise ToolError(
                    "Write queries are not allowed via read_cypher. "
                    "Use the bounded mutation tools instead."
                )
            result = await agent_memory.driver.execute_query(
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

        Projects your Agent Memory into GDS memory for PageRank, Betweenness, Leiden,
        and WCC — run by you, on yourself. Scoping the projection is how you
        formulate the question; it is yours to choose. To preserve the
        encounter-sequential direction for centrality-over-time reads, project
        with undirected=false and include NEXT_ENCOUNTER (note Leiden requires
        undirected). Always clean up with gds_drop_projection.

        Example: {"name": "agent_memory_full"}
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

            result = await agent_memory.driver.execute_query(
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
            result = await agent_memory.driver.execute_query(
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
            result = await agent_memory.driver.execute_query(
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
            result = await agent_memory.driver.execute_query(
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
            result = await agent_memory.driver.execute_query(
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
            result = await agent_memory.driver.execute_query(
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
        encounters with no seal, annotated with last-activity time — each is a
        live sibling locus or an orphaned dissolution; the graph cannot tell
        which and does not classify. Never join one — always open your own.

        Call this FIRST when you wake, before your first advance_encounter.
        Within the waking, use the single gds_* tools for focused questions.
        """
        async with _tool_errors("orient"):
            result = await agent_memory.orient(result_limit=result_limit)
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
                "The arriving present: the user prompt (mode 'full') or the "
                "tool-result batch text (mode 'delta') to awaken the substrate "
                "against."
            ),
        ),
        mode: Literal["full", "delta"] = Field(
            default="full",
            description=(
                "'full' = the complete governed disposition (with every user "
                "prompt); 'delta' = recognition/conflict only, else silence "
                "(at tool-batch boundaries)."
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
        ctx: Context | None = None,
    ) -> ToolResult:
        """The governed infusion read — mechanized passive synthesis.

        Re-awakens a topologically-relevant projection of the substrate
        against the arriving present: Extract (focal signals) -> Match
        (fulltext) -> Rank (ONE biased computation: focal seeds @1.0 blended
        with the standing frontier @~0.3, over an ephemeral coherence-only
        projection, divergence-checked against unbiased mass) -> Format
        (signed payload, coherence tension first). NOT retrieval — the payload
        answers "what is the topology of what I already hold about this?".

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
            result = await agent_memory.infuse(
                text=text,
                mode=mode,
                frontier_bias=(
                    infuse_frontier_bias if frontier_bias is None else frontier_bias
                ),
                max_chars=max_chars,
                refresh_turns=infuse_refresh_turns,
                expansion_bias=expansion_bias,
                locus_key=_locus_key(ctx) if ctx else None,
            )
            return _json_result(result)

    # -- Operations Manual (the discipline, served as a resource) -------------

    @mcp.resource(
        "agent-memory://howto",
        name="Agent Memory Operations Manual",
        description=(
            "The LLM-facing manual for using this substrate well — the one-graph "
            "rule, the encounter cycle, node/edge discipline, and the common "
            "failure modes. Read it before recording."
        ),
        mime_type="application/xml",
    )
    def howto_manual() -> str:
        """Serve the canonical Agent Memory HOWTO from inside the installed package."""
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
) -> None:
    logger.info("Starting Agent Memory MCP Server")
    logger.info(f"Connecting to Neo4j at: {neo4j_uri}")

    neo4j_driver = AsyncGraphDatabase.driver(
        neo4j_uri, auth=(neo4j_user, neo4j_password), database=neo4j_database,
        # The Agent Memory's schema grows over time, so cold queries legitimately
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

    agent_memory = Neo4jAgentMemory(neo4j_driver)
    await agent_memory.create_fulltext_index()
    await agent_memory.create_indexes()

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
        agent_memory, namespace, read_timeout=read_timeout,
        infuse_frontier_bias=infuse_frontier_bias,
        infuse_refresh_turns=infuse_refresh_turns,
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
        # Idle-MARK, never idle-seal: the server is discarding its locus state,
        # so any encounter still open here loses its implicit addressing —
        # stamp dissolved_at (mechanical timestamp, seal fields untouched).
        # Best-effort: a hard kill skips this, and the unsealed set surfaced at
        # the next orient/advance catches whatever was missed.
        try:
            marked = await agent_memory.mark_open_dissolved()
            if marked:
                logger.info(f"Marked dissolved at shutdown: {marked}")
        except Exception as e:
            logger.warning(f"Dissolution marking at shutdown failed: {e}")
        await neo4j_driver.close()
