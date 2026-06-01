# mcp-agent-memory

**Structured, persistent working memory for LLM agents, backed by Neo4j.** An MCP
server that lets an agent accumulate what it learns about a subject across many
separate sessions, recall it on re-entry, and reason over it — with the graph's
integrity enforced by the tools, not by instructions. No raw writes; the agent
physically can't corrupt its own memory.

Most memory tools give an agent a flat pile of notes. This one gives it a
*temporally ordered* memory: each session is an `Encounter`, the encounters chain
forward in time, and everything the agent records hangs off the session it was
learned in. That ordering is what lets the agent walk back through its own
history — "what did I figure out last time, and what's still open" — instead of
re-deriving it every session.

> Under the hood this is an **encounter-sequential semantic graph**: a process
> spine (the encounter chain) carrying a semantic layer (observations, questions,
> hypotheses, concepts) and a reference layer (bookmarks into the sources it
> read). You don't need that framing to use it — but if it helps, that's the shape.

> **License:** this project is **source-available** under the Hippocratic License
> 3.0 — *not* OSI "open source." See `LICENSE`.

## The three layers

- **Process** — `Encounter` (one work session) chained by `NEXT_ENCOUNTER`. The
  agent's own ordering of its memory. Written only by the guarded
  `advance_encounter` / `close_encounter` — the spine stays a single clean chain.
- **Semantic** — `Observation` / `Question` / `Hypothesis` / `Concept` / `Note`.
  What the agent has come to understand: noticings, open threads, working
  explanations, syntheses.
- **Reference** — `Component` (a bookmark into a source: `source_kind` +
  `source_label` + `source_key`) and `Citation` (provenance: `kind`, `uri`/`query`,
  `snapshot`). The agent annotates its sources; it never copies them.

## Tool surface

- **Session:** `advance_encounter` (opens a session, returns the re-entry payload —
  recent sessions, open questions, live hypotheses, recent concepts),
  `close_encounter` (annotate the session at the end).
- **Write:** `create_entities` (auto-anchored to the open session),
  `create_relations` (the agent's authored links), `delete_entities`,
  `delete_relations`.
- **Read:** `search`, `find_by_name`, `read_cypher` (EXPLAIN-gated, read-only),
  `get_schema`, `list_node_types`, `list_relation_types`, `list_vocabulary`.
- **Analytics (GDS):** `gds_create_projection` → `gds_pagerank` /
  `gds_betweenness` / `gds_louvain` / `gds_wcc` → `gds_drop_projection`.

The operations manual is served as the MCP resource `agent-memory://howto` and
also ships at `src/mcp_agent_memory/HOWTO.xml` — read it before recording.

## Quick start

```bash
uv sync
uv run mcp-agent-memory --db-url bolt://localhost:7687
```

## Docker stack (self-contained unit)

`compose.yml` brings up Neo4j **and** the server together:

```bash
cp .env.example .env        # set a strong NEO4J_AGENT_MEMORY_PASSWORD
docker compose up --build   # streamable-http on :8003
```

Neo4j Community with APOC + Graph Data Science auto-installed; a query-level
healthcheck gates startup so the server never races an unready database; Community
allows one user database, so the stack names it via `initial.dbms.default_database`.

## Configuration

CLI args → env vars → defaults: `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`,
`NEO4J_DATABASE`, `NEO4J_TRANSPORT` (`stdio` | `streamable-http` | `sse`),
`NEO4J_MCP_SERVER_*`, `NEO4J_NAMESPACE`, `NEO4J_READ_TIMEOUT`,
`NEO4J_MCP_STATELESS_HTTP` (default `true`; set `false` for a session-bridged
client that needs `Mcp-Session-Id` + `DELETE` teardown).

Single-tenant: one agent, one memory graph, one Neo4j database.
