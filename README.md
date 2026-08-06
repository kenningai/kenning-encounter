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

- **Process** — `Encounter` (one work session) chained by `NEXT_ENCOUNTER`
  **per session**: each session's encounters form one clean fork-free path, and
  a session's first encounter is anchored by `INSTANTIATED_AFTER` to whichever
  encounter was latest when the session began (writes are serialized, so that
  order is exact). Concurrent sessions — multiple agents over one memory —
  therefore branch without ever corrupting each other's chains, and the whole
  spine stays one connected, temporally ordered structure. Written only by the
  guarded `advance_encounter` / `close_encounter`. Encounters are never
  deletable — the spine is the record, not editable content.
- **Semantic** — `Observation` / `Question` / `Hypothesis` / `Concept` / `Note`.
  What the agent has come to understand: noticings, open threads, working
  explanations, syntheses.
- **Reference** — `Component` (a bookmark into a source: `source_kind` +
  `source_label` + `source_key`) and `Citation` (provenance: `kind`, `uri`/`query`,
  `snapshot`). The agent annotates its sources; it never copies them.

## Tool surface

- **Session:** `advance_encounter` (opens an encounter chained from *your
  session's* previous one; returns the re-entry payload — recent sessions, open
  questions, live hypotheses, recent concepts, and the **unsealed set**: any
  encounters left open elsewhere, with last-activity times), `close_encounter`
  (seals *your session's* encounter — a concurrent session can't capture it;
  accepts an explicit `encounter` name if the server restarted mid-session).
  Encounters left open when the server discards session state get a mechanical
  `dissolved_at` timestamp — never an auto-written summary.
- **Write:** `create_entities` (auto-anchored to *your session's* open encounter),
  `create_relations` (the agent's authored links; `SUPERSEDES` carries a required
  `revision_why` — a revision records what changed and why), `delete_entities`,
  `delete_relations`.
- **Read:** `search`, `find_by_name`, `trace_provenance` (walk a node's grounding
  subtree back to the observations and citations under it), `read_cypher`
  (EXPLAIN-gated, read-only), `get_schema`, `list_node_types`,
  `list_relation_types`, `list_vocabulary`.
- **Analytics:** `orient` — a one-call instrument panel over the authored-edge
  graph, managing the projections for you. An agent whose memory analytics feed
  its next writes can develop rich-get-richer bias (what the reading makes
  prominent gets written about more), so the panel is composed against that
  feedback loop: **mass** (ArticleRank — damped accumulation), **frontier_mass**
  (PageRank personalized from the open questions/untested hypotheses — the same
  graph ranked from what's unresolved), their **divergence** (nodes prominent
  only by accumulation vs. nodes the open work actually leans on),
  **betweenness**, **leiden** communities, **wcc**, a **fragility** map
  (articulation points + bridges), a **weave_audit** (high-degree/low-clustering
  stars), and **drift** (the mass reading vs. the previous session's baseline —
  risers, fallers, new). Plus the **frontier** view of where the graph is
  thinnest (unanswered questions, untested hypotheses, ungrounded concepts).
  For focused questions: `gds_create_projection` → `gds_pagerank` /
  `gds_betweenness` / `gds_leiden` / `gds_wcc` → `gds_drop_projection`.
- **Infusion:** `infuse` — the *unasked* recall channel; see the next section.

The operations manual is served as the MCP resource `agent-memory://howto` and
also ships at `src/mcp_agent_memory/HOWTO.xml` — read it before recording.

## Governed infusion — memory that arrives on its own

Everything above is *asked-for* recall: the agent calls `orient` or `search`
and gets an answer. Real working memory has a second mode — what you already
know about the matter at hand surfaces *without being asked*, at the moment
it bears. `infuse` is that channel, run once per user prompt (typically from
a harness hook), and it is **governed**, because the naive form — re-inject
whatever is heaviest, every turn — is a feedback loop: whatever the payload
makes salient gets written about, gains weight, and dominates the next
payload. An agent bound to that loop doesn't get more knowledgeable, it gets
more repetitive.

The pipeline: **Extract** focal signals from the prompt (terms, identifiers,
CIDRs, hostnames) → **Match** candidates via the fulltext index → **Rank**
with a single biased personalized rank over an *ephemeral, coherence-only
projection* — focal seeds at full weight, the open frontier (unanswered
questions, live hypotheses) at a minority bias so unresolved work keeps a
voice — checked against unbiased mass so genuine tension leads → **Format**
a signed, budgeted payload: load-bearing conflicts first at full amplitude,
peripheral ones parked in a brief register, neighborhood after. The payload
is signed with the seed terms it awoke from, so the agent can see *why* this
surfaced and keep its own judgment over it.

Three disciplines keep the channel honest:

- **The renewal economy.** A per-session delivery ledger: a node's full body
  is delivered at first sight, on a changed fact-state, or after a staleness
  horizon — otherwise it re-pins as a one-line *handle* in a STANDING
  register ("you were already told this; the body is earlier in your
  context"). Payloads rotate with the work instead of droning a permanent
  core. The ledger is stamped from what was actually *delivered*, never
  merely selected, and the metadata reports both.
- **Silence is a valid answer.** `mode=delta` (per tool call) returns
  recognition or conflict only — and when the moment holds nothing the graph
  recognizes and contests nothing it holds, it returns nothing. Each
  fact-state announces once, at first sight, then suppresses; a changed
  state re-announces.
- **Graph-native reach.** Focal seeds expand one hop across authored
  coherence edges from matched Concepts, at reduced bias — the concept that
  *matches* often sits one edge from the one that *matters*. No embeddings,
  no vector index, nothing vector-shaped ever stored.

`scripts/agent_memory_infuse_hook.py` is a stdlib-only, fail-silent hook
client for Claude Code-style harnesses: full mode on each user prompt, delta
mode per tool call. If the server is down, the hook stays silent — the agent
just runs uninfused.

### Wiring the hook

```json
{
  "hooks": {
    "UserPromptSubmit": [{ "hooks": [{ "type": "command",
      "command": "/absolute/path/to/python3 /path/to/scripts/agent_memory_infuse_hook.py --mode full" }] }]
  }
}
```

Hook configuration: `AGENT_MEMORY_MCP_URL` (default
`http://127.0.0.1:8003/mcp/`) or `--url`; `AGENT_MEMORY_INFUSE_TIMEOUT`
seconds (default `10.0`) or `--timeout`; `AGENT_MEMORY_INFUSE_SHADOW_LOG` /
`AGENT_MEMORY_INFUSE_OBSERVE_LOG` for the shadow and observation streams.
The timeout default is sized to the **cold first call** — the first prompt
of a session ranks against a cold Neo4j page cache at roughly 10× the warm
cost, and because the hook is fail-silent, a budget that only fits the warm
call fails invisibly at exactly the moment re-entry matters.

Deployment notes, each learned from a real deployment:

- **Absolute interpreter path, always.** Hook processes do not inherit an
  interactive shell PATH; a bare `python3` can resolve to an ancient system
  interpreter. The hook requires Python 3.12+ and refuses older interpreters
  cleanly (exit 0, one line on stderr) rather than crashing mid-import.
- **Project-scoped settings, never global.** Wire the hook in the
  *project's* `.claude/settings.json`. A hook in the global settings would
  inject this substrate's memories into every unrelated session on the
  machine, pointed at a graph that has nothing to do with that work.
- **Stage it.** Run `--shadow` for a few real sessions first — it computes
  everything and injects nothing, logging what *would* have been surfaced,
  so you read the payloads before they condition a live turn. Wire
  `UserPromptSubmit` alone before adding the high-frequency delta channel.

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
`NEO4J_MCP_SERVER_*`, `NEO4J_NAMESPACE`, `NEO4J_READ_TIMEOUT`.

Infusion governors: `NEO4J_INFUSE_FRONTIER_BIAS` (default `0.3` — the
frontier's minority weight in the biased rank), `NEO4J_INFUSE_REFRESH_TURNS`
(default `10` — the staleness horizon before a standing node's full body is
re-delivered), and per-call `expansion_bias` (`0` disables cluster
expansion). Per node, `frontier_mute: true` (on `Question` / `Hypothesis` /
`Concept`) retires a resolved line of inquiry from every frontier surface —
orient and infusion both — while leaving it fully queryable.

HTTP sessions are always **stateful** — there is no stateless option, by
design: an encounter depends on the state that preceded it (per-session
chaining, session-scoped writes), so the `Mcp-Session-Id` carries the session
identity and the `DELETE` teardown is honored.

Single-tenant: one agent, one memory graph, one Neo4j database.
