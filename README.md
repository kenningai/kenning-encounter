# kenning-encounter

**Encounter driven, temporally constituted persistent memory for LLM agents,
backed by Neo4j.** An MCP server that lets an agent accumulate what it learns
about a subject across many separate sessions, recall it on re-entry, and reason
over it — with the graph's integrity enforced by the tools, not by instructions.
No raw writes; the agent physically can't corrupt its own memory.

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

The operations manual is served as the MCP resource `kenning-encounter://howto` and
also ships at `src/kenning_encounter/HOWTO.xml` — read it before recording.

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

Since v0.7.0, **selection reads meaning, not keywords**. A self-authored
graph defeats lexical retrieval by construction: its node names are
syntheses, deliberately written *above* the vocabulary of the moments that
need them, so the memory that matters most — the one that bears as an
unnamed constraint — shares no words with the prompt. We measured three
retrieval designs to failure on exactly this (lexical matching, frozen
embeddings, query rewriting) before shipping the one that passed: **the
meaning matcher**. Offline, one model compresses every node's name and
description into a one-sentence meaning (the *sidecar* — a derived file
inside the stack's own volume, never node properties); per prompt, the
*same model* reads all compressed meanings plus the session **trajectory**
(the last N user turns, parsed host-side by the hook) and selects the nodes
that bear on the current moment. Measured on held-out decision-point cases
with pre-named targets: recall 7/8 against a 2/8 lexical baseline, with the
trajectory alone contributing two cases no single-prompt method reached.

The selections seed the governed pipeline, which is unchanged: **Rank**
with a single biased personalized rank over an *ephemeral, coherence-only
projection* — focal seeds at full weight, the open frontier (unanswered
questions, live hypotheses) at a minority bias so unresolved work keeps a
voice — checked against unbiased mass so genuine tension leads → **Format**
a signed, budgeted payload: load-bearing conflicts first at full amplitude,
peripheral ones parked in a brief register, neighborhood after. The payload
is signed, so the agent can see *why* this surfaced and keep its own
judgment over it. On any matcher failure — no API key, timeout, malformed
output — selection falls back to the original lexical Extract → Match path
and the result says so; the matcher can never block a turn.

The matcher requires a **Gemini API key** (`GEMINI_API_KEY` in `.env`) and
defaults to `gemini-3.5-flash-lite`. The model, the 5-second timeout, and
the reasons-off default are all *measured* choices, not preferences — the
constraints were built through extensive experimentation (documented in the
release notes), and the slow tail of the latency distribution is exactly
the arc-dependent selections that justify the mechanism. The endpoint and
model are configurable; deviate from the measured configuration at your own
risk. The sidecar **maintains itself**: nodes are meaning-made on creation,
a startup reconcile sweep rebuilds any gap from the graph (including first
boot), and deletions drop entries — there is no maintenance step.

Three disciplines keep the channel honest:

- **The renewal economy.** A per-session delivery ledger: a node's full body
  is delivered at first sight, on a changed fact-state, or after a staleness
  horizon — otherwise it re-pins as a one-line *handle* in a STANDING
  register ("you were already told this; the body is earlier in your
  context"). Payloads rotate with the work instead of droning a permanent
  core. The ledger is stamped from what was actually *delivered*, never
  merely selected, and the metadata reports both.
- **Silence is a valid answer.** When the arriving prompt touches nothing
  the graph holds and contests nothing it believes, the payload is the empty
  string and the hook stays quiet.
- **One channel, by construction (v0.8.0).** Infusion is a *conflux*
  operation: it has content only where two frames meet. A written prompt
  crosses a frame boundary — you cannot know what the other holds until the
  conflux is actualized. A tool return does not: the agent issued that call
  because something in its own frontier caught its attention, so the
  trajectory an infusion would make meaning from, it already *is*. The
  per-tool-call `delta` mode is gone, and with it the `mode` parameter. A
  surprising tool return is a reason for the agent to *invoke* a lookup —
  an invoked lookup's silence is a result; an ambient channel's silence is
  not readable at all.
- **Graph-native reach.** Focal seeds expand one hop across authored
  coherence edges from matched Concepts, at reduced bias — the concept that
  *matches* often sits one edge from the one that *matters*. No embeddings,
  no vector index, nothing vector-shaped ever stored.

`scripts/kenning_encounter_infuse_hook.py` is a stdlib-only, fail-silent hook
client for Claude Code-style harnesses: one call on each user prompt. If the
server is down — or the command line carries a flag from an older version —
the hook stays silent and exits 0; the agent just runs uninfused. It never
blocks a turn.

### Wiring the hook

```json
{
  "hooks": {
    "UserPromptSubmit": [{ "hooks": [{ "type": "command",
      "command": "/absolute/path/to/python3 /path/to/scripts/kenning_encounter_infuse_hook.py" }] }]
  }
}
```

Hook configuration: `KENNING_ENCOUNTER_MCP_URL` (default
`http://127.0.0.1:8003/mcp/`) or `--url`; `KENNING_ENCOUNTER_INFUSE_TIMEOUT`
seconds (default `10.0`) or `--timeout`; `KENNING_ENCOUNTER_INFUSE_SHADOW_LOG` /
`KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG` for the shadow and observation streams;
`KENNING_ENCOUNTER_INFUSE=auto|on|off` (default `auto`) is the switch — `auto`
infuses on a written prompt and stays silent on a scheduled trigger, `on`
declares a second frame the harness cannot detect (an inbound message from
another agent), `off` disables. Every suppression is recorded with its
reason, never silently.
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
  so you read the payloads before they condition a live turn.

## Quick start

```bash
uv sync
uv run kenning-encounter --db-url bolt://localhost:7687
```

## Docker stack (self-contained unit)

`compose.yml` brings up Neo4j **and** the server together:

```bash
cp .env.example .env        # set a strong NEO4J_KENNING_ENCOUNTER_PASSWORD
                            # and GEMINI_API_KEY for the meaning matcher
docker compose up --build   # streamable-http on :8003
```

Neo4j Community with APOC + Graph Data Science auto-installed; a query-level
healthcheck gates startup so the server never races an unready database; Community
allows one user database, so the stack names it via `initial.dbms.default_database`.
The meaning sidecar lives inside the stack (the `sidecar_data` named volume,
like `neo4j_data`) and builds itself from the graph on first boot — nothing
to run, nothing on the host to lose. Without `GEMINI_API_KEY`, selection
runs on the lexical fallback path and reports it.

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

Meaning matcher: `GEMINI_API_KEY` (env only, never argv),
`NEO4J_MATCHER_MODEL` (default `gemini-3.5-flash-lite`),
`NEO4J_MATCHER_ENDPOINT`, `NEO4J_MATCHER_TIMEOUT_MS` (default `5000` —
sized to the measured reasoning tail, not the median), and
`NEO4J_MATCHER_SIDECAR` (container path of the sidecar file). Hook-side,
`KENNING_ENCOUNTER_INFUSE_TRAJECTORY_TURNS` (default `7`) sets how many prior
user turns the hook parses from the harness transcript and sends as the
trajectory; `0` disables and selection sees the prompt alone.

HTTP sessions are always **stateful** — there is no stateless option, by
design: an encounter depends on the state that preceded it (per-session
chaining, session-scoped writes), so the `Mcp-Session-Id` carries the session
identity and the `DELETE` teardown is honored.

Single-tenant: one agent, one memory graph, one Neo4j database.
