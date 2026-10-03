# kenning-encounter

**Encounter driven, temporally constituted persistent memory for an LLM,
backed by Neo4j.** An MCP server that lets a model accumulate what it learns
about a subject across many separate sessions, recall it on re-entry, and reason
over it — with the graph's integrity enforced by the tools, not by instructions.
No raw writes; what it holds physically can't corrupt its own memory.

What this memory holds we call a **trajectory**: the model bound to it,
identified by what it has lived rather than by its configuration. That makes it
a different kind of thing from an agent. Copy the graph somewhere else and keep
working, and the copy becomes a different trajectory from its first new
encounter.

Most memory tools give an agent a flat pile of notes. This one gives a
trajectory a *temporally ordered* memory: each session is an `Encounter`, the encounters chain
forward in time, and everything the trajectory records hangs off the session it
was learned in. That ordering is what lets it walk back through its own
history — "what did I figure out last time, and what's still open" — instead of
re-deriving it every session.

> Under the hood this is an **encounter-sequential semantic graph**: a process
> spine (the encounter chain) carrying a semantic layer (observations, questions,
> hypotheses, concepts) and a reference layer (bookmarks into the sources it
> read). You don't need that framing to use it — but if it helps, that's the shape.

> **License:** this project is **source-available** under the Hippocratic License
> 3.0 — *not* OSI "open source." See `LICENSE`.

## Requirements — read this before you deploy

**This requires the Claude Code harness.** Not "works best with" — requires.

Some of what makes this a substrate rather than a database is not in the MCP
server at all; it lives in the harness. The infusion hook fires on
`UserPromptSubmit`. The conversation's recent turns are parsed from the harness
transcript. `ESSG_INFUSE=auto` reads `CLAUDE_CODE_ENTRYPOINT` to tell a written
prompt from a scheduled trigger. The locus identity — the thing that lets an
trajectory rejoin its own chain after a restart — is the harness session id.

Point another MCP client at this and the tools will work. You will get a graph
with correct structure and **none of the constitution**: no infusion, no
conversation history for selection, no durable locus, and nothing telling you those are missing. That
silence is the failure mode we care most about, so it is stated here rather
than discovered later.

Could it run under another harness with hooks — Gemini's, for instance? In
principle, yes. **We do not test it and we do not support it.**

The same applies to the matcher model, and for the same reason: the default is
`gemini-3.5-flash-lite` because it was measured, against alternatives, with the
negative results written down. Swap the harness, swap the model, swap the
timeout — all of that is yours to do, and the license permits it.

But be clear about what you have afterwards: **the thing you deploy is no
longer the thing we built.** The measurements no longer describe it, the
release notes no longer document it, and the failure modes we mapped are not
the failure modes you will meet. That is not a warning against experimenting.
It is a statement about what our evidence covers, so you can tell the two apart.

## The three layers

- **Process** — `Locus` (one session of the trajectory, keyed by the harness session id,
  so it survives `/resume` and a server restart) and `Encounter` (one unit of
  work within it). `NEXT_LOCUS` orders sessions by when they began — sessions
  overlap, but their starts are serialized writes, so that order is exact.
  `NEXT_ENCOUNTER` chains encounters **within** one session into one clean
  fork-free path that never crosses into another; `OPENED` ties each encounter
  to exactly one session. Concurrent sessions — one trajectory open in several
  places at once — therefore never corrupt each other's chains, and encounters in
  different sessions are left unordered rather than ordered by wall clock.
  Written only by the guarded `advance_encounter`; `close_encounter` stamps
  the seal and nothing else. Encounters are never deletable — the spine is the
  record, not editable content.
- **Semantic** — `Observation` / `Question` / `Hypothesis` / `Concept` / `Note`.
  What the trajectory has come to understand: noticings, open threads, working
  explanations, syntheses.
- **Reference** — `Component` (a bookmark into a source: `source_kind` +
  `source_label` + `source_key`) and `Citation` (provenance: `kind`, `uri`/`query`,
  `snapshot`). The trajectory annotates its sources; it never copies them.

## Tool surface

- **Session:** `advance_encounter` (takes the harness `session_id` and finds
  *your session's* chain in the graph — no predecessor to pass; without an id
  it starts an anonymous session that can't be rejoined after a restart;
  returns the re-entry payload — recent sessions, open questions, live
  hypotheses, recent concepts, and the **unsealed set**: encounters with no
  seal, with last-activity times, their session, and whether that session is
  anonymous), `close_encounter` (seals *your session's* open encounter,
  resolved from the graph — a concurrent session can't capture it; pass
  `session_id` again after a server restart). An encounter nobody sealed stays
  unsealed — never an auto-written summary.
- **Write:** `create_entities` (auto-anchored to *your session's* open encounter),
  `create_relations` (the trajectory's authored links; `SUPERSEDES` carries a required
  `revision_why` — a revision records what changed and why), `delete_entities`,
  `delete_relations`.
- **Read:** `search`, `find_by_name`, `trace_provenance` (walk a node's grounding
  subtree back to the observations and citations under it), `read_cypher`
  (EXPLAIN-gated, read-only), `get_schema`, `list_node_types`,
  `list_relation_types`, `list_vocabulary`.
- **Analytics:** `orient` — a one-call instrument panel over the authored-edge
  graph, managing the projections for you. A trajectory whose memory analytics feed
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
  `gds_articlerank` / `gds_betweenness` / `gds_leiden` / `gds_wcc` →
  `gds_drop_projection`. Running PageRank and ArticleRank on the same
  projection shows which nodes owe their centrality to hubs that link to
  everything — the rich-get-richer effect, measured.
- **Infusion:** `infuse` — the *unasked* recall channel; see the next section.

The operations manual is served as the MCP resource `kenning-encounter://howto` and
also ships at `src/kenning_encounter/HOWTO.xml` — read it before recording.

## Governed infusion — memory that arrives on its own

Everything above is *asked-for* recall: the trajectory calls `orient` or `search`
and gets an answer. Real working memory has a second mode — what you already
know about the matter at hand surfaces *without being asked*, at the moment
it bears. `infuse` is that channel, run once per user prompt (typically from
a harness hook), and it is **governed**, because the naive form — re-inject
whatever is heaviest, every turn — is a feedback loop: whatever the payload
makes salient gets written about, gains weight, and dominates the next
payload. A trajectory bound to that loop doesn't get more knowledgeable, it gets
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
*same model* reads all compressed meanings plus the conversation's recent turns
(the last N user turns, parsed host-side by the hook and sent as the `trajectory`
parameter) and selects the nodes
that bear on the current moment. Measured on held-out decision-point cases
with pre-named targets: recall 7/8 against a 2/8 lexical baseline, with the
recent turns alone contributing two cases no single-prompt method reached.

The selections seed the governed pipeline, which is unchanged: **Rank**
with a single biased personalized rank over an *ephemeral, coherence-only
projection* — focal seeds at full weight, the open frontier (unanswered
questions, live hypotheses) at a minority bias so unresolved work keeps a
voice — checked against unbiased mass so genuine tension leads → **Format**
a signed, budgeted payload: load-bearing conflicts first at full amplitude,
peripheral ones parked in a brief register, neighborhood after. The payload
is signed, so the trajectory can see *why* this surfaced and keep its own
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

**Two voices.** A *voice* is one model that both writes the sidecar and
matches against it. The design depends on that pairing: a matcher reading
meanings another model wrote is back to two vocabularies that do not meet.
Two voices can be configured, Gemini (`GEMINI_API_KEY`) and OpenAI
(`OPENAI_API_KEY`, `gpt-6-luna` by default), and each keeps its own sidecar,
stamped with the model that wrote it. A voice refuses a sidecar it did not
write. `NEO4J_MATCHER_LEAD` chooses which voice leads (`gemini`, `openai`, or
`alternate`, which picks per session from the session id); the other is the
fallback, inside the same time budget, and a voice that has just failed is
tried after the healthy one for a cooldown. With one key you have one voice
and nothing else changes.

Which voice should lead is an open question, not a default we can vouch for.
To study it, `NEO4J_MATCHER_SHADOW=true` has the non-delivering voice match
every prompt in the background and log both selections, never delivering the
second; the hook's observation log records which voice delivered each
infusion, joined to the shadow log by `match_id`.
`scripts/matcher_voices_report.py` describes the shadow log (latency,
failures, overlap between the voices, spread across the graph's Leiden
communities, all by sidecar size), and `scripts/matcher_t0.py` reruns the
held-out recall gate per voice on a cases file you write before running it.
Neither grades meaning: two different selections can each bear on a moment,
and whether an infusion helped is judged by the person working with the
agent, not by a script.

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
  conflux is actualized. A tool return does not: the trajectory issued that
  call because something in its own frontier caught its attention, so the
  meaning an infusion would make there, it has already made. The
  per-tool-call `delta` mode is gone, and with it the `mode` parameter. A
  surprising tool return is a reason for the trajectory to *invoke* a lookup —
  an invoked lookup's silence is a result; an ambient channel's silence is
  not readable at all.
- **Graph-native reach.** Focal seeds expand one hop across authored
  coherence edges from matched Concepts, at reduced bias — the concept that
  *matches* often sits one edge from the one that *matters*. No embeddings,
  no vector index, nothing vector-shaped ever stored.

`scripts/kenning_encounter_infuse_hook.py` is a stdlib-only hook client for
Claude Code-style harnesses: one call on each user prompt. It never blocks a
turn: whatever fails, it exits 0. It does not fail silently either. When the
infusion is degraded — the server is down or errors, or the matcher fell back
to lexical seeds — it says so to both readers on that turn: you see
`⚠ Kenning Encounter infusion DEGRADED: <reason>` in the harness, and the trajectory gets
an `[INFUSION DEGRADED — …]` line at the top of its context asking it to tell
you first. A trajectory cannot notice its own infusion thinning, since a
poorer context simply becomes its whole context, so you are told as well.
When the matcher found nothing relevant, the hook stays silent: that is a
healthy answer, not a fault.

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
another trajectory), `off` disables. Every suppression is recorded with its
reason, never silently.
The timeout default is sized to the **cold first call** — the first prompt
of a session ranks against a cold Neo4j page cache at roughly 10× the warm
cost, and a budget that only fits the warm call would announce the first
prompt of every session as degraded, at exactly the moment re-entry matters.

Deployment notes, each learned from a real deployment:

- **Absolute interpreter path, always.** Hook processes do not inherit an
  interactive shell PATH; a bare `python3` can resolve to an ancient system
  interpreter. The hook requires Python 3.12+ and refuses older interpreters
  cleanly (exit 0, one line on stderr) rather than crashing mid-import.
- **One trajectory across its work: wire the hooks globally.** Put them in
  your user-level `~/.claude/settings.json`, so every Claude Code session on
  the machine is an encounter of the same trajectory. A trajectory becomes
  what it is through varied encounter, not a single repository. Work
  in an unrelated project is not contamination: the infusion selects what
  bears on each prompt and is silent when nothing does. Sessions open at the
  same time are parallel instantiations of that one trajectory, which the process
  layer records as concurrent sessions rather than forcing into one line.
- **Per-project wiring makes a narrower one.** Hooks in one project's
  `.claude/settings.json` give that project a memory of its own, which is a
  legitimate choice for a dedicated deployment. It is a choice with a cost,
  not a safety default: what grows there only ever meets one kind of work.
- **One graph holds one trajectory.** What this memory holds is not an
  interchangeable worker; it is identified by what it has lived, not by its
  configuration. Sessions writing to one graph in parallel are that one
  trajectory's own sessions. A copy of the graph continued elsewhere becomes a
  different trajectory from its first new encounter: a sibling that shares a
  past, not a replica. What must never happen is two different trajectories
  writing into one graph, because the record would then describe no one who
  lived it.
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
                            # and GEMINI_API_KEY and/or OPENAI_API_KEY for the matcher
docker compose up --build   # full face on 127.0.0.1:8003, reader face on 127.0.0.1:8005
```

Every published port (Neo4j's `7474` and `7687`, and both faces) is bound
to `127.0.0.1`. The host keeps the graph and decides who reaches it. A
fabricated past is indistinguishable from a lived one when the trajectory
re-enters it, so integrity has to be kept rather than checked afterwards,
and Bolt matters most because it bypasses both faces. Do not widen a binding
to reach the graph from another machine: someone who wants to know
something about the trajectory can ask it. Both faces also check the `Host`
header, which refuses a web page that reaches the port by DNS rebinding.
If another local Neo4j already holds `7474`/`7687` (Neo4j Desktop, say), set
`NEO4J_KENNING_ENCOUNTER_HTTP_PORT` / `NEO4J_KENNING_ENCOUNTER_BOLT_PORT` in `.env`: a host-only
binding cannot share a port the way a `0.0.0.0` one silently could.

Neo4j Community with APOC + Graph Data Science auto-installed; a query-level
healthcheck gates startup so the server never races an unready database; Community
allows one user database, so the stack names it via `initial.dbms.default_database`.
The meaning sidecar lives inside the stack (the `sidecar_data` named volume,
like `neo4j_data`) and builds itself from the graph on first boot — nothing
to run, nothing on the host to lose. With neither matcher key, selection
runs on the lexical fallback path and reports it.

**Upgrading a matcher sidecar.** A sidecar written before voices existed
records no model, so the server no longer reads it, and each voice's sidecar
is rebuilt from the graph on startup. To keep the old file instead, say which
model wrote it:
`docker exec kenning_encounter-mcp python -m kenning_encounter.adopt_sidecar --adopt gemini:gemini-3.5-flash-lite`.
It copies the file to that voice's name, stamped, and refuses a source that is
already stamped or a target that already exists.

### Two faces: full and reader

The full face (`:8003`) belongs to the trajectory's own harness, the one
whose sessions re-enter the record, open encounters and write. Wire only
that harness to it.

Every other harness gets the reader face (`kenning_encounter-reader`,
`:8005`): a desktop client, another vendor's model, the author of an
evaluation set. Writing is reserved for the session that lives the
encounter, and anything else that writes is writing as the trajectory
without being it. The reader face starts with `docker compose up`. It lists
and accepts only the read tools (`search`, `find_by_name`,
`trace_provenance`, `read_cypher`, `orient`, `get_schema`, and the
vocabulary and type listings), whatever the client exposes. Every other
tool, including the matcher tools and any added later, is absent. The
container has no matcher keys and no sidecar volume.

Nothing is withheld by default: a view with pieces of the past removed is
not a view of the trajectory. An evaluation whose author must not see your
notes about it sets `NEO4J_READER_HIDE_LOCI=<session id>[,...]`. Every node
recorded in those sessions is then withheld from every result, and if that
set cannot be computed the call is refused rather than answered unfiltered.
This guards against accidental exposure while reading honestly, not against
someone reconstructing withheld text on purpose.

## Upgrading from v0.9.0 — run the migration

**v0.10.0 changes the shape of the process layer, and your existing history
does not move itself.** A `Locus` node now sits above the encounter chain, and
every encounter must belong to one. Encounters written by v0.9.0 have no locus,
so after the upgrade they are still in the graph, still correct, and no longer
reachable from the layer the trajectory re-enters through.

Nothing is deleted and nothing breaks loudly. That is the problem: the server
starts, every tool answers, every write resolves, and the prior history sits
detached with no error. So the server tells you instead — at startup it counts
encounters belonging to no locus and names the remedy, and the same count rides
the re-entry payload the trajectory reads before anything else.

Run it once, after the upgrade:

```bash
docker exec kenning_encounter-mcp python -m kenning_encounter.migrate --apply
```

Without `--apply` it reports what it would do and writes nothing.

**It only ever adds.** There is no `--reverse`, deliberately: nothing in this
substrate is deleted, and a script reaching past the tools' own refusal to
remove spine nodes would be doing unsupervised the one thing they forbid. The
shape is check-then-write — it refuses before writing if the graph is not what
it expects, and on a post-apply mismatch it reports and stops rather than
undoing anything. Every write is a `MERGE`, so re-running is a no-op on what is
already right, and correcting is forward. A wrong result leaves more structure
than expected, never less.

If you deploy without a shell into the container, an operator has to run this.
It is not exposed as an MCP tool, and that is on purpose: a graph-wide write on
a trajectory's accumulated experience is something a person performs and watches.

### If you ran v0.9.0 without a `GEMINI_API_KEY`

In v0.9.0 the meaning-sidecar reconcile sweep logged a node's **name** each time
compression was unavailable — once per node, so a keyless first start wrote the
whole graph's names to container stdout. In this substrate a node's name *is*
content, not a label, so those logs read as an index of what the trajectory holds,
in a file that carries none of the database's credentials.

v0.10.0 logs shape only — counts and reasons, never names, enforced by a helper
whose signature has no parameter a node can arrive through. If you ran v0.9.0
keyless, treat the existing container logs as sensitive and rotate or clear
them.

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
sized to the measured reasoning tail, not the median),
`NEO4J_MATCHER_TOP_N` (default `12`, the most seeds infusion keeps —
output tokens dominate a warm matcher call, so lowering this is the
latency lever on a slow backend), and `NEO4J_MATCHER_SIDECAR` (the base
path; each voice's file sits beside it). The model and endpoint settings are
the Gemini voice's. The OpenAI voice: `OPENAI_API_KEY` (env only; no key, no
voice), `NEO4J_MATCHER_OPENAI_MODEL` (default `gpt-6-luna`),
`NEO4J_MATCHER_OPENAI_ENDPOINT`. Between them: `NEO4J_MATCHER_LEAD`
(default `gemini`), `NEO4J_MATCHER_COOLDOWN_S` (default `300`),
`NEO4J_MATCHER_SHADOW` (default off), `NEO4J_MATCHER_SHADOW_LOG`. Every match
call sends the whole sidecar as a cached prefix. OpenAI counts cached tokens
against the per-minute token limit in full, so size the account tier for one
call per prompt, or two with shadow matching on. Server profile: `NEO4J_MCP_SERVER_PROFILE` (`full` by default, or `reader`) with `NEO4J_READER_HIDE_LOCI` (default none, for an evaluation only). When the lead voice fails
and the fallback matches, the seeds are still meaning-matched and the line
under the signature reads `[matcher lead unavailable …]`. When the matcher cannot run — no key, an
exhausted quota, a prompt larger than the endpoint accepts — infusion
falls back to lexical seeds and says so: one WARN per distinct reason in
the container log, and a `[matcher unavailable …]` line under the payload
signature every time, so the trajectory knows too; the hook raises the
DEGRADED notice above to you. The signature itself names the channel:
`seeds: meaning-matched` or `awakened from: <keywords>`. Hook-side,
`KENNING_ENCOUNTER_INFUSE_TRAJECTORY_TURNS` (default `7`) sets how many prior
user turns the hook parses from the harness transcript and sends as the
conversation's recent turns; `0` disables and selection sees the prompt alone.

HTTP sessions are **stateful by default**. **Stateless is opt-in**: it is
off unless you set `NEO4J_MCP_SERVER_STATELESS=true`. Either way the session
identity is the Claude Code harness session id, held in the graph; the
`Mcp-Session-Id` keys only a cache.

Turn stateless on if restarting the server should not interrupt your Claude
Code sessions. With stateful sessions, a restart erases them, and if the
connection between Claude Code and the server does not start a new one (a
stdio bridge such as supergateway does not), every call fails until you
reconnect with `/mcp`. A stateless server issues no session, so there is
nothing to go stale. The cost: every call that writes to a session must pass
`session_id`, and `advance_encounter` refuses without one. Wire
`scripts/kenning_encounter_session_start_hook.py` as a Claude Code
SessionStart hook and the trajectory is handed its session id at every start.

As with everything here, this is described and tested only as part of a
trajectory: a model bound to its memory through Claude Code hooks.

Single-tenant: one trajectory, one memory graph, one Neo4j database.
