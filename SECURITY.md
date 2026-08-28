# Security

## Reporting

Write to **contact@kenningai.com**. There is no bug bounty and no formal SLA;
this is a single-author project and reports are read by a person, not a queue.
Please include what you did, what happened, and what you expected. If a report
is sensitive, say so and we will agree to a disclosure timeline before anything is
published.

Please do not open a public issue for a vulnerability. Pull requests are not
accepted here (see `TRADEMARKS.md`), so a report is the contribution.

## The thing you most need to know

**The graph is trusted input by construction.**

Kenning Encounter does not merely store content — it *injects* it. The infusion
path selects held material and places it into the model's context ahead of a
decision. Anything written to the graph is therefore reachable by the model as
though it were context the operator supplied.

The consequence follows directly: **a substrate that an untrusted party can
write to is a prompt-injection surface.** Text placed into a node's name or
description can be selected by the matcher and delivered into a prompt. No
sanitisation of natural-language content is performed, and none is possible
without destroying the thing the substrate is for.

This is a design property, not a defect, and it dictates deployment:

- Treat write access to the database as equivalent to write access to the
  model's prompt. Grant it to exactly the agent whose substrate it is.
- Partition along **trust domains**, not projects. One substrate per trust
  boundary is the intended shape; a graph shared across boundaries commingles
  material that should not meet.
- Do not expose the MCP endpoint to an untrusted network. It is designed to sit
  beside its agent, not in front of the internet.
- Treat a restored or imported graph as untrusted until you know its origin.
  Provenance is recorded (`RECORDED`, `CONSULTED`, `trace_provenance`), so it can
  be inspected — but only if someone inspects it.

## What the software does guard

- **Read tools cannot write.** `read_cypher` is gated by `EXPLAIN`: any query
  whose plan reports a write operation is refused before execution.
- **The process layer is closed to generic CRUD.** Encounters and the chain
  edges between them are written only by the guarded advance path and can never
  be deleted through the entity tools.
- **Writes are scoped to the calling session.** A parallel client cannot seal
  another's encounter or adopt its nodes.
- **Output is sanitised and bounded.** Embedding-shaped values and very long
  lists are stripped from results; read output is truncated at 200 KB.

None of these guard against the injection surface above. They guard the
database's integrity, not the meaning of what is in it.

## Dependencies and secrets

Database credentials are supplied by environment or CLI and are never written to
the graph. If a deployment passes an API key for the meaning matcher, that key
lives in the process environment and is not persisted. Report any place you find
either in stored data — that would be a real defect.

## Supported versions

The most recent release only. This project is pre-1.0; fixes land in a new
version rather than as patches to a sealed one.

---

Kenning AI LLC · https://kenningai.com · contact@kenningai.com
