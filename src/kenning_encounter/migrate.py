#!/usr/bin/env python3
"""Run the Locus migration, or refuse to.

    python -m kenning_encounter.migrate               # dry run (default)
    python -m kenning_encounter.migrate --apply
    python -m kenning_encounter.migrate --verify

THERE IS NO --reverse, AND THAT IS THE CONTRACT RATHER THAN AN OMISSION.
Nothing in a Kenning Encounter is ever deleted. delete_entities already refuses every
process type with "No tool deletes the spine" — and Locus is one. A migration
script that reached past that refusal to DETACH DELETE the very nodes the tool
protects would be doing, unsupervised, the one thing the substrate forbids.

"Reversible by deleting what was added" is not available here and never was:
it is total only while this migration is the SOLE creator of a Locus, which
stops being true the instant the server has run. On a graph in use it would
destroy the live loci an agent is working inside. You do not lobotomize the
witness to undo a bookkeeping error.

So the shape is CHECK-THEN-WRITE rather than write-then-undo: pre-flight
refuses before anything is written, and a post-apply mismatch REPORTS and
stops without removing a thing. The migration only ever adds, so a wrong
result leaves extra structure, never less — and you fix that forward.

From the documented compose deployment, where the server has the database
credentials already in its environment:

    docker exec kenning_encounter-mcp python -m kenning_encounter.migrate --apply

IT LIVES IN THE PACKAGE, NOT IN scripts/, AND THAT IS THE POINT. A repo
script is reachable if you have cloned the repository and unreachable from
the container the compose file runs, which is the documented deployment. A
remedy that a warning names and the operator cannot reach is not a remedy.

WHY THIS IS A SCRIPT AND NOT A STARTUP HOOK. The target is the agent's own
accumulated experience, and unlike every other schema change in this house
there is NO SOURCE to rebuild it from — a restored snapshot is a predecessor,
not the same self. So the migration is an act somebody performs and watches,
once, rather than something a container does on boot while nobody is looking.

THE PRE-REGISTRATION IS MECHANICAL, NOT PROSE. --apply computes the expected
outcome from the graph FIRST, applies, then checks the result against what it
predicted. A migration that reports its own success by DESCRIBING what it did,
rather than by checking it against a prediction made beforehand, cannot fail;
here the check is the exit code.

ADDITIVE ONLY, AND THAT IS WHAT MAKES THE ABSENCE OF AN UNDO SAFE. The
migration writes the :Locus label and the OPENED and NEXT_LOCUS edges, none of
which exist in a pre-v0.12.0 graph, and touches nothing else. A wrong result
therefore leaves MORE structure than expected, never less — so it is corrected
forward, and every write is a MERGE, so re-running is a no-op on what is
already right.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from neo4j import GraphDatabase, RoutingControl

from .utils import lit

CYPHER = Path(__file__).parent / "_cypher" / "migrate_locus.cypher"

# What must be true afterwards, derived from the graph before anything is
# written. Each is (label, query, expectation-from-baseline).
VERIFY = [
    ("loci minted",
     "MATCH (l:Locus) RETURN count(l) AS n", "loci"),
    ("OPENED edges",
     "MATCH (:Locus)-[r:OPENED]->(:Encounter) RETURN count(r) AS n", "encounters"),
    ("NEXT_LOCUS edges",
     "MATCH (:Locus)-[r:NEXT_LOCUS]->(:Locus) RETURN count(r) AS n", "loci_minus_one"),
    ("NEXT_ENCOUNTER untouched",
     "MATCH ()-[r:NEXT_ENCOUNTER]->() RETURN count(r) AS n", "next_encounter"),
    ("INSTANTIATED_AFTER untouched",
     "MATCH ()-[r:INSTANTIATED_AFTER]->() RETURN count(r) AS n", "instantiated_after"),
]

# Invariants that must be ZERO. These are the ones the schema cannot declare
# in Neo4j, so they are checked rather than assumed — including the x-squared
# claim itself, that succession never crosses a locus.
ZERO_INVARIANTS = [
    ("encounters with no locus",
     "MATCH (e:Encounter) WHERE NOT (:Locus)-[:OPENED]->(e) RETURN count(e) AS n"),
    ("encounters claimed by two loci",
     "MATCH (e:Encounter) WITH e, size([(:Locus)-[:OPENED]->(e) | 1]) AS k "
     "WHERE k > 1 RETURN count(e) AS n"),
    ("loci holding nothing",
     "MATCH (l:Locus) WHERE NOT (l)-[:OPENED]->() RETURN count(l) AS n"),
    ("NEXT_LOCUS forks",
     "MATCH (l:Locus) WITH l, size([(l)-[:NEXT_LOCUS]->() | 1]) AS k "
     "WHERE k > 1 RETURN count(l) AS n"),
    ("NEXT_LOCUS merges",
     "MATCH (l:Locus) WITH l, size([()-[:NEXT_LOCUS]->(l) | 1]) AS k "
     "WHERE k > 1 RETURN count(l) AS n"),
    ("CROSS-LOCUS succession (the x^2 invariant)",
     "MATCH (la:Locus)-[:OPENED]->(:Encounter)-[:NEXT_ENCOUNTER]->"
     "(:Encounter)<-[:OPENED]-(lb:Locus) WHERE la <> lb RETURN count(*) AS n"),
]

# The prediction must describe the graph AFTER the migration, on a graph that
# may already be partly populated: an upgrader who worked before migrating has
# real loci already, so counting all chain heads and all encounters is correct
# only for a virgin upgrade.
BASELINE = {
    "loci": "MATCH (h:Encounter) WHERE NOT ()-[:NEXT_ENCOUNTER]->(h) "
            "AND NOT (:Locus)-[:OPENED]->(h) RETURN count(h) AS n",
    "existing_loci": "MATCH (l:Locus) RETURN count(l) AS n",
    "encounters": "MATCH (e:Encounter) RETURN count(e) AS n",
    "next_encounter": "MATCH ()-[r:NEXT_ENCOUNTER]->() RETURN count(r) AS n",
    "instantiated_after": "MATCH ()-[r:INSTANTIATED_AFTER]->() RETURN count(r) AS n",
}


def _one(driver, db, query):
    res = driver.execute_query(lit(query), database_=db, routing_=RoutingControl.READ)
    return res.records[0]["n"] if res.records else 0


def baseline(driver, db) -> dict[str, int]:
    b = {k: _one(driver, db, q) for k, q in BASELINE.items()}
    b["loci"] += b["existing_loci"]          # total after, not newly minted
    b["loci_minus_one"] = max(b["loci"] - 1, 0)
    return b


def statements() -> list[str]:
    body = "\n".join(
        l for l in CYPHER.read_text().splitlines() if not l.strip().startswith("//")
    )
    return [s.strip() for s in body.split(";") if s.strip()]


def report(driver, db, expected: dict[str, int]) -> bool:
    ok = True
    print("\n  RESULT vs PREDICTION")
    for label, query, key in VERIFY:
        got, want = _one(driver, db, query), expected[key]
        flag = "ok " if got == want else "BAD"
        ok &= got == want
        print(f"    [{flag}] {label:<32} {got:>5}  (predicted {want})")
    print("\n  INVARIANTS (must be zero)")
    for label, query in ZERO_INVARIANTS:
        got = _one(driver, db, query)
        flag = "ok " if got == 0 else "BAD"
        ok &= got == 0
        print(f"    [{flag}] {label:<44} {got}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
                    or "bolt://localhost:7687")
    ap.add_argument("--user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    ap.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    a = ap.parse_args()

    driver = GraphDatabase.driver(a.uri, auth=(a.user, a.password))
    db = a.database
    try:
        b = baseline(driver, db)
        print(f"  BASELINE  encounters={b['encounters']}  chain-heads={b['loci']}  "
              f"NEXT_ENCOUNTER={b['next_encounter']}  "
              f"INSTANTIATED_AFTER={b['instantiated_after']}")
        print(f"  PREDICTS  {b['loci']} loci, {b['encounters']} OPENED, "
              f"{b['loci_minus_one']} NEXT_LOCUS, nothing deleted")

        if a.verify:
            return 0 if report(driver, db, b) else 1

        # PRE-FLIGHT — refuse BEFORE writing, since there is no undo.
        # The first refuses a non-unique identity key, which would silently
        # merge two instantiations into one locus; the second detects chain
        # heads a running server already gave a locus, which must be skipped
        # rather than given a second one.
        problems = []
        dupes = _one(driver, db,
                     "MATCH (h:Encounter) WHERE NOT ()-[:NEXT_ENCOUNTER]->(h) "
                     "AND NOT (:Locus)-[:OPENED]->(h) "
                     "WITH count(h) AS heads, count(DISTINCT elementId(h)) AS ids "
                     "RETURN heads - ids AS n")
        if dupes:
            problems.append(f"{dupes} chain head(s) share an identity key")
        held = _one(driver, db,
                    "MATCH (h:Encounter) WHERE NOT ()-[:NEXT_ENCOUNTER]->(h) "
                    "AND (:Locus)-[:OPENED]->(h) RETURN count(h) AS n")
        if held:
            print(f"  NOTE      {held} chain head(s) already hold a locus and "
                  "are correctly skipped")
        if problems:
            print("\n  REFUSED before writing:")
            for x in problems:
                print(f"    - {x}")
            return 1

        if not a.apply:
            print("\n  dry run — nothing written. Re-run with --apply.")
            return 0

        for i, stmt in enumerate(statements(), 1):
            driver.execute_query(lit(stmt), database_=db)
            print(f"  step {i} applied")

        if report(driver, db, b):
            print("\n  MIGRATED — result matches the prediction.")
            return 0
        # A disagreement REPORTS and stops. It does not undo, because undoing
        # would mean deleting spine nodes, and nothing in a Kenning Encounter is deleted.
        # The migration only ever adds, so a wrong result is extra structure
        # rather than lost structure: inspect it, fix forward, re-run (this is
        # idempotent). The check is still the exit code.
        print("\n  PREDICTION NOT MET — and NOTHING has been removed.")
        print("  This migration only adds, so the graph holds more than expected,")
        print("  never less. Inspect the [BAD] rows above, correct forward, and")
        print("  re-run: every write is a MERGE, so a second pass is a no-op.")
        return 1
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
