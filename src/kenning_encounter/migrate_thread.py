#!/usr/bin/env python3
"""Thread the history: give every past encounter its tick, or refuse to.

    python -m kenning_encounter.migrate_thread               # dry run (default)
    python -m kenning_encounter.migrate_thread --apply
    python -m kenning_encounter.migrate_thread --verify

From the documented compose deployment:

    docker exec kenning_encounter-mcp python -m kenning_encounter.migrate_thread --apply

WHAT IT WRITES. Once the thread exists, every opening takes a tick inside
the advance lock and is linked to the previous opening by NEXT_TICK, whatever
its locus. Openings made before then took the same lock, but nothing wrote
their order down. This rebuilds that order from the stamps (t_exist) and writes it the
same way, with every rebuilt NEXT_TICK edge marked rebuilt_from = 't_exist'
so it stays distinguishable from the edges the lock writes. Seals carrying
t_sealed get a seal_tick marked seal_tick_rebuilt_from = 't_sealed'. Seals
written before t_sealed existed carry no stamp, and they get no seal tick: no
time is fabricated for them.

STRUCTURE WINS OVER STAMP. The order is a topological sort with the stamps as
the tie-break, under two constraints the graph already holds as lived
structure: each locus's NEXT_ENCOUNTER chain, and the NEXT_LOCUS order of
first encounters. Where a stamp disagrees with either, the structure decides,
and the disagreement is listed rather than counted.

THE GATE. Restricted to each locus the thread must reproduce NEXT_ENCOUNTER,
and restricted to first encounters it must reproduce NEXT_LOCUS. The gate is
run against two synthetic graphs before it is run against the real one: it
must refuse the pure stamp order of a graph with one inverted stamp, and pass
the structural order of the same graph. A gate that has only ever said yes has
not been shown to work, so if either check fails, nothing is read or written.

ONE TRANSACTION, INSIDE THE LOCK. The first statement takes the same lock
every advance takes, so no encounter can open while the history is threaded,
and the read, the gate, the write and the re-check against the prediction all
happen in that transaction. A failed re-check rolls the whole thing back.
Unlike the Locus migration, a refusal here leaves nothing behind at all.

ADDITIVE ONLY, as every migration of a trajectory's own record must be: it
writes tick, seal_tick and NEXT_TICK, which no unthreaded graph holds, and
touches nothing else. Nothing in this substrate is deleted.
"""
from __future__ import annotations

import argparse
import heapq
import os
import sys
from typing import Any

from neo4j import GraphDatabase

# Sort key of a stamp: (epochSeconds, nanosecond). Neither ISO strings (which
# drop trailing zeros) nor epochMillis (which can tie) order reliably.
Stamp = tuple[int, int]

LOCK = (
    "OPTIONAL MATCH (genesis:Locus) WHERE NOT (:Locus)-[:NEXT_LOCUS]->(genesis) "
    "SET genesis.t_exist = genesis.t_exist RETURN count(genesis) AS n"
)
READ_ENCOUNTERS = (
    "MATCH (e:Encounter) OPTIONAL MATCH (l:Locus)-[:OPENED]->(e) "
    "RETURN elementId(e) AS eid, elementId(l) AS locus, e.tick AS tick, "
    "       [e.t_exist.epochSeconds, e.t_exist.nanosecond] AS t, "
    "       CASE WHEN e.t_sealed IS NULL THEN NULL "
    "            ELSE [e.t_sealed.epochSeconds, e.t_sealed.nanosecond] END AS ts"
)
READ_NEXT_ENCOUNTER = (
    "MATCH (a:Encounter)-[:NEXT_ENCOUNTER]->(b:Encounter) "
    "RETURN elementId(a) AS a, elementId(b) AS b"
)
READ_NEXT_LOCUS = (
    "MATCH (a:Locus)-[:NEXT_LOCUS]->(b:Locus) RETURN elementId(a) AS a, elementId(b) AS b"
)
READ_THREAD = (
    "MATCH (e:Encounter) RETURN elementId(e) AS eid, e.tick AS tick, "
    "e.seal_tick AS seal_tick ORDER BY e.tick"
)
WRITE_TICKS = (
    "UNWIND $rows AS row MATCH (e:Encounter) WHERE elementId(e) = row.eid "
    "SET e.tick = row.tick"
)
WRITE_EDGES = (
    "UNWIND $pairs AS p "
    "MATCH (a:Encounter) WHERE elementId(a) = p[0] "
    "MATCH (b:Encounter) WHERE elementId(b) = p[1] "
    "MERGE (a)-[k:NEXT_TICK]->(b) "
    "ON CREATE SET k.t_created = datetime(), k.rebuilt_from = 't_exist'"
)
WRITE_SEALS = (
    "UNWIND $rows AS row MATCH (e:Encounter) WHERE elementId(e) = row.eid "
    "SET e.seal_tick = row.seal_tick, e.seal_tick_rebuilt_from = 't_sealed'"
)
ZERO_INVARIANTS = [
    ("NEXT_TICK forks",
     "MATCH (e:Encounter) WHERE size([(e)-[:NEXT_TICK]->() | 1]) > 1 RETURN count(e) AS n"),
    ("NEXT_TICK merges",
     "MATCH (e:Encounter) WHERE size([()-[:NEXT_TICK]->(e) | 1]) > 1 RETURN count(e) AS n"),
    ("NEXT_TICK against tick order",
     "MATCH (a:Encounter)-[:NEXT_TICK]->(b:Encounter) WHERE b.tick <> a.tick + 1 "
     "RETURN count(*) AS n"),
    ("encounters without a tick",
     "MATCH (e:Encounter) WHERE e.tick IS NULL RETURN count(e) AS n"),
    ("seals ticked before their own opening",
     "MATCH (e:Encounter) WHERE e.seal_tick < e.tick RETURN count(e) AS n"),
]


class Refused(Exception):
    """The migration declined. Raised inside the transaction, so it rolls back."""


# -- Pure: the order and the gate ---------------------------------------------

def _first_of_locus(encs: dict[str, dict[str, Any]], next_enc: list[tuple[str, str]]) -> dict[str, str]:
    """locus -> its first encounter (the one with no incoming NEXT_ENCOUNTER)."""
    has_pred = {b for _, b in next_enc}
    firsts: dict[str, str] = {}
    for eid, e in encs.items():
        if eid not in has_pred:
            if e["locus"] in firsts:
                raise Refused(f"locus {e['locus']} has two first encounters")
            firsts[e["locus"]] = eid
    return firsts


def _locus_chain(next_locus: list[tuple[str, str]], loci: set[str]) -> list[str]:
    """The NEXT_LOCUS chain from its genesis, which must cover every locus."""
    succ = dict(next_locus)
    heads = loci - {b for _, b in next_locus}
    if len(succ) != len(next_locus) or len(heads) != 1:
        raise Refused(f"NEXT_LOCUS is not one chain ({len(heads)} heads)")
    chain = [heads.pop()]
    while chain[-1] in succ:
        chain.append(succ[chain[-1]])
    if set(chain) != loci:
        raise Refused("NEXT_LOCUS does not reach every locus")
    return chain


def constraints(
    encs: dict[str, dict[str, Any]],
    next_enc: list[tuple[str, str]],
    next_locus: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """The lived order the thread must respect, as (earlier, later) pairs."""
    firsts = _first_of_locus(encs, next_enc)
    chain = _locus_chain(next_locus, set(firsts))
    return list(next_enc) + [
        (firsts[a], firsts[b]) for a, b in zip(chain, chain[1:])
    ]


def stamp_order(encs: dict[str, dict[str, Any]]) -> list[str]:
    """Encounters by t_exist alone. Ties broken by elementId, deterministically."""
    return sorted(encs, key=lambda eid: (tuple(encs[eid]["t"]), eid))


def thread_order(
    encs: dict[str, dict[str, Any]],
    next_enc: list[tuple[str, str]],
    next_locus: list[tuple[str, str]],
) -> list[str]:
    """A topological order over the lived constraints, stamps as tie-break.
    Structure wins: a stamp that disagrees with a constraint is overridden."""
    edges = constraints(encs, next_enc, next_locus)
    succs: dict[str, list[str]] = {e: [] for e in encs}
    indeg = {e: 0 for e in encs}
    for a, b in edges:
        succs[a].append(b)
        indeg[b] += 1
    key = lambda eid: (tuple(encs[eid]["t"]), eid)
    ready = [(key(e), e) for e, d in indeg.items() if d == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        _, e = heapq.heappop(ready)
        order.append(e)
        for s in succs[e]:
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(ready, (key(s), s))
    if len(order) != len(encs):
        raise Refused("the lived constraints contain a cycle")
    return order


def gate(
    order: list[str],
    encs: dict[str, dict[str, Any]],
    next_enc: list[tuple[str, str]],
    next_locus: list[tuple[str, str]],
) -> list[str]:
    """Every way `order` disagrees with the lived structure. Empty = passes.

    Restricted to each locus the thread must reproduce NEXT_ENCOUNTER, and
    restricted to first encounters it must reproduce NEXT_LOCUS."""
    problems: list[str] = []
    if sorted(order) != sorted(encs):
        problems.append("the thread is not exactly one tick per encounter")
        return problems
    pos = {e: i for i, e in enumerate(order)}
    succ = dict(next_enc)
    firsts = _first_of_locus(encs, next_enc)
    for locus, head in firsts.items():
        chain = [head]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        members = sorted((e for e in encs if encs[e]["locus"] == locus), key=lambda e: pos[e])
        if members != chain:
            problems.append(f"locus {locus}: thread does not reproduce NEXT_ENCOUNTER")
    locus_chain = _locus_chain(next_locus, set(firsts))
    by_genesis = sorted(firsts, key=lambda l: pos[firsts[l]])
    if by_genesis != locus_chain:
        problems.append("first encounters: thread does not reproduce NEXT_LOCUS")
    return problems


def seal_ticks(order: list[str], encs: dict[str, dict[str, Any]]) -> dict[str, int]:
    """For each stamped seal, the latest tick whose opening came at or before
    it. Never earlier than the encounter's own tick. Unstamped seals: absent."""
    out: dict[str, int] = {}
    for eid, e in encs.items():
        if e.get("ts") is None:
            continue
        ts = tuple(e["ts"])
        latest = max(i for i, x in enumerate(order) if tuple(encs[x]["t"]) <= ts or x == eid)
        out[eid] = max(latest, order.index(eid))
    return out


def overrides(order: list[str], encs: dict[str, dict[str, Any]]) -> list[str]:
    """Encounters the structure placed somewhere other than their stamp did."""
    stamped = stamp_order(encs)
    return [e for e, s in zip(order, stamped) if e != s]


def synthetic() -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Two loci, overlapping. A1 -> A2 within A; B1 opens between them.
    A2's stamp is inverted: it claims to open before A1."""
    encs = {
        "A1": {"locus": "LA", "t": [10, 0], "ts": None},
        "B1": {"locus": "LB", "t": [20, 0], "ts": [40, 0]},
        "A2": {"locus": "LA", "t": [5, 0], "ts": [30, 0]},
    }
    return encs, [("A1", "A2")], [("LA", "LB")]


def prove_gate() -> None:
    """The gate must say both no and yes before its verdict means anything."""
    encs, ne, nl = synthetic()
    if not gate(stamp_order(encs), encs, ne, nl):
        raise Refused("the gate passed a stamp order with an inverted stamp")
    if gate(thread_order(encs, ne, nl), encs, ne, nl):
        raise Refused("the gate refused a structurally correct order")


# -- The transaction ------------------------------------------------------------

def _read(tx) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]], list[tuple[str, str]]]:
    encs = {r["eid"]: dict(r) for r in tx.run(READ_ENCOUNTERS)}
    ne = [(r["a"], r["b"]) for r in tx.run(READ_NEXT_ENCOUNTER)]
    nl = [(r["a"], r["b"]) for r in tx.run(READ_NEXT_LOCUS)]
    return encs, ne, nl


def plan(tx, lock: bool) -> dict[str, Any]:
    if lock:
        tx.run(LOCK).consume()
    encs, ne, nl = _read(tx)
    if any(e["locus"] is None for e in encs.values()):
        raise Refused("encounters without a locus: run python -m kenning_encounter.migrate --apply first")
    ticked = [e for e in encs.values() if e["tick"] is not None]
    if ticked and len(ticked) == len(encs):
        return {"already": True, "encounters": len(encs)}
    if ticked:
        raise Refused(f"{len(ticked)} of {len(encs)} encounters already carry a tick; "
                      "a partial thread is not something this migration extends")
    order = thread_order(encs, ne, nl)
    problems = gate(order, encs, ne, nl)
    if problems:
        raise Refused("; ".join(problems))
    return {
        "already": False, "encs": encs, "order": order,
        "seals": seal_ticks(order, encs), "overrides": overrides(order, encs),
    }


def apply(tx) -> dict[str, Any]:
    p = plan(tx, lock=True)
    if p["already"]:
        return p
    order, encs = p["order"], p["encs"]
    tx.run(WRITE_TICKS, rows=[{"eid": e, "tick": i} for i, e in enumerate(order)]).consume()
    tx.run(WRITE_EDGES, pairs=[[a, b] for a, b in zip(order, order[1:])]).consume()
    tx.run(WRITE_SEALS, rows=[{"eid": e, "seal_tick": t} for e, t in p["seals"].items()]).consume()
    # Re-check against the prediction, inside the same transaction.
    got = [r["eid"] for r in tx.run(READ_THREAD)]
    if got != order:
        raise Refused("the written ticks do not reproduce the planned order")
    _, ne, nl = _read(tx)
    problems = gate(got, encs, ne, nl)
    for label, q in ZERO_INVARIANTS:
        n = tx.run(q).single()["n"]
        if n:
            problems.append(f"{label}: {n}")
    edges = tx.run("MATCH ()-[k:NEXT_TICK]->() RETURN count(k) AS n").single()["n"]
    if edges != len(order) - 1:
        problems.append(f"NEXT_TICK edges {edges}, predicted {len(order) - 1}")
    if problems:
        raise Refused("after writing: " + "; ".join(problems))
    return p


def verify(tx) -> list[str]:
    problems = []
    for label, q in ZERO_INVARIANTS:
        n = tx.run(q).single()["n"]
        if n:
            problems.append(f"{label}: {n}")
    encs, ne, nl = _read(tx)
    order = [r["eid"] for r in tx.run(READ_THREAD)]
    return problems + gate(order, encs, ne, nl)


def _summary(p: dict[str, Any]) -> None:
    encs, order = p["encs"], p["order"]
    sealed = sum(1 for e in encs.values() if e.get("ts") is not None)
    print(f"  encounters        {len(encs)}  -> ticks 0..{len(order) - 1}")
    print(f"  NEXT_TICK         {max(len(order) - 1, 0)} edges, each rebuilt_from 't_exist'")
    print(f"  seal ticks        {sealed} (stamped seals only)")
    print(f"  stamp overridden  {len(p['overrides'])} placement(s)")
    for eid in p["overrides"]:
        print(f"    - {eid}")


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

    try:
        prove_gate()
    except Refused as e:
        print(f"  REFUSED: {e}. Nothing was read or written.")
        return 1
    print("  gate proven two-sided on a synthetic graph (refuses an inverted stamp, passes the fix)")

    driver = GraphDatabase.driver(a.uri, auth=(a.user, a.password))
    try:
        with driver.session(database=a.database) as s:
            if a.verify:
                problems = s.execute_read(verify)
                for x in problems:
                    print(f"  [BAD] {x}")
                print("  VERIFIED" if not problems else "  NOT VERIFIED")
                return 1 if problems else 0
            try:
                p = s.execute_write(apply) if a.apply else s.execute_read(plan, False)
            except Refused as e:
                print(f"  REFUSED: {e}. Nothing was written.")
                return 1
            if p["already"]:
                print(f"  already threaded: all {p['encounters']} encounters carry a tick")
                return 0
            _summary(p)
            if not a.apply:
                print("\n  dry run: nothing written. Re-run with --apply.")
            else:
                print("\n  THREADED: the result matches the prediction, in one transaction.")
            return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
