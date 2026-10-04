"""Against a real Neo4j: the thread, the lock, and the migration.

The unit suite mocks the driver, so a defect living in a .cypher file is
invisible to it. This suite runs the real cypher. It is skipped unless a
THROWAWAY database is named, because every test wipes it:

    NEO4J_TEST_URL=bolt://localhost:7699 NEO4J_TEST_PASSWORD=... \\
        uv run pytest tests/integration

Never point it at a trajectory's graph.
"""
from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase, GraphDatabase

from kenning_encounter import migrate_thread as mt
from kenning_encounter.kenning_encounter import Neo4jKenningEncounter
from kenning_encounter.utils import load_cypher

URL = os.getenv("NEO4J_TEST_URL")
AUTH = ("neo4j", os.getenv("NEO4J_TEST_PASSWORD", ""))

pytestmark = [
    pytest.mark.skipif(not URL, reason="NEO4J_TEST_URL not set (throwaway DB only)"),
    pytest.mark.asyncio,
]


@pytest_asyncio.fixture
async def kenning_encounter():
    driver = AsyncGraphDatabase.driver(URL, auth=AUTH)
    await driver.execute_query("MATCH (n) DETACH DELETE n")
    e = Neo4jKenningEncounter(driver)
    await e.create_indexes()
    await e.create_fulltext_index()
    yield e
    await driver.close()


async def thread_problems(driver) -> list[str]:
    """Everything that would make the thread not one chain. Empty = linear."""
    checks = {
        "fork": "MATCH (e:Encounter) WHERE size([(e)-[:NEXT_TICK]->() | 1]) > 1 RETURN count(e) AS n",
        "merge": "MATCH (e:Encounter) WHERE size([()-[:NEXT_TICK]->(e) | 1]) > 1 RETURN count(e) AS n",
        "unticked": "MATCH (e:Encounter) WHERE e.tick IS NULL RETURN count(e) AS n",
        "duplicate tick": "MATCH (e:Encounter) WITH e.tick AS t, count(*) AS k WHERE k > 1 RETURN count(t) AS n",
        "edge off by one": "MATCH (a)-[:NEXT_TICK]->(b) WHERE b.tick <> a.tick + 1 RETURN count(*) AS n",
    }
    out = []
    for label, q in checks.items():
        n = (await driver.execute_query(q)).records[0]["n"]
        if n:
            out.append(f"{label}: {n}")
    return out


# -- the detector says both yes and no -------------------------------------------

async def test_the_detector_flags_a_fork_and_passes_a_chain(kenning_encounter):
    d = kenning_encounter.driver
    await d.execute_query(
        "CREATE (a:Encounter {name:'a', tick:0})-[:NEXT_TICK]->(:Encounter {name:'b', tick:1})")
    assert await thread_problems(d) == []
    await d.execute_query(
        "MATCH (a:Encounter {name:'a'}) CREATE (a)-[:NEXT_TICK]->(:Encounter {name:'c', tick:1})")
    assert any(p.startswith("fork") for p in await thread_problems(d))


# -- the thread ------------------------------------------------------------------

async def test_openings_thread_across_loci(kenning_encounter):
    a1 = await kenning_encounter.advance_encounter("A1", session_id="A")
    b1 = await kenning_encounter.advance_encounter("B1", session_id="B")
    a2 = await kenning_encounter.advance_encounter("A2", session_id="A")
    ticks = [x["encounter"]["tick"] for x in (a1, b1, a2)]
    assert ticks == [0, 1, 2]
    rows = (await kenning_encounter.driver.execute_query(
        "MATCH (x)-[:NEXT_TICK]->(y) RETURN x.name AS a, y.name AS b ORDER BY x.tick")).records
    assert [(r["a"], r["b"]) for r in rows] == [("A1", "B1"), ("B1", "A2")]
    # NEXT_ENCOUNTER still never crosses a locus.
    ne = (await kenning_encounter.driver.execute_query(
        "MATCH (a)-[:NEXT_ENCOUNTER]->(b) RETURN a.name AS a, b.name AS b")).records
    assert [(r["a"], r["b"]) for r in ne] == [("A1", "A2")]
    assert await thread_problems(kenning_encounter.driver) == []


async def test_the_seal_records_the_latest_tick(kenning_encounter):
    await kenning_encounter.advance_encounter("A1", session_id="A")
    await kenning_encounter.advance_encounter("B1", session_id="B")
    await kenning_encounter.advance_encounter("C1", session_id="C")
    sealed = await kenning_encounter.close_encounter(summary="s", session_id="A")
    assert sealed["seal_tick"] == 2


async def test_an_unthreaded_history_is_not_ticked_over(kenning_encounter):
    """New openings wait for the migration rather than start a second thread."""
    await kenning_encounter.driver.execute_query(
        "CREATE (:Locus {name:'Locus old', t_exist: datetime()})-[:OPENED]->"
        "(:Encounter {name:'old', t_exist: datetime()})")
    out = await kenning_encounter.advance_encounter("new", session_id="X")
    assert out["encounter"]["tick"] is None
    assert out["reentry"]["thread_unmigrated"]["encounters_without_a_tick"] == 2


# -- the lock: two-sided ---------------------------------------------------------

async def _hold_lock_then_advance(kenning_encounter, n: int) -> tuple[int, list[str]]:
    """Hold the genesis lock in an open transaction, start n advances, count
    how many committed while it was held, release, and wait for the rest."""
    await kenning_encounter.advance_encounter("genesis", session_id="G")
    session = kenning_encounter.driver.session()
    tx = await session.begin_transaction()
    await tx.run("MATCH (g:Locus) WHERE NOT (:Locus)-[:NEXT_LOCUS]->(g) "
                 "SET g.t_exist = g.t_exist")
    tasks = [asyncio.create_task(
        kenning_encounter.advance_encounter(f"E{i}", session_id=f"S{i}")) for i in range(n)]
    await asyncio.sleep(1.5)
    during = sum(t.done() for t in tasks)
    await tx.commit()
    await session.close()
    await asyncio.gather(*tasks)
    return during, await thread_problems(kenning_encounter.driver)


async def test_held_lock_blocks_every_advance_and_the_thread_stays_linear(kenning_encounter):
    during, problems = await _hold_lock_then_advance(kenning_encounter, 8)
    assert during == 0, "an advance committed while the spine lock was held"
    assert problems == []
    n = (await kenning_encounter.driver.execute_query("MATCH (e:Encounter) RETURN count(e) AS n")).records[0]["n"]
    assert n == 9


async def _hold_tail_then_advance(kenning_encounter, n: int) -> list[str]:
    """Hold a write lock on the thread's TAIL encounter while n advances start.
    Every advance that gets past step 1 reads that tail and then waits to
    write NEXT_TICK from it. Deterministic, unlike a bare race.

    The sessions are opened once BEFORE the hold, so the contended advances
    mint no locus. A mint writes NEXT_LOCUS from the tail locus, and that
    write serializes on its own; only advances in existing loci share
    nothing but the thread, which is the ordinary case and the one the lock
    is for."""
    await kenning_encounter.advance_encounter("genesis", session_id="G")
    for i in range(n):
        await kenning_encounter.advance_encounter(f"first-{i}", session_id=f"S{i}")
    session = kenning_encounter.driver.session()
    tx = await session.begin_transaction()
    await tx.run("MATCH (t:Encounter) WHERE NOT (t)-[:NEXT_TICK]->() SET t.tick = t.tick")
    tasks = [asyncio.create_task(
        kenning_encounter.advance_encounter(f"E{i}", session_id=f"S{i}")) for i in range(n)]
    await asyncio.sleep(1.5)
    await tx.commit()
    await session.close()
    await asyncio.gather(*tasks)
    return await thread_problems(kenning_encounter.driver)


async def test_with_the_lock_a_held_tail_still_yields_one_thread(kenning_encounter):
    assert await _hold_tail_then_advance(kenning_encounter, 6) == []
    n = (await kenning_encounter.driver.execute_query("MATCH (e:Encounter) RETURN count(e) AS n")).records[0]["n"]
    assert n == 13


async def test_without_the_lock_the_same_load_forks(kenning_encounter, monkeypatch):
    """The other side, under the identical load: strip step 1 and the thread
    forks. If this ever passes linear, the test above proves nothing."""
    real = load_cypher("encounter_advance")
    lockless = real.replace("SET genesis.t_exist = genesis.t_exist", "")
    assert lockless != real
    import kenning_encounter.kenning_encounter as mod
    monkeypatch.setattr(mod, "load_cypher",
                        lambda name, **k: lockless if name == "encounter_advance" else load_cypher(name, **k))
    problems = await _hold_tail_then_advance(kenning_encounter, 6)
    assert any(p.startswith("fork") for p in problems), problems


# -- item 1: atomic batches and honest anchors ------------------------------------

async def test_a_failing_batch_writes_nothing(kenning_encounter):
    await kenning_encounter.advance_encounter("E", session_id="A")
    with pytest.raises(ValueError, match="Nothing was written"):
        await kenning_encounter.create_entities([
            {"type": "Observation", "name": "o", "description": "d", "t_observed": "2026-10-04T00:00:00Z"},
            {"type": "Question", "name": "q", "description": "d", "priority": None},
        ], session_id="A")
    n = (await kenning_encounter.driver.execute_query("MATCH (o:Observation) RETURN count(o) AS n")).records[0]["n"]
    assert n == 0


async def test_a_merge_hit_reports_its_original_anchor(kenning_encounter):
    await kenning_encounter.advance_encounter("E1", session_id="A")
    note = {"type": "Note", "name": "n", "description": "d"}
    first = await kenning_encounter.create_entities([note], session_id="A")
    await kenning_encounter.close_encounter(summary="s", session_id="A")
    await kenning_encounter.advance_encounter("E2", session_id="A")
    again = await kenning_encounter.create_entities([note], session_id="A")
    assert first[0]["anchored_to"] == "E1"
    assert again[0] == {**again[0], "created": False, "anchored_to": "E1"}


# -- item 2: REBINDS -------------------------------------------------------------

async def test_only_the_head_of_a_rebinding_chain_is_frontier(kenning_encounter):
    await kenning_encounter.advance_encounter("E", session_id="A")
    await kenning_encounter.create_entities([
        {"type": "Question", "name": "q-earlier", "description": "d", "status": "open"},
        {"type": "Question", "name": "q-later", "description": "d", "status": "open"},
    ], session_id="A")
    with pytest.raises(ValueError, match="why"):
        await kenning_encounter.create_relations([{"source": "q-later", "target": "q-earlier", "type": "REBINDS"}])
    await kenning_encounter.create_relations([{"source": "q-later", "target": "q-earlier", "type": "REBINDS",
                                  "properties": {"why": "the second look asked what a judgment responds to"}}])
    names = [q["name"] for q in (await kenning_encounter._frontier())["unanswered_questions"]]
    assert names == ["q-later"]


# -- only the head of a supersession chain is live ---------------------------------

async def test_a_superseded_claim_leaves_every_directive_read(kenning_encounter):
    await kenning_encounter.advance_encounter("E", session_id="A")
    obs = lambda n: {"type": "Observation", "name": n, "description": "d",
                     "t_observed": "2026-10-04T00:00:00Z"}
    await kenning_encounter.create_entities([
        {"type": "Hypothesis", "name": "h-old", "description": "d", "status": "proposed"},
        {"type": "Hypothesis", "name": "h-new", "description": "d", "status": "proposed"},
        {"type": "Hypothesis", "name": "h-alone", "description": "d", "status": "proposed"},
        {"type": "Hypothesis", "name": "h-retired", "description": "d", "status": "retired"},
        obs("for"), obs("against"),
    ], session_id="A")
    for h in ("h-old", "h-alone", "h-retired"):
        await kenning_encounter.create_relations([
            {"source": "for", "target": h, "type": "SUPPORTS"},
            {"source": "against", "target": h, "type": "CHALLENGES"},
        ])
    contested = lambda f: {r["name"] for r in f["contested_hypotheses"]}

    # Before the supersession: both contested claims are live (the gate can say yes).
    before = await kenning_encounter._frontier()
    assert contested(before) == {"h-old", "h-alone"}

    await kenning_encounter.create_relations([{"source": "h-new", "target": "h-old", "type": "SUPERSEDES",
                                  "properties": {"revision_why": "w"}}])
    after = await kenning_encounter._frontier()
    assert contested(after) == {"h-alone"}
    assert {r["name"] for r in after["untested_hypotheses"]} == {"h-new"}
    conflicts = await kenning_encounter._conflicts_among(["h-old", "h-new", "h-alone", "h-retired"])
    assert {c["to_name"] for c in conflicts if c["kind"] == "challenge"} == {"h-alone"}
    out = await kenning_encounter.advance_encounter("E2", session_id="A")
    live = {h["name"] for h in out["reentry"]["live_hypotheses"]}
    assert "h-old" not in live and {"h-new", "h-alone"} <= live
    # The predecessor's authored status is untouched.
    st = (await kenning_encounter.driver.execute_query(
        "MATCH (h:Hypothesis {name:'h-old'}) RETURN h.status AS s")).records[0]["s"]
    assert st == "proposed"


# -- item 3: the edges around a Concept's name -----------------------------------

async def test_a_concept_arrives_with_its_incoming_edges(kenning_encounter):
    await kenning_encounter.advance_encounter("E", session_id="A")
    await kenning_encounter.create_entities([
        {"type": "Concept", "name": "c-old", "description": "d"},
        {"type": "Concept", "name": "c-new", "description": "d"},
    ], session_id="A")
    await kenning_encounter.create_relations([{"source": "c-new", "target": "c-old", "type": "SUPERSEDES",
                                  "properties": {"revision_why": "w"}}])
    found = await kenning_encounter.find_by_name(["c-old"])
    old = found["entities"][0]
    assert old["edge_count"] == 1
    assert old["edges"] == [{"direction": "in", "rel": "SUPERSEDES", "other": "c-new",
                             "other_type": "Concept", "props": {"revision_why": "w"}}]


# -- item 5: the migration ---------------------------------------------------------

def _legacy_graph(sync_driver) -> None:
    """Two overlapping loci, no ticks. B's genesis falls inside A's lifetime,
    and A2's stamp is INVERTED: it claims to open before A1."""
    sync_driver.execute_query("MATCH (n) DETACH DELETE n")
    sync_driver.execute_query(
        "CREATE (la:Locus {name:'Locus A', t_exist: datetime('2026-01-01T00:00:00Z')}) "
        "CREATE (lb:Locus {name:'Locus B', t_exist: datetime('2026-01-01T00:00:20Z')}) "
        "CREATE (la)-[:NEXT_LOCUS]->(lb) "
        "CREATE (a1:Encounter {name:'A1', t_exist: datetime('2026-01-01T00:00:10Z'), summary:'old'}) "
        "CREATE (a2:Encounter {name:'A2', t_exist: datetime('2026-01-01T00:00:05Z'), "
        "        t_sealed: datetime('2026-01-01T00:00:30Z'), summary:'s'}) "
        "CREATE (b1:Encounter {name:'B1', t_exist: datetime('2026-01-01T00:00:20Z')}) "
        "CREATE (la)-[:OPENED]->(a1) CREATE (la)-[:OPENED]->(a2) CREATE (lb)-[:OPENED]->(b1) "
        "CREATE (a1)-[:NEXT_ENCOUNTER]->(a2)")


async def test_migration_threads_history_with_structure_winning(kenning_encounter):
    sync = GraphDatabase.driver(URL, auth=AUTH)
    try:
        _legacy_graph(sync)
        with sync.session() as s:
            p = s.execute_write(mt.apply)
            assert [r["name"] for r in s.run("MATCH (e:Encounter) RETURN e.name AS name ORDER BY e.tick")] \
                == ["A1", "A2", "B1"]
            assert len(p["overrides"]) == 2           # A1/A2 swapped against their stamps
            seal = s.run("MATCH (e:Encounter {name:'A2'}) RETURN e.seal_tick AS t, "
                         "e.seal_tick_rebuilt_from AS f").single()
            assert (seal["t"], seal["f"]) == (2, "t_sealed")
            # The unstamped older seal gets no tick: nothing fabricated.
            assert s.run("MATCH (e:Encounter {name:'A1'}) RETURN e.seal_tick AS t").single()["t"] is None
            assert {r["f"] for r in s.run("MATCH ()-[k:NEXT_TICK]->() RETURN k.rebuilt_from AS f")} == {"t_exist"}
            assert s.execute_read(mt.verify) == []
            # Idempotent.
            assert s.execute_write(mt.apply)["already"] is True
        # And the next live opening continues the thread from its tail.
        nxt = await kenning_encounter.advance_encounter("C1", session_id="C")
        assert nxt["encounter"]["tick"] == 3
        assert await thread_problems(kenning_encounter.driver) == []
    finally:
        sync.close()


async def test_migration_refuses_a_partial_thread_and_writes_nothing(kenning_encounter):
    sync = GraphDatabase.driver(URL, auth=AUTH)
    try:
        _legacy_graph(sync)
        sync.execute_query("MATCH (e:Encounter {name:'B1'}) SET e.tick = 0")
        with sync.session() as s, pytest.raises(mt.Refused, match="partial thread"):
            s.execute_write(mt.apply)
        n = sync.execute_query("MATCH ()-[k:NEXT_TICK]->() RETURN count(k) AS n").records[0]["n"]
        assert n == 0
    finally:
        sync.close()
