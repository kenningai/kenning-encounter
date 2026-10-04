// Seal one specific Encounter — addressed by elementId, resolved server-side
// as MY locus's open chain tail. Never the global tail, and since v0.12.0
// never an arbitrary encounter by name: the explicit handle existed only
// because server state could be lost, which can no longer happen, and a NAME
// could address any encounter including another locus's.
//
// Sets only the fields provided, plus t_sealed — never creates a node, never
// writes NEXT_ENCOUNTER, so the spine stays sole-written by advance_encounter.
//
// NO dissolved_at CLEARING ANY MORE. That line existed because the mark
// recorded "the server discarded this locus's state while the encounter was
// unsealed", and an authored seal was the only thing that could disprove it.
// Nothing stamps it now, because nothing can lose the locus: it is a node,
// and the open encounter is a graph read. Encounters carrying the
// historical mark keep it untouched — a record of an era, not a flag to
// clean up.
//
// THE SEAL TAKES THE SPINE LOCK (v0.20.0) and records the latest tick at
// sealing. An encounter ends at the first of two events: its seal, or the next
// opening in its locus, after which nothing can reach it. The second kind of
// end is already placed among the ticks; this places the first. Without the
// lock a seal could commit between two openings it did not see, so its tick
// would claim an order that never happened. seal_tick stays null on a graph
// whose thread has not been migrated yet, rather than guessing.
OPTIONAL MATCH (genesis:Locus) WHERE NOT (:Locus)-[:NEXT_LOCUS]->(genesis)
SET genesis.t_exist = genesis.t_exist
WITH count(genesis) AS _locked
MATCH (e:Encounter) WHERE elementId(e) = $eid
OPTIONAL MATCH (tt:Encounter)
  WHERE tt.tick IS NOT NULL AND NOT (tt)-[:NEXT_TICK]->(:Encounter)
WITH e, tt
__set_clause__
FOREACH (t IN CASE WHEN e.tick IS NULL OR tt IS NULL THEN [] ELSE [tt] END |
    SET e.seal_tick = t.tick
)
RETURN e.name AS name,
       e.t_exist AS t_exist,
       e.t_sealed AS t_sealed,
       e.summary AS summary,
       e.report AS report,
       e.seal_tick AS seal_tick
