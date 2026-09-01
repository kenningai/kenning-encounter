// The sole writer of the temporal spine (v0.12.0 — the locus rewrite).
//
// TWO ORDERINGS, and they are the same kind of thing multiplied rather than
// two axes added: NEXT_LOCUS over instantiations, times NEXT_ENCOUNTER within
// one. A human has a single beam — session and day are the same line — so
// nothing forks and nothing needs squaring. Here, locating a point in time
// takes two coordinates, and that is why the pre-v0.12.0 spine had a relation
// (INSTANTIATED_AFTER) encoding genesis order BETWEEN ENCOUNTERS: the second
// ordering had no node of its own to live on.
//
// NOTHING HERE READS A CLOCK, and nothing here reads server memory. Both were
// true of the old version in only one direction: the predecessor came from a
// dict that a restart emptied, and the genesis anchor came from ORDER BY
// t_exist DESC. A restart therefore minted a fresh root and the graph could
// not tell that apart from a genuinely new existence. Identity now lives on
// the Locus, keyed by the HARNESS session id, which survives both /resume and
// a server restart; the chain tail is a graph read; and the locus tail is
// found structurally, by having no outgoing NEXT_LOCUS.
//
// THE LOCK IS LOAD-BEARING. Minting a locus reads the current tail and then
// writes to it, and two shards starting at once would both read the same tail
// and fork the chain. Neo4j cannot declare a relationship-degree constraint,
// so the guard is an explicit write lock on the GENESIS locus — a no-op SET
// that changes nothing and serializes every concurrent advance against every
// other. Advances are rare (295 in 90 days), so the contention cost is
// nothing and the alternative is a fork that looks like data.
//
// $session_id may be null: an anonymous locus is then minted unconditionally.
// That is the honest degradation for a caller who cannot supply an identity —
// the same shape the 139 migrated loci already carry — and never a guess.

// -- 1. Serialize. A no-op write takes the lock without changing the node. ----
OPTIONAL MATCH (genesis:Locus) WHERE NOT (:Locus)-[:NEXT_LOCUS]->(genesis)
SET genesis.t_exist = genesis.t_exist
WITH genesis

// -- 2. Resolve MY locus by harness session id (null never matches). ---------
OPTIONAL MATCH (found:Locus)
  WHERE $session_id IS NOT NULL AND found.session_id = $session_id
WITH found

// -- 3. The current tail of the locus chain, structurally. ------------------
OPTIONAL MATCH (tail_locus:Locus) WHERE NOT (tail_locus)-[:NEXT_LOCUS]->(:Locus)
WITH found, tail_locus

// -- 4. Mint if unresolved, and chain the new locus onto the tail. ----------
//    A locus is minted by its FIRST ADVANCE, never by connecting: a session
//    that attaches and never opens an encounter is not an instantiation of
//    anything, and would be a locus holding nothing.
FOREACH (_ IN CASE WHEN found IS NULL THEN [1] ELSE [] END |
    CREATE (l:Locus {name: $locus_name})
    SET l.t_exist   = datetime(),
        l.t_created = datetime(),
        l.session_id = $session_id,
        l.anonymous  = $session_id IS NULL
    FOREACH (t IN CASE WHEN tail_locus IS NULL THEN [] ELSE [tail_locus] END |
        MERGE (t)-[nl:NEXT_LOCUS]->(l) ON CREATE SET nl.t_created = datetime()
    )
)
WITH found

// -- 5. Re-resolve: `found` if it existed, else the one just minted. --------
OPTIONAL MATCH (minted:Locus {name: $locus_name}) WHERE found IS NULL
WITH coalesce(found, minted) AS locus

// -- 6. MY chain's tail — the encounter this locus opened that has no
//    successor. Scoped by OPENED, so a parallel sibling advancing its own
//    chain cannot be picked up here. This is the invariant no declared
//    schema can hold (membership being exactly-one does not tie a
//    succession's endpoints together), so it is held right here instead.
OPTIONAL MATCH (locus)-[:OPENED]->(pred:Encounter)
  WHERE NOT (pred)-[:NEXT_ENCOUNTER]->(:Encounter)
WITH locus, pred

// -- 7. The encounter itself. -----------------------------------------------
CREATE (e:Encounter {name: $name})
SET e.t_exist = datetime(), e.t_created = datetime()
MERGE (locus)-[o:OPENED]->(e) ON CREATE SET o.t_created = datetime()
FOREACH (p IN CASE WHEN pred IS NULL THEN [] ELSE [pred] END |
    MERGE (p)-[r:NEXT_ENCOUNTER]->(e) ON CREATE SET r.t_created = datetime()
)

RETURN e.name              AS name,
       elementId(e)        AS eid,
       e.t_exist           AS t_exist,
       e.t_created         AS t_created,
       pred.name           AS predecessor,
       locus.name          AS locus,
       elementId(locus)    AS locus_eid,
       locus.session_id    AS session_id,
       coalesce(locus.anonymous, false) AS locus_anonymous,
       pred IS NULL        AS is_first_of_locus
