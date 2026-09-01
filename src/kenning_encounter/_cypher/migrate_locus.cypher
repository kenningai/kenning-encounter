// ============================================================================
// v0.12.0 — LOCUS MIGRATION.  ADDITIVE ONLY.  DELETES NOTHING.
//
// Derives the Locus layer from what the spine already records.  Adds three
// things that do not exist in this graph today — the :Locus label and the
// OPENED and NEXT_LOCUS edge types — and touches no existing node or edge.
// NEXT_ENCOUNTER and INSTANTIATED_AFTER are left exactly as they are: the
// v0.3.0 seam discipline, leave the braid, mark the seam.
//
// THERE IS NO REVERSAL — see the tail of this file for why that is the
// contract rather than an omission. What follows from writing only new
// structure is not undoability but SAFETY WITHOUT AN UNDO: a wrong result
// leaves more than expected, never less, and is corrected forward.
//
// ON THE CLOCK.  Step 3 orders by t_exist.  That is deliberate and it is the
// only place in the new model a clock is read: the genesis order of the
// historical loci really happened, and the timestamps are the ONLY surviving
// witness to it — the locus identities that would have carried it lived in
// server memory and were never recorded.  Check it before trusting it on your
// own graph: chain-head t_exist values should be distinct (ties mean the order
// is not total), and where an INSTANTIATED_AFTER anchor exists it should be
// the immediate predecessor in that order.  Then this RECOVERS the order the
// edges already encode rather than inventing one.  The live path never reads a
// clock: NEXT_LOCUS is set structurally from the current tail at mint time.
// ============================================================================

// --- STEP 1 -----------------------------------------------------------------
// One anonymous Locus per chain head.  Keyed on the head encounter, not on an
// ordinal, so re-running is safe no matter what order the rows arrive in.
//
// ANONYMOUS IS THE POINT.  Derived loci carry no session_id and never will:
// identity lived in server memory and died with each restart.  An anonymous
// locus is a real instantiation whose identity was never recorded, and saying
// so is better than minting a plausible one.
// KEYED ON elementId, NOT ON THE NAME — and that is not a detail. Encounter
// names were never unique: a double-advance seconds apart leaves two distinct
// chain heads carrying an identical name, and MERGE on that name collapses two
// instantiations into one locus. The retired name-handle resolver carried an
// explicit "ambiguous" branch for the same reason. elementId is unique within
// the database being migrated, which is exactly the scope of a one-time
// migration, and it keeps the re-run idempotent. The pre-flight refuses if the
// identity key is not unique, so this fails closed rather than merging.
// AND NOT ALREADY HELD. Found by testing the realistic order — upgrade, work
// for a while, THEN read the warning and migrate. A chain head created by the
// running server already has its own Locus; minting a second one for it gives
// that encounter two loci and breaks the exactly-one rule the schema exists to
// enforce. The migration is for what predates it, never for what the server
// has since made.
MATCH (head:Encounter)
WHERE NOT ()-[:NEXT_ENCOUNTER]->(head)
  AND NOT (:Locus)-[:OPENED]->(head)
MERGE (l:Locus {derived_from_eid: elementId(head)})
  ON CREATE SET l.name       = 'Locus of ' + head.name,
                l.derived_from = head.name,
                l.t_exist    = head.t_exist,
                l.anonymous  = true,
                l.derived_at = datetime();

// --- STEP 2 -----------------------------------------------------------------
// OPENED: every encounter to the locus of its chain head.
//
// Total by construction: every encounter resolves to exactly ONE head, because
// NEXT_ENCOUNTER chains are fork-free and merge-free — the invariant the single
// guarded writer maintains. Encounters with no summary need no special case:
// they are adopted like every other, and no seal is fabricated for any.
MATCH (e:Encounter) WHERE NOT (:Locus)-[:OPENED]->(e)
CALL (e) {
  MATCH (h:Encounter)-[:NEXT_ENCOUNTER*0..]->(e)
  WHERE NOT ()-[:NEXT_ENCOUNTER]->(h)
  RETURN h LIMIT 1
}
MATCH (l:Locus {derived_from_eid: elementId(h)})
MERGE (l)-[r:OPENED]->(e)
  ON CREATE SET r.t_created = datetime(), r.derived = true;

// --- STEP 3 -----------------------------------------------------------------
// NEXT_LOCUS in genesis order.
//
// NOT from INSTANTIATED_AFTER.  That derivation looks right — injective at the
// ENCOUNTER level, one anchor each — and FORKS at the LOCUS level wherever
// several loci each began at different points inside one predecessor's
// lifetime.  Correct data; not a chain, and the schema refuses forks.  NEXT_LOCUS is the total order over
// GENESES, which is what "locus lifetimes overlap but geneses are points and
// points totally order" has always meant.
MATCH (l:Locus) WITH l ORDER BY l.t_exist
WITH collect(l) AS ordered
UNWIND range(0, size(ordered) - 2) AS i
WITH ordered[i] AS a, ordered[i + 1] AS b
MERGE (a)-[r:NEXT_LOCUS]->(b)
  ON CREATE SET r.t_created = datetime(), r.derived = true;

// ============================================================================
// THERE IS NO REVERSAL, AND THAT IS THE CONTRACT.
//
// Nothing in a Kenning Encounter is ever deleted. delete_entities refuses every process
// type — "No tool deletes the spine" — and Locus is one. A migration that
// reached past that refusal to remove the very nodes the tool protects would
// be doing, unsupervised, the one thing the substrate forbids. You do not
// lobotomize the witness to undo a bookkeeping error.
//
// This is safe to have no undo BECAUSE IT ONLY ADDS. A wrong result leaves
// more structure than expected, never less. So the shape is check-then-write:
// pre-flight refuses before anything is written, a post-apply mismatch reports
// and stops, and every write is a MERGE, so correcting forward and re-running
// is a no-op on what is already right.
//
// "Reversible by deleting what was added" is not available here and never was:
// it is total only while this migration is the SOLE creator of a Locus, which
// stops being true the instant the server has run. On a graph in use it would
// remove the live loci an agent is working inside.
