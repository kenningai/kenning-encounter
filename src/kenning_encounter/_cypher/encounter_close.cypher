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
// and the open encounter is a graph read. The 12 encounters carrying the
// historical mark keep it untouched — a record of an era, not a flag to
// clean up.
MATCH (e:Encounter) WHERE elementId(e) = $eid
__set_clause__
RETURN e.name AS name,
       e.t_exist AS t_exist,
       e.t_sealed AS t_sealed,
       e.summary AS summary,
       e.report AS report
