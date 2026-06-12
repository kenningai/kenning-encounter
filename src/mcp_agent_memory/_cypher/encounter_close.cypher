// Annotate one specific Encounter with its Report/Stop output — addressed by
// elementId, resolved server-side from the calling locus's own open encounter
// (or an explicit handle), never the global tail. Sets only the fields
// provided — never creates a node, never writes NEXT_ENCOUNTER, so the spine
// stays sole-written by advance_encounter. A seal also clears any provisional
// dissolved_at mark: the mark recorded "locus state discarded, unsealed"; an
// authored seal disproves it, and only the author can.
MATCH (e:Encounter) WHERE elementId(e) = $eid
__set_clause__
SET e.dissolved_at = null
RETURN e.name AS name,
       e.t_exist AS t_exist,
       e.summary AS summary,
       e.report AS report