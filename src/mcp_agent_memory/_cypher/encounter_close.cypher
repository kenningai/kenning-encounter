// Annotate the current (tail) Encounter with its Report/Stop output. Sets only
// the fields provided — never creates a node, never writes NEXT_ENCOUNTER, so
// the spine stays sole-written by advance_encounter. The EERRS "Report / Stop"
// step lands here; everything else about the encounter is already recorded.
MATCH (tail:Encounter) WHERE NOT (tail)-[:NEXT_ENCOUNTER]->(:Encounter)
WITH tail ORDER BY tail.t_exist DESC LIMIT 1
__set_clause__
RETURN tail.name AS name,
       tail.t_exist AS t_exist,
       tail.summary AS summary,
       tail.report AS report
