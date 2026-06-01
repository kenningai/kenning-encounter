// The sole writer of the temporal spine. Takes NO predecessor argument: it
// finds the current chain tail server-side (the Encounter with no outgoing
// NEXT_ENCOUNTER), creates the new Encounter, and links tail -> new in one
// transaction. First-ever encounter: no tail, becomes the chain head. Because
// the caller cannot pass a predecessor, it cannot fork the chain with a stale
// one. t_exist is the server-stamped instantiation moment — the agent's "now."
OPTIONAL MATCH (tail:Encounter) WHERE NOT (tail)-[:NEXT_ENCOUNTER]->(:Encounter)
WITH tail ORDER BY tail.t_exist DESC LIMIT 1
CREATE (e:Encounter {name: $name})
SET e.t_exist = datetime(), e.t_created = datetime()
WITH e, tail
FOREACH (_ IN CASE WHEN tail IS NOT NULL THEN [1] ELSE [] END |
    MERGE (tail)-[r:NEXT_ENCOUNTER]->(e)
    ON CREATE SET r.t_created = datetime()
)
RETURN e.name AS name,
       e.t_exist AS t_exist,
       e.t_created AS t_created,
       tail.name AS predecessor,
       tail IS NULL AS is_first
