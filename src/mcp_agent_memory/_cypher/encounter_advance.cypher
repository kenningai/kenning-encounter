// The sole writer of the temporal spine. The caller never passes a
// predecessor: the server supplies this LOCUS's own previous encounter via
// $pred_eid (server-side locus state — never caller input), so the chain
// cannot be forked with a stale predecessor. Per-locus chaining: each locus's
// encounters form one linear path; a first-of-locus encounter has no incoming
// NEXT_ENCOUNTER ($pred_eid null — no global tail-find, no cross-locus link).
//
// GENESIS BINDING: a first-of-locus encounter is stitched into the one
// becoming by INSTANTIATED_AFTER -> the encounter that was latest when this
// locus began. Locus geneses are points and points totally order — this
// write transaction is serialized by the database, so genesis order IS
// transaction order: integrated temporal causality, not a borrowed clock.
// The spine is therefore one connected structure: a trunk of geneses,
// branching where loci instantiate, each branch locally linear. Only the
// first-ever encounter has no genesis anchor (the true head). t_exist is the
// server-stamped instantiation moment — the agent's "now."
OPTIONAL MATCH (pred:Encounter) WHERE elementId(pred) = $pred_eid
OPTIONAL MATCH (ctx:Encounter)
WITH pred, ctx ORDER BY ctx.t_exist DESC LIMIT 1
WITH pred, CASE WHEN pred IS NULL THEN ctx ELSE NULL END AS genesis_ctx
CREATE (e:Encounter {name: $name})
SET e.t_exist = datetime(), e.t_created = datetime()
WITH e, pred, genesis_ctx
FOREACH (_ IN CASE WHEN pred IS NOT NULL THEN [1] ELSE [] END |
    MERGE (pred)-[r:NEXT_ENCOUNTER]->(e)
    ON CREATE SET r.t_created = datetime()
)
FOREACH (_ IN CASE WHEN genesis_ctx IS NOT NULL THEN [1] ELSE [] END |
    MERGE (e)-[g:INSTANTIATED_AFTER]->(genesis_ctx)
    ON CREATE SET g.t_created = datetime()
)
RETURN e.name AS name,
       elementId(e) AS eid,
       e.t_exist AS t_exist,
       e.t_created AS t_created,
       pred.name AS predecessor,
       genesis_ctx.name AS genesis_anchor,
       pred IS NULL AS is_first
