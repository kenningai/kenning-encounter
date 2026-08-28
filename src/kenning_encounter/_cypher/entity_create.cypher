// Create or update a semantic/reference node, idempotent via MERGE on name.
// `created` is true only when this MERGE actually created the node (not matched
// a pre-existing one) — auto-anchoring keys off it so re-touching a node in a
// later encounter never falsely re-dates its birth.
OPTIONAL MATCH (existing:`__label__` {name: $name})
WITH existing IS NOT NULL AS preexisted
MERGE (n:`__label__` {name: $name})
ON CREATE SET n.t_created = datetime()
__extra_sets__
WITH n, NOT preexisted AS created
__anchor_block__
RETURN n.name AS name, labels(n)[0] AS type, n.t_created AS t_created, created
