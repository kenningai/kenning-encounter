MATCH (source) WHERE elementId(source) = $source_eid
MATCH (target) WHERE elementId(target) = $target_eid
MERGE (source)-[r:`__rel_type__`]->(target)
ON CREATE SET r.t_created = datetime()__prop_clause__
RETURN type(r) AS type
