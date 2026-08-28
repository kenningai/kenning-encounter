MATCH (source {name: $source})-[r:`__rel_type__`]->(target {name: $target})
DELETE r
RETURN count(r) AS deleted
