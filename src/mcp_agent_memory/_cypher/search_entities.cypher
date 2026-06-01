CALL db.index.fulltext.queryNodes('agent_memory_index', $query)
YIELD node, score
WITH node, score
LIMIT $limit
RETURN node.name AS name,
       labels(node)[0] AS type,
       node.description AS description,
       score,
       properties(node) AS properties
ORDER BY score DESC
