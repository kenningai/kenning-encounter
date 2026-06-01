MATCH (n)
WHERE n.name IN $names
RETURN n.name AS name,
       labels(n)[0] AS type,
       properties(n) AS properties
LIMIT $limit
