MATCH (n {name: $name})
DETACH DELETE n
RETURN count(n) AS deleted
