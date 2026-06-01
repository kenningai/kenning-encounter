MATCH (source)-[r]->(target)
WHERE source.name IN $names AND target.name IN $names
RETURN source.name AS source,
       target.name AS target,
       type(r) AS type
