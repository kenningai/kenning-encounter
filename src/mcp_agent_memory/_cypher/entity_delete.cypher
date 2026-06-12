// Deletes every NON-PROCESS node with this name. The spine (Encounter) is
// excluded structurally here, not only in the Python guard: names are not
// unique across labels, so a semantic node and an Encounter could share one —
// the label filter keeps the sweep off the temporal record no matter what the
// preview matched.
MATCH (n {name: $name})
WHERE __process_guard__
DETACH DELETE n
RETURN count(n) AS deleted
