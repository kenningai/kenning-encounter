CREATE FULLTEXT INDEX kenning_encounter_index IF NOT EXISTS
FOR (n:Encounter|Component|Concept|Observation|Question|Hypothesis|Citation|Note)
ON EACH [n.name, n.description]
