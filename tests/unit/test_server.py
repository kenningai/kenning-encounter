"""Unit tests for mcp-agent-memory.

Tests validation logic without requiring a Neo4j instance.
"""

import pytest

from mcp_agent_memory.agent_memory import (
    Neo4jAgentMemory,
    NodeType,
    RelationType,
    validate_entity,
    validate_relation,
    NODE_SCHEMAS,
    RELATION_SCHEMAS,
    PROCESS_TYPES,
    PROCESS_EDGES,
    ANCHOR_FOR,
    DATETIME_CAST_PROPS,
)
from mcp_agent_memory.utils import format_namespace


# -- Fake driver (DB-free) ----------------------------------------------------

class _FakeResult:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    """Scripted stand-in for the async Neo4j driver.

    `script` is a list of (query_substring, records) pairs; the first pair
    whose substring appears in the executed query supplies the records
    (queries with no match return no records). Every call is captured in
    `calls` as (query, params) so tests can assert what was sent."""

    def __init__(self, script=None):
        self.script = script or []
        self.calls: list[tuple[str, dict | None]] = []

    async def execute_query(self, query, params=None, parameters_=None, **kwargs):
        q = str(query)
        self.calls.append((q, params if params is not None else parameters_))
        for substring, records in self.script:
            if substring in q:
                return _FakeResult(records)
        return _FakeResult([])


# -- NodeType Enum Tests ------------------------------------------------------

class TestNodeTypeEnum:
    def test_all_types_have_schemas(self):
        for nt in NodeType:
            assert nt.value in NODE_SCHEMAS, f"Missing schema for {nt.value}"

    def test_enum_count(self):
        # Encounter + 5 semantic (Observation/Question/Hypothesis/Concept/Note)
        # + 2 reference (Component/Citation)
        assert len(NodeType) == 8


# -- RelationType Enum Tests --------------------------------------------------

class TestRelationTypeEnum:
    def test_all_types_have_schemas(self):
        for rt in RelationType:
            assert rt.value in RELATION_SCHEMAS, f"Missing schema for {rt.value}"

    def test_enum_count(self):
        # 4 provenance (INSTANTIATED_AFTER added in v0.3.0) + 11 coherence
        # (GROUNDS added in v0.2.0)
        assert len(RelationType) == 15

    def test_process_edges_are_relation_types(self):
        for e in PROCESS_EDGES:
            assert e in RELATION_SCHEMAS


# -- Process-Layer Guard Tests ------------------------------------------------

class TestProcessLayer:
    def test_encounter_is_process_type(self):
        assert PROCESS_TYPES == {"Encounter"}

    def test_anchor_mapping(self):
        # Epistemic nodes anchor via RECORDED; Citation via CONSULTED.
        for t in ["Observation", "Question", "Hypothesis", "Concept", "Note"]:
            assert ANCHOR_FOR[t] == "RECORDED"
        assert ANCHOR_FOR["Citation"] == "CONSULTED"

    def test_component_is_not_anchored(self):
        assert "Component" not in ANCHOR_FOR

    def test_encounter_is_not_anchored(self):
        assert "Encounter" not in ANCHOR_FOR


# -- Re-entry Gradient Tests --------------------------------------------------

class TestReentryGradient:
    """The re-entry payload surfaces the coherence edges incident to the
    tension nodes. The gradient query excludes PROCESS_EDGES so the auto-written
    spine/provenance edges never become artificial centrality hubs. These guard
    the one-source-of-truth contract that query depends on (DB-free)."""

    def test_process_edges_are_the_auto_written_four(self):
        # The exact set the gradient query excludes via $process_edges.
        assert PROCESS_EDGES == {
            "NEXT_ENCOUNTER", "INSTANTIATED_AFTER", "RECORDED", "CONSULTED",
        }

    def test_coherence_edges_are_the_nonprocess_complement(self):
        # Every authored edge the gradient may surface = all relations minus the
        # process edges. Adding a coherence edge type extends the gradient;
        # adding a process edge keeps it out. Guards against silent drift.
        coherence = {r.value for r in RelationType} - PROCESS_EDGES
        assert coherence == set(RELATION_SCHEMAS) - PROCESS_EDGES
        assert coherence, "coherence set must be non-empty"
        assert PROCESS_EDGES.isdisjoint(coherence)


# -- Entity Validation Tests --------------------------------------------------

class TestEntityValidation:
    def test_observation_valid(self):
        result = validate_entity("Observation", {
            "name": "All prod services depend on node-02",
            "description": "Four production services all DEPEND_ON node-02.",
            "t_observed": "2026-05-29T14:05:00Z",
            "confidence": "high",
        })
        assert result["name"] == "All prod services depend on node-02"
        assert result["t_observed"] == "2026-05-29T14:05:00Z"

    def test_observation_requires_t_observed(self):
        with pytest.raises(ValueError, match="requires properties"):
            validate_entity("Observation", {
                "name": "X", "description": "noticed something",
            })

    def test_component_requires_source_kind_and_key(self):
        with pytest.raises(ValueError, match="requires properties"):
            validate_entity("Component", {"name": "node-02 bookmark"})

    def test_component_valid_with_source_label(self):
        result = validate_entity("Component", {
            "name": "node-02 bookmark",
            "source_kind": "infra_graph",
            "source_label": "Host",
            "source_key": "node-02",
        })
        assert result["source_kind"] == "infra_graph"
        assert result["source_label"] == "Host"
        assert result["source_key"] == "node-02"

    def test_component_source_label_optional(self):
        result = validate_entity("Component", {
            "name": "X", "source_kind": "snow_cmdb", "source_key": "CI0001",
        })
        assert "source_label" not in result

    def test_citation_requires_kind(self):
        with pytest.raises(ValueError, match="requires properties"):
            validate_entity("Citation", {"name": "graph read"})

    def test_citation_valid(self):
        result = validate_entity("Citation", {
            "name": "graph read @ refresh 2026-05-01",
            "kind": "infra_graph",
            "snapshot": "RefreshEvent_2026-05-01",
            "t_consulted": "2026-05-29T14:00:00Z",
        })
        assert result["kind"] == "infra_graph"
        assert result["snapshot"] == "RefreshEvent_2026-05-01"

    def test_concept_status_enum(self):
        for status in ["forming", "stable", "revising", "retired"]:
            result = validate_entity("Concept", {
                "name": "Failover topology", "description": "the shape of failover",
                "status": status,
            })
            assert result["status"] == status

    def test_concept_invalid_status(self):
        with pytest.raises(ValueError, match="Invalid value"):
            validate_entity("Concept", {
                "name": "X", "description": "y", "status": "halfbaked",
            })

    def test_question_status_enum(self):
        for status in ["open", "answered", "abandoned"]:
            result = validate_entity("Question", {
                "name": "Why SPOF?", "description": "why a single point of failure",
                "status": status,
            })
            assert result["status"] == status

    def test_question_invalid_status(self):
        with pytest.raises(ValueError, match="Invalid value"):
            validate_entity("Question", {
                "name": "X", "description": "y", "status": "maybe",
            })

    def test_hypothesis_confidence_enum(self):
        for conf in ["low", "medium", "high"]:
            result = validate_entity("Hypothesis", {
                "name": "H1", "description": "a guess", "confidence": conf,
            })
            assert result["confidence"] == conf

    def test_hypothesis_invalid_confidence(self):
        with pytest.raises(ValueError, match="Invalid value"):
            validate_entity("Hypothesis", {
                "name": "H1", "description": "a guess", "confidence": "certain",
            })

    def test_hypothesis_status_enum(self):
        for status in ["proposed", "supported", "challenged", "confirmed", "falsified", "retired"]:
            result = validate_entity("Hypothesis", {
                "name": "H1", "description": "a guess", "status": status,
            })
            assert result["status"] == status

    def test_note_valid(self):
        result = validate_entity("Note", {
            "name": "draft thought", "description": "needs crystallizing", "tag": "draft",
        })
        assert result["tag"] == "draft"

    def test_type_property_rejected(self):
        with pytest.raises(ValueError, match="'type' property is forbidden"):
            validate_entity("Concept", {
                "name": "X", "description": "y", "type": "Concept",
            })

    def test_unknown_property_rejected(self):
        with pytest.raises(ValueError, match="Unknown property"):
            validate_entity("Concept", {
                "name": "X", "description": "y", "flavor": "vanilla",
            })

    def test_unknown_node_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown node type"):
            validate_entity("Widget", {"name": "X"})

    def test_t_created_stripped(self):
        result = validate_entity("Concept", {
            "name": "X", "description": "y", "t_created": "2024-01-01",
        })
        assert "t_created" not in result

    def test_datetime_props_declared(self):
        # The temporal annotation fields the agent may supply.
        assert {"t_observed", "t_raised", "t_resolved", "t_proposed", "t_consulted"} <= DATETIME_CAST_PROPS


# -- Relation Validation Tests ------------------------------------------------

class TestRelationValidation:
    def test_about_observation_to_component(self):
        assert validate_relation("ABOUT", "Observation", "Component") == {}

    def test_about_hypothesis_to_concept(self):
        validate_relation("ABOUT", "Hypothesis", "Concept")

    def test_about_wrong_source(self):
        with pytest.raises(ValueError, match="requires source"):
            validate_relation("ABOUT", "Citation", "Concept")

    def test_about_wrong_target(self):
        with pytest.raises(ValueError, match="requires target"):
            validate_relation("ABOUT", "Observation", "Citation")

    def test_observed_at_direction(self):
        validate_relation("OBSERVED_AT", "Observation", "Citation")
        with pytest.raises(ValueError, match="requires source"):
            validate_relation("OBSERVED_AT", "Citation", "Observation")

    def test_raises_and_resolves(self):
        validate_relation("RAISES", "Observation", "Question")
        validate_relation("RESOLVES", "Observation", "Question")
        with pytest.raises(ValueError, match="requires target"):
            validate_relation("RAISES", "Observation", "Concept")

    def test_supports_challenges_observation_to_hypothesis(self):
        validate_relation("SUPPORTS", "Observation", "Hypothesis")
        validate_relation("CHALLENGES", "Observation", "Hypothesis")
        with pytest.raises(ValueError, match="requires target"):
            validate_relation("SUPPORTS", "Observation", "Concept")

    def test_supports_wrong_source(self):
        with pytest.raises(ValueError, match="requires source"):
            validate_relation("SUPPORTS", "Hypothesis", "Hypothesis")

    def test_informs_concept_to_concept_or_component(self):
        validate_relation("INFORMS", "Concept", "Concept")
        validate_relation("INFORMS", "Concept", "Component")
        with pytest.raises(ValueError, match="requires source"):
            validate_relation("INFORMS", "Component", "Concept")

    def test_grounds_observation_to_concept(self):
        assert validate_relation("GROUNDS", "Observation", "Concept") == {}

    def test_grounds_wrong_source(self):
        # GROUNDS is Observation -> Concept only; a Concept cannot ground.
        with pytest.raises(ValueError, match="requires source"):
            validate_relation("GROUNDS", "Concept", "Concept")

    def test_grounds_wrong_target(self):
        # Constrained to Concept — an Observation does not GROUND a Component.
        with pytest.raises(ValueError, match="requires target"):
            validate_relation("GROUNDS", "Observation", "Component")

    def test_composes_decomposes_concept_only(self):
        validate_relation("COMPOSES", "Concept", "Concept")
        validate_relation("DECOMPOSES", "Concept", "Concept")
        with pytest.raises(ValueError, match="requires target"):
            validate_relation("COMPOSES", "Concept", "Component")

    def test_supersedes_same_type_valid(self):
        why = {"revision_why": "found a clearer framing"}
        validate_relation("SUPERSEDES", "Hypothesis", "Hypothesis", why)
        validate_relation("SUPERSEDES", "Concept", "Concept", why)
        validate_relation("SUPERSEDES", "Note", "Note", why)

    def test_supersedes_requires_revision_why(self):
        with pytest.raises(ValueError, match="requires property 'revision_why'"):
            validate_relation("SUPERSEDES", "Concept", "Concept")

    def test_supersedes_empty_revision_why_rejected(self):
        with pytest.raises(ValueError, match="Invalid value"):
            validate_relation(
                "SUPERSEDES", "Concept", "Concept", {"revision_why": "   "}
            )

    def test_supersedes_revision_why_returned(self):
        result = validate_relation(
            "SUPERSEDES", "Concept", "Concept",
            {"revision_why": "the v1 view missed a second failover tier"},
        )
        assert result["revision_why"] == "the v1 view missed a second failover tier"

    def test_supersedes_cross_type_rejected(self):
        with pytest.raises(ValueError, match="share a type"):
            validate_relation("SUPERSEDES", "Hypothesis", "Concept")

    def test_supersedes_invalid_type_rejected(self):
        # Observation is not a supersedable type.
        with pytest.raises(ValueError, match="requires source"):
            validate_relation("SUPERSEDES", "Observation", "Observation")

    def test_recorded_documented_direction(self):
        # Provenance edges are documented in the schema (Encounter -> epistemic),
        # even though create_relations rejects authoring them.
        validate_relation("RECORDED", "Encounter", "Observation")
        validate_relation("CONSULTED", "Encounter", "Citation")
        validate_relation("NEXT_ENCOUNTER", "Encounter", "Encounter")

    def test_unknown_relation_type(self):
        with pytest.raises(ValueError, match="Unknown relation type"):
            validate_relation("RELATES_TO", "Component", "Component")

    def test_unknown_property_rejected(self):
        with pytest.raises(ValueError, match="Unknown property"):
            validate_relation("ABOUT", "Observation", "Concept", {"weight": 1})

    def test_t_created_stripped(self):
        result = validate_relation(
            "ABOUT", "Observation", "Concept", {"t_created": "2024-01-01"}
        )
        assert "t_created" not in result


# -- Utility Tests ------------------------------------------------------------

class TestUtils:
    def test_format_namespace(self):
        assert format_namespace("") == ""
        assert format_namespace("agent_memory") == "agent_memory-"
        assert format_namespace("agent_memory-") == "agent_memory-"


# -- Schema Completeness Tests ------------------------------------------------

class TestSchemaCompleteness:
    """Ensure the schema registry is self-consistent."""

    def test_all_node_types_have_name(self):
        for node_type, schema in NODE_SCHEMAS.items():
            assert "name" in schema["required"], (
                f"{node_type} must have 'name' as required property"
            )

    def test_all_relation_schemas_have_source_and_target_types(self):
        for rel_type, schema in RELATION_SCHEMAS.items():
            assert "source_types" in schema, f"{rel_type} must define source_types"
            assert "target_types" in schema, f"{rel_type} must define target_types"

    def test_relation_source_types_are_valid_node_types(self):
        valid_types = set(NODE_SCHEMAS.keys())
        for rel_type, schema in RELATION_SCHEMAS.items():
            for st in (schema.get("source_types") or set()):
                assert st in valid_types, (
                    f"{rel_type} references unknown source type '{st}'"
                )

    def test_relation_target_types_are_valid_node_types(self):
        valid_types = set(NODE_SCHEMAS.keys())
        for rel_type, schema in RELATION_SCHEMAS.items():
            for tt in (schema.get("target_types") or set()):
                assert tt in valid_types, (
                    f"{rel_type} references unknown target type '{tt}'"
                )

    def test_anchor_targets_are_epistemic_or_citation(self):
        valid = set(NODE_SCHEMAS.keys())
        for node_type in ANCHOR_FOR:
            assert node_type in valid

    def test_encounter_has_dissolved_at(self):
        # Server-written dissolution mark (idle-MARK, never idle-seal).
        assert "dissolved_at" in NODE_SCHEMAS["Encounter"]["optional"]


# -- Locus-Scoped Write Targeting Tests (DB-free) ------------------------------

# Substrings that identify each templated/inline query against the FakeDriver.
_ADVANCE = "CREATE (e:Encounter {name: $name})"
_RESOLVE_BY_NAME = "MATCH (e:Encounter {name: $name})"
_RESOLVE_BY_EID = "RETURN e.name AS name, elementId(e) AS eid"
_CLOSE = "SET e.dissolved_at = null"
_MARK_DISSOLVED = "SET e.dissolved_at = datetime()"

_ADVANCE_RECORD = {
    "name": "E1", "eid": "eid-1", "t_exist": None, "t_created": None,
    "predecessor": None, "genesis_anchor": None, "is_first": True,
}


def _advance_call(driver):
    """The (query, params) pair of the spine-advance write."""
    calls = [(q, p) for q, p in driver.calls if _ADVANCE in q]
    assert calls, "advance cypher was never executed"
    return calls[-1]


class TestLocusScopedWrites:
    """The continuity fix: writes address the encounter the CALLING LOCUS
    opened (per-locus chaining, locus-scoped close/create), never a global
    chain tail. All DB-free via FakeDriver."""

    pytestmark = pytest.mark.asyncio

    def _agent_memory(self, extra_script=None):
        script = [(_ADVANCE, [dict(_ADVANCE_RECORD)])] + (extra_script or [])
        return Neo4jAgentMemory(FakeDriver(script))  # type: ignore[arg-type]

    async def test_first_of_locus_has_no_predecessor(self):
        agent_memory = self._agent_memory()
        await agent_memory.advance_encounter(name="E1", locus_key="locus-A")
        _, params = _advance_call(agent_memory.driver)
        assert params["pred_eid"] is None

    async def test_advance_chains_from_own_previous_encounter(self):
        agent_memory = self._agent_memory()
        await agent_memory.advance_encounter(name="E1", locus_key="locus-A")
        await agent_memory.advance_encounter(name="E2", locus_key="locus-A")
        _, params = _advance_call(agent_memory.driver)
        assert params["pred_eid"] == "eid-1"

    async def test_parallel_locus_never_chains_from_sibling(self):
        agent_memory = self._agent_memory()
        await agent_memory.advance_encounter(name="E1", locus_key="locus-A")
        await agent_memory.advance_encounter(name="E2", locus_key="locus-B")
        _, params = _advance_call(agent_memory.driver)
        assert params["pred_eid"] is None  # B roots its own branch

    async def test_advance_without_locus_key_roots_and_stores_nothing(self):
        agent_memory = self._agent_memory()
        await agent_memory.advance_encounter(name="E1")
        _, params = _advance_call(agent_memory.driver)
        assert params["pred_eid"] is None
        assert agent_memory._locus_open == {}

    async def test_advance_stores_locus_state(self):
        agent_memory = self._agent_memory()
        await agent_memory.advance_encounter(name="E1", locus_key="locus-A")
        assert agent_memory._locus_open == {"locus-A": "eid-1"}

    async def test_advance_surfaces_genesis_anchor(self):
        # The genesis binding rides the payload: a locus root reports which
        # encounter was latest when its locus began.
        record = dict(_ADVANCE_RECORD, genesis_anchor="E0")
        agent_memory = Neo4jAgentMemory(FakeDriver([(_ADVANCE, [record])]))  # type: ignore[arg-type]
        result = await agent_memory.advance_encounter(name="E1", locus_key="locus-B")
        assert result["encounter"]["genesis_anchor"] == "E0"

    async def test_reentry_payload_carries_unsealed_set(self):
        agent_memory = self._agent_memory()
        result = await agent_memory.advance_encounter(name="E1", locus_key="locus-A")
        assert "unsealed" in result["reentry"]
        assert "dissolution" not in result["reentry"]

    async def test_advance_after_close_chains_from_sealed_predecessor(self):
        # The chain predecessor survives the seal: one session holds many
        # sealed work-units, each chaining from the last. Only the write
        # target (_locus_open) ends at the seal.
        agent_memory = self._agent_memory(extra_script=[
            (_CLOSE, [{"name": "E1", "t_exist": None, "summary": "x", "report": None}]),
            (_RESOLVE_BY_EID, [{"name": "E1", "eid": "eid-1"}]),
        ])
        await agent_memory.advance_encounter(name="E1", locus_key="locus-A")
        await agent_memory.close_encounter(summary="x", locus_key="locus-A")
        await agent_memory.advance_encounter(name="E2", locus_key="locus-A")
        _, params = _advance_call(agent_memory.driver)
        assert params["pred_eid"] == "eid-1"

    async def test_close_without_locus_or_handle_refuses(self):
        agent_memory = self._agent_memory()
        with pytest.raises(ValueError, match="No open Encounter for this locus"):
            await agent_memory.close_encounter(summary="x")

    async def test_close_seals_own_encounter_and_clears_state(self):
        agent_memory = self._agent_memory(extra_script=[
            (_CLOSE, [{"name": "E1", "t_exist": None, "summary": "x", "report": None}]),
            (_RESOLVE_BY_EID, [{"name": "E1", "eid": "eid-1"}]),
        ])
        await agent_memory.advance_encounter(name="E1", locus_key="locus-A")
        result = await agent_memory.close_encounter(summary="x", locus_key="locus-A")
        assert result["name"] == "E1"
        assert agent_memory._locus_open == {}
        close_calls = [(q, p) for q, p in agent_memory.driver.calls if _CLOSE in q]
        assert close_calls[-1][1]["eid"] == "eid-1"

    async def test_close_by_explicit_handle(self):
        agent_memory = self._agent_memory(extra_script=[
            (_CLOSE, [{"name": "E9", "t_exist": None, "summary": "x", "report": None}]),
            (_RESOLVE_BY_NAME, [{"name": "E9", "eid": "eid-9"}]),
        ])
        result = await agent_memory.close_encounter(summary="x", encounter="E9")
        assert result["name"] == "E9"
        close_calls = [(q, p) for q, p in agent_memory.driver.calls if _CLOSE in q]
        assert close_calls[-1][1]["eid"] == "eid-9"

    async def test_ambiguous_handle_refuses(self):
        agent_memory = self._agent_memory(extra_script=[
            (_RESOLVE_BY_NAME, [
                {"name": "E9", "eid": "eid-9a"},
                {"name": "E9", "eid": "eid-9b"},
            ]),
        ])
        with pytest.raises(ValueError, match="ambiguous"):
            await agent_memory.close_encounter(summary="x", encounter="E9")

    async def test_create_entities_without_locus_refuses_with_guard_message(self):
        agent_memory = self._agent_memory()
        with pytest.raises(ValueError, match="has not advanced or has already closed"):
            await agent_memory.create_entities(
                [{"type": "Note", "name": "n", "description": "d"}]
            )

    async def test_delete_entities_refuses_encounter(self):
        # The spine is the record of lived time, not editable content.
        agent_memory = self._agent_memory(extra_script=[
            ("OPTIONAL MATCH (n)-[r]-()", [{
                "type": "Encounter", "name": "E1",
                "description": None, "rel_count": 3,
            }]),
        ])
        with pytest.raises(ValueError, match="No tool deletes the spine"):
            await agent_memory.delete_entities(["E1"])

    async def test_delete_cypher_excludes_process_labels_structurally(self):
        # Names are not unique across labels: even if the preview matches a
        # same-named semantic node, the sweep must not touch the spine.
        agent_memory = self._agent_memory(extra_script=[
            ("OPTIONAL MATCH (n)-[r]-()", [{
                "type": "Note", "name": "shared-name",
                "description": "d", "rel_count": 0,
            }]),
            ("DETACH DELETE n", [{"deleted": 1}]),
        ])
        await agent_memory.delete_entities(["shared-name"])
        delete_calls = [q for q, _ in agent_memory.driver.calls if "DETACH DELETE n" in q]
        assert delete_calls and "NOT n:`Encounter`" in delete_calls[-1]

    async def test_mark_open_dissolved_marks_and_clears(self):
        agent_memory = self._agent_memory(extra_script=[
            (_MARK_DISSOLVED, [{"name": "E1"}]),
        ])
        await agent_memory.advance_encounter(name="E1", locus_key="locus-A")
        marked = await agent_memory.mark_open_dissolved()
        assert marked == ["E1"]
        assert agent_memory._locus_open == {}
