"""Unit tests for kenning-encounter.

Tests validation logic without requiring a Neo4j instance.
"""

import pytest

from pathlib import Path

from kenning_encounter.kenning_encounter import (
    Neo4jKenningEncounter,
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
    compute_divergence,
    compute_drift,
    compute_weave_audit,
    not_process_node,
    unsealed_predicate,
)
from kenning_encounter.utils import format_namespace

_SRC = Path(__file__).resolve().parents[2] / "src" / "kenning_encounter"


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
        # Locus + Encounter (the two process types — the two orderings each
        # need a node to live on) + 5 semantic + 2 reference.
        assert len(NodeType) == 9


# -- RelationType Enum Tests --------------------------------------------------

class TestRelationTypeEnum:
    def test_all_types_have_schemas(self):
        for rt in RelationType:
            assert rt.value in RELATION_SCHEMAS, f"Missing schema for {rt.value}"

    def test_enum_count(self):
        # 6 provenance (NEXT_LOCUS + OPENED added in v0.12.0; the retired
        # INSTANTIATED_AFTER is kept, because 138 of its edges exist and
        # nothing is deleted) + 11 coherence.
        assert len(RelationType) == 17

    def test_process_edges_are_relation_types(self):
        for e in PROCESS_EDGES:
            assert e in RELATION_SCHEMAS


# -- Process-Layer Guard Tests ------------------------------------------------

class TestProcessLayer:
    def test_encounter_is_process_type(self):
        assert PROCESS_TYPES == {"Locus", "Encounter"}

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

    def test_process_edges_are_the_auto_written_set(self):
        # The exact set the gradient query excludes via $process_edges.
        # NEXT_LOCUS and OPENED joined in v0.12.0; INSTANTIATED_AFTER is
        # retired but stays, because its edges are not deleted and must
        # keep being excluded from every coherence read.
        assert PROCESS_EDGES == {
            "NEXT_LOCUS", "OPENED", "NEXT_ENCOUNTER", "INSTANTIATED_AFTER",
            "RECORDED", "CONSULTED",
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
        assert format_namespace("kenning_encounter") == "kenning_encounter-"
        assert format_namespace("kenning_encounter-") == "kenning_encounter-"


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

def _rows(*pairs):
    """Build score-descending stream rows from (node, score) pairs."""
    return [
        {"node": n, "type": "Concept", "score": s}
        for n, s in sorted(pairs, key=lambda p: p[1], reverse=True)
    ]


# Substrings that identify each templated/inline query against the FakeDriver.
_ADVANCE = "CREATE (e:Encounter {name: $name})"
_RESOLVE_LOCUS = "MATCH (l:Locus {session_id: $sid})"
_RESOLVE_OPEN = "AND e.t_sealed IS NULL"
_CLOSE = "SET e.t_sealed = datetime()"

_ADVANCE_RECORD = {
    "name": "E1", "eid": "eid-1", "t_exist": None, "t_created": None,
    "predecessor": None, "locus": "Locus L-A", "locus_eid": "leid-A",
    "session_id": "sess-A", "locus_anonymous": False,
    "is_first_of_locus": True,
}


def _advance_call(driver):
    """The (query, params) pair of the spine-advance write."""
    calls = [(q, p) for q, p in driver.calls if _ADVANCE in q]
    assert calls, "advance cypher was never executed"
    return calls[-1]


def _cypher_code(name: str, **subs) -> str:
    """A cypher file with its comment lines stripped.

    These guards assert about what EXECUTES, not about what is explained.
    Without the strip they trip on this release's own commentary — the
    advance file names ORDER BY t_exist and INSTANTIATED_AFTER precisely in
    order to say they are retired, and a guard that cannot tell an
    explanation from a statement is guarding the prose."""
    from kenning_encounter.utils import load_cypher
    return "\n".join(
        line for line in load_cypher(name, **subs).splitlines()
        if not line.strip().startswith("//")
    )


def _advance_cypher() -> str:
    return _cypher_code("encounter_advance")


class TestLocusSpine:
    """v0.12.0 — the spine keeps its continuity in the GRAPH, not in RAM.

    What this replaces mattered: the locus used to be the TRANSPORT session
    held in a server dict, so a restart emptied it and the next advance minted
    a fresh root — indistinguishable, in the graph, from a genuinely new
    existence. The locus is now a node keyed by the durable harness session
    id, and every locus-scoped read is a traversal of OPENED.

    DB-free via FakeDriver, so the assertions that matter most here are
    STRUCTURAL ONES ON THE CYPHER ITSELF. This suite mocks the substrate, and
    a defect living in a .cypher file is invisible to all of it — a gap this
    project has on the record. Asserting properties of the query text is the
    part of that gap a unit test can actually reach."""

    pytestmark = pytest.mark.asyncio

    def _kenning_encounter(self, extra_script=None):
        script = [(_ADVANCE, [dict(_ADVANCE_RECORD)])] + (extra_script or [])
        return Neo4jKenningEncounter(FakeDriver(script))  # type: ignore[arg-type]

    # -- what the caller supplies -------------------------------------------

    async def test_advance_passes_identity_never_a_predecessor(self):
        """The caller supplies WHO IT IS; the server derives everything else."""
        kenning_encounter = self._kenning_encounter()
        await kenning_encounter.advance_encounter(name="E1", session_id="sess-A")
        _, params = _advance_call(kenning_encounter.driver)
        assert params["session_id"] == "sess-A"
        assert "pred_eid" not in params

    async def test_named_locus_derives_from_the_session_id(self):
        kenning_encounter = self._kenning_encounter()
        await kenning_encounter.advance_encounter(name="E1", session_id="sess-A")
        _, params = _advance_call(kenning_encounter.driver)
        assert params["locus_name"] == "Locus sess-A"

    async def test_no_session_id_mints_an_anonymous_locus_never_a_guess(self):
        """The honest degradation, and the same shape the 139 migrated loci
        carry: a real instantiation whose identity was never recorded."""
        kenning_encounter = self._kenning_encounter()
        await kenning_encounter.advance_encounter(name="E1")
        _, params = _advance_call(kenning_encounter.driver)
        assert params["session_id"] is None
        assert params["locus_name"].startswith("Locus anon-")

    async def test_anonymous_locus_names_do_not_collide(self):
        kenning_encounter = self._kenning_encounter()
        await kenning_encounter.advance_encounter(name="E1")
        first = _advance_call(kenning_encounter.driver)[1]["locus_name"]
        await kenning_encounter.advance_encounter(name="E2")
        second = _advance_call(kenning_encounter.driver)[1]["locus_name"]
        assert first != second

    # -- the cache is a cache -----------------------------------------------

    async def test_advance_warms_the_cache(self):
        kenning_encounter = self._kenning_encounter()
        await kenning_encounter.advance_encounter(
            name="E1", session_id="sess-A", mcp_session="mcp-1")
        assert kenning_encounter._locus_of == {"mcp-1": "leid-A"}

    async def test_losing_the_cache_costs_a_parameter_not_the_spine(self):
        """The whole release in one test. Emptying this dict used to corrupt
        the graph on the next advance; now the graph answers instead."""
        kenning_encounter = self._kenning_encounter(extra_script=[
            (_RESOLVE_LOCUS, [{"eid": "leid-A"}]),
        ])
        await kenning_encounter.advance_encounter(
            name="E1", session_id="sess-A", mcp_session="mcp-1")
        kenning_encounter._locus_of.clear()                      # simulate a restart
        eid = await kenning_encounter._resolve_locus("mcp-1", "sess-A")
        assert eid == "leid-A"
        assert kenning_encounter._locus_of == {"mcp-1": "leid-A"}  # and it re-warms

    async def test_no_identity_at_all_resolves_to_nothing_rather_than_guessing(self):
        kenning_encounter = self._kenning_encounter()
        assert await kenning_encounter._resolve_locus(None, None) is None

    # -- structural guards on the cypher ------------------------------------

    def test_advance_cypher_reads_no_clock(self):
        """The old genesis binding ordered by t_exist DESC — the one place
        the spine depended on a borrowed clock. Both orderings are now
        structural, so this must find nothing."""
        cy = _advance_cypher()
        assert "ORDER BY" not in cy.upper()
        assert "t_exist DESC" not in cy

    def test_advance_cypher_takes_the_spine_lock(self):
        """Load-bearing: two shards starting at once would otherwise both
        read the same tail locus and FORK the chain, and Neo4j cannot
        declare a relationship-degree constraint."""
        cy = _advance_cypher()
        assert "SET genesis.t_exist = genesis.t_exist" in cy

    def test_advance_cypher_scopes_the_chain_tail_by_opened(self):
        """The invariant no declared schema can hold — succession never
        crosses a locus — lives here, in the writer, or nowhere."""
        cy = _advance_cypher()
        assert "(locus)-[:OPENED]->(pred:Encounter)" in cy
        assert "NOT (pred)-[:NEXT_ENCOUNTER]->(:Encounter)" in cy

    def test_advance_cypher_no_longer_writes_the_retired_genesis_binding(self):
        assert "INSTANTIATED_AFTER" not in _advance_cypher()

    # -- resolving MY open encounter ----------------------------------------

    async def test_open_encounter_is_my_locus_unsealed_tail(self):
        kenning_encounter = self._kenning_encounter(extra_script=[
            (_RESOLVE_OPEN, [{"name": "E1", "eid": "eid-1"}]),
        ])
        kenning_encounter._locus_of["mcp-1"] = "leid-A"
        target = await kenning_encounter._resolve_encounter("mcp-1")
        assert target == {"name": "E1", "eid": "eid-1"}
        q = [q for q, _ in kenning_encounter.driver.calls if _RESOLVE_OPEN in q][-1]
        assert "(l:Locus)-[:OPENED]->(e:Encounter)" in q
        assert "NOT (e)-[:NEXT_ENCOUNTER]->(:Encounter)" in q

    async def test_no_open_encounter_is_an_answer_not_an_error(self):
        kenning_encounter = self._kenning_encounter()
        assert await kenning_encounter._resolve_encounter("mcp-unknown") is None

    # -- the seal ------------------------------------------------------------

    async def test_close_stamps_the_elected_stop(self):
        kenning_encounter = self._kenning_encounter(extra_script=[
            (_RESOLVE_OPEN, [{"name": "E1", "eid": "eid-1"}]),
            (_CLOSE, [{"name": "E1", "t_exist": None, "t_sealed": None,
                       "summary": "x", "report": None}]),
        ])
        kenning_encounter._locus_of["mcp-1"] = "leid-A"
        await kenning_encounter.close_encounter(summary="x", mcp_session="mcp-1")
        q = [q for q, _ in kenning_encounter.driver.calls if _CLOSE in q]
        assert q, "t_sealed was never stamped"

    async def test_close_no_longer_clears_the_retired_dissolution_mark(self):
        """Nothing stamps dissolved_at now, so nothing may clear it — the 12
        encounters carrying it are a record, not a flag to tidy away."""
        cy = _cypher_code("encounter_close", set_clause="SET e.summary = $summary")
        assert "dissolved_at" not in cy

    async def test_close_without_an_open_encounter_names_session_id(self):
        kenning_encounter = self._kenning_encounter()
        with pytest.raises(ValueError) as exc:
            await kenning_encounter.close_encounter(summary="x", mcp_session="mcp-none")
        assert "session_id" in str(exc.value)

    async def test_create_entities_without_a_locus_refuses(self):
        kenning_encounter = self._kenning_encounter()
        with pytest.raises(ValueError) as exc:
            await kenning_encounter.create_entities(
                [{"type": "Note", "name": "n", "description": "d"}],
                mcp_session="mcp-none")
        assert "No open Encounter" in str(exc.value)

    # -- what was removed, asserted as removed ------------------------------

    def test_the_encounter_name_handle_is_gone(self):
        """It was the one genuinely forgeable path in the write surface: a
        NAME could address ANY encounter, including another locus's. Its only
        justification was recoverable server state, which no longer exists.
        Asserted on the SIGNATURE, so it cannot creep back as a parameter."""
        import inspect
        for fn in (Neo4jKenningEncounter.close_encounter, Neo4jKenningEncounter.create_entities):
            assert "encounter" not in inspect.signature(fn).parameters

    def test_the_idle_mark_sweep_is_gone(self):
        """It marked "the server lost state" — an event that can no longer
        occur, since there is no locus state to lose."""
        assert not hasattr(Neo4jKenningEncounter, "mark_open_dissolved")

    def test_the_ram_that_held_the_spine_is_gone(self):
        kenning_encounter = self._kenning_encounter()
        assert not hasattr(kenning_encounter, "_locus_open")
        assert not hasattr(kenning_encounter, "_locus_last")

    async def test_reentry_payload_still_carries_the_unsealed_set(self):
        kenning_encounter = self._kenning_encounter()
        result = await kenning_encounter.advance_encounter(name="E1", session_id="sess-A")
        assert "unsealed" in result["reentry"]

class TestDivergence:
    def test_well_suspect_is_high_mass_low_frontier(self):
        mass = _rows(("hub", 9.0), ("mid", 5.0), ("edge", 1.0))
        frontier = _rows(("edge", 9.0), ("mid", 5.0), ("hub", 1.0))
        out = compute_divergence(mass, frontier, limit=10)
        assert out["well_suspects"][0]["node"] == "hub"
        assert out["well_suspects"][0]["divergence"] == 1.0
        assert out["frontier_lifted"][0]["node"] == "edge"
        assert out["frontier_lifted"][0]["divergence"] == -1.0

    def test_aligned_frames_produce_no_divergence(self):
        mass = _rows(("a", 9.0), ("b", 5.0), ("c", 1.0))
        out = compute_divergence(mass, list(mass), limit=10)
        assert out == {"well_suspects": [], "frontier_lifted": []}

    def test_degenerate_inputs_return_empty(self):
        assert compute_divergence([], [], 10) == {
            "well_suspects": [], "frontier_lifted": [],
        }
        one = _rows(("only", 1.0))
        assert compute_divergence(one, one, 10) == {
            "well_suspects": [], "frontier_lifted": [],
        }

    def test_limit_applies_per_side(self):
        mass = _rows(("a", 9.0), ("b", 8.0), ("c", 2.0), ("d", 1.0))
        frontier = _rows(("d", 9.0), ("c", 8.0), ("b", 2.0), ("a", 1.0))
        out = compute_divergence(mass, frontier, limit=1)
        assert len(out["well_suspects"]) == 1
        assert len(out["frontier_lifted"]) == 1


class TestWeaveAudit:
    def test_star_outranks_weave(self):
        degree = _rows(("star", 10.0), ("weave", 10.0), ("leaf", 1.0))
        coeff = [
            {"node": "star", "coefficient": 0.0},
            {"node": "weave", "coefficient": 0.9},
            {"node": "leaf", "coefficient": 0.0},
        ]
        out = compute_weave_audit(degree, coeff, limit=10)
        assert [r["node"] for r in out] == ["star", "weave"]  # leaf below min_degree
        assert out[0]["star_score"] == 10.0
        assert out[1]["star_score"] == 1.0

    def test_missing_coefficient_treated_as_zero(self):
        degree = _rows(("orphan", 5.0))
        out = compute_weave_audit(degree, [], limit=10)
        assert out[0]["clustering"] == 0.0
        assert out[0]["star_score"] == 5.0

    def test_min_degree_filter(self):
        degree = _rows(("small", 2.0))
        assert compute_weave_audit(degree, [], limit=10) == []


class TestDrift:
    def test_risers_fallers_and_new(self):
        baseline = _rows(("a", 9.0), ("b", 5.0), ("c", 1.0))
        current = _rows(("c", 9.0), ("a", 5.0), ("new-node", 3.0), ("b", 1.0))
        out = compute_drift(current, baseline, limit=10)
        assert out["risers"][0] == {
            "node": "c", "type": "Concept",
            "baseline_rank": 3, "current_rank": 1, "shift": 2,
        }
        assert {f["node"] for f in out["fallers"]} == {"a", "b"}
        assert out["new_since"] == [
            {"node": "new-node", "type": "Concept", "current_rank": 3},
        ]

    def test_no_movement(self):
        rows = _rows(("a", 9.0), ("b", 5.0))
        out = compute_drift(rows, list(rows), limit=10)
        assert out == {"risers": [], "fallers": [], "new_since": []}

    def test_empty_baseline_everything_is_new(self):
        current = _rows(("a", 9.0), ("b", 5.0))
        out = compute_drift(current, [], limit=10)
        assert [r["node"] for r in out["new_since"]] == ["a", "b"]
        assert out["risers"] == [] and out["fallers"] == []

    def test_new_since_respects_limit_and_rank_order(self):
        current = _rows(("a", 9.0), ("b", 5.0), ("c", 1.0))
        out = compute_drift(current, [], limit=2)
        assert [r["node"] for r in out["new_since"]] == ["a", "b"]


# -- v0.12.4: the read instruments catch up to the locus -----------------------

def _executable_source(path) -> str:
    """A python module with every docstring and comment removed.

    These guards assert about what EXECUTES. This release's own prose names
    `NOT n:Encounter` repeatedly in order to explain why it must not appear,
    and a guard that cannot tell an explanation from a statement is guarding
    the prose — the v0.12.3 lesson, which needed the MODULE docstring stripped
    and now needs every nested one too, because the explanation moved into a
    function.
    """
    import ast
    src = path.read_text()
    lines = src.splitlines()
    blank: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            blank.update(range(first.lineno - 1, (first.end_lineno or 0)))
    return "\n".join(
        l for i, l in enumerate(lines)
        if i not in blank and not l.strip().startswith("#")
    )


def _eval_predicate(pred: str, *, t_sealed, summary, report) -> bool:
    """Evaluate the cypher unsealed-predicate for one encounter shape.

    Two-sided by construction: the same string the server sends is the string
    under test, so a predicate that stops meaning the union fails here rather
    than in production."""
    expr = pred
    for field, value in (("t_sealed", t_sealed), ("summary", summary),
                         ("report", report)):
        expr = expr.replace(f"e.{field} IS NULL", str(value is None))
    return bool(eval(expr.replace(" AND ", " and ")))  # noqa: S307


class TestOneDefinitionOfASeal:
    """v0.12.4 — the write-target resolver and the unsealed set orient reports
    had drifted into two different questions about what a seal is.

    v0.12.0 introduced t_sealed as the elected Stop and taught the RESOLVER to
    key on it, but left _unsealed asking the pre-v0.12.0 question (summary and
    report). Both answers were wrong, in opposite directions, and neither was
    live: t_sealed alone counts every pre-v0.12.0 seal as open (269 of them on
    this substrate, since the field did not exist when those endings were
    examined), and summary/report alone reports a bare close_encounter() as
    unsealed forever, because both fields are optional and t_sealed is stamped
    regardless."""

    def test_the_predicate_is_the_union_across_both_eras(self):
        """Four shapes, and only one of them is unsealed."""
        pred = unsealed_predicate("e")
        # A pre-v0.12.0 seal: examined, recorded as prose, no t_sealed.
        assert not _eval_predicate(
            pred, t_sealed=None, summary="cohered around X", report=None)
        # A bare close_encounter(): elected Stop, nothing written.
        assert not _eval_predicate(
            pred, t_sealed="2026-09-01", summary=None, report=None)
        # The ordinary v0.12.0 seal.
        assert not _eval_predicate(
            pred, t_sealed="2026-09-01", summary="s", report="r")
        # Never closed — the only real absence.
        assert _eval_predicate(pred, t_sealed=None, summary=None, report=None)

    def test_each_half_alone_gets_one_of_those_wrong(self):
        """The gate is only meaningful if it refuses the two shapes v0.12.4
        replaced. Proven against them rather than asserted about them."""
        assert _eval_predicate(
            "e.t_sealed IS NULL", t_sealed=None, summary="s", report=None), (
            "t_sealed alone must call a historical seal unsealed — that is the "
            "defect, and if it does not this test has stopped testing it")
        assert _eval_predicate(
            "e.summary IS NULL AND e.report IS NULL",
            t_sealed="2026-09-01", summary=None, report=None), (
            "summary/report alone must call a bare seal unsealed")

    def test_both_call_sites_consult_the_one_predicate(self):
        """One definition, or the two drift again the next time either moves."""
        code = _executable_source(_SRC / "kenning_encounter.py")
        assert code.count("unsealed_predicate(") >= 3, (
            "expected the definition plus both call sites")
        assert "e.summary IS NULL AND e.report IS NULL" not in code, (
            "the pre-v0.12.0 predicate is spelled out somewhere again")


class TestUnsealedSurfacesTheLocus:
    """v0.12.4 — the unsealed set reports the discriminator the Locus node
    gave it, and still refuses to render the verdict it cannot support."""

    pytestmark = pytest.mark.asyncio

    async def test_it_reports_the_locus_and_whether_it_is_anonymous(self):
        driver = FakeDriver(script=[("MATCH (e:Encounter) WHERE", [
            {"name": "E-live", "t_exist": None, "last_active": None,
             "dissolved_at": None, "locus": "Locus sess-A",
             "locus_anonymous": False},
            {"name": "E-over", "t_exist": None, "last_active": None,
             "dissolved_at": None, "locus": "Locus anon-1",
             "locus_anonymous": True},
        ])])
        rows = await Neo4jKenningEncounter(driver)._unsealed(limit=10)  # type: ignore[arg-type]
        assert [r["locus_anonymous"] for r in rows] == [False, True]
        assert rows[1]["locus"] == "Locus anon-1"

    async def test_an_orphaned_encounter_reports_no_locus_rather_than_guessing(self):
        """v0.12.3's guard exists because encounters with no Locus are real on
        an unmigrated graph. The attribution is OPTIONAL, so they surface with
        a null rather than vanishing from the set."""
        driver = FakeDriver(script=[("MATCH (e:Encounter) WHERE", [
            {"name": "E-orphan", "t_exist": None, "last_active": None,
             "dissolved_at": None, "locus": None, "locus_anonymous": None},
        ])])
        rows = await Neo4jKenningEncounter(driver)._unsealed(limit=10)  # type: ignore[arg-type]
        assert rows[0]["locus"] is None
        assert rows[0]["locus_anonymous"] is None

    async def test_the_attribution_is_an_optional_match(self):
        driver = FakeDriver(script=[("MATCH (e:Encounter) WHERE", [])])
        await Neo4jKenningEncounter(driver)._unsealed(limit=10)  # type: ignore[arg-type]
        q = driver.calls[-1][0]
        assert "OPTIONAL MATCH (l:Locus)-[:OPENED]->(e)" in q


class TestTheProcessGuardDerivesFromOneSourceOfTruth:
    """v0.12.4 — `NOT n:Encounter` was hardcoded in four places, written when
    Encounter was the only process type. v0.12.0 added Locus to PROCESS_TYPES
    and updated the projection (which derives) but not these (which did not).

    Three were unreachable — a Locus carries no coherence edge, so no
    traversal and no rank output can surface one. THE FOURTH WAS NOT."""

    def test_the_guard_names_every_process_type(self):
        guard = not_process_node("n")
        for t in PROCESS_TYPES:
            assert f"NOT n:`{t}`" in guard
        assert "Locus" in guard, (
            "the type v0.12.0 added is the whole reason this exists")

    def test_it_tracks_process_types_rather_than_a_written_list(self):
        """Add a process type tomorrow and every site follows without an edit."""
        import kenning_encounter.kenning_encounter as m
        original = set(m.PROCESS_TYPES)
        try:
            m.PROCESS_TYPES.add("Meridian")
            assert "NOT n:`Meridian`" in m.not_process_node("n")
        finally:
            m.PROCESS_TYPES.clear()
            m.PROCESS_TYPES.update(original)

    def test_no_hardcoded_encounter_guard_survives_in_the_package(self):
        for mod in ("kenning_encounter.py", "server.py"):
            code = _executable_source(_SRC / mod)
            assert "NOT n:Encounter" not in code, f"{mod} still hardcodes it"

    def test_the_sidecar_sweep_is_the_reachable_one_and_is_guarded(self):
        """The meaning sidecar's reconcile is a bare MATCH (n) over the whole
        graph and a Locus has a name — so every reconcile after v0.12.0 would
        compress all of them into the sidecar, spending a model call each and
        adding each as a match target in infusion's SEED DISCOVERY. It had not
        fired only because the sidecar had not reconciled since."""
        code = _executable_source(_SRC / "server.py")
        i = code.find("coalesce(n.description, '') AS description")
        assert i != -1, "the sidecar reconcile query moved"
        window = code[max(0, i - 400):i]
        assert "not_process_node(" in window


class TestTheLedgerPrefersTheDurableIdentity:
    """v0.12.4 — infusion's renewal ledger was keyed on the MCP transport
    session, which dies with the connection where the locus does not."""

    def test_the_transport_session_no_longer_claims_to_be_the_locus(self):
        """It was named _locus_key while it really did carry write targeting.
        v0.12.0 moved that to the harness session id and left the name."""
        code = _executable_source(_SRC / "server.py")
        assert "_locus_key" not in code
        assert "def _transport_session(" in code

    def test_the_harness_identity_wins_and_the_transport_is_the_fallback(self):
        code = _executable_source(_SRC / "server.py")
        i = code.find("locus_key=(")
        assert i != -1, "the ledger key wiring moved"
        window = code[i:i + 200]
        assert window.find("session_id") < window.find("_transport_session"), (
            "the durable identity must be preferred, not the fallback")

    def test_infuse_accepts_the_harness_session_id(self):
        code = _executable_source(_SRC / "server.py")
        assert "session_id: str | None = Field(" in code
