"""Unit tests for the governed infusion pipeline (v0.5.0).

The judgment-bearing arithmetic — Extract, the Lucene Match query, the
conflict triage, Format — is pure and tested DB-free; the orchestration is
tested against the scripted FakeDriver, matching the repo's discipline.
"""

import argparse

import pytest

from mcp_agent_memory.agent_memory import Neo4jAgentMemory, frontier_seed_candidates, validate_entity
from mcp_agent_memory.infuse import (
    commit_progression,
    delta_novelty,
    extract_focal_signals,
    format_delta,
    commit_delivery,
    format_payload,
    format_progression,
    group_progressions,
    lucene_query,
    progression_renewal,
    renewal_filter_edges,
    renewal_partition,
    triage_conflicts,
)
from mcp_agent_memory.utils import process_config

from .test_server import FakeDriver


# -- Extract --------------------------------------------------------------------

class TestExtract:
    def test_empty_text_yields_no_signals(self):
        assert extract_focal_signals("") == []
        assert extract_focal_signals("   \n  ") == []

    def test_ip_addresses_are_hard_signals(self):
        out = extract_focal_signals("What application runs on 53.112.30.104?")
        terms = [s["term"] for s in out]
        assert "53.112.30.104" in terms
        # Hard signal outweighs the prose words around it.
        ip = next(s for s in out if s["term"] == "53.112.30.104")
        word = next(s for s in out if s["term"] == "application")
        assert ip["weight"] > word["weight"]

    def test_cidr_and_dotted_hostnames(self):
        out = extract_focal_signals(
            "hosts in 53.112.24.0/24 behind lb-east.example.corp"
        )
        terms = [s["term"] for s in out]
        assert "53.112.24.0/24" in terms
        assert "lb-east.example.corp" in terms

    def test_identifier_shaped_tokens(self):
        out = extract_focal_signals(
            "why does advance_encounter chain from the ddc-checkpoint-omu repo"
        )
        terms = [s["term"] for s in out]
        assert "advance_encounter" in terms
        assert "ddc-checkpoint-omu" in terms

    def test_stopwords_and_short_words_excluded(self):
        out = extract_focal_signals("what is the of and to it up")
        assert out == []

    def test_question_stem_verbs_excluded(self):
        # "What do we know about X?" — the asking vocabulary is not the
        # subject. Domain nouns (failover) survive; stems (know, tell) do not.
        out = extract_focal_signals(
            "What do we know about the failover? Tell me what you think."
        )
        terms = {s["term"].lower() for s in out}
        assert "failover" in terms
        assert terms.isdisjoint({"know", "tell", "think"})

    def test_split_and_weight_on_long_documents(self):
        # A long paste: the question lives at the edges; the middle is noise
        # except for a repeated term the frequency sample must surface.
        head = "frontier divergence " * 100          # 200 tokens
        middle = ("lorem ipsum dolor sit amet " * 100) + (" checkpoint" * 12)
        tail = " ".join(["articulation"] * 100)      # 100 tokens
        out = extract_focal_signals(head + middle + " " + tail)
        by_term = {s["term"].lower(): s["weight"] for s in out}
        assert "frontier" in by_term and "articulation" in by_term
        assert "checkpoint" in by_term  # frequency-sampled from the middle
        # Edge terms carry triple weight per occurrence; a middle term of
        # matching frequency cannot outrank them.
        assert by_term["frontier"] > by_term["checkpoint"]

    def test_max_terms_cap(self):
        text = " ".join(f"uniqueterm{i}" for i in range(100))
        assert len(extract_focal_signals(text, max_terms=10)) == 10

    def test_harness_envelope_noise_excluded(self):
        # A PostToolUse batch arrives wrapped in transport: content-block keys,
        # tool-use ids, mcp__ tool names, notification field names. None of it
        # is the subject; all of it is identifier-shaped enough to have ridden
        # the hard-signal weight before the filter existed.
        out = extract_focal_signals(
            '{"type": "text", "text": "corridor annotated", '
            '"tool_use_id": "toolu_01AbCdEfGh", '
            '"tool_name": "mcp__agent_memory__advance_encounter"} '
            "<task-notification><task-id>b452tvc2l</task-id>"
            "<tool-use-id>x</tool-use-id><output-file>y</output-file>"
        )
        terms = {s["term"].lower() for s in out}
        assert terms.isdisjoint(
            {"type", "text", "tool_use_id", "tool_name",
             "task-notification", "task-id", "tool-use-id", "output-file"}
        )
        assert not any(t.startswith(("mcp__", "toolu_")) for t in terms)
        # The content inside the envelope survives.
        assert "corridor" in terms and "annotated" in terms

    def test_unicode_escape_artifacts_excluded(self):
        # A serialized em-dash reaches the word regex as the literal text
        # u2014; the fragment class, not just the one codepoint, is filtered.
        # The input must carry the ESCAPED form — a real em-dash never produces
        # the artifact, and would pass vacuously without the filter.
        out = extract_focal_signals(
            "the corridor \\u2014 eligibility \\u00e9 refused aliasing"
        )
        terms = {s["term"].lower() for s in out}
        assert terms.isdisjoint({"u2014", "u00e9"})
        assert {"corridor", "eligibility", "refused", "aliasing"} <= terms

    def test_domain_identifiers_survive_envelope_filter(self):
        # Two-sided: the filter must say no to plumbing and yes to real
        # snake/kebab domain names of the same shape.
        out = extract_focal_signals(
            "why does paloalto_flow_upsert differ from ddc-systems-omu extraction"
        )
        terms = [s["term"] for s in out]
        assert "paloalto_flow_upsert" in terms
        assert "ddc-systems-omu" in terms


# -- Match query ------------------------------------------------------------------

class TestLuceneQuery:
    def test_terms_are_ored_with_boost(self):
        q = lucene_query(
            [{"term": "checkpoint", "weight": 3.0}, {"term": "omu", "weight": 1.0}]
        )
        assert q == "checkpoint^3 OR omu"

    def test_specials_are_escaped(self):
        q = lucene_query([{"term": "53.112.24.0/24", "weight": 3.0}])
        assert "\\/" in q
        q = lucene_query([{"term": "a:b", "weight": 1.0}])
        assert "a\\:b" in q

    def test_boost_clamped_to_three(self):
        q = lucene_query([{"term": "hot", "weight": 17.0}])
        assert q == "hot^3"


# -- Triage -----------------------------------------------------------------------

class TestTriage:
    def test_core_vs_parked_by_constitutive_proximity(self):
        # The biased rank IS proximity: a conflict on a top-ranked node leads;
        # one the focal point barely reaches parks — carried, not collapsed on.
        conflicts = [
            {"kind": "challenge", "from_name": "obs-core", "to_name": "hyp-core"},
            {"kind": "challenge", "from_name": "obs-far", "to_name": "hyp-far"},
        ]
        scores = {"hyp-core": 0.9, "obs-core": 0.1, "hyp-far": 0.01, "obs-far": 0.0}
        core, parked = triage_conflicts(conflicts, scores)
        assert [c["to_name"] for c in core] == ["hyp-core"]
        assert [c["to_name"] for c in parked] == ["hyp-far"]
        assert core[0]["severity"] == 1.0
        assert 0 <= parked[0]["severity"] < 0.25

    def test_no_scores_means_everything_parks(self):
        core, parked = triage_conflicts(
            [{"kind": "challenge", "from_name": "a", "to_name": "b"}], {}
        )
        assert core == [] and len(parked) == 1

    def test_severity_sorts_descending(self):
        conflicts = [
            {"kind": "challenge", "from_name": f"o{i}", "to_name": f"h{i}"}
            for i in range(3)
        ]
        scores = {"h0": 0.3, "h1": 0.9, "h2": 0.6}
        core, _ = triage_conflicts(conflicts, scores)
        assert [c["to_name"] for c in core] == ["h1", "h2", "h0"]


# -- Delta novelty ----------------------------------------------------------------

class TestDeltaNovelty:
    R1 = {"name": "node-A", "type": "Observation", "encounters": ["E1"]}
    R2 = {"name": "node-B", "type": "Concept", "encounters": []}
    C1 = {"kind": "challenge", "from_name": "obs-X", "to_name": "hyp-Y"}

    def test_first_sight_passes_everything(self):
        fresh_r, fresh_c, new_keys = delta_novelty([self.R1, self.R2], [self.C1], set())
        assert fresh_r == [self.R1, self.R2] and fresh_c == [self.C1]
        assert len(new_keys) == 3

    def test_repeat_is_suppressed(self):
        seen: set[str] = set()
        _, _, keys = delta_novelty([self.R1, self.R2], [self.C1], seen)
        seen |= keys
        fresh_r, fresh_c, new_keys = delta_novelty([self.R1, self.R2], [self.C1], seen)
        assert fresh_r == [] and fresh_c == [] and new_keys == set()

    def test_new_item_still_fires_among_repeats(self):
        seen: set[str] = set()
        _, _, keys = delta_novelty([self.R1], [], seen)
        seen |= keys
        fresh_r, fresh_c, _ = delta_novelty([self.R1, self.R2], [self.C1], seen)
        assert fresh_r == [self.R2] and fresh_c == [self.C1]

    def test_conflict_identity_is_kind_and_endpoints(self):
        # A dissonance and a challenge on the same node are different findings.
        d = {"kind": "dissonance", "from_name": "hyp-Y", "detail": "confidence:high"}
        seen: set[str] = set()
        _, _, keys = delta_novelty([], [self.C1], seen)
        seen |= keys
        _, fresh_c, _ = delta_novelty([], [d], seen)
        assert fresh_c == [d]

    def test_changed_state_is_a_new_first_sight(self):
        # The key fingerprints CONTENT, not identity: a suppressed conflict
        # whose state changes (props gained, detail shifted) re-announces —
        # first sight of the new state. Identity-only keying would suppress
        # the change forever.
        seen: set[str] = set()
        _, _, keys = delta_novelty([self.R1], [self.C1], seen)
        seen |= keys
        revised_c = {**self.C1, "props": {"revision_why": "standby verified"}}
        changed_r = {**self.R1, "encounters": ["E1", "E2"]}
        fresh_r, fresh_c, _ = delta_novelty([changed_r], [revised_c], seen)
        assert fresh_c == [revised_c]
        assert fresh_r == [changed_r]

    def test_dissonance_detail_shift_reannounces(self):
        d1 = {"kind": "dissonance", "from_name": "hyp-Y",
              "detail": "confidence:high, 1 SUPPORTS"}
        d2 = {"kind": "dissonance", "from_name": "hyp-Y",
              "detail": "confidence:high, 0 SUPPORTS"}
        seen: set[str] = set()
        _, _, keys = delta_novelty([], [d1], seen)
        seen |= keys
        _, fresh_c, _ = delta_novelty([], [d2], seen)
        assert fresh_c == [d2]


# -- Renewal (the delivery ledger) ---------------------------------------------------


def _deliver(nodes, ledger, turn, refresh_turns=None):
    """Partition and commit every selected body — the all-fits case."""
    kw = {} if refresh_turns is None else {"refresh_turns": refresh_turns}
    fresh, standing, fps = renewal_partition(nodes, ledger, turn, **kw)
    commit_delivery(ledger, fps, [n["name"] for n in fresh], turn)
    return fresh, standing

class TestRenewalPartition:
    N1 = {"name": "concept-A", "type": "Concept", "description": "held synthesis"}
    N2 = {"name": "obs-B", "type": "Observation", "description": "a noticing"}

    def test_first_sight_is_fresh_and_recorded(self):
        ledger: dict = {}
        fresh, standing, fps = renewal_partition([self.N1, self.N2], ledger, turn=1)
        assert fresh == [self.N1, self.N2] and standing == []
        # v0.7.1: partition NEVER writes. The ledger records delivery only.
        assert ledger == {}
        commit_delivery(ledger, fps, [n["name"] for n in fresh], turn=1)
        assert set(ledger) == {"concept-A", "obs-B"}

    def test_verbatim_repeat_becomes_a_handle(self):
        ledger: dict = {}
        _deliver([self.N1], ledger, turn=1)
        fresh, standing = _deliver([self.N1], ledger, turn=2)
        assert fresh == []
        assert standing == [{"name": "concept-A", "type": "Concept"}]

    def test_changed_state_is_fresh_again(self):
        # Same discipline as the delta gate: a changed fact-state is a new
        # first sight — the description shifted, the full body re-delivers.
        ledger: dict = {}
        _deliver([self.N1], ledger, turn=1)
        revised = {**self.N1, "description": "revised synthesis"}
        fresh, standing = _deliver([revised], ledger, turn=2)
        assert fresh == [revised] and standing == []

    def test_staleness_refreshes_the_body(self):
        # Residual salience decays: past the refresh horizon the handle's
        # bridge is presumed too lossy and the body re-delivers.
        ledger: dict = {}
        _deliver([self.N1], ledger, turn=1)
        fresh, _ = _deliver([self.N1], ledger, turn=5, refresh_turns=10)
        assert fresh == []  # inside the horizon: handle
        fresh, standing = _deliver([self.N1], ledger, turn=11, refresh_turns=10)
        assert fresh == [self.N1] and standing == []
        # And the refresh resets the clock.
        fresh, _ = _deliver([self.N1], ledger, turn=12, refresh_turns=10)
        assert fresh == []

    def test_mixed_turn_fresh_and_standing_coexist(self):
        ledger: dict = {}
        _deliver([self.N1], ledger, turn=1)
        fresh, standing = _deliver([self.N1, self.N2], ledger, turn=2)
        assert fresh == [self.N2]
        assert standing == [{"name": "concept-A", "type": "Concept"}]

    def test_edge_renewal_suppresses_repeats_and_refreshes_stale(self):
        # An edge has no handle form: print at first sight or stale, omit
        # otherwise — the replay measured unledgered edges reabsorbing every
        # character the body ledger freed.
        e1 = {"from_name": "obs-B", "rel": "GROUNDS", "to_name": "concept-A"}
        e2 = {"from_name": "obs-B", "rel": "SUPPORTS", "to_name": "hyp-C"}
        ledger: dict = {}
        assert renewal_filter_edges([e1], ledger, turn=1) == [e1]
        # Verbatim repeat suppressed; a new edge still prints.
        assert renewal_filter_edges([e1, e2], ledger, turn=2) == [e2]
        # Stale refresh past the horizon, and the clock resets.
        assert renewal_filter_edges([e1], ledger, turn=11, refresh_turns=10) == [e1]
        assert renewal_filter_edges([e1], ledger, turn=12, refresh_turns=10) == []
        # Node and edge entries share one ledger without key collisions.
        _deliver([{"name": "obs-B", "type": "Observation",
                   "description": "d"}], ledger, turn=12)
        assert "obs-B" in ledger and "edge|obs-B|GROUNDS|concept-A" in ledger


# -- Format -----------------------------------------------------------------------

def _payload(**overrides):
    args = dict(
        seed_terms=["checkpoint", "0590_fw"],
        seed_mode="blended",
        core_conflicts=[
            {"kind": "challenge", "from_name": "the forest observation",
             "to_name": "chain-suffices hypothesis", "severity": 0.8},
        ],
        parked_conflicts=[
            {"kind": "dissonance", "from_name": "stable-but-ungrounded",
             "detail": "status:stable, no observational grounding",
             "severity": 0.05},
        ],
        neighborhood_nodes=[
            {"name": "Coherence is in the edges, not the nodes",
             "type": "Concept", "description": "The tension lists alone are a gradientless bag."},
        ],
        neighborhood_edges=[
            {"from_name": "obs-1", "rel": "GROUNDS", "to_name": "concept-1"},
        ],
        open_threads=[{"type": "Question", "name": "an open question"}],
        trajectory=[{"encounter": "Encounter 001", "t_exist": "2026-05-30", "name": "obs-1"}],
    )
    args.update(overrides)
    return format_payload(**args)[0]



class TestFormatPayload:
    def test_signature_leads(self):
        p = _payload()
        first = p.splitlines()[0]
        assert first.startswith("[substrate proposal — awakened from: checkpoint")
        assert "not a conclusion" in first

    def test_tension_first_parked_last(self):
        p = _payload()
        assert p.index("CORE TENSION") < p.index("NEIGHBORHOOD")
        assert p.index("NEIGHBORHOOD") < p.index("OPEN THREADS")
        assert p.index("OPEN THREADS") < p.index("TRAJECTORY")
        assert p.index("TRAJECTORY") < p.index("PARKED TENSIONS")
        assert "noted, unresolved, non-blocking" in p

    def test_budget_protects_the_ends(self):
        # A huge middle must yield; the signature, core tension, and parked
        # register survive — the tension is never dropped.
        many_nodes = [
            {"name": f"node-{i}", "type": "Concept", "description": "d " * 50}
            for i in range(200)
        ]
        p = _payload(neighborhood_nodes=many_nodes, max_chars=2000)
        assert len(p) <= 2000
        assert "CORE TENSION" in p
        assert "PARKED TENSIONS" in p

    def test_degradation_order_protects_the_frontier(self):
        # Under a squeeze the middle yields trajectory-first, neighborhood
        # second; the open threads (the frontier's voice — the governance
        # mechanism) survive, even though they display AFTER neighborhood.
        many_nodes = [
            {"name": f"node-{i}", "type": "Concept", "description": "d " * 50}
            for i in range(200)
        ]
        long_traj = [
            {"encounter": f"Encounter {i:03d}", "name": f"node-{i}"}
            for i in range(50)
        ]
        p = _payload(
            neighborhood_nodes=many_nodes, trajectory=long_traj, max_chars=2000
        )
        assert len(p) <= 2000
        assert "OPEN THREADS" in p and "an open question" in p
        assert "TRAJECTORY" not in p  # first to yield
        assert "CORE TENSION" in p and "PARKED TENSIONS" in p

    def test_frontier_only_header(self):
        p = _payload(seed_terms=[], seed_mode="frontier_only")
        assert "(frontier only)" in p.splitlines()[0]

    def test_standing_register_renders_after_neighborhood(self):
        p = _payload(standing_nodes=[
            {"name": "The two-party test", "type": "Concept"},
            {"name": "omu-core mortar", "type": "Component"},
        ])
        assert "STANDING — delivered earlier this waking" in p
        assert "  • The two-party test (Concept)" in p
        assert p.index("NEIGHBORHOOD") < p.index("STANDING") < p.index("OPEN THREADS")

    def test_standing_yields_before_threads_but_after_trajectory(self):
        # Survival: threads > neighborhood > standing > trajectory.
        many_standing = [
            {"name": f"standing-node-{i:03d} with a long semantic name", "type": "Concept"}
            for i in range(60)
        ]
        long_traj = [
            {"encounter": f"Encounter {i:03d}", "name": f"n-{i}"} for i in range(50)
        ]
        p = _payload(standing_nodes=many_standing, trajectory=long_traj, max_chars=2000)
        assert len(p) <= 2000
        assert "OPEN THREADS" in p and "an open question" in p
        assert "TRAJECTORY" not in p  # still first to yield

    def test_high_tension_guard_fails_honest(self):
        # When core tension ALONE busts the budget, whole lines yield
        # lowest-severity-first with a withheld count — never a mid-line cut,
        # because half a contradiction can assert something neither node
        # claims. The parked register collapses to a carried-count line.
        many_core = [
            {"kind": "challenge", "from_name": f"obs-{i:02d} with a long name",
             "to_name": f"hyp-{i:02d} with a long name",
             "severity": round(1.0 - i * 0.01, 2)}
            for i in range(60)
        ]
        p = _payload(core_conflicts=many_core, max_chars=1000)
        assert len(p) <= 1000
        lines = p.splitlines()
        # No mid-line cut: every line is complete (no trailing ellipsis).
        assert not any(ln.endswith("…") for ln in lines)
        assert "core tensions withheld — substrate in high-tension state" in p
        assert "withheld under budget; carried, not dropped" in p
        # Highest severity survives; the lowest-severity kept outranks
        # everything dropped (drop order is lowest-first).
        assert "obs-00 with a long name" in p
        kept_idx = [i for i in range(60) if f"obs-{i:02d}" in p]
        assert kept_idx == list(range(len(kept_idx)))  # a prefix, no gaps


class TestFormatDelta:
    def test_silence_when_nothing_recognized_and_nothing_contested(self):
        assert format_delta([], []) == ""

    def test_recognition_names_the_constituting_encounters(self):
        p = format_delta(
            [{"name": "FCCC", "type": "Component",
              "encounters": ["Encounter 012", "Encounter 017"]}],
            [],
        )
        assert "already held: FCCC" in p
        assert "Encounter 012, Encounter 017" in p
        assert p.splitlines()[0].startswith("[substrate delta")

    def test_conflict_is_the_finding(self):
        p = format_delta(
            [],
            [{"kind": "challenge", "from_name": "new result",
              "to_name": "held hypothesis", "severity": 1.0}],
        )
        assert "CONTESTS what you hold" in p


# -- frontier_mute (v0.6.0) --------------------------------------------------------

class TestFrontierMute:
    def test_accepted_as_bool_on_candidate_types(self):
        for t, req in (
            ("Question", {"name": "q", "description": "d"}),
            ("Hypothesis", {"name": "h", "description": "d"}),
            ("Concept", {"name": "c", "description": "d"}),
        ):
            cleaned = validate_entity(t, {**req, "frontier_mute": True})
            assert cleaned["frontier_mute"] is True

    def test_rejected_on_non_candidate_types_and_wrong_type(self):
        with pytest.raises(ValueError, match="Unknown property"):
            validate_entity("Observation", {
                "name": "o", "description": "d",
                "t_observed": "2026-01-01T00:00:00Z", "frontier_mute": True,
            })
        with pytest.raises(ValueError, match="must be bool"):
            validate_entity("Hypothesis", {
                "name": "h", "description": "d", "frontier_mute": "yes",
            })

    def test_every_frontier_cut_honors_the_mute(self):
        # The mute must hold at every candidate source, or bookkeeping
        # leaks back into payloads through the unfiltered cut.
        driver = FakeDriver()
        agent_memory = Neo4jAgentMemory(driver)
        import asyncio
        asyncio.run(agent_memory._frontier())
        frontier_queries = [q for q, _ in driver.calls]
        assert len(frontier_queries) == 5
        for q in frontier_queries:
            assert "frontier_mute" in q, f"unfiltered cut: {q[:80]}"


# -- Frontier seed candidates (shared with orient) ---------------------------------

class TestFrontierSeedCandidates:
    def test_primary_and_fallback(self):
        frontier = {
            "unanswered_questions": [{"name": "q1"}],
            "untested_hypotheses": [{"name": "h1"}],
            "ungrounded_concepts": [{"name": "c1"}],
            "confidence_dissonance": [{"name": "d1"}],
            "contested_hypotheses": [{"name": "x1"}],
        }
        primary, fallback = frontier_seed_candidates(frontier)
        assert primary == ["q1", "h1"]
        assert fallback == ["q1", "h1", "c1", "d1", "x1"]


# -- Orchestration (FakeDriver) ------------------------------------------------------

@pytest.mark.asyncio
class TestInfuseOrchestration:
    async def test_mode_is_validated(self):
        agent_memory = Neo4jAgentMemory(FakeDriver())
        with pytest.raises(ValueError, match="Unknown infuse mode"):
            await agent_memory.infuse("text", mode="firehose")

    async def test_delta_silence_when_nothing_matches(self):
        agent_memory = Neo4jAgentMemory(FakeDriver())  # fulltext returns no records
        out = await agent_memory.infuse("a batch about nothing held", mode="delta")
        assert out["silence"] is True and out["payload"] == ""

    async def test_delta_recognition_and_conflict(self):
        driver = FakeDriver(script=[
            ("db.index.fulltext.queryNodes", [
                {"name": "FCCC bookmark", "type": "Component",
                 "description": None, "score": 2.0},
            ]),
            ("collect(e.name) AS encounters", [
                {"name": "FCCC bookmark", "encounters": ["Encounter 012"]},
            ]),
            ("-[r:CHALLENGES]->", [
                {"from_name": "new obs", "to_name": "held hyp", "props": {}},
            ]),
        ])
        out = await agent_memory_infuse_delta(driver)
        assert out["silence"] is False
        assert "already held: FCCC bookmark" in out["payload"]
        assert "Encounter 012" in out["payload"]
        assert "CONTESTS what you hold" in out["payload"]
        # Delta computes no rank, so it asserts no amplitude: a severity here
        # would contradict full mode's computed severity for the same edge.
        assert all("severity" not in c for c in out["conflicts"])
        # Delta never touches GDS: no projection created.
        assert not any("gds.graph.project" in q for q, _ in driver.calls)

    async def test_delta_novelty_announces_once_per_locus(self):
        driver = FakeDriver(script=[
            ("db.index.fulltext.queryNodes", [
                {"name": "FCCC bookmark", "type": "Component",
                 "description": None, "score": 2.0},
            ]),
            ("collect(e.name) AS encounters", [
                {"name": "FCCC bookmark", "encounters": ["Encounter 012"]},
            ]),
            ("-[r:CHALLENGES]->", [
                {"from_name": "new obs", "to_name": "held hyp", "props": {}},
            ]),
        ])
        agent_memory = Neo4jAgentMemory(driver)
        first = await agent_memory.infuse(
            "result mentioning FCCC", mode="delta", locus_key="locus-1"
        )
        assert first["silence"] is False
        assert first["suppressed"] == {"recognitions": 0, "conflicts": 0}
        # Same content, same locus: the debounce — everything already
        # announced, so the payload is silence and the counts say why.
        second = await agent_memory.infuse(
            "result mentioning FCCC", mode="delta", locus_key="locus-1"
        )
        assert second["silence"] is True and second["payload"] == ""
        assert second["suppressed"] == {"recognitions": 1, "conflicts": 1}
        # A DIFFERENT locus has been told nothing: first sight again.
        other = await agent_memory.infuse(
            "result mentioning FCCC", mode="delta", locus_key="locus-2"
        )
        assert other["silence"] is False

    async def test_delta_without_locus_never_suppresses(self):
        driver = FakeDriver(script=[
            ("db.index.fulltext.queryNodes", [
                {"name": "FCCC bookmark", "type": "Component",
                 "description": None, "score": 2.0},
            ]),
        ])
        agent_memory = Neo4jAgentMemory(driver)
        for _ in range(2):
            out = await agent_memory.infuse("result mentioning FCCC", mode="delta")
            assert out["silence"] is False
            assert out["suppressed"] == {"recognitions": 0, "conflicts": 0}

    async def test_full_silence_when_no_seeds_anywhere(self):
        # Matches exist but carry no coherence edges; frontier empty.
        driver = FakeDriver(script=[
            ("db.index.fulltext.queryNodes", [
                {"name": "isolated note", "type": "Note",
                 "description": None, "score": 1.0},
            ]),
            ("AS eligible", [{"eligible": []}]),
        ])
        agent_memory = Neo4jAgentMemory(driver)
        out = await agent_memory.infuse("something new", mode="full")
        assert out["silence"] is True
        assert out["seed_mode"] == "none"
        assert not any("gds.graph.project" in q for q, _ in driver.calls)

    async def test_full_payload_and_projection_lifecycle(self):
        driver = FakeDriver(script=[
            ("db.index.fulltext.queryNodes", [
                {"name": "concept-A", "type": "Concept",
                 "description": "held synthesis", "score": 3.0},
            ]),
            ("AS eligible", [{"eligible": ["concept-A"]}]),
            ("AS pairs", [  # the single biased computation
                {"node": "concept-A", "type": "Concept", "score": 0.9},
                {"node": "hyp-B", "type": "Hypothesis", "score": 0.5},
                {"node": "obs-C", "type": "Observation", "score": 0.4},
            ]),
            ("gds.articleRank.stream", [
                {"node": "concept-A", "type": "Concept", "score": 0.7},
                {"node": "hyp-B", "type": "Hypothesis", "score": 0.6},
                {"node": "obs-C", "type": "Observation", "score": 0.5},
            ]),
            ("-[r:CHALLENGES]->", [
                {"from_name": "obs-C", "to_name": "hyp-B",
                 "props": {"note": "x"}},
            ]),
            ("MATCH (a)-[r]-(b)", [
                {"from_name": "obs-C", "rel": "GROUNDS", "to_name": "concept-A"},
            ]),
            ("MATCH (n) WHERE n.name IN $names AND NOT n:Encounter", [
                {"name": "concept-A", "type": "Concept", "description": "held synthesis"},
                {"name": "hyp-B", "type": "Hypothesis", "description": None},
                {"name": "obs-C", "type": "Observation", "description": None},
            ]),
            ("ORDER BY e.t_exist ASC", [
                {"encounter": "Encounter 001", "t_exist": None, "name": "obs-C"},
            ]),
        ])
        agent_memory = Neo4jAgentMemory(driver)
        out = await agent_memory.infuse("tell me about concept A", mode="full")
        assert out["silence"] is False
        assert out["seed_mode"] == "focal_only"
        assert out["rank_mode"] == "biased_single"
        p = out["payload"]
        assert p.splitlines()[0].startswith("[substrate proposal")
        assert "CORE TENSION" in p  # hyp-B ranks 0.5/0.9 — core
        assert "obs-C CHALLENGES hyp-B" in p
        assert "NEIGHBORHOOD" in p and "concept-A (Concept)" in p
        assert "TRAJECTORY" in p and "Encounter 001" in p
        # Projection lifecycle: created once, dropped once.
        projects = [q for q, _ in driver.calls if "gds.graph.project" in q]
        drops = [q for q, _ in driver.calls if "gds.graph.drop" in q]
        assert len(projects) == 1 and len(drops) == 1

    async def test_full_renewal_across_turns_same_locus(self):
        # Turn 1: full bodies. Turn 2 (same locus, same graph): the
        # unchanged neighborhood re-pins as STANDING handles and the
        # payload shrinks — the renewal economy end to end.
        script = [
            ("db.index.fulltext.queryNodes", [
                {"name": "concept-A", "type": "Concept",
                 "description": "held synthesis", "score": 3.0},
            ]),
            ("AS eligible", [{"eligible": ["concept-A"]}]),
            ("AS pairs", [
                {"node": "concept-A", "type": "Concept", "score": 0.9},
                {"node": "obs-C", "type": "Observation", "score": 0.4},
            ]),
            ("gds.articleRank.stream", [
                {"node": "concept-A", "type": "Concept", "score": 0.7},
                {"node": "obs-C", "type": "Observation", "score": 0.5},
            ]),
            ("MATCH (n) WHERE n.name IN $names AND NOT n:Encounter", [
                {"name": "concept-A", "type": "Concept",
                 "description": "held synthesis " * 12},
                {"name": "obs-C", "type": "Observation",
                 "description": "a long noticing about the thing " * 6},
            ]),
        ]
        agent_memory = Neo4jAgentMemory(FakeDriver(script=script))
        one = await agent_memory.infuse("about concept A", mode="full", locus_key="L1")
        assert one["turn"] == 1
        assert one["counts"]["payload_nodes"] == 2 and one["counts"]["standing"] == 0
        assert "STANDING" not in one["payload"]
        two = await agent_memory.infuse("about concept A again", mode="full", locus_key="L1")
        assert two["turn"] == 2
        assert two["counts"]["payload_nodes"] == 0 and two["counts"]["standing"] == 2
        assert "STANDING — delivered earlier this waking" in two["payload"]
        assert len(two["payload"]) < len(one["payload"])
        # A different locus is fresh again; a stateless caller always is.
        other = await agent_memory.infuse("about concept A", mode="full", locus_key="L2")
        assert other["counts"]["standing"] == 0
        stateless = await agent_memory.infuse("about concept A", mode="full")
        assert stateless["counts"]["standing"] == 0 and stateless["turn"] == 0

    async def test_expansion_tier_enters_the_blend_and_can_be_disabled(self):
        # EXPERIMENT-BARLOW B2: concept-cluster siblings enter the biased
        # pairs at the expansion bias; expansion_bias=0 disables the reach
        # entirely (the benchmark's OFF arm).
        script = [
            ("db.index.fulltext.queryNodes", [
                {"name": "concept-A", "type": "Concept",
                 "description": "held synthesis", "score": 3.0},
            ]),
            ("AS eligible", [{"eligible": ["concept-A"]}]),
            ("MATCH (s:Concept)-[", [{"name": "sibling-X"}, {"name": "concept-A"}]),
            ("AS pairs", [
                {"node": "concept-A", "type": "Concept", "score": 0.9},
                {"node": "sibling-X", "type": "Observation", "score": 0.5},
            ]),
            ("gds.articleRank.stream", [
                {"node": "concept-A", "type": "Concept", "score": 0.7},
            ]),
            ("MATCH (n) WHERE n.name IN $names AND NOT n:Encounter", [
                {"name": "concept-A", "type": "Concept", "description": "d"},
                {"name": "sibling-X", "type": "Observation", "description": "d2"},
            ]),
        ]
        driver = FakeDriver(script=script)
        agent_memory = Neo4jAgentMemory(driver)
        out = await agent_memory.infuse("about concept A", mode="full")
        # The already-focal name is deduped out of the expansion set.
        assert out["expansion_seeds"] == ["sibling-X"]
        assert out["expansion_bias"] == 0.5
        pairs_call = next(p for q, p in driver.calls if q and "AS pairs" in q)
        assert pairs_call["expansion"] == ["sibling-X"]
        assert pairs_call["exp_bias"] == 0.5
        # OFF arm: bias 0 issues no expansion query at all.
        driver2 = FakeDriver(script=script)
        out2 = await Neo4jAgentMemory(driver2).infuse(
            "about concept A", mode="full", expansion_bias=0.0
        )
        assert out2["expansion_seeds"] == []
        assert not any("MATCH (s:Concept)-[" in (q or "") for q, _ in driver2.calls)

    async def test_full_result_carries_renewal_metadata(self):
        script = [
            ("db.index.fulltext.queryNodes", [
                {"name": "concept-A", "type": "Concept",
                 "description": None, "score": 3.0},
            ]),
            ("AS eligible", [{"eligible": ["concept-A"]}]),
            ("AS pairs", [{"node": "concept-A", "type": "Concept", "score": 0.9}]),
            ("gds.articleRank.stream", [
                {"node": "concept-A", "type": "Concept", "score": 0.7},
            ]),
            ("MATCH (n) WHERE n.name IN $names AND NOT n:Encounter", [
                {"name": "concept-A", "type": "Concept", "description": "d"},
            ]),
        ]
        agent_memory = Neo4jAgentMemory(FakeDriver(script=script))
        one = await agent_memory.infuse("concept A", mode="full", locus_key="L9")
        assert one["renewal"]["fresh"] == ["concept-A"]
        assert one["renewal"]["standing"] == []
        # v0.7.1: selection and delivery are reported separately, and with a
        # roomy budget they agree.
        assert one["renewal"]["delivered_bodies"] == ["concept-A"]
        assert one["renewal"]["dropped_bodies"] == []
        assert one["renewal"]["body_origin"]["concept-A"] in {
            "focal", "expansion", "frontier", "ranked"
        }
        two = await agent_memory.infuse("concept A", mode="full", locus_key="L9")
        assert two["renewal"]["fresh"] == []
        assert two["renewal"]["standing"] == ["concept-A"]
        assert two["renewal"]["delivered_handles"] == ["concept-A"]

    async def test_dropped_body_does_not_stamp_the_ledger(self):
        """v0.7.1 regression: a body squeezed out by the budget must NOT
        reset its clock. Stamping from selection produced 31 phantom
        delivery records in a 7-turn session and would have handed
        EXPERIMENT-BARLOW B1 a corrupted independent variable."""
        ledger: dict = {}
        n = {"name": "concept-A", "type": "Concept", "description": "x" * 400}
        fresh, standing, fps = renewal_partition([n], ledger, turn=1)
        assert fresh == [n] and ledger == {}
        # Nothing delivered -> nothing stamped -> still fresh next turn.
        commit_delivery(ledger, fps, [], turn=1)
        assert ledger == {}
        fresh2, standing2, _ = renewal_partition([n], ledger, turn=2)
        assert fresh2 == [n] and standing2 == []

    async def test_standing_outranks_neighborhood_under_squeeze(self):
        """v0.7.1: handles are cheaper AND their loss is worse — a dropped
        handle silently withdraws an established node from the standing
        picture, while a dropped body simply arrives next turn."""
        payload, delivered = format_payload(
            seed_terms=["t"], seed_mode="blended",
            core_conflicts=[], parked_conflicts=[],
            neighborhood_nodes=[
                {"name": f"body-{i}", "type": "Observation", "description": "y" * 300}
                for i in range(6)
            ],
            neighborhood_edges=[],
            open_threads=[],
            trajectory=[],
            standing_nodes=[{"name": f"hand-{i}", "type": "Concept"} for i in range(6)],
            max_chars=900,
        )
        assert delivered["handles"], "standing must survive the squeeze"
        assert len(delivered["handles"]) > len(delivered["bodies"])
        assert "STANDING" in payload

    async def test_full_drops_projection_on_rank_failure(self):
        class ExplodingDriver(FakeDriver):
            async def execute_query(self, query, params=None, parameters_=None, **kwargs):
                q = str(query)
                if "gds.articleRank.stream" in q:
                    self.calls.append((q, params))
                    raise RuntimeError("gds fell over")
                return await super().execute_query(
                    query, params=params, parameters_=parameters_, **kwargs
                )

        driver = ExplodingDriver(script=[
            ("db.index.fulltext.queryNodes", [
                {"name": "concept-A", "type": "Concept",
                 "description": None, "score": 3.0},
            ]),
            ("AS eligible", [{"eligible": ["concept-A"]}]),
            ("AS pairs", [{"node": "concept-A", "type": "Concept", "score": 0.9}]),
        ])
        agent_memory = Neo4jAgentMemory(driver)
        with pytest.raises(RuntimeError, match="gds fell over"):
            await agent_memory.infuse("boom", mode="full")
        assert any("gds.graph.drop" in q for q, _ in driver.calls)


async def agent_memory_infuse_delta(driver):
    agent_memory = Neo4jAgentMemory(driver)
    return await agent_memory.infuse("result mentioning FCCC", mode="delta")


# -- Config ---------------------------------------------------------------------------

class TestInfuseConfig:
    def _args(self, **overrides):
        base = dict(
            db_url="bolt://x", username="u", password="p", database="d",
            namespace=None, transport=None, server_host=None, server_port=None,
            server_path=None, allow_origins=None, allowed_hosts=None,
            read_timeout=None, infuse_frontier_bias=None,
            infuse_refresh_turns=None,
            matcher_endpoint=None, matcher_model=None,
            matcher_timeout_ms=None, matcher_sidecar=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_default_bias(self, monkeypatch):
        monkeypatch.delenv("NEO4J_INFUSE_FRONTIER_BIAS", raising=False)
        assert process_config(self._args())["infuse_frontier_bias"] == 0.3

    def test_cli_wins_and_clamps(self, monkeypatch):
        monkeypatch.setenv("NEO4J_INFUSE_FRONTIER_BIAS", "0.9")
        assert process_config(
            self._args(infuse_frontier_bias=0.5)
        )["infuse_frontier_bias"] == 0.5
        assert process_config(
            self._args(infuse_frontier_bias=7.0)
        )["infuse_frontier_bias"] == 1.0

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("NEO4J_INFUSE_FRONTIER_BIAS", "0.15")
        assert process_config(self._args())["infuse_frontier_bias"] == 0.15

    def test_refresh_turns_default_env_cli_clamp(self, monkeypatch):
        monkeypatch.delenv("NEO4J_INFUSE_REFRESH_TURNS", raising=False)
        assert process_config(self._args())["infuse_refresh_turns"] == 10
        monkeypatch.setenv("NEO4J_INFUSE_REFRESH_TURNS", "6")
        assert process_config(self._args())["infuse_refresh_turns"] == 6
        assert process_config(
            self._args(infuse_refresh_turns=15)
        )["infuse_refresh_turns"] == 15
        assert process_config(
            self._args(infuse_refresh_turns=0)
        )["infuse_refresh_turns"] == 1

    def test_matcher_config_default_env_cli_clamp(self, monkeypatch):
        for var in ("GEMINI_API_KEY", "NEO4J_MATCHER_ENDPOINT",
                    "NEO4J_MATCHER_MODEL", "NEO4J_MATCHER_TIMEOUT_MS",
                    "NEO4J_MATCHER_SIDECAR"):
            monkeypatch.delenv(var, raising=False)
        cfg = process_config(self._args())
        assert cfg["matcher_api_key"] == ""
        assert cfg["matcher_endpoint"] == (
            "https://generativelanguage.googleapis.com/v1beta"
        )
        assert cfg["matcher_model"] == "gemini-3.5-flash-lite"
        assert cfg["matcher_timeout_ms"] == 5000  # measured renegotiation
        assert cfg["matcher_sidecar"] == "models/meaning_sidecar.json"
        monkeypatch.setenv("GEMINI_API_KEY", "k-123")
        monkeypatch.setenv("NEO4J_MATCHER_MODEL", "gemini-other")
        monkeypatch.setenv("NEO4J_MATCHER_TIMEOUT_MS", "99999")
        monkeypatch.setenv("NEO4J_MATCHER_SIDECAR", "/app/models/m.json")
        cfg = process_config(self._args())
        assert cfg["matcher_api_key"] == "k-123"
        assert cfg["matcher_model"] == "gemini-other"
        assert cfg["matcher_timeout_ms"] == 10_000  # ceiling
        assert cfg["matcher_sidecar"] == "/app/models/m.json"
        assert process_config(
            self._args(matcher_timeout_ms=50)
        )["matcher_timeout_ms"] == 100  # floor


# -- Progression assembly (v0.8.0, the progression reform) -----------------------


def _dell_steps():
    """The Dell/CNTLM progression — the standing falsification case."""
    mk = lambda name, sk, type_="Observation", desc="": {
        "name": name, "type": type_, "description": desc,
        "sort_key": sk, "encounter": "Encounter Aug", "enc_t": sk,
    }
    return [
        mk("XE8640 discovery: sole constraint is stale cntlm credentials",
           "2026-08-03T17:10:00Z",
           desc="4x H100 healthy; cntlm creds stale on the jump-box proxy."),
        mk("Conor repointed cntlm's parent endpoint and full egress returned",
           "2026-08-03T17:35:00Z", desc="HF weights stream; pypi passes."),
        mk("First light: gpt-oss-20b at 323 tok/s", "2026-08-03T18:55:00Z"),
        mk("gpt-oss-120b live at 222 tok/s", "2026-08-04T02:45:00Z"),
        mk("Mistral Large 2411 serving", "2026-08-08T00:55:00Z"),
        mk("Medium 3.5 serving at 256k", "2026-08-08T09:50:00Z",
           desc="497k-token KV cache; tokenizer-mode mistral."),
        mk("What is the production serving configuration?",
           "2026-08-08T10:00:00Z", type_="Question"),
    ]


def _rows_for(component, steps):
    return [{**s, "component": component} for s in steps]


class TestGroupProgressions:
    def test_one_progression_per_component_regardless_of_seeds(self):
        rows = _rows_for("stnamcvdl200", _dell_steps())
        groups = group_progressions(
            rows,
            ["XE8640 discovery: sole constraint is stale cntlm credentials",
             "Medium 3.5 serving at 256k"],
        )
        assert len(groups) == 1
        assert groups[0]["component"] == "stnamcvdl200"
        assert len(groups[0]["seed_names"]) == 2

    def test_steps_ordered_by_coalesced_noticing_time(self):
        rows = list(reversed(_rows_for("comp", _dell_steps())))
        groups = group_progressions(rows, ["Mistral Large 2411 serving"])
        names = [s["name"] for s in groups[0]["steps"]]
        assert names[0].startswith("XE8640 discovery")
        assert names[-1].startswith("What is the production")

    def test_encounter_t_exist_breaks_noticing_time_ties(self):
        # Structural tiebreak: same noticing stamp, different serialized
        # encounter creation order — the later encounter's step sorts later.
        mk = lambda name, enc_t: {
            "name": name, "type": "Observation", "description": "",
            "sort_key": "2026-08-03T17:00:00Z", "encounter": "E",
            "enc_t": enc_t, "component": "comp",
        }
        rows = [mk("later-encounter step", "2026-08-03T18:00"),
                mk("earlier-encounter step", "2026-08-03T16:00")]
        groups = group_progressions(rows, ["earlier-encounter step"])
        names = [s["name"] for s in groups[0]["steps"]]
        assert names == ["earlier-encounter step", "later-encounter step"]

    def test_single_step_component_is_not_a_progression(self):
        rows = _rows_for("solo", _dell_steps()[:1])
        assert group_progressions(rows, [rows[0]["name"]]) == []

    def test_component_without_a_selected_member_is_excluded(self):
        rows = _rows_for("comp", _dell_steps())
        assert group_progressions(rows, ["unrelated node"]) == []

    def test_multi_component_membership_appears_in_both(self):
        shared = _dell_steps()[0]
        rows = (_rows_for("comp-a", _dell_steps()[:3])
                + _rows_for("comp-b", [shared, _dell_steps()[3]]))
        groups = group_progressions(rows, [shared["name"]])
        assert {g["component"] for g in groups} == {"comp-a", "comp-b"}


class TestProgressionRenewal:
    def test_first_sight_is_full(self):
        form, fp, new = progression_renewal("comp", _dell_steps(), {}, 1)
        assert form == "full" and new == [] and fp.startswith("prog|")

    def test_unchanged_and_fresh_is_standing(self):
        steps = _dell_steps()
        ledger: dict = {}
        _, fp, _ = progression_renewal("comp", steps, ledger, 1)
        commit_progression(ledger, "comp", fp, [s["name"] for s in steps], 1)
        form, _, _ = progression_renewal("comp", steps, ledger, 3)
        assert form == "standing"

    def test_unchanged_and_stale_is_full_again(self):
        steps = _dell_steps()
        ledger: dict = {}
        _, fp, _ = progression_renewal("comp", steps, ledger, 1)
        commit_progression(ledger, "comp", fp, [s["name"] for s in steps], 1)
        form, _, _ = progression_renewal("comp", steps, ledger, 11)
        assert form == "full"

    def test_new_step_is_advance_with_the_delta_named(self):
        steps = _dell_steps()
        ledger: dict = {}
        _, fp, _ = progression_renewal("comp", steps[:-2], ledger, 1)
        commit_progression(
            ledger, "comp", fp, [s["name"] for s in steps[:-2]], 1
        )
        form, _, new = progression_renewal("comp", steps, ledger, 2)
        assert form == "advance"
        assert set(new) == {"Medium 3.5 serving at 256k",
                            "What is the production serving configuration?"}

    def test_revised_content_without_new_steps_is_full(self):
        steps = _dell_steps()
        ledger: dict = {}
        _, fp, _ = progression_renewal("comp", steps, ledger, 1)
        commit_progression(ledger, "comp", fp, [s["name"] for s in steps], 1)
        revised = [dict(s) for s in steps]
        revised[-2]["description"] = "revised terminus body"
        form, _, new = progression_renewal("comp", revised, ledger, 2)
        assert form == "full" and new == []


class TestFormatProgression:
    def test_terminus_is_last_state_step_not_trailing_question(self):
        lines, delivered = format_progression(
            "stnamcvdl200", _dell_steps(), ["Mistral Large 2411 serving"]
        )
        now = next(ln for ln in lines if ln.startswith("  NOW"))
        assert "Medium 3.5 serving at 256k" in now
        assert "Medium 3.5 serving at 256k" in delivered["bodies"]

    def test_frontier_step_renders_as_open_line(self):
        lines, _ = format_progression("comp", _dell_steps(), ["Mistral Large 2411 serving"])
        assert any(
            ln.startswith("  ? open (Question)") for ln in lines
        )

    def test_stale_step_arrives_tensed_and_before_its_resolution(self):
        # The standing falsification test (design of record §8): the stale
        # CNTLM claim appears as a DATED step, temporally before the
        # resolution — never as an unmodalized present-tense assertion.
        lines, _ = format_progression(
            "stnamcvdl200", _dell_steps(),
            ["XE8640 discovery: sole constraint is stale cntlm credentials"],
        )
        text = "\n".join(lines)
        stale = next(ln for ln in lines if "stale cntlm" in ln)
        assert "2026-08-03" in stale
        assert text.index("stale cntlm") < text.index("repointed cntlm")
        assert text.index("repointed cntlm") < text.index("NOW (2026-08-08)")

    def test_seed_distinct_from_terminus_gets_a_body(self):
        lines, delivered = format_progression(
            "comp", _dell_steps(),
            ["XE8640 discovery: sole constraint is stale cntlm credentials"],
        )
        seed_line = next(ln for ln in lines if ln.startswith("  SEED"))
        assert "jump-box proxy" in seed_line  # the description, not a handle
        assert len(delivered["bodies"]) == 2

    def test_protected_step_survives_while_middles_fold(self):
        mk = lambda i: {
            "name": f"middle step {i:02d}", "type": "Observation",
            "description": "", "sort_key": f"2026-07-{i + 1:02d}T00:00:00Z",
            "encounter": "E", "enc_t": "",
        }
        steps = [mk(i) for i in range(20)]
        steps[5]["name"] = "the previously-delivered standalone claim"
        lines, _ = format_progression(
            "comp", steps, [steps[-1]["name"]],
            protected_names={"the previously-delivered standalone claim"},
        )
        text = "\n".join(lines)
        assert "the previously-delivered standalone claim" in text
        assert "more steps folded" in text
        assert "middle step 00" in text  # origin survives

    def test_cap_is_respected(self):
        mk = lambda i: {
            "name": f"step {i:02d} with a long semantic name " + "x" * 60,
            "type": "Observation", "description": "d " * 120,
            "sort_key": f"2026-07-{(i % 28) + 1:02d}T00:00:00Z",
            "encounter": "E", "enc_t": "",
        }
        steps = [mk(i) for i in range(30)]
        lines, _ = format_progression("comp", steps, [steps[0]["name"]])
        assert sum(len(ln) + 1 for ln in lines) <= 1200 + 400  # bodies floor

    def test_standing_form_is_one_line(self):
        lines, delivered = format_progression(
            "comp", _dell_steps(), ["Mistral Large 2411 serving"],
            form="standing",
        )
        assert len(lines) == 1
        assert "progression standing" in lines[0]
        assert "Medium 3.5" in lines[0]  # the terminus is named
        assert delivered["bodies"] == []

    def test_advance_form_is_terminus_plus_delta(self):
        lines, delivered = format_progression(
            "comp", _dell_steps(), ["Mistral Large 2411 serving"],
            form="advance",
            new_step_names=["Medium 3.5 serving at 256k"],
        )
        text = "\n".join(lines)
        assert "progression advanced" in lines[0]
        assert "NOW (2026-08-08)" in text
        # The old resolved steps do NOT re-render in an advance.
        assert "repointed cntlm" not in text
        assert "Medium 3.5 serving at 256k" in delivered["bodies"]

    def test_terminus_grounding_renders_inline(self):
        lines, _ = format_progression(
            "comp", _dell_steps(), ["Mistral Large 2411 serving"],
            grounding=["GROUNDS → A GPU is vacated when the driver finishes teardown"],
        )
        assert any("A GPU is vacated" in ln for ln in lines)


class TestFormatPayloadProgressions:
    def _block(self, component="stnamcvdl200"):
        lines, delivered = format_progression(
            component, _dell_steps(), ["Mistral Large 2411 serving"]
        )
        return {"component": component, "lines": lines,
                "bodies": delivered["bodies"], "steps": delivered["steps"]}

    def test_progressions_render_before_neighborhood(self):
        p, delivered = format_payload(
            seed_terms=["dell"], seed_mode="blended",
            core_conflicts=[], parked_conflicts=[],
            neighborhood_nodes=[{"name": "hubless", "type": "Concept",
                                 "description": "d"}],
            neighborhood_edges=[], open_threads=[], trajectory=[],
            progressions=[self._block()],
        )
        assert "PROGRESSIONS — what you hold about these" in p
        assert p.index("PROGRESSIONS") < p.index("NEIGHBORHOOD")
        assert delivered["progressions"] == ["stnamcvdl200"]
        assert "Medium 3.5 serving at 256k" in delivered["prog_bodies"]

    def test_blocks_yield_atomically_never_mid_block(self):
        # A squeezed progression could strand the stale step without its
        # resolution — blocks are kept whole or dropped whole.
        blocks = [self._block("comp-a"), self._block("comp-b")]
        block_chars = sum(len(ln) + 1 for ln in blocks[0]["lines"])
        p, delivered = format_payload(
            seed_terms=["dell"], seed_mode="blended",
            core_conflicts=[], parked_conflicts=[],
            neighborhood_nodes=[], neighborhood_edges=[],
            open_threads=[], trajectory=[],
            progressions=blocks,
            max_chars=block_chars + 250,  # room for one block, not two
        )
        assert delivered["progressions"] == ["comp-a"]
        assert "comp-b" not in p

    def test_as_of_tense_label_renders(self):
        p = _payload(neighborhood_nodes=[
            {"name": "a hubless claim", "type": "Observation",
             "description": "d", "as_of": "2026-08-03"},
        ])
        assert "a hubless claim (Observation) [as of 2026-08-03]" in p

    def test_convergence_annotation_renders(self):
        p = _payload(neighborhood_nodes=[
            {"name": "an instance", "type": "Observation", "description": "d",
             "convergence": 'instantiates "the principle" — one of 5 grounding instances'},
        ])
        assert '↳ instantiates "the principle" — one of 5' in p


class TestFormatDeltaProgression:
    def test_recognition_carries_progression_and_terminus(self):
        p = format_delta(
            [{"name": "XE8640 discovery", "type": "Observation",
              "encounters": ["Encounter Aug 3"],
              "progression": {"component": "stnamcvdl200",
                              "terminus": "Medium 3.5 serving at 256k"}}],
            [],
        )
        assert "step in progression stnamcvdl200" in p
        assert "current terminus: Medium 3.5 serving at 256k" in p

    def test_recognition_that_is_the_terminus_says_so(self):
        p = format_delta(
            [{"name": "Medium 3.5 serving at 256k", "type": "Observation",
              "encounters": [],
              "progression": {"component": "stnamcvdl200",
                              "terminus": "Medium 3.5 serving at 256k"}}],
            [],
        )
        assert "terminus of progression stnamcvdl200" in p
        assert "current terminus:" not in p
