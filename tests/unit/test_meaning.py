"""Unit tests for the meaning matcher (T0, temporary — infuse_meaning).

Prefix construction, trajectory assembly, transcript parsing, matcher
output parsing, sidecar loading, and report formatting are pure and tested
here. The generation call is transport; every failure mode collapses to
MeaningUnavailable and the fallback path is the tool's guarantee.
"""

import json

import pytest

from kenning_encounter.meaning import (
    MATCHER_HEADER,
    MEANING_PROMPT,
    MeaningIndex,
    MeaningUnavailable,
    assemble_trajectory,
    build_prefix,
    compress_meaning,
    content_hash,
    deferral_summary,
    format_meaning_report,
    match_meanings,
    parse_matcher_output,
    record_deferral,
    sidecar_diff,
    user_turns_from_transcript,
)
from kenning_encounter.meaning import _DEFERRAL_REASON_CAP

NODES = {
    "Zeta concept": {"type": "Concept", "meaning": "About z.", "hash": "b"},
    "Alpha observation": {"type": "Observation", "meaning": "About  a.", "hash": "a"},
    "Mid note": {"type": "Note", "meaning": "About m.", "hash": "c"},
}


class TestBuildPrefix:
    def test_insertion_numbering_and_header(self):
        prefix, ordered = build_prefix(NODES)
        assert ordered == ["Zeta concept", "Alpha observation", "Mid note"]
        assert prefix.startswith(MATCHER_HEADER)
        assert "1. Zeta concept :: About z." in prefix
        assert "3. Mid note :: About m." in prefix

    def test_deterministic_bytes(self):
        assert build_prefix(NODES) == build_prefix(dict(NODES.items()))

    def test_order_is_the_sidecars_not_the_alphabets(self):
        """Two-sided against the old behaviour: the same nodes in a
        different file order number differently — sorting is gone."""
        assert build_prefix(NODES) != build_prefix(dict(reversed(NODES.items())))


class TestPrefixSurvivesWrites:
    """The prefix is a cache key: a write must leave every line before it
    byte-identical, through the real store, not only through build_prefix."""

    def _index(self, tmp_path):
        idx = MeaningIndex(tmp_path / "sidecar.json")
        idx.upsert(dict(NODES))
        return idx

    def test_append_leaves_existing_prefix_byte_identical(self, tmp_path):
        idx = self._index(tmp_path)
        before, _ = idx.prefix()
        # "Aardvark" sorts FIRST — under name order this renumbered all.
        idx.upsert({"Aardvark": {"type": "Note", "meaning": "New.", "hash": "d"}})
        after, ordered = idx.prefix()
        assert after.startswith(before)
        assert ordered[-1] == "Aardvark"

    def test_edit_keeps_position(self, tmp_path):
        idx = self._index(tmp_path)
        idx.upsert({"Alpha observation": {"type": "Observation", "meaning": "Revised.", "hash": "e"}})
        prefix, ordered = idx.prefix()
        assert ordered.index("Alpha observation") == 1
        assert "1. Zeta concept :: About z." in prefix  # line before it untouched
        assert "2. Alpha observation :: Revised." in prefix

    def test_meaning_whitespace_collapsed(self):
        prefix, _ = build_prefix(NODES)
        assert "About  a." not in prefix  # double space collapsed


class TestParseMatcherOutput:
    ORDERED = ["Alpha observation", "Mid note", "Zeta concept"]

    def test_number_dash_reason_lines(self):
        raw = "3 - bears on z\n1 - the a case"
        out = parse_matcher_output(raw, self.ORDERED, 10)
        assert [s["name"] for s in out] == ["Zeta concept", "Alpha observation"]
        assert out[0]["reason"] == "bears on z"

    def test_tolerates_numbering_punctuation_variants(self):
        raw = "2. because\n 3: why\n1 — reason"
        out = parse_matcher_output(raw, self.ORDERED, 10)
        assert [s["name"] for s in out] == [
            "Mid note", "Zeta concept", "Alpha observation",
        ]

    def test_out_of_range_dropped_not_guessed(self):
        out = parse_matcher_output("7 - nope\n2 - ok\n0 - nope", self.ORDERED, 10)
        assert [s["name"] for s in out] == ["Mid note"]

    def test_dedup_and_cap(self):
        raw = "1 - a\n1 - again\n2 - b\n3 - c"
        out = parse_matcher_output(raw, self.ORDERED, 2)
        assert [s["name"] for s in out] == ["Alpha observation", "Mid note"]

    def test_bare_number_lines_parse_without_reasons(self):
        # The default request shape: numbers only, one per line.
        out = parse_matcher_output("2\n3\n1", self.ORDERED, 10)
        assert [s["name"] for s in out] == [
            "Mid note", "Zeta concept", "Alpha observation",
        ]
        assert all(s["reason"] == "" for s in out)

    def test_prose_lines_ignored(self):
        raw = "Here are my selections:\n2 - the one\nHope that helps!"
        out = parse_matcher_output(raw, self.ORDERED, 10)
        assert [s["name"] for s in out] == ["Mid note"]

    def test_empty_is_malformed_signal(self):
        assert parse_matcher_output("", self.ORDERED, 10) == []
        assert parse_matcher_output("no numbers here", self.ORDERED, 10) == []


class TestAssembleTrajectory:
    def test_current_prompt_last_and_turn_markers(self):
        text, meta = assemble_trajectory(["first", "second"], "now")
        assert text.rstrip().endswith("[current prompt] now")
        assert "[turn -2] first" in text
        assert "[turn -1] second" in text
        assert meta == {"verbatim_turns": 2, "digested_turns": 0,
                        "dropped_turns": 0}

    def test_prompt_only(self):
        text, meta = assemble_trajectory([], "just this")
        assert "[current prompt] just this" in text
        assert meta["verbatim_turns"] == 0

    def test_budget_digests_older_half_recent_verbatim(self):
        turns = [f"turn {i} " + ("x" * 3000) for i in range(8)]
        text, meta = assemble_trajectory(turns, "now", max_chars=8000)
        assert meta["verbatim_turns"] >= 1
        assert meta["digested_turns"] >= 1
        assert "story so far (older turns, digested):" in text
        # The newest prior turn survives verbatim; the oldest does not.
        assert "turn 7" in text
        # Budget is fixed: total stays near the cap, not the 24K input.
        assert len(text) < 10_000

    def test_empty_turns_filtered(self):
        _, meta = assemble_trajectory(["", "   ", "real"], "now")
        assert meta["verbatim_turns"] == 1


class TestUserTurnsFromTranscript:
    def _write(self, tmp_path, entries):
        p = tmp_path / "t.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in entries))
        return p

    def test_extracts_last_n_user_text_turns(self, tmp_path):
        entries = [
            {"type": "user", "message": {"role": "user", "content": "one"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "reply"}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "two"}]}},
            {"type": "user", "message": {"role": "user", "content": "three"}},
        ]
        assert user_turns_from_transcript(self._write(tmp_path, entries), 2) == [
            "two", "three",
        ]

    def test_skips_tool_results_meta_and_reminders(self, tmp_path):
        entries = [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "content": "output"}]}},
            {"type": "user", "isMeta": True,
             "message": {"role": "user", "content": "meta"}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>injected</system-reminder>"},
                {"type": "text", "text": "the real ask"}]}},
        ]
        assert user_turns_from_transcript(self._write(tmp_path, entries), 5) == [
            "the real ask",
        ]

    def test_garbage_lines_skipped_missing_file_empty(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text("not json\n" + json.dumps(
            {"type": "user", "message": {"role": "user", "content": "ok"}}))
        assert user_turns_from_transcript(p, 3) == ["ok"]
        assert user_turns_from_transcript(tmp_path / "absent.jsonl", 3) == []


class TestMeaningIndex:
    def test_missing_sidecar_raises_with_pointer(self, tmp_path):
        idx = MeaningIndex(tmp_path / "absent.json")
        with pytest.raises(MeaningUnavailable) as exc:
            idx.prefix()
        assert "build_meaning_sidecar" in str(exc.value)

    def test_loads_and_reloads_on_mtime_change(self, tmp_path):
        import os

        p = tmp_path / "sidecar.json"
        p.write_text(json.dumps({"nodes": NODES}))
        idx = MeaningIndex(p)
        prefix1, ordered1 = idx.prefix()
        assert idx.size == 3 and len(ordered1) == 3
        grown = {**NODES, "New node": {"type": "Concept", "meaning": "n", "hash": "d"}}
        p.write_text(json.dumps({"nodes": grown}))
        os.utime(p, (1, 2_000_000_000))
        prefix2, ordered2 = idx.prefix()
        assert len(ordered2) == 4 and prefix2 != prefix1

    def test_empty_sidecar_unavailable(self, tmp_path):
        p = tmp_path / "sidecar.json"
        p.write_text(json.dumps({"nodes": {}}))
        with pytest.raises(MeaningUnavailable):
            MeaningIndex(p).prefix()


class TestDeferralReporting:
    """The keyless-start disclosure fix, two-sided: the summary must SAY
    enough to act on and must NOT carry a node name.

    Keyless, compress_meaning raises before any network call, so the
    reconcile sweep's failure branch fires once per node; the old per-node
    warning named it, so a keyless first start wrote the agent's whole
    corpus of node names into `docker logs`, readable by anything on the
    host. Node names in a Kenning Encounter are the substrate's content, not
    identifiers."""

    # Headline-shaped, because the leak is that node names ARE prose — a
    # name-as-identifier would have been a much smaller disclosure.
    CORPUS = [
        "a gate's model of what ships is not the shipping mechanism",
        "a mark held for custody, not commerce",
        "isolation is absorbing: a node born unwired cannot be reached",
    ]

    def _keyless_run(self, corpus):
        """The keyless path: every node fails with the same reason."""
        counts: dict[str, int] = {}
        for _ in corpus:
            record_deferral(
                counts,
                MeaningUnavailable("compression not configured (GEMINI_API_KEY)"),
            )
        return counts

    def test_summary_reports_total_and_reason(self):
        """Says enough: a bare count hides WHICH failure, a bare reason hides
        HOW MUCH of the corpus is missing. Both, or the report conceals."""
        line = deferral_summary(self._keyless_run(self.CORPUS * 200))
        assert "600 deferred" in line
        assert "1 distinct reason" in line
        assert "GEMINI_API_KEY" in line

    def test_summary_never_carries_a_node_name(self):
        """The refusing half. 600 failures, zero names."""
        line = deferral_summary(self._keyless_run(self.CORPUS * 200))
        for name in self.CORPUS:
            assert name not in line
        for fragment in ("shipping mechanism", "custody", "isolation is absorbing"):
            assert fragment not in line

    def test_node_is_not_a_parameter_so_a_name_cannot_be_passed(self):
        """The guard is the signature, not the call site. record_deferral
        takes the exception and the tally — there is no argument a node
        could arrive through, so this cannot regress by someone adding an
        f-string back."""
        import inspect

        params = list(inspect.signature(record_deferral).parameters)
        assert params == ["counts", "exc"]

    def test_distinct_reasons_are_kept_and_ordered_by_frequency(self):
        counts: dict[str, int] = {}
        for _ in range(3):
            record_deferral(counts, MeaningUnavailable("timeout"))
        for _ in range(7):
            record_deferral(counts, ValueError("bad json"))
        line = deferral_summary(counts)
        assert "10 deferred (2 distinct reasons)" in line
        assert line.index("7x ValueError") < line.index("3x MeaningUnavailable")

    def test_reason_is_capped_so_an_exception_cannot_dump_content(self):
        """The reason is the one field that could reintroduce the leak if a
        future exception ever embedded what it was compressing."""
        counts: dict[str, int] = {}
        record_deferral(counts, MeaningUnavailable("x" * 5000))
        (reason,) = counts
        assert len(reason) == _DEFERRAL_REASON_CAP

    def test_empty_tally_is_empty_string_not_a_reassuring_line(self):
        """Nothing deferred must print nothing — a '0 deferred' line is a
        clean verdict from a run that may not have scanned anything."""
        assert deferral_summary({}) == ""


class TestSidecarAutomation:
    """The guard-in-code layer: on-write upsert, delete removal, and the
    startup reconcile diff. A human told to rebuild the sidecar will
    forget; these paths are why they never have to remember."""

    def test_content_hash_stable_and_content_sensitive(self):
        a = content_hash("node", "desc")
        assert a == content_hash("node", "desc")
        assert a != content_hash("node", "desc changed")
        assert a != content_hash("node2", "desc")

    def test_sidecar_diff_new_changed_unchanged_departed(self):
        graph = [
            {"name": "kept", "type": "Concept", "description": "same"},
            {"name": "edited", "type": "Note", "description": "new text"},
            {"name": "born", "type": "Observation", "description": "d"},
        ]
        sidecar = {
            "kept": {"type": "Concept", "meaning": "m",
                     "hash": content_hash("kept", "same")},
            "edited": {"type": "Note", "meaning": "m",
                       "hash": content_hash("edited", "old text")},
            "departed": {"type": "Concept", "meaning": "m", "hash": "x"},
        }
        to_compress, to_remove = sidecar_diff(graph, sidecar)
        assert sorted(n["name"] for n in to_compress) == ["born", "edited"]
        assert all("hash" in n for n in to_compress)
        assert to_remove == ["departed"]

    def test_upsert_merges_persists_atomically_and_reloads(self, tmp_path):
        p = tmp_path / "sidecar.json"
        p.write_text(json.dumps({"nodes": NODES}))
        idx = MeaningIndex(p)
        _, before = idx.prefix()
        idx.upsert({"Born node": {"type": "Observation", "hash": "h",
                                  "meaning": "fresh meaning"}})
        # Persisted (a second, independent index sees it)…
        again = json.loads(p.read_text())["nodes"]
        assert again["Born node"]["meaning"] == "fresh meaning"
        assert set(NODES) <= set(again)  # merged over, not clobbered
        # …and the live index reloads without an mtime tick.
        _, after = idx.prefix()
        assert "Born node" in after and len(after) == len(before) + 1
        assert not (tmp_path / "sidecar.tmp").exists()

    def test_upsert_creates_missing_sidecar(self, tmp_path):
        idx = MeaningIndex(tmp_path / "fresh" / "sidecar.json")
        idx.upsert({"n": {"type": "Note", "hash": "h", "meaning": "m"}})
        assert idx.size == 1

    def test_remove_drops_and_tolerates_unknown(self, tmp_path):
        p = tmp_path / "sidecar.json"
        p.write_text(json.dumps({"nodes": NODES}))
        idx = MeaningIndex(p)
        idx.remove(["Zeta concept", "never existed"])
        _, ordered = idx.prefix()
        assert "Zeta concept" not in ordered and len(ordered) == 2
        idx.remove(["also unknown"])  # no-op, no error, no rewrite

    def test_version_mismatch_forces_full_recompress(self):
        graph = [{"name": "kept", "type": "Concept", "description": "same"}]
        sidecar = {"kept": {"type": "Concept", "meaning": "m",
                            "hash": content_hash("kept", "same")}}
        # Same content, matching hash — but written under an older prompt:
        # everything recompresses. A prompt change ships as a code change.
        to_compress, _ = sidecar_diff(graph, sidecar, sidecar_version="v1")
        assert [n["name"] for n in to_compress] == ["kept"]
        to_compress, _ = sidecar_diff(graph, sidecar)  # current version
        assert to_compress == []

    def test_file_version_read_and_stamped_on_upsert(self, tmp_path):
        from kenning_encounter.meaning import PROMPT_VERSION

        p = tmp_path / "sidecar.json"
        p.write_text(json.dumps({"prompt_version": "v1", "nodes": NODES}))
        idx = MeaningIndex(p)
        assert idx.file_version() == "v1"
        idx.upsert({"n": {"type": "Note", "hash": "h", "meaning": "m"}})
        assert idx.file_version() == PROMPT_VERSION
        assert MeaningIndex(tmp_path / "absent.json").file_version() == ""

    @pytest.mark.asyncio
    async def test_compress_unconfigured_and_dead_endpoint_collapse(self):
        with pytest.raises(MeaningUnavailable):
            await compress_meaning({"name": "n"}, api_key="", model="m",
                                   endpoint="http://x")
        with pytest.raises(MeaningUnavailable) as exc:
            await compress_meaning(
                {"name": "n", "type": "Note", "description": "d"},
                api_key="k", model="m",
                endpoint="http://127.0.0.1:9/v1beta", timeout_s=0.3,
            )
        assert "compress failed" in str(exc.value)


class TestMatchConfigGuard:
    @pytest.mark.asyncio
    async def test_missing_key_raises_unavailable(self):
        with pytest.raises(MeaningUnavailable) as exc:
            await match_meanings("p", ["a"], "t", 5, api_key="", model="m",
                                 endpoint="http://x")
        assert "not configured" in str(exc.value)

    @pytest.mark.asyncio
    async def test_dead_endpoint_collapses_to_unavailable(self):
        with pytest.raises(MeaningUnavailable) as exc:
            await match_meanings(
                "p", ["a"], "t", 5, api_key="k", model="m",
                endpoint="http://127.0.0.1:9/v1beta", timeout_ms=300,
            )
        assert "match failed" in str(exc.value)


class TestFormatMeaningReport:
    def _rows(self):
        return [{"name": "Alpha observation", "type": "Observation",
                 "score": 0.4, "channels": ["meaning"]}]

    def test_sections_order_and_cache_evidence(self):
        out = format_meaning_report(
            [{"name": "Zeta concept", "reason": "bears on z"}],
            self._rows(), {"verbatim_turns": 3, "digested_turns": 1,
                           "dropped_turns": 0},
            match_ms=700.0, pipeline_ms=90.0,
            cached_tokens=14000, prompt_tokens=15500, sidecar_size=519,
        )
        assert out.startswith("ARM M — MEANING MATCHER")
        assert "TRAJECTORY: 3 verbatim turns + 1 digested" in out
        assert out.index("MATCHER SELECTION") < out.index("PIPELINE CANDIDATES")
        assert "debugging artifact, not evidence" in out
        assert "↳ bears on z" in out
        assert "cached 14000/15500 prompt tokens" in out
        assert "sidecar 519 meanings" in out

    def test_fallback_stated_with_reason(self):
        out = format_meaning_report(
            [], self._rows(), {}, match_ms=None, pipeline_ms=80.0,
            fallback=True, fallback_reason="meaning sidecar missing at x",
        )
        assert "FALLBACK" in out
        assert "sidecar missing" in out
        assert "ran current Extract behaviour instead" in out
        assert "match — | pipeline 80.0 ms" in out

    def test_meaning_prompt_embeds_all_three_fields(self):
        p = MEANING_PROMPT.format(type="Concept", name="N", description="D")
        assert "TYPE: Concept" in p and "NAME: N" in p and "DESCRIPTION: D" in p
