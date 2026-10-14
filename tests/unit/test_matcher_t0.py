"""The T0-by-voice runner's corpus checks (scripts/matcher_t0.py).

These are the gates that run before any provider call: a malformed corpus
is refused, an arc that names its target's words is a leak, and recall is
computed per repeat. Each is shown saying yes and no.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "matcher_t0", Path(__file__).resolve().parents[2] / "scripts" / "matcher_t0.py"
)
t0 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(t0)


def _case(**over):
    c = {
        "id": "H01", "set": "heldout", "stratum": "unnamed-constraint",
        "target": "A guard in code, not in vigilance",
        "decoy": "The public face is a build, not a sync",
        "trajectory": ["we keep forgetting the rebuild", "it went stale for days",
                       "nobody noticed until the outage"],
        "prompt": "should I add a reminder to the README?",
        "rationale": "the fix proposed is a human reminder; the held principle says otherwise",
    }
    c.update(over)
    return c


def _corpus(*cases):
    return {"corpus": {"author": "Desktop", "written": "2026-10-02"}, "cases": list(cases)}


class TestValidate:
    def test_a_well_formed_corpus_passes(self):
        assert t0.validate_corpus(_corpus(_case())) == []

    def test_every_problem_is_listed(self):
        bad = _case(id="H01", trajectory=["one", "two"], stratum="vibes", set="test",
                    rationale="", decoy="A guard in code, not in vigilance")
        problems = t0.validate_corpus(_corpus(bad, _case()))
        text = "\n".join(problems)
        for needle in ("3-6", "stratum", "set must", "rationale", "decoy is the target",
                       "duplicate id"):
            assert needle in text

    def test_corpus_metadata_is_required(self):
        assert "corpus.author missing" in t0.validate_corpus({"cases": [_case()]})

    def test_not_an_object(self):
        assert t0.validate_corpus([_case()])


class TestLeaks:
    def test_clean_arc_has_no_leak(self):
        assert t0.leaks(_case()) == []

    def test_a_target_word_in_the_arc_is_a_leak(self):
        c = _case(prompt="maybe a guard would help, not vigilance")
        assert t0.leaks(c) == ["guard", "vigilance"]

    def test_stopwords_and_short_words_do_not_count(self):
        # "code" is a content word; "not", "in", "a" are not.
        assert t0.leaks(_case(prompt="not in a")) == []
        assert t0.leaks(_case(prompt="the code")) == ["code"]


class TestFunctionWords:
    def test_a_modal_is_not_a_leak(self):
        # The first held-out corpus was refused on "will" used as a modal in
        # a target's name. A modal is a function word.
        c = _case(target="A published schedule says a backup will run, not that it ran",
                  prompt="it will be fine once it ships")
        assert t0.leaks(c) == []

    def test_a_content_word_still_is(self):
        c = _case(target="A published schedule says a backup will run, not that it ran",
                  prompt="the published plan will be fine")
        assert t0.leaks(c) == ["published"]


class TestTransient:
    def test_provider_failures_retry(self):
        for e in ("match failed: ReadTimeout: ",
                  "match failed: HTTPStatusError: Server error '503 Service Unavailable'",
                  "match failed: HTTPStatusError: Client error '429 Too Many Requests'",
                  "match failed: HTTPStatusError: Server error '529 '"):
            assert t0.transient(e)

    def test_the_voices_own_failures_do_not(self):
        for e in ("malformed matcher output (12 chars, 0 selections)",
                  "match failed: HTTPStatusError: Client error '400 Bad Request'",
                  "match failed: ValueError: the model declined the request (stop_reason refusal)",
                  # A length that reads like a status is still the voice's answer.
                  "malformed matcher output (512 chars, 0 selections)",
                  "malformed matcher output (429 chars, 0 selections)"):
            assert not t0.transient(e)

    def test_an_account_failure_pauses_and_is_never_a_result(self):
        # Auth and billing are the provider's state. Not retried, since
        # retrying will not clear them; not recorded as the arm's answer.
        for e in ("match failed: HTTPStatusError: Client error '402 Payment Required'",
                  "match failed: HTTPStatusError: Client error '401 Unauthorized'",
                  "match failed: HTTPStatusError: Client error '403 Forbidden'"):
            assert t0.account_failure(e) and not t0.transient(e)
        for e in ("match failed: HTTPStatusError: Client error '400 Bad Request'",
                  "malformed matcher output (402 chars, 0 selections)",
                  "match failed: ReadTimeout: "):
            assert not t0.account_failure(e)


class TestArms:
    def test_every_voice_runs_in_both_shapes(self):
        assert set(t0.ARMS.values()) == {
            (p, shape) for p in t0.MODEL_ENV for shape in ("flat", "split")}

    def test_every_pair_of_voices_is_contrasted(self):
        assert {frozenset(pair) for pair in t0.VOICE_PAIRS} == {
            frozenset((a, b)) for a in t0.MODEL_ENV for b in t0.MODEL_ENV if a != b}

    def test_a_key_shaped_like_any_provider_is_redacted(self):
        out = t0.redact("x sk-ant-api03-ABCDEFGHIJKLMNOP y")
        assert "ABCDEFGHIJKLMNOP" not in out


class TestSummarize:
    def test_recall_per_repeat(self):
        assert t0.summarize([3, None, 14], 30) == 2 / 3
        assert t0.summarize([3, None, 14], 12) == 1 / 3
        assert t0.summarize([], 30) == 0.0


class TestSharedOrder:
    def test_one_order_over_the_common_nodes(self):
        g = {"a": {}, "b": {}, "c": {}, "x": {}}
        o = {"c": {}, "b": {}, "a": {}, "y": {}}
        order, dropped = t0.shared_order({"gemini": g, "openai": o}, "gemini")
        assert order == ["a", "b", "c"]          # gemini's order, common nodes only
        assert dropped == {"gemini": ["x"], "openai": ["y"]}
        order2, _ = t0.shared_order({"gemini": g, "openai": o}, "openai")
        assert order2 == ["c", "b", "a"]


class TestSchedule:
    def test_seeded_complete_and_interleaved(self):
        a = t0.schedule(["H01", "H02"], ["gemini-flat", "openai-split"], 3, seed=7)
        b = t0.schedule(["H01", "H02"], ["gemini-flat", "openai-split"], 3, seed=7)
        assert a == b and len(a) == 2 * 2 * 2 * 3 and len(set(a)) == len(a)
        arms_in_first_half = {t[0] for t in a[: len(a) // 2]}
        assert arms_in_first_half == {"gemini-flat", "openai-split"}  # not arm-by-arm
        assert t0.schedule(["H01", "H02"], ["gemini-flat", "openai-split"], 3, seed=8) != a


class TestCaseScores:
    def test_errors_are_neither_hit_nor_miss(self):
        s = t0.case_scores([{"rank": 1}, {"error": "503"}, {"rank": None}])
        assert s == {"recall@30": 0.5, "recall@12": 0.5, "mrr": 0.5}
        assert t0.case_scores([{"error": "x"}]) is None

    def test_rank_beyond_twelve(self):
        s = t0.case_scores([{"rank": 20}])
        assert s["recall@30"] == 1.0 and s["recall@12"] == 0.0 and s["mrr"] == 1 / 20


class TestSignFlip:
    def test_no_difference_is_p_one(self):
        assert t0.sign_flip_p([0.0, 0.0, 0.0]) == 1.0

    def test_a_consistent_difference_is_small(self):
        assert t0.sign_flip_p([1.0] * 12) < 0.001

    def test_a_balanced_difference_is_large(self):
        assert t0.sign_flip_p([1.0, -1.0] * 6) > 0.5

    def test_the_seeded_path_for_many_cases(self):
        p = t0.sign_flip_p([0.5] * 20, seed=1, n=2000)
        assert p < 0.01


class TestClaim:
    def _scores(self, vals):
        return {f"H{i:02d}": {"recall@30": v, "recall@12": v, "mrr": v} for i, v in enumerate(vals)}

    def test_claimed_when_outside_band_and_significant(self):
        r = t0.contrast(self._scores([1.0] * 10), self._scores([0.0] * 10), "recall@30")
        assert r["diff"] == 10 and r["claim"]

    def test_not_claimed_inside_the_band(self):
        r = t0.contrast(self._scores([1.0, 1.0] + [0.0] * 8), self._scores([0.0] * 10), "recall@30")
        assert r["diff"] == 2 and not r["claim"]

    def test_not_claimed_when_not_significant(self):
        a = self._scores([1, 0, 1, 0, 1, 0, 1, 1])
        b = self._scores([0, 1, 0, 1, 0, 1, 0, 0])
        r = t0.contrast(a, b, "recall@30")
        assert abs(r["diff"]) <= 2 or r["p"] >= 0.05
        assert not r["claim"]


class TestIneligible:
    def test_a_target_or_decoy_on_the_list_is_named(self):
        names = {"A guard in code, not in vigilance", "Some decoy"}
        assert t0.ineligible_uses([_case()], names) == [
            "H01: target is ineligible (A guard in code, not in vigilance)"]
        assert t0.ineligible_uses([_case(target="Other", decoy="Some decoy")], names) == [
            "H01: decoy is ineligible (Some decoy)"]

    def test_none_when_clear(self):
        assert t0.ineligible_uses([_case()], {"Unrelated node"}) == []


class TestAuthorBriefGate:
    """scripts/t0_author_brief.py: the author's copy may carry no result or
    operator material. Two-sided on constructed text, independent of the
    lab/ files (which never ship)."""

    def _gate(self):
        import pytest

        path = Path(__file__).resolve().parents[2] / "scripts" / "t0_author_brief.py"
        if not path.exists():  # lab tooling: not part of every tree
            pytest.skip("t0_author_brief.py is not in this tree")
        spec = importlib.util.spec_from_file_location("t0_author_brief", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_result_material_is_caught(self):
        g = self._gate()
        for leak in ("gpt-6-luna is the winning voice",
                     "MRR 0.77 against 0.30",
                     "recall@30 with the arc",
                     "see A4 for the verdict",
                     "## 6 · How the results will be read"):
            assert g.forbidden_hits(leak), leak

    def test_a_clean_brief_passes(self):
        g = self._gate()
        clean = ("## 1 · What you are building\nTwo voices exist: Gemini "
                 "gemini-3.5-flash-lite and OpenAI gpt-6-luna. Which one should "
                 "lead is the question this corpus serves.\n")
        assert g.forbidden_hits(clean) == []


class TestThreeVoiceReport:
    """The report over three voices, on synthetic runs: a real gap is
    claimed and counted, and no gap claims nothing."""

    CASES = [{"id": f"H{i:02d}", "stratum": "old", "target": "t", "set": "heldout"}
             for i in range(1, 13)]

    def _state(self, rank_by_provider):
        runs = {}
        for arm, (provider, _) in t0.ARMS.items():
            for c in self.CASES:
                for mode in t0.MODES:
                    for rep in range(2):
                        runs[t0.task_key(arm, c["id"], mode, rep)] = {
                            "rank": rank_by_provider[provider], "decoy_rank": None, "ms": 900.0,
                            "prompt_tokens": 100, "cached_tokens": 90,
                            "output_tokens": 20, "reasoning_tokens": 0,
                        }
        meta = {"corpus": "c.json", "corpus_sha256": "0" * 64, "set": "heldout", "seed": 1,
                "arms": list(t0.ARMS), "order_from": "gemini", "order_size": 10,
                "sidecar_sha256": {p: "0" * 64 for p in t0.MODEL_ENV},
                "sidecar_sizes": {p: 10 for p in t0.MODEL_ENV},
                "dropped": {p: [] for p in t0.MODEL_ENV}, "excluded": {},
                "models": {p: p + "-model" for p in t0.MODEL_ENV},
                "native_shapes": {p: "split" for p in t0.MODEL_ENV}, "started": "now"}
        return {"meta": meta, "repeats": 2, "runs": runs}

    def test_a_gap_is_claimed_for_the_pair_that_has_it(self):
        out = t0.render(self._state({"gemini": 1, "openai": 1, "anthropic": None}),
                        self.CASES, list(t0.ARMS), 0, {})
        assert "OpenAI vs Anthropic, recall@12: a MODEL difference is claimed" in out
        assert "Gemini vs OpenAI, recall@12: no model difference is claimed" in out
        assert "calls that reasoned 0 of" in out
        assert "head of the list in order (scored as misses): 0 of" in out
        assert "Multiplicity: 27 tests" in out
        assert "(Bonferroni): none" not in out

    def test_no_gap_claims_nothing(self):
        out = t0.render(self._state({"gemini": 1, "openai": 1, "anthropic": 1}),
                        self.CASES, list(t0.ARMS), 0, {})
        assert "a MODEL difference is claimed" not in out
        assert "(Bonferroni): none." in out


class TestCrossedArms:
    """One voice matching over another's meanings: the arms that measure
    whether a voice reads its own sentences better."""

    def test_every_ordered_pair_exists_once_and_is_not_a_default(self):
        assert len(t0.CROSSED) == 6
        assert t0.arm_spec("gemini-reads-anthropic") == ("gemini", "split", "anthropic")
        assert t0.arm_spec("anthropic-flat") == ("anthropic", "flat", "anthropic")
        assert not set(t0.CROSSED) & set(t0.ARMS)

    def _scores(self, own, crossed, better=0.0):
        # per-case MRR for the four arms of the gemini/anthropic pair
        ids = [f"H{i:02d}" for i in range(12)]
        mk = lambda v: {c: {"recall@30": 1.0, "recall@12": 1.0, "mrr": v} for c in ids}
        return {"gemini-split": mk(own + better), "anthropic-split": mk(own),
                "gemini-reads-anthropic": mk(crossed + better), "anthropic-reads-gemini": mk(crossed)}

    def test_an_own_meanings_advantage_is_found(self):
        r = t0.interaction(self._scores(0.9, 0.4), "gemini", "anthropic", "mrr")
        assert r["diff"] == pytest.approx(0.5) and r["claim"]

    def test_a_voice_that_simply_matches_better_cancels_out(self):
        r = t0.interaction(self._scores(0.5, 0.5, better=0.4), "gemini", "anthropic", "mrr")
        assert r["diff"] == pytest.approx(0.0) and not r["claim"]

    def test_it_can_read_negative(self):
        r = t0.interaction(self._scores(0.3, 0.8), "gemini", "anthropic", "mrr")
        assert r["diff"] == pytest.approx(-0.5) and r["claim"]

    def test_none_when_an_arm_was_not_run(self):
        sc = self._scores(0.9, 0.4)
        del sc["anthropic-reads-gemini"]
        assert t0.interaction(sc, "gemini", "anthropic", "mrr") is None


class TestRawArms:
    def test_one_per_voice_split_and_arc_only(self):
        assert set(t0.RAW) == {f"{m}-reads-raw" for m in t0.MODEL_ENV}
        assert t0.arm_spec("anthropic-reads-raw") == ("anthropic", "split", "raw")

    def test_the_line_is_the_authors_opening_words_at_a_meanings_length(self):
        desc = " ".join(f"w{i}" for i in range(80))
        out = t0.raw_meanings({"n": desc, "empty": ""}, ["n", "empty"])
        assert out["n"]["meaning"].split() == [f"w{i}" for i in range(t0.RAW_WORDS)]
        assert out["empty"]["meaning"] == ""

    def test_a_dump_missing_a_node_is_refused(self):
        with pytest.raises(SystemExit):
            t0.raw_meanings({"n": "x"}, ["n", "absent"])

    def test_crossed_and_raw_arms_run_with_the_arc_only(self):
        tasks = t0.schedule(["H01"], ["gemini-split", "gemini-reads-raw", "gemini-reads-openai"], 2, seed=1)
        modes = {}
        for arm, _, mode, _ in tasks:
            modes.setdefault(arm, set()).add(mode)
        assert modes == {"gemini-split": {"arc", "prompt_alone"},
                         "gemini-reads-raw": {"arc"}, "gemini-reads-openai": {"arc"}}

    def _sc(self, v):
        return {f"H{i:02d}": {"recall@30": 1.0, "recall@12": 1.0, "mrr": v} for i in range(12)}

    def test_a_voice_at_home_in_the_authors_words_loses_less(self):
        sc = {"anthropic-split": self._sc(0.8), "anthropic-reads-raw": self._sc(0.75),
              "gemini-split": self._sc(0.8), "gemini-reads-raw": self._sc(0.3)}
        r = t0.raw_cost_gap(sc, "gemini", "anthropic", "mrr")
        assert r["diff"] == pytest.approx(0.45) and r["claim"]

    def test_equal_costs_read_zero_whatever_the_voices_levels(self):
        sc = {"anthropic-split": self._sc(0.9), "anthropic-reads-raw": self._sc(0.7),
              "gemini-split": self._sc(0.5), "gemini-reads-raw": self._sc(0.3)}
        r = t0.raw_cost_gap(sc, "gemini", "anthropic", "mrr")
        assert r["diff"] == pytest.approx(0.0) and not r["claim"]
        del sc["gemini-reads-raw"]
        assert t0.raw_cost_gap(sc, "gemini", "anthropic", "mrr") is None


class TestAgreement:
    def _runs(self, tops):
        return {t0.task_key(arm, "H01", "arc", rep): {"top": top}
                for (arm, rep), top in tops.items()}

    def test_identical_and_disjoint_selections(self):
        runs = self._runs({("a", 0): [1, 2, 3], ("b", 0): [1, 2, 3], ("c", 0): [7, 8, 9]})
        assert t0.agreement(runs, "a", "b", ["H01"], 1) == 1.0
        assert t0.agreement(runs, "a", "c", ["H01"], 1) == 0.0

    def test_an_arm_against_itself_compares_different_repeats_only(self):
        runs = self._runs({("a", 0): [1, 2], ("a", 1): [2, 3]})
        assert t0.agreement(runs, "a", "a", ["H01"], 2) == pytest.approx(1 / 3)

    def test_none_for_runs_made_before_selections_were_kept(self):
        runs = {t0.task_key("a", "H01", "arc", 0): {"rank": 1}}
        assert t0.agreement(runs, "a", "a", ["H01"], 1) is None

