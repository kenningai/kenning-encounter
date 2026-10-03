"""The T0-by-voice runner's corpus checks (scripts/matcher_t0.py).

These are the gates that run before any provider call: a malformed corpus
is refused, an arc that names its target's words is a leak, and recall is
computed per repeat. Each is shown saying yes and no.
"""

import importlib.util
from pathlib import Path

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
                  "match failed: HTTPStatusError: Client error '429 Too Many Requests'"):
            assert t0.transient(e)

    def test_the_voices_own_failures_do_not(self):
        for e in ("malformed matcher output (12 chars, 0 selections)",
                  "match failed: HTTPStatusError: Client error '400 Bad Request'"):
            assert not t0.transient(e)


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
