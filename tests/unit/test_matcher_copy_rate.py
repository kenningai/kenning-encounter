"""The copy-rate instrument must read high on a sidecar that quotes its
nodes and low on one that paraphrases them, or its figure means nothing."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "matcher_copy_rate", _ROOT / "scripts" / "matcher_copy_rate.py")
assert _spec is not None and _spec.loader is not None
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

DESC = ("The verify step scans the build output for forbidden markers and "
        "exits non-zero, writing nothing, if any survive the rename pass.")
NODES = {"A guard in code": DESC}


def test_longest_run_counts_consecutive_words_only():
    assert cr.longest_run(cr.words("scans the build output"), cr.words(DESC)) == 4
    assert cr.longest_run(cr.words("output build the scans"), cr.words(DESC)) == 1
    assert cr.longest_run(cr.words("entirely unrelated wording"), cr.words(DESC)) == 0


def test_a_quoting_sidecar_reads_high():
    quoted = {"A guard in code": {"meaning": "Scans the build output for forbidden "
              "markers and exits non-zero, writing nothing, if any survive."}}
    r = cr.measure(quoted, NODES)
    assert r["longest_run_max"] >= cr.LONG_RUN
    assert r["share5_p50"] > 0.9 and r["share5_half_or_more"] == 1


def test_a_paraphrasing_sidecar_reads_low():
    para = {"A guard in code": {"meaning": "A release check that refuses to publish "
            "when banned terms remain; relevant whenever safety rests on remembering."}}
    r = cr.measure(para, NODES)
    assert r["longest_run_max"] < 3
    assert r["share5_p50"] == 0.0 and r["share5_half_or_more"] == 0


def test_the_name_counts_as_source_and_missing_nodes_are_counted_not_scored():
    side = {"A guard in code": {"meaning": "A guard in code beats vigilance."},
            "Gone": {"meaning": "anything"}}
    r = cr.measure(side, NODES)
    assert r["longest_run_max"] == 4 and r["nodes"] == 1 and r["not_in_dump"] == 1


def test_the_report_carries_no_node_text():
    out = cr.render({"x:y": cr.measure({"A guard in code": {"meaning": DESC}}, NODES)},
                    cr.datetime(2026, 10, 9, tzinfo=cr.timezone.utc), 1)
    assert "forbidden markers" not in out and "x:y" in out
