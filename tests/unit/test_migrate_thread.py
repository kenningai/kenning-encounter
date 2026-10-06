"""Gate the thread migration's pure half: the order and the gate.

The live run of the migration is in tests/integration. What is gated here,
DB-free, is that the gate says both no and yes, and that structure wins over
stamp in the order it is asked to judge.
"""
from __future__ import annotations

import pytest

from kenning_encounter import migrate_thread as mt


def test_the_gate_refuses_an_inverted_stamp():
    encs, ne, nl = mt.synthetic()
    assert mt.gate(mt.stamp_order(encs), encs, ne, nl)


def test_the_gate_passes_the_structural_order():
    encs, ne, nl = mt.synthetic()
    assert mt.gate(mt.thread_order(encs, ne, nl), encs, ne, nl) == []


def test_structure_wins_over_stamp_and_the_override_is_listed():
    encs, ne, nl = mt.synthetic()
    order = mt.thread_order(encs, ne, nl)
    assert order == ["A1", "A2", "B1"]
    assert mt.overrides(order, encs) == ["A1", "A2"]


def test_a_stamp_order_that_agrees_with_structure_passes():
    encs, ne, nl = mt.synthetic()
    encs["A2"]["t"] = [15, 0]
    assert mt.gate(mt.stamp_order(encs), encs, ne, nl) == []
    assert mt.overrides(mt.thread_order(encs, ne, nl), encs) == []


def test_genesis_order_is_a_constraint_not_only_a_stamp():
    """B's stamp says it opened first, but NEXT_LOCUS says A began first."""
    encs, ne, nl = mt.synthetic()
    encs["B1"]["t"] = [1, 0]
    order = mt.thread_order(encs, ne, nl)
    assert order.index("A1") < order.index("B1")
    assert mt.gate(mt.stamp_order(encs), encs, ne, nl)


def test_seal_ticks_only_for_stamped_seals_and_never_before_opening():
    encs, ne, nl = mt.synthetic()
    encs["A1"]["ts"] = None
    order = mt.thread_order(encs, ne, nl)
    ticks = mt.seal_ticks(order, encs)
    assert "A1" not in ticks
    for eid, t in ticks.items():
        assert t >= order.index(eid)


def test_a_cycle_in_the_lived_constraints_refuses():
    encs, ne, nl = mt.synthetic()
    with pytest.raises(mt.Refused):
        mt.thread_order(encs, ne + [("A2", "A1")], nl)


def test_prove_gate_runs_clean():
    mt.prove_gate()


def test_the_migration_writes_nothing_but_the_thread():
    """Additive only: every write statement targets tick, seal_tick or NEXT_TICK."""
    writes = [mt.WRITE_TICKS, mt.WRITE_EDGES, mt.WRITE_SEALS]
    for w in writes:
        assert "DELETE" not in w.upper()
        assert "REMOVE" not in w.upper()
    assert "rebuilt_from = 't_exist'" in mt.WRITE_EDGES
    assert "seal_tick_rebuilt_from = 't_sealed'" in mt.WRITE_SEALS


def _seal_ticks_by_scan(order, encs):
    """The definition, read literally: O(n^2), kept only as the oracle."""
    out = {}
    for eid, e in encs.items():
        if e.get("ts") is None:
            continue
        ts = tuple(e["ts"])
        latest = max(i for i, x in enumerate(order) if tuple(encs[x]["t"]) <= ts or x == eid)
        out[eid] = max(latest, order.index(eid))
    return out


def _shuffled_history(rng, n):
    """Stamps deliberately out of thread order, with ties, so the running
    maximum is what carries the answer rather than the sort alone."""
    order = [f"E{i}" for i in range(n)]
    encs = {}
    for e in order:
        t = [rng.randrange(n // 2 + 1), 0]
        ts = None if rng.random() < 0.3 else [t[0] + rng.randrange(-2, n // 3 + 1), 0]
        encs[e] = {"locus": "L", "t": t, "ts": ts}
    return order, encs


def test_seal_ticks_agrees_with_its_definition_on_shuffled_histories():
    import random
    rng = random.Random(20261006)
    for n in (1, 2, 5, 40, 300):
        for _ in range(20):
            order, encs = _shuffled_history(rng, n)
            assert mt.seal_ticks(order, encs) == _seal_ticks_by_scan(order, encs)


def test_the_agreement_check_can_fail():
    """Two-sided: dropping the running maximum (taking the last stamp's own
    tick) must disagree with the definition somewhere in the same corpus."""
    import bisect
    import random

    def no_running_max(order, encs):
        pos = {e: i for i, e in enumerate(order)}
        by_stamp = sorted(order, key=lambda x: tuple(encs[x]["t"]))
        stamps = [tuple(encs[x]["t"]) for x in by_stamp]
        out = {}
        for eid, e in encs.items():
            if e.get("ts") is None:
                continue
            k = bisect.bisect_right(stamps, tuple(e["ts"]))
            out[eid] = max(pos[by_stamp[k - 1]], pos[eid]) if k else pos[eid]
        return out

    rng = random.Random(20261006)
    histories = [_shuffled_history(rng, 40) for _ in range(20)]
    assert any(no_running_max(o, e) != _seal_ticks_by_scan(o, e) for o, e in histories)


def test_seal_ticks_scales_to_a_long_history():
    """A long synthetic history, every opening sealed: the scan would be ~2.5e9 steps."""
    import random
    import time
    rng = random.Random(7)
    order, encs = _shuffled_history(rng, 50_000)
    for e in encs.values():
        e["ts"] = e["ts"] or [e["t"][0] + 1, 0]
    start = time.perf_counter()
    mt.seal_ticks(order, encs)
    assert time.perf_counter() - start < 2.0
