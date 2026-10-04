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
