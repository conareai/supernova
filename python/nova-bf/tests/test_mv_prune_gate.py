"""The engage/disengage gate: prune only while it is actually paying.

Pass one is not free, so at a low prune rate it costs more than it saves —
`Fp16State` therefore watches the live fraction and stops paying when it exceeds
`params.multivector_min_prune_rate` (0.70 by default: keep pruning while at
least 70% of pairs are being ruled out). Stated as a PRUNE RATE because that is
what this path reports everywhere else; the dense two-pass states the same idea
inverted, as a live fraction (`twopass.DEFAULT_THRESHOLD = 0.50`).

THE PART THAT IS EASY TO GET WRONG, and what most of these tests are about:
the gate must NOT latch off. The prune rate CLIMBS during a run, because the
bound prunes against the running top-K threshold and that starts empty —
measured 1.6% pruned over one corpus file at k=10 against 99.1% over two. A
gate that measured once at the start would see 1.6%, disengage, and never
discover it would have reached 99%.

These exercise the state machine directly, with no GPU: it is fed plain counts.
"""
from __future__ import annotations

import pytest

from nova_bf.mv_fp16 import Fp16State


def _state(min_prune_rate=0.70):
    return Fp16State(min_prune_rate=min_prune_rate)


def test_the_gate_is_open_before_any_evidence():
    """Nothing observed yet means prune and find out -- the alternative is
    never starting."""
    assert _state()._gate_open()


def test_a_good_prune_rate_keeps_the_gate_open():
    st = _state()
    for _ in range(20):
        st.note_pruned(95, 100)              # 5% live == 95% pruned
        assert st._gate_open()


def test_a_poor_prune_rate_closes_the_gate():
    st = _state()
    for _ in range(20):
        st.note_pruned(20, 100)             # 80% live == 20% pruned
    assert not st._gate_open()


def test_the_gate_reopens_periodically_rather_than_latching():
    """The whole point. A closed gate must retry, because the rate improves as
    the run's top-K fills."""
    st = _state()
    for _ in range(20):
        st.note_pruned(10, 100)
    reopened = sum(1 for _ in range(Fp16State._REPROBE_EVERY * 3)
                   if st._gate_open())
    assert reopened >= 2, "a closed gate never retried — it latched off"


def test_a_run_that_starts_badly_and_improves_ends_up_pruning():
    """The measured trajectory: 1.6% pruned on one file, 99.1% on two. The
    gate must recover, not write the run off on its early slices."""
    st = _state()
    for _ in range(40):                   # early: almost nothing prunable
        if st._gate_open():
            st.note_pruned(16, 1000)
    assert not st._gate_open(), "should have disengaged while unproductive"

    # The top-K fills and the rate jumps; the re-probe has to notice.
    engaged = 0
    for _ in range(Fp16State._REPROBE_EVERY * 4):
        if st._gate_open():
            engaged += 1
            st.note_pruned(991, 1000)         # 99.1% pruned
    assert engaged > 5, "the gate never recovered after the rate improved"
    assert st._gate_open(), "the gate should be open again once it is paying"


def test_the_floor_is_respected():
    """A caller that lowers the floor tolerates a worse rate, not a better."""
    strict, loose = _state(min_prune_rate=0.90), _state(min_prune_rate=0.10)
    for _ in range(20):
        strict.note_pruned(50, 100)
        loose.note_pruned(50, 100)
    assert not strict._gate_open()
    assert loose._gate_open()


def test_a_floor_of_zero_disables_the_gate_entirely():
    st = _state(min_prune_rate=0.0)
    for _ in range(50):
        st.note_pruned(0, 1000)          # nothing pruned at all
        assert st._gate_open(), "min_prune_rate=0.0 must always prune"


def test_the_default_state_has_no_gate():
    """`Fp16State()` with no floor always prunes; the gate is opt-in at
    construction so a direct caller is not silently governed."""
    st = Fp16State()
    for _ in range(20):
        st.note_pruned(0, 1000)
    assert st._gate_open()


def test_an_empty_slice_does_not_move_the_estimate():
    st = _state()
    st.note_pruned(0, 0)
    assert st._prune_ema is None


def test_one_bad_slice_does_not_flip_a_healthy_gate():
    """Smoothing exists so a single unusual slice cannot disengage a run that
    is otherwise pruning well."""
    st = _state()
    for _ in range(30):
        st.note_pruned(98, 100)
    st.note_pruned(0, 100)
    assert st._gate_open()


def test_a_closed_gate_asks_to_observe_unpruned_slices():
    """How the gate learns while disengaged. Dense reads its hint off one-pass
    scoring; this reads it off a slice scored without the prune. Without it the
    only way to learn is to pay for a speculative pass one.

    `wants_probe` reports only WHETHER observing would help. How often that
    happens is the caller's one-shot per-batch latch, mirroring dense's
    `tp_probe` — so this is a plain predicate, not a counter."""
    st = _state()
    for _ in range(20):
        st.note_pruned(10, 100)
    assert not st._gate_open()
    assert st.wants_probe()
    assert st.wants_probe(), "asking twice must not consume anything"


def test_an_open_gate_does_not_ask_to_probe():
    """It is already learning from its own output; sampling again would buy a
    device sync for nothing."""
    st = _state()
    for _ in range(20):
        st.note_pruned(98, 100)
    assert st._gate_open()
    assert not st.wants_probe()


def test_a_gate_with_no_evidence_yet_does_not_ask_to_probe():
    """Nothing observed means the gate is open and pruning anyway."""
    assert not _state().wants_probe()


def test_observing_unpruned_slices_reopens_the_gate():
    """The end-to-end behaviour that matters: a run disengages while
    unproductive, keeps watching the slices it scored normally, and re-engages
    once the top-K has filled enough to make pruning pay."""
    st = _state()
    for _ in range(30):
        st.note_pruned(50, 1000)
    assert not st._gate_open()
    # A handful of batches later, unpruned slices show a better fraction.
    for _ in range(20):
        if st.wants_probe():
            st.note_pruned(980, 1000)
    assert st._gate_open(), "observing better slices did not reopen the gate"
