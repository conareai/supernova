"""The engage/disengage gate: prune only while it is actually paying.

Pass one is not free, so at a low prune rate it costs more than it saves.
`Fp16State` therefore latches pruning ON after `_LATCH_AFTER` consecutive
probes at or above `params.multivector_min_prune_rate` (0.70 by default), and
back OFF after `_UNLATCH_AFTER` consecutive pruned slices below it. Stated as a
PRUNE RATE because that is what this path reports everywhere else; the dense
two-pass states the same idea inverted, as a live fraction
(`twopass.DEFAULT_THRESHOLD = 0.50`).

WHY THE TWO THRESHOLDS DIFFER. The errors are not symmetric. Failing to prune a
run that would pay costs up to 2x (measured 577.7 s against 303.8 s on 8
pubmed files at k=1000); pruning a run that does not pay costs pass one's GEMM,
~195 ms against ~800 ms slices, and only until the next measurement. So the
gate engages on thin evidence and disengages on thick.

WHY THE TWO DIRECTIONS ARE MEASURED DIFFERENTLY. While the gate is off, the
only available measurement is `note_probe`, taken from a slice scored the
ordinary way, and it costs a device->host sync -- so the caller spends one per
BATCH GROUP. Once latched, `note_pruned` gets the achieved rate for free, from
counts the caller already materialized for its own tally, so that direction
watches every SLICE.

WHAT REPLACED AN EMA, and why. The prune rate CLIMBS during a run: the bound
prunes against the running top-K threshold, and that starts empty -- measured
1.6% pruned over one corpus file at k=10 against 99.1% over two. A smoothed
average of those early slices told the gate the prune was not paying, and on 8
pubmed files it shut off for 279 of 373 slices. The streak rule needs no
warm-up detection to avoid that: during warm-up the rate is genuinely low, so
no streak forms and the gate simply stays off until the evidence changes.

These exercise the state machine directly, with no GPU: it is fed plain counts.
"""
from __future__ import annotations

import pytest

from nova_bf import mv_fp16
from nova_bf.mv_fp16 import Fp16State

GOOD = (90, 100)          # 90% pruned, above the 0.70 floor
BAD = (10, 100)           # 10% pruned, below it


def _state(min_prune_rate=0.70):
    return Fp16State(min_prune_rate=min_prune_rate)


def _probe(st, n, sample=GOOD):
    for _ in range(n):
        st.note_probe(*sample)


def _prune(st, n, sample=BAD):
    for _ in range(n):
        st.note_pruned(*sample)


# --- engaging -----------------------------------------------------------------

def test_the_gate_starts_closed():
    """Nothing has been measured, so nothing justifies pass one's GEMM yet.
    The first slices are scored the ordinary way and measured for free."""
    assert not _state().gate_open()


def test_a_streak_of_good_probes_engages_it():
    st = _state()
    _probe(st, Fp16State._LATCH_AFTER - 1)
    assert not st.gate_open(), "engaged before the streak was complete"
    _probe(st, 1)
    assert st.gate_open()


def test_one_bad_probe_resets_the_streak():
    """Consecutive, not cumulative: two good probes either side of a bad one
    are not evidence of a sustained rate."""
    st = _state()
    _probe(st, Fp16State._LATCH_AFTER - 1)
    st.note_probe(*BAD)
    _probe(st, Fp16State._LATCH_AFTER - 1)
    assert not st.gate_open()


def test_a_rate_exactly_on_the_floor_counts_as_good():
    """The floor is a minimum to KEEP pruning, so equality keeps it."""
    st = _state(0.70)
    _probe(st, Fp16State._LATCH_AFTER, sample=(70, 100))
    assert st.gate_open()


def test_a_warming_up_run_simply_never_forms_a_streak():
    """The replaced failure mode, stated directly: early slices prune almost
    nothing because hardly any query has a full top-K. No special case detects
    that -- the rate is low, so no streak forms."""
    st = _state()
    _probe(st, 50, sample=(2, 100))
    assert not st.gate_open()
    # ... and once the top-K fills, it engages on the ordinary rule.
    _probe(st, Fp16State._LATCH_AFTER)
    assert st.gate_open()


def test_a_zero_floor_engages_on_the_first_probes():
    """`multivector_min_prune_rate: 0.0` is the documented setting for short
    runs and benchmarks: every rate clears a zero floor."""
    st = _state(0.0)
    _probe(st, Fp16State._LATCH_AFTER, sample=(0, 100))
    assert st.gate_open()


# --- disengaging --------------------------------------------------------------

def test_a_sustained_bad_streak_disengages_it():
    st = _state()
    _probe(st, Fp16State._LATCH_AFTER)
    _prune(st, Fp16State._UNLATCH_AFTER - 1)
    assert st.gate_open(), "disengaged before the streak was complete"
    _prune(st, 1)
    assert not st.gate_open()


def test_one_good_slice_resets_the_bad_streak():
    """A single unproductive slice is not a reason to stop: the gate tolerates
    nine of them so long as a good one intervenes."""
    st = _state()
    _probe(st, Fp16State._LATCH_AFTER)
    _prune(st, Fp16State._UNLATCH_AFTER - 1)
    st.note_pruned(*GOOD)
    _prune(st, Fp16State._UNLATCH_AFTER - 1)
    assert st.gate_open()


def test_it_takes_more_measurements_to_disengage_than_to_engage():
    """The asymmetry is the design, so pin it rather than leaving it to the
    constants drifting apart later.

    Read this for what it is: a comparison of MEASUREMENT COUNTS, not of wall
    time. The two are denominated differently — `note_pruned` gets one per
    slice, `note_probe` one per BATCH GROUP — so 10 slices to disengage can be
    a shorter stretch than 3 batch groups to re-engage. In wall-clock terms
    the asymmetry is therefore weaker than these constants suggest, and a run
    hovering just under the floor pays the re-engage latency each cycle. That
    is a throughput cost only, and it is the price of `note_probe` needing a
    device->host sync that would be unaffordable per slice."""
    assert Fp16State._UNLATCH_AFTER > Fp16State._LATCH_AFTER


def test_a_run_can_re_engage_after_disengaging():
    """Disengaging returns it to the probe path, not to a dead end."""
    st = _state()
    _probe(st, Fp16State._LATCH_AFTER)
    _prune(st, Fp16State._UNLATCH_AFTER)
    assert not st.gate_open()
    _probe(st, Fp16State._LATCH_AFTER)
    assert st.gate_open()


# --- which measurement is admissible in which state ---------------------------

def test_a_latched_gate_ignores_probes():
    """Once latched, the achieved rate is available free on every slice, so the
    optimistic probe estimate must not be able to re-latch or reset anything."""
    st = _state()
    _probe(st, Fp16State._LATCH_AFTER)
    _prune(st, Fp16State._UNLATCH_AFTER - 1)
    _probe(st, 5)                      # must not clear the bad streak
    _prune(st, 1)
    assert not st.gate_open()


def test_a_closed_gate_ignores_pruned_counts():
    """Symmetric: nothing was pruned while the gate was closed, so a stray
    achieved-rate report must not disengage what is already disengaged."""
    st = _state()
    _prune(st, 20)
    _probe(st, Fp16State._LATCH_AFTER)
    assert st.gate_open()


def test_only_the_probe_is_wanted_while_closed():
    """`wants_probe` is what stops the caller buying a sync it cannot use."""
    st = _state()
    assert st.wants_probe()
    _probe(st, Fp16State._LATCH_AFTER)
    assert not st.wants_probe()


def test_empty_slices_are_not_evidence_either_way():
    st = _state()
    for _ in range(20):
        st.note_probe(0, 0)
    assert not st.gate_open()
    _probe(st, Fp16State._LATCH_AFTER)
    _prune(st, 20, sample=(0, 0))
    assert st.gate_open()


# --- reporting ----------------------------------------------------------------

def test_the_gate_reports_why_a_run_did_not_prune():
    """`declined` lumps a closed gate together with a refused certification,
    and the two call for opposite responses — one is a tuning knob, the other
    means the prune cannot run at all."""
    mv_fp16.reset_gate_stats()
    st = _state()
    for _ in range(4):
        st.gate_open()
    _probe(st, Fp16State._LATCH_AFTER)
    st.gate_open()

    g = mv_fp16.gate_stats()
    assert g["closed"] == 4 and g["open"] == 1
    assert g["latched"] == 1 and g["unlatched"] == 0
    mv_fp16.reset_gate_stats()
    assert set(mv_fp16.gate_stats().values()) == {0}


def test_disengaging_is_reported_too():
    mv_fp16.reset_gate_stats()
    st = _state()
    _probe(st, Fp16State._LATCH_AFTER)
    _prune(st, Fp16State._UNLATCH_AFTER)
    assert mv_fp16.gate_stats()["unlatched"] == 1
    mv_fp16.reset_gate_stats()


# --- the zero floor means what it says ----------------------------------------

def test_a_zero_floor_prunes_from_the_first_slice():
    """`multivector_min_prune_rate: 0.0` is documented as "always prune", and
    it is the setting for short runs and benchmarks. Starting closed and
    demanding a streak cost 77 of 358 slices on an 8-file run, because a probe
    fires once per BATCH GROUP and three of them is several batches however low
    the floor is — so the delay would be most of a short measurement."""
    assert _state(0.0).gate_open()


def test_a_zero_floor_can_never_disengage():
    """Every rate clears a zero floor, so there is no evidence that could turn
    it off — including a slice that pruned nothing at all."""
    st = _state(0.0)
    _prune(st, Fp16State._UNLATCH_AFTER * 3, sample=(0, 100))
    assert st.gate_open()


def test_a_nonzero_floor_still_starts_closed():
    """The optimism is specific to a zero floor: with a real floor the gate
    must not prune until something has cleared it."""
    assert not _state(0.01).gate_open()


def test_the_unlatch_counts_slices_not_member_measurements():
    """`note_pruned` runs once per (slice, member) in `compute`, so the caller
    sums a slice's members and reports ONCE. Reporting per member made
    `_UNLATCH_AFTER` mean 10/M slices — with 5 multivector searches the gate
    disengaged on 2 slices of evidence while still needing 3 batch groups to
    return, inverting the asymmetry the two constants exist to encode."""
    st = _state()
    _probe(st, Fp16State._LATCH_AFTER)
    # Five members' worth of bad pairs, summed into one slice report.
    for _ in range(Fp16State._UNLATCH_AFTER - 1):
        st.note_pruned(10 * 5, 100 * 5)
    assert st.gate_open(), "a slice's members counted as separate evidence"
    st.note_pruned(10 * 5, 100 * 5)
    assert not st.gate_open()
