"""The prune audit grades correctly, and a violation actually stops the run.

The audit's whole value is that it FAILS when the bound is wrong. A grader
that only ever reports success on a healthy run has not been tested — so every
case here feeds it a decision it must call wrong, and checks both the verdict
and the consequence.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nova_bf import mv_fp16


@pytest.fixture(autouse=True)
def _clean_audit():
    mv_fp16.reset_audit()
    yield
    mv_fp16.reset_audit()


def test_a_clean_slice_grades_as_all_correct():
    exact = torch.tensor([[5.0, 1.0], [4.0, 0.5]])
    thr = torch.tensor([2.0, 2.0])
    dead = torch.tensor([[False, True], [False, True]])      # low pairs pruned
    upper = exact + 1.0

    got = mv_fp16.audit_decisions(exact, dead, thr, upper=upper)
    assert got["false_prune"] == 0 and got["bound_violations"] == 0
    assert got["correct_prune"] == 2 and got["correct_live"] == 2
    assert mv_fp16.audit_stats()["graded_pairs"] == 4
    assert mv_fp16.audit_stats()["disabled"] is None


def test_a_pruned_pair_that_reaches_its_threshold_is_a_false_prune():
    """The observable failure: a candidate that belonged in the top-K was
    dropped."""
    exact = torch.tensor([[5.0, 3.0]])
    thr = torch.tensor([3.0])                                # equality is LIVE
    dead = torch.tensor([[False, True]])
    got = mv_fp16.audit_decisions(exact, dead, thr, upper=exact + 1.0)

    assert got["false_prune"] == 1
    assert mv_fp16.audit_stats()["worst_false_prune"] == pytest.approx(0.0)


def test_a_bound_below_the_exact_score_is_caught_even_with_no_false_prune():
    """The bound check is the STRONGER of the two: it fires while the
    threshold still happens to be saving us, which is the only warning you get
    before a candidate is actually lost."""
    exact = torch.tensor([[5.0]])
    thr = torch.tensor([99.0])            # nothing can reach it, so no false prune
    upper = torch.tensor([[4.0]])         # ... but the bound is below the truth
    got = mv_fp16.audit_decisions(exact, torch.tensor([[True]]), thr, upper=upper)

    assert got["false_prune"] == 0
    assert got["bound_violations"] == 1
    assert mv_fp16.audit_stats()["worst_bound_gap"] == pytest.approx(1.0)


def test_a_violation_disables_pruning_for_the_rest_of_the_process():
    """If the bound is wrong on one slice every later slice is suspect, and a
    run that kept pruning would be quietly incomplete."""
    state = mv_fp16.Fp16State()
    mv_fp16.audit_decisions(torch.tensor([[5.0]]), torch.tensor([[True]]),
                            torch.tensor([0.0]), upper=torch.tensor([[1.0]]))

    assert mv_fp16.audit_stats()["disabled"]
    assert state.score(torch.randn(2, 4), torch.randn(2, 4),
                       torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]),
                       torch.zeros(2)) is None


def test_reset_clears_the_disable_as_well_as_the_counters():
    mv_fp16.audit_decisions(torch.tensor([[5.0]]), torch.tensor([[True]]),
                            torch.tensor([0.0]), upper=torch.tensor([[1.0]]))
    mv_fp16.reset_audit()
    assert mv_fp16.audit_stats()["disabled"] is None
    assert mv_fp16.audit_stats()["graded_pairs"] == 0


def test_wasted_live_counts_pairs_kept_that_could_have_been_pruned():
    """Not a failure — it is how much the bound is leaving on the table."""
    exact = torch.tensor([[0.5, 0.4]])
    got = mv_fp16.audit_decisions(exact, torch.tensor([[False, False]]),
                                  torch.tensor([9.0]), upper=exact + 1.0)
    assert got["wasted_live"] == 2 and got["correct_live"] == 0


def test_a_non_finite_score_carries_no_verdict():
    """Zero-token queries and documents score `-inf`: there is nothing to
    grade, for either the decision or the bound."""
    exact = torch.tensor([[float("-inf"), 1.0]])
    got = mv_fp16.audit_decisions(exact, torch.tensor([[True, False]]),
                                  torch.tensor([0.0]), upper=exact + 1.0)
    assert got["graded"] == 1


def test_a_row_with_no_threshold_is_still_graded_for_the_BOUND():
    """The decision needs a threshold to be judged against; `upper >= exact`
    does not. Rows whose top-K is still filling, and rows another member owns,
    carry `-inf` thresholds and are most of the matrix in exactly the runs the
    audit is for — grading them as clean reported a PASS on a bound 9.0 below
    the truth."""
    exact = torch.tensor([[0.5, 0.5], [9.0, 9.0]])
    upper = torch.tensor([[1.0, 1.0], [0.0, 0.0]])      # row 1 is 9.0 short
    got = mv_fp16.audit_decisions(
        exact, torch.tensor([[True, True], [False, False]]),
        torch.tensor([1.0, float("-inf")]), upper=upper)

    assert got["bound_violations"] == 2, "an unthresholded bad bound went unseen"
    assert mv_fp16.audit_stats()["worst_bound_gap"] == pytest.approx(9.0)
    assert got["graded"] == 2, "the DECISION count must still exclude that row"


def test_a_bound_of_the_wrong_shape_is_refused_not_broadcast():
    """Same broadcast hazard `_pruned_maxsim_scores` guards for thresholds: a
    `(n_q, 1)` bound silently covers every document."""
    with pytest.raises(ValueError, match="does not match the exact"):
        mv_fp16.audit_decisions(
            torch.full((2, 2), 5.0), torch.ones(2, 2, dtype=torch.bool),
            torch.tensor([1.0, 1.0]), upper=torch.tensor([[9.0], [9.0]]))


def test_the_grader_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="do not match"):
        mv_fp16.audit_decisions(torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.bool),
                                torch.zeros(2))
    with pytest.raises(ValueError, match="threshold length"):
        mv_fp16.audit_decisions(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool),
                                torch.zeros(5))


@pytest.mark.parametrize("raw,want", [
    ("0", 0), ("", 0), ("1", 1), ("8", 8), ("-3", 0), ("nonsense", 0),
])
def test_the_audit_rate_is_off_unless_asked_for(monkeypatch, raw, want):
    """Off by default: grading costs the exact full-slice MaxSim that the
    prune exists to avoid.

    This reader never raises — it runs per slice, and aborting a run hours in
    over an environment variable would be worse than the typo. Rejecting the
    typo is `check_audit_env`'s job, once, at startup."""
    if raw:
        monkeypatch.setenv("NOVA_BF_MV_PRUNE_AUDIT", raw)
    else:
        monkeypatch.delenv("NOVA_BF_MV_PRUNE_AUDIT", raising=False)
    assert mv_fp16.audit_rate() == want


@pytest.mark.parametrize("bad", ["ture", "1x", "1.5", "on", "-3", "0x4"])
def test_a_malformed_audit_rate_is_refused_at_startup(monkeypatch, bad):
    """Setting this variable is an explicit request to have the run verified,
    so a typo must NOT quietly downgrade to "not audited" — that is the same
    outcome as an audit that checks nothing, and the operator sees a clean run
    and believes it was graded."""
    monkeypatch.setenv("NOVA_BF_MV_PRUNE_AUDIT", bad)
    with pytest.raises(ValueError, match="NOVA_BF_MV_PRUNE_AUDIT"):
        mv_fp16.check_audit_env()


@pytest.mark.parametrize("ok,want", [("0", 0), ("1", 1), ("64", 64)])
def test_a_valid_audit_rate_passes_startup_validation(monkeypatch, ok, want):
    monkeypatch.setenv("NOVA_BF_MV_PRUNE_AUDIT", ok)
    assert mv_fp16.check_audit_env() == want


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_an_unset_or_blank_audit_rate_is_not_an_error(monkeypatch, blank):
    """Unset means "no audit wanted", which is the default and not a typo.
    Blank is treated the same way: `export VAR=` is how a shell clears one."""
    if blank is None:
        monkeypatch.delenv("NOVA_BF_MV_PRUNE_AUDIT", raising=False)
    else:
        monkeypatch.setenv("NOVA_BF_MV_PRUNE_AUDIT", blank)
    assert mv_fp16.check_audit_env() == 0


def test_sampling_grades_one_slice_in_n(monkeypatch):
    monkeypatch.setenv("NOVA_BF_MV_PRUNE_AUDIT", "3")
    got = [mv_fp16._audit_should_grade() for _ in range(7)]
    assert got == [True, False, False, True, False, False, True]


def test_no_sampling_happens_when_the_audit_is_off(monkeypatch):
    monkeypatch.delenv("NOVA_BF_MV_PRUNE_AUDIT", raising=False)
    assert not any(mv_fp16._audit_should_grade() for _ in range(5))
    assert mv_fp16.audit_stats()["offered"] == 0


def test_counter_reset_leaves_a_failed_audit_disabled():
    """A second run in the same process must get its own tallies without
    inheriting run 1's numbers or re-emitting its failure — and without
    quietly re-enabling a prune a failed audit turned off, since the same code
    would fail the same way."""
    mv_fp16.audit_decisions(torch.tensor([[5.0]]), torch.tensor([[True]]),
                            torch.tensor([0.0]), upper=torch.tensor([[1.0]]))
    assert mv_fp16.audit_disabled()

    mv_fp16.reset_audit_counters()
    assert mv_fp16.audit_stats()["graded_pairs"] == 0
    assert mv_fp16.audit_stats()["disabled"] is None, "counters not cleared"
    assert mv_fp16.audit_disabled(), "a failed audit must stay disabling"

    # Only the explicit full reset clears the disable.
    mv_fp16.reset_audit()
    assert mv_fp16.audit_disabled() is None


def test_a_wrong_survivor_score_is_caught():
    """The audit's blind spot before this: it graded the DECISION and the
    BOUND but never the scores the run keeps. `score_survivors` is the only
    exact scorer left in the tree, so nothing else cross-checks it — and a
    wrong survivor score is worse than a false prune, because a false prune
    omits a candidate while a wrong score can PROMOTE a non-candidate."""
    exact = torch.tensor([[5.0, 4.0]])
    dead = torch.tensor([[False, False]])
    kept = torch.tensor([[5.0, -999.0]])           # second survivor corrupt
    got = mv_fp16.audit_decisions(exact, dead, torch.tensor([1.0]),
                                  upper=exact + 1.0, scores=kept)

    assert got["score_mismatch"] == 1
    assert got["bound_violations"] == 0, "the bound was fine; the score was not"
    assert mv_fp16.audit_stats()["disabled"], "a wrong score must stop the run"


def test_a_survivor_left_at_negative_infinity_is_caught():
    """Not just wrong VALUES: a pair the prune kept but never scored."""
    got = mv_fp16.audit_decisions(
        torch.tensor([[5.0]]), torch.tensor([[False]]), torch.tensor([1.0]),
        upper=torch.tensor([[6.0]]),
        scores=torch.tensor([[float("-inf")]]))
    assert got["score_mismatch"] == 1


def test_last_bit_disagreement_between_exact_scorers_is_not_a_failure():
    """The two scorers do float32 dots and differ in reduction order, and the
    survivor fold runs in float64 — so they legitimately disagree in the last
    bits. Flagging that would make the audit useless."""
    exact = torch.tensor([[12.345678, -3.5]])
    kept = exact + torch.tensor([[2.0e-6, -1.0e-6]])
    got = mv_fp16.audit_decisions(exact, torch.zeros(1, 2, dtype=torch.bool),
                                  torch.tensor([-10.0]), upper=exact + 1.0,
                                  scores=kept)
    assert got["score_mismatch"] == 0
    assert mv_fp16.audit_stats()["graded_scores"] == 2


def test_dead_pairs_are_not_graded_as_scores():
    """A pruned pair is `-inf` by design; only survivors carry a score."""
    got = mv_fp16.audit_decisions(
        torch.tensor([[5.0, 0.1]]), torch.tensor([[False, True]]),
        torch.tensor([1.0]), upper=torch.tensor([[6.0, 1.0]]),
        scores=torch.tensor([[5.0, float("-inf")]]))
    assert got["score_mismatch"] == 0
    assert mv_fp16.audit_stats()["graded_scores"] == 1


def test_kept_scores_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="kept scores"):
        mv_fp16.audit_decisions(
            torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool),
            torch.zeros(2), scores=torch.zeros(2, 4))


def test_a_non_candidate_scored_zero_is_caught():
    """THE regression this check exists for, and the one it originally missed.

    Masking the score check on `isfinite(exact)` drops every zero-token query
    and empty document — exactly the pairs `score_survivors`' write mask has to
    keep at `-inf`. A run scoring three of them `0.0` then graded as a clean
    PASS, and `0.0` outranks every negative MaxSim, so all three are promoted
    into the top-K."""
    exact = torch.tensor([[float("-inf"), float("-inf")],
                          [2.0, float("-inf")]])
    kept = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    got = mv_fp16.audit_decisions(exact, torch.zeros(2, 2, dtype=torch.bool),
                                  torch.tensor([1.0, 1.0]),
                                  upper=torch.full((2, 2), 9.0), scores=kept)

    assert got["score_mismatch"] == 3, "non-candidates scored 0.0 went unseen"
    assert mv_fp16.audit_stats()["disabled"]


def test_the_structure_check_covers_pairs_with_no_threshold():
    """Same masking hazard one level out: these pairs have no threshold, so
    the DECISION cannot be graded — but the structure still can."""
    got = mv_fp16.audit_decisions(
        torch.tensor([[float("-inf")]]), torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([float("-inf")]), upper=torch.tensor([[1.0]]),
        scores=torch.tensor([[0.0]]))
    assert got["score_mismatch"] == 1


def test_a_survivor_scored_nan_is_caught():
    got = mv_fp16.audit_decisions(
        torch.tensor([[3.0]]), torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([1.0]), upper=torch.tensor([[4.0]]),
        scores=torch.tensor([[float("nan")]]))
    assert got["score_mismatch"] == 1


# --- non-finite values must not escape either check ---------------------------

INF, NAN = float("inf"), float("nan")


@pytest.mark.parametrize("bad_upper,why", [
    (-INF, "the largest shortfall expressible"),
    (NAN, "a meaningless bound"),
])
def test_a_non_finite_bound_below_the_truth_is_caught(bad_upper, why):
    """Masking the bound check on `isfinite(upper)` excused exactly the bounds
    most obviously broken. `~(upper >= exact)` needs no mask: every comparison
    against NaN is False, so NaN and `-inf` both flag."""
    got = mv_fp16.audit_decisions(
        torch.tensor([[5.0]]), torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([1.0]), upper=torch.tensor([[bad_upper]]))
    assert got["bound_violations"] == 1, why


def test_an_infinite_bound_is_valid_if_useless():
    """`+inf` dominates every exact score, so it is a correct bound — merely
    one that prunes nothing. Flagging it would be a false failure."""
    got = mv_fp16.audit_decisions(
        torch.tensor([[5.0]]), torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([1.0]), upper=torch.tensor([[INF]]))
    assert got["bound_violations"] == 0


def test_min_headroom_survives_a_non_finite_pair():
    """`-inf - -inf` is NaN and would poison the running minimum, so headroom
    is measured only where the subtraction means something."""
    mv_fp16.audit_decisions(
        torch.tensor([[-INF, 5.0]]), torch.zeros(1, 2, dtype=torch.bool),
        torch.tensor([1.0]), upper=torch.tensor([[-INF, 6.0]]))
    hr = mv_fp16.audit_stats()["min_headroom"]
    assert hr is not None and hr == pytest.approx(1.0), hr


@pytest.mark.parametrize("bad_score", [NAN, INF])
def test_a_non_finite_survivor_score_is_caught(bad_score):
    """Comparing `isfinite` alone called `-inf` and NaN both "non-finite", so
    a survivor scored NaN where the reference is `-inf` compared EQUAL and
    passed — contradicting the comment that said NaN was caught."""
    got = mv_fp16.audit_decisions(
        torch.tensor([[-INF]]), torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([1.0]), upper=torch.tensor([[INF]]),
        scores=torch.tensor([[bad_score]]))
    assert got["score_mismatch"] == 1


def test_a_matching_negative_infinity_is_not_a_mismatch():
    """The legitimate case this must not break: a non-candidate correctly left
    at `-inf` by both."""
    got = mv_fp16.audit_decisions(
        torch.tensor([[-INF]]), torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([1.0]), upper=torch.tensor([[INF]]),
        scores=torch.tensor([[-INF]]))
    assert got["score_mismatch"] == 0


_CATEGORIES = {"nan": NAN, "neg_inf": -INF, "finite": 1.0, "pos_inf": INF}


@pytest.mark.parametrize("name_s", list(_CATEGORIES))
@pytest.mark.parametrize("name_e", list(_CATEGORIES))
def test_survivor_scores_must_match_the_reference_category(name_e, name_s):
    """The exact-category rule, exhaustively: the two sides must fall in the
    SAME class of {NaN, -inf, finite, +inf}, and NaN is never acceptable on
    either side.

    Comparing by `isfinite` alone lumps `-inf`, `+inf` and NaN together, and
    each of these pairings passed at some point during review: a NaN score
    against a `-inf` reference, a finite score against a `+inf` reference, a
    `+inf` score against a NaN reference. Different bugs, none benign, all
    invisible to the looser test. Parametrized over the full 4x4 rather than
    the cases anyone thought of."""
    e, s = _CATEGORIES[name_e], _CATEGORIES[name_s]
    got = mv_fp16.audit_decisions(
        torch.tensor([[e]]), torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([-1e30]), upper=torch.tensor([[INF]]),
        scores=torch.tensor([[s]]))

    # Legitimate only when the categories agree and neither side is NaN.
    should_pass = (name_e == name_s and name_e != "nan")
    assert (got.get("score_mismatch", 0) == 0) is should_pass, (
        f"exact={name_e} scores={name_s}: "
        f"{'flagged a legitimate pair' if should_pass else 'let a bad pair through'}")
