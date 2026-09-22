"""Adversarial tests for the float16 MaxSim bound.

The only failure that matters is a LIVE pair being pruned, so every test here
is built to produce that if the slack is understated -- not to confirm it on
easy data. That means: queries aligned with corpus tokens so scores crowd the
threshold, thresholds placed exactly at a live score, component magnitudes in
float16's subnormal range where the relative rounding bound does not hold,
magnitudes near float16's ceiling, and unnormalized tokens.

Several tests pin an individual TERM of the slack rather than the total: a
bound can be correct in aggregate on benign data while a term that only bites
in a corner has been dropped. Those tests fail if the term is removed.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="float16 pass one requires CUDA")

from nova_bf import mv_fp16

DIM = 128


@pytest.fixture(scope="module")
def certified():
    """Certify once, the way a run does, so each test is not paying for the
    device probes. A device that cannot be certified skips rather than
    silently testing an unused path."""
    reason = mv_fp16.certify(DIM, "cuda")
    if reason:
        pytest.skip(f"device not certified for the float16 bound: {reason}")
    return True


def _offsets(lengths):
    off = torch.zeros(len(lengths) + 1, dtype=torch.int64, device="cuda")
    off[1:] = torch.tensor(lengths, device="cuda").cumsum(0)
    return off


def naive_maxsim(q_flat, q_off, c_flat, c_off):
    """Reference MaxSim, one pair at a time, in float64.

    Deliberately shares no code with the kernels: it is the thing the bound is
    checked against, so it must not inherit their reduction order or dtype.
    """
    n_q, n_d = q_off.numel() - 1, c_off.numel() - 1
    out = torch.full((n_q, n_d), float("-inf"), dtype=torch.float64, device="cuda")
    qd, cd = q_flat.double(), c_flat.double()
    for i in range(n_q):
        a, b = int(q_off[i]), int(q_off[i + 1])
        if b <= a:
            continue
        for j in range(n_d):
            c0, c1 = int(c_off[j]), int(c_off[j + 1])
            if c1 <= c0:
                continue
            out[i, j] = (qd[a:b] @ cd[c0:c1].T).max(dim=1).values.sum()
    return out


def exact_path(q_flat, q_off, c_flat, c_off):
    """What the UNPRUNED run produces -- float32, the value the bound must
    dominate (not the real-arithmetic value, which nobody ever sees)."""
    from nova_bf.multivector_kernels import fused_ragged_maxsim_reduce

    n_q, n_d = q_off.numel() - 1, c_off.numel() - 1
    out = torch.full((n_q, n_d), float("-inf"), device="cuda")
    fused_ragged_maxsim_reduce(q_flat @ c_flat.T, q_off, c_off, query_start=0,
                               query_token_base=0, n_queries=n_q, out=out)
    return out


def _assert_admissible(q_flat, q_off, c_flat, c_off, thr):
    """The whole contract: nothing at or above threshold may be pruned, and
    every survivor carries the exact score."""
    res = mv_fp16._pruned_maxsim_scores(q_flat, c_flat, q_off, c_off, thr,
                                       certified=True)
    assert res is not None, "float16 pass declined; the test would be vacuous"
    scores, dead, _ = res
    ref = exact_path(q_flat, q_off, c_flat, c_off)
    at_or_above = ref >= thr[:, None]
    dropped = int((dead & at_or_above).sum())
    assert dropped == 0, f"{dropped} pair(s) at/above threshold were pruned"
    live = ~dead & torch.isfinite(ref)
    assert torch.allclose(scores[live], ref[live], rtol=1e-4, atol=1e-3)
    return scores, dead


@pytest.fixture
def rand_case():
    g = torch.Generator(device="cuda").manual_seed(3)
    q_len = [0, 1, 6, 13, 4, 9, 2, 7]
    c_len = [0, 1, 30, 12, 44, 5, 19]
    q = torch.nn.functional.normalize(
        torch.randn(sum(q_len), DIM, device="cuda", generator=g), dim=1)
    c = torch.nn.functional.normalize(
        torch.randn(sum(c_len), DIM, device="cuda", generator=g), dim=1)
    return q.contiguous(), _offsets(q_len), c.contiguous(), _offsets(c_len)


@pytest.mark.parametrize("quantile", [0.0, 0.5, 0.9, 0.999, 1.0])
def test_bound_never_prunes_a_live_pair(rand_case, certified, quantile):
    q, q_off, c, c_off = rand_case
    ref = exact_path(q, q_off, c, c_off)
    thr = torch.nan_to_num(ref, neginf=0.0).quantile(quantile, dim=1)
    _assert_admissible(q, q_off, c, c_off, thr)


def test_threshold_placed_exactly_on_a_live_score(rand_case, certified):
    """The adversarial case: every query's threshold IS one of its scores, so
    an off-by-an-epsilon bound prunes something it must keep."""
    q, q_off, c, c_off = rand_case
    ref = exact_path(q, q_off, c, c_off)
    finite = torch.where(torch.isfinite(ref), ref, torch.full_like(ref, -1e30))
    thr = finite.max(dim=1).values          # exactly the best score per query
    _, dead = _assert_admissible(q, q_off, c, c_off, thr)
    # And the best document must survive for every query that has one.
    best = finite.argmax(dim=1)
    has_any = torch.isfinite(ref).any(dim=1)
    assert not bool(dead[has_any, best[has_any]].any())


def test_queries_drawn_from_the_corpus_itself(rand_case, certified):
    """Self-similar data crowds scores together, so the ranking is decided by
    differences the size of the rounding error the bound must cover."""
    _, _, c, c_off = rand_case
    q = c.clone()
    q_off = c_off.clone()
    ref = exact_path(q, q_off, c, c_off)
    thr = torch.nan_to_num(ref, neginf=0.0).quantile(0.95, dim=1)
    _assert_admissible(q, q_off, c, c_off, thr)


def test_components_in_float16_subnormal_range(rand_case, certified):
    """Below float16's smallest normal there is no relative precision, only the
    absolute floor `eta` -- which `eps_dot` defaults to zero and must be passed.

    This test used to scale random gaussians by 1e-5 and passed even while the
    bound was missing that term, because random signs make the per-component
    rounding errors incoherent and they cancel across the dot product. It is
    now deterministic and COHERENT: every component sits half-way between two
    float16 subnormals, so all of them round to even in the same direction and
    the errors accumulate. See `test_mv_fp16_slack_cpu.py` for the same
    construction exercised without a GPU.
    """
    _, q_off, _, c_off = rand_case
    n_qtok, n_ctok = int(q_off[-1]), int(c_off[-1])
    half_grid = 2.0 ** -24
    q = torch.full((n_qtok, DIM), 0.5 * half_grid, device="cuda")
    c = torch.full((n_ctok, DIM), 4.5 * half_grid, device="cuda")
    ref = exact_path(q, q_off, c, c_off)
    thr = torch.nan_to_num(ref, neginf=0.0).quantile(0.9, dim=1)
    _assert_admissible(q, q_off, c, c_off, thr)


def test_components_near_the_float16_ceiling(rand_case, certified):
    q, q_off, c, c_off = rand_case
    q = (q * 200.0).contiguous()
    c = (c * 200.0).contiguous()
    ref = exact_path(q, q_off, c, c_off)
    thr = torch.nan_to_num(ref, neginf=0.0).quantile(0.9, dim=1)
    _assert_admissible(q, q_off, c, c_off, thr)


def test_unnormalized_tokens_of_wildly_mixed_magnitude(rand_case, certified):
    """Nothing in the derivation assumes unit norms; this checks that."""
    g = torch.Generator(device="cuda").manual_seed(19)
    q, q_off, c, c_off = rand_case
    scale_q = torch.pow(10.0, torch.randint(-3, 2, (q.shape[0], 1),
                                            device="cuda", generator=g).float())
    scale_c = torch.pow(10.0, torch.randint(-3, 2, (c.shape[0], 1),
                                            device="cuda", generator=g).float())
    q = (q * scale_q).contiguous()
    c = (c * scale_c).contiguous()
    ref = exact_path(q, q_off, c, c_off)
    thr = torch.nan_to_num(ref, neginf=0.0).quantile(0.9, dim=1)
    _assert_admissible(q, q_off, c, c_off, thr)


def test_values_beyond_float16_range_make_the_pass_decline(rand_case, certified):
    """Overflow must be declined, never rounded to inf and bounded anyway."""
    q, q_off, c, c_off = rand_case
    c = (c * 1e5).contiguous()
    thr = torch.zeros(q_off.numel() - 1, device="cuda")
    assert mv_fp16._pruned_maxsim_scores(q, c, q_off, c_off, thr) is None


def test_the_bound_actually_prunes_something(rand_case, certified):
    """A bound that prunes nothing passes every admissibility test above while
    being worthless, so pin a floor on the rate."""
    q, q_off, c, c_off = rand_case
    ref = exact_path(q, q_off, c, c_off)
    thr = torch.nan_to_num(ref, neginf=0.0).quantile(0.8, dim=1)
    _, dead = _assert_admissible(q, q_off, c, c_off, thr)
    finite = torch.isfinite(ref)
    rate = float((dead & finite).sum()) / max(int(finite.sum()), 1)
    assert rate > 0.5, f"only {rate:.1%} of scorable pairs pruned at the 80th pct"


# --- pinning the lift this module is responsible for -----------------------
#
# The per-dot allowance is `closed_form.eps_dot`, tested in its own suite. What
# is pinned here is only how this module LIFTS it through MaxSim's reductions,
# which is where a multivector-specific mistake would live.

def test_slack_scales_at_least_linearly_with_the_query_token_count(certified):
    """MaxSim sums over a query's tokens, so the allowance must be multiplied
    by that count -- using one per-dot allowance unscaled would understate a
    30-token query's slack thirtyfold.

    At LEAST linear, not exactly linear: the float32 summation term grows with
    the token count too, so a longer query is charged slightly more per token.
    Sublinear growth is the failure this guards against.
    """
    few = float(mv_fp16.slack(torch.tensor([1], device="cuda"), 1.0, 1.0, DIM)[0])
    many = float(mv_fp16.slack(torch.tensor([30], device="cuda"), 1.0, 1.0, DIM)[0])
    assert many >= 30.0 * few
    assert many < 31.0 * few, "growth far above linear suggests a wrong factor"


def test_slack_exceeds_the_pass_one_allowance_alone(certified):
    """The bound must dominate the score the UNPRUNED run produces, not the
    real-arithmetic MaxSim. That run computes a float32 length-d dot and a
    float32 sum over the query's tokens, either of which can land ABOVE the
    real value -- so `n * eps_dot` on its own is too small.

    The reference must be built with the SAME `eps_dot` call the code makes,
    including `eta_q`/`eta_c`. An earlier version used the eta-free call, which
    is so much smaller that the assertion passed even with every float32 charge
    deleted -- it proved nothing.
    """
    import numpy as np

    from nova_bf import closed_form as cf

    n = 12
    fmt = cf.FP16
    got = float(mv_fp16.slack(torch.tensor([n], device="cuda"), 1.0, 1.0, DIM)[0])
    pass_one_only = n * float(cf.eps_dot(
        DIM, fmt.u_t, fmt.u_t, np.asarray([1.0]), 1.0,
        cf.C_HW_ENVELOPE, False, fmt.eta_t, fmt.eta_t, fmt.lam_t)[0])
    assert got > pass_one_only


def test_norm_upper_bounds_the_true_norm(certified):
    """`slack` is evaluated at these norms, so one that came out BELOW the true
    norm would make the bound smaller than the mathematics requires.

    The comparison is against a binary64 norm, not the float32 one: float32's
    own norm can land either side of the truth, and it is the truth this has
    to dominate.
    """
    g = torch.Generator(device="cuda").manual_seed(31)
    for scale in (1e-3, 1.0, 1e3):
        x = torch.randn(2048, DIM, device="cuda", generator=g) * scale
        true_max = float(x.double().norm(dim=1).max())
        assert mv_fp16.norm_upper(x) >= true_max


def test_norm_upper_handles_an_empty_matrix(certified):
    x = torch.zeros((0, DIM), device="cuda")
    assert mv_fp16.norm_upper(x) == 0.0


def test_slack_is_monotone_in_both_norm_bounds(certified):
    """`slack` evaluates `eps_dot` at the slice-wide MAXIMUM norms and calls
    that conservative. It only is if the allowance is non-decreasing in both,
    so that is checked rather than assumed."""
    base = float(mv_fp16.slack(torch.tensor([4], device="cuda"), 1.0, 1.0, DIM)[0])
    bigger_q = float(mv_fp16.slack(torch.tensor([4], device="cuda"), 4.0, 1.0, DIM)[0])
    bigger_d = float(mv_fp16.slack(torch.tensor([4], device="cuda"), 1.0, 4.0, DIM)[0])
    assert bigger_q >= base and bigger_d >= base


def test_slack_is_strictly_positive_for_a_single_token(certified):
    s = mv_fp16.slack(torch.tensor([1], device="cuda"), 1.0, 1.0, DIM)
    assert float(s[0]) > 0.0


def test_zero_token_queries_stay_non_candidates(certified):
    """A query with no tokens has no score; it must stay -inf rather than
    becoming 0 via an empty sum."""
    q, q_off, c, c_off = _fixed_case()
    thr = torch.full((q_off.numel() - 1,), -1e30, device="cuda")
    scores, _, _ = mv_fp16._pruned_maxsim_scores(q, c, q_off, c_off, thr,
                                                certified=True)
    empty = q_off.diff() == 0
    assert bool(empty.any()), "the fixture no longer has a zero-token query"
    assert bool(torch.isneginf(scores[empty]).all())


def _fixed_case():
    g = torch.Generator(device="cuda").manual_seed(3)
    q_len = [0, 1, 6, 13, 4, 9, 2, 7]
    c_len = [0, 1, 30, 12, 44, 5, 19]
    q = torch.nn.functional.normalize(
        torch.randn(sum(q_len), DIM, device="cuda", generator=g), dim=1)
    c = torch.nn.functional.normalize(
        torch.randn(sum(c_len), DIM, device="cuda", generator=g), dim=1)
    return q.contiguous(), _offsets(q_len), c.contiguous(), _offsets(c_len)


def test_agrees_with_the_naive_float64_reference_on_survivors(rand_case, certified):
    q, q_off, c, c_off = rand_case
    ref64 = naive_maxsim(q, q_off, c, c_off)
    thr = torch.nan_to_num(ref64.float(), neginf=0.0).quantile(0.7, dim=1)
    scores, dead = _assert_admissible(q, q_off, c, c_off, thr)
    live = ~dead & torch.isfinite(ref64)
    assert torch.allclose(scores[live].double(), ref64[live], rtol=1e-4, atol=1e-3)
