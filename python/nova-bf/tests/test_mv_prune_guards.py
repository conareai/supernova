"""Every guard on the float16 multivector prune actually refuses.

A guard nobody tests is a comment. Each case here constructs the SPECIFIC
malformed input the guard exists for, and asserts the refusal — because the
failure mode being prevented is silent in every one of them: a broadcast
threshold prunes against another query's top-K, shifted offsets score against
the wrong rows, a stale float16 cache scores the wrong queries while the slack
describes the right ones. None of those raise on their own.

Most of these run on the CPU because the checks they cover sit AHEAD of
certification, which declines on a CPU. The offset-partition check sits behind
it, so it is exercised directly here and end-to-end in
`tests/parity/test_parity_mv_prune.py`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nova_bf import mv_fp16


def _slice(n_q=3, n_d=2, dim=4, tok=2):
    """A well-formed argument set, for one field at a time to be broken."""
    q_off = torch.arange(n_q + 1, dtype=torch.int64) * tok
    d_off = torch.arange(n_d + 1, dtype=torch.int64) * tok
    q = torch.randn(n_q * tok, dim)
    c = torch.randn(n_d * tok, dim)
    thr = torch.zeros(n_q)
    return dict(q_flat=q, c_flat=c, q_offsets=q_off, doc_offsets=d_off,
                thresholds=thr)


def _call(**over):
    a = _slice() | over
    return mv_fp16._pruned_maxsim_scores(
        a["q_flat"], a["c_flat"], a["q_offsets"], a["doc_offsets"],
        a["thresholds"], **{k: v for k, v in over.items()
                            if k in ("q_half", "q_norm_max", "certified")})


# --- the token matrices -------------------------------------------------------

def test_a_1d_token_matrix_is_refused():
    with pytest.raises(ValueError, match="2D"):
        _call(q_flat=torch.randn(6))


def test_float64_tokens_are_refused():
    """The error model is float16-against-float32; another width is not the
    thing the bound describes."""
    with pytest.raises(TypeError, match="float32"):
        _call(c_flat=torch.randn(4, 4, dtype=torch.float64))


def test_a_dimension_mismatch_is_refused():
    """`dim` for the whole bound comes from the corpus side while `norm_upper`
    measures each side as handed to it, so a mismatch would evaluate the bound
    at one width for norms taken at another."""
    with pytest.raises(ValueError, match="dimension"):
        _call(q_flat=torch.randn(6, 8))


# --- thresholds ---------------------------------------------------------------

def test_a_length_one_threshold_is_refused_not_broadcast():
    """`prune_mask` compares against `thresholds[:, None]`; a length-1 vector
    would silently prune every query against ONE query's top-K."""
    with pytest.raises(ValueError, match="threshold length"):
        _call(thresholds=torch.zeros(1))


def test_a_threshold_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="threshold length"):
        _call(thresholds=torch.zeros(7))


def test_a_2d_threshold_is_refused():
    with pytest.raises(ValueError, match="threshold length"):
        _call(thresholds=torch.zeros(3, 1))


# --- the cached float16 queries -----------------------------------------------

@pytest.mark.parametrize("bad,why", [
    (torch.zeros(4, 4, dtype=torch.float16), "shape"),
    (torch.zeros(6, 4, dtype=torch.float32), "dtype"),
])
def test_a_cached_half_that_does_not_match_q_flat_is_refused(bad, why):
    """A mismatched cache scores the wrong queries in pass one while `slack`
    describes `q_flat` — the bound would then be applied to the wrong scores."""
    with pytest.raises(ValueError, match="q_half"):
        _call(q_half=bad)


def test_a_matching_cached_half_is_accepted():
    """The guard must not reject the legitimate cache it exists to police."""
    a = _slice()
    qh = mv_fp16.to_half(a["q_flat"], n_max=mv_fp16.norm_upper(a["q_flat"]))
    # Declines on the CPU (certification), but reaching the decline means it
    # got PAST the cache check rather than raising.
    assert mv_fp16._pruned_maxsim_scores(
        a["q_flat"], a["c_flat"], a["q_offsets"], a["doc_offsets"],
        a["thresholds"], q_half=qh) is None


# --- offsets ------------------------------------------------------------------

def test_empty_offsets_are_refused():
    """An empty query offset array gives `n_q == -1` and a nonsense shape."""
    with pytest.raises(ValueError, match="non-empty"):
        _call(q_offsets=torch.zeros(0, dtype=torch.int64))


@pytest.mark.parametrize("off,n,why", [
    (torch.tensor([1, 3, 6]), 6, "shifted, so it spans the right total"),
    (torch.tensor([0, 2, 4]), 6, "truncated, leaving rows unowned"),
    (torch.tensor([0, 5, 3]), 3, "non-monotonic"),
])
def test_offsets_that_are_not_an_exact_partition_are_refused(off, n, why):
    """A range check alone would pass the first two: they address the wrong
    rows while every downstream shape check still agrees."""
    with pytest.raises(ValueError):
        mv_fp16._check_partition(off, n, "query")


def test_a_valid_partition_is_accepted():
    mv_fp16._check_partition(torch.tensor([0, 2, 5]), 5, "query")


def test_torch_offsets_are_normalized_to_contiguous_int64():
    """The Triton kernels require int64 and read offsets as a raw linear
    buffer, so a strided or narrower tensor would address the wrong rows."""
    strided = torch.arange(0, 12, dtype=torch.int32)[::2]
    assert not strided.is_contiguous()
    got = mv_fp16._as_offsets(strided, torch.device("cpu"))
    assert got.dtype == torch.int64 and got.is_contiguous()
    assert got.tolist() == [0, 2, 4, 6, 8, 10]


@pytest.mark.parametrize("bad", [
    torch.tensor([0.0, 2.0, 4.0]),
    [0.0, 2.5, 4.0],
])
def test_float_offsets_are_refused_not_truncated(bad):
    """An offset is an exact position; a value that needed rounding was never
    a valid one."""
    with pytest.raises(TypeError, match="integral"):
        mv_fp16._as_offsets(bad, torch.device("cpu"))


# --- the reductions -----------------------------------------------------------

def test_counts_that_do_not_partition_the_token_rows_are_refused():
    """An all-zero count vector used to skip the `index_add_` and return an
    all-`-inf` slice: every pair silently a non-candidate."""
    per_tok = torch.randn(6, 3)
    with pytest.raises(ValueError, match="token rows"):
        mv_fp16._sum_rows(per_tok, torch.zeros(2, dtype=torch.int64), 3)
    with pytest.raises(ValueError, match="token rows"):
        mv_fp16._sum_rows(per_tok, torch.tensor([2, 2]), 3)


def test_a_document_width_mismatch_is_refused():
    with pytest.raises(ValueError, match="documents"):
        mv_fp16._sum_rows(torch.randn(4, 3), torch.tensor([2, 2]), 5)


def test_a_survivor_mask_of_the_wrong_shape_is_refused():
    """`score_survivors` decides which pairs get scored at all."""
    a = _slice()
    with pytest.raises(ValueError, match="does not match the output"):
        mv_fp16.score_survivors(
            torch.zeros(3, 5, dtype=torch.bool), a["q_flat"], a["c_flat"],
            a["q_offsets"], a["doc_offsets"], torch.zeros(3, 2))


def test_norm_upper_refuses_a_1d_input():
    """The derived inflation needs `x.shape[1]` to mean the dimension."""
    with pytest.raises(ValueError, match="2D"):
        mv_fp16.norm_upper(torch.randn(8))


def test_norm_upper_is_an_upper_bound():
    x = torch.randn(64, 128)
    assert mv_fp16.norm_upper(x) >= float(x.double().norm(dim=1).max())


# --- the gate knob ------------------------------------------------------------

@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf")])
def test_an_out_of_range_min_prune_rate_is_refused(bad):
    """NaN is the nasty one: every comparison in the gate goes false, leaving
    it shut with only the periodic re-probe running — slow, and silent."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mv_fp16.Fp16State(min_prune_rate=bad)


@pytest.mark.parametrize("ok", [0.0, 0.7, 1.0])
def test_a_valid_min_prune_rate_is_accepted(ok):
    assert mv_fp16.Fp16State(min_prune_rate=ok).min_prune_rate == ok


def test_norm_upper_propagates_a_nan_instead_of_swallowing_it():
    """`max(0.0, nan)` is 0.0 in Python — it evaluates `nan > 0.0` as False —
    so a chunked accumulator seeded at 0.0 turned a NaN row into a ZERO
    "upper bound", which is not an upper bound at all. The single-shot
    version it replaced propagated the NaN."""
    import math

    for bad_row in (0, 3, 7):
        x = torch.randn(8, 4)
        x[bad_row, 1] = float("nan")
        assert math.isnan(mv_fp16.norm_upper(x)), f"NaN in row {bad_row} lost"


def test_norm_upper_handles_an_infinite_row():
    assert mv_fp16.norm_upper(
        torch.tensor([[1.0, 2.0], [float("inf"), 0.0]])) == float("inf")


def test_norm_upper_bounds_across_chunk_boundaries(monkeypatch):
    """Force one row per chunk so every row crosses a boundary."""
    monkeypatch.setattr(mv_fp16, "_NORM_CHUNK_BYTES", 1)
    for scale in (1e-3, 1.0, 50.0):
        x = torch.randn(37, 129) * scale
        assert mv_fp16.norm_upper(x) >= float(x.double().norm(dim=1).max())
