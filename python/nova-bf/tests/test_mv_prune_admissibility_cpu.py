"""The pruning contract, end to end, WITHOUT a GPU.

WHY THIS FILE EXISTS. `tests/test_mv_fp16.py` — the file that actually tests
the bound — is entirely CUDA-gated, because pass one is a Triton kernel. On a
laptop it reports 20 skips, so nothing exercises the bound, the threshold
comparison, the survivor rescoring or the `-inf` structure at all. Three
adversarial review rounds each found a correctness bug in that code, and none
of them was caught by the suite; two of the three were introduced by the
previous round's fix. That is what an unreachable test path costs.

WHAT THIS TESTS, AND WHAT IT DOES NOT. Pass one is replaced by a CPU
emulation of its documented contract (float16 inputs, float32 accumulation,
per-document max, `-inf` for an empty document). So this covers how the bound
is APPLIED — lifted through MaxSim's two reductions, compared against the
running threshold, and the survivors rescored — and it covers the `-inf`
structure of the result.

It does NOT cover the hardware error model. A CPU float16 GEMM is more
accurate than a tensor-core one, which is what `closed_form.C_HW_ENVELOPE`
exists to bound; `tests/test_mv_fp16_slack_cpu.py` and the `closed_form`
oracle cover that half. To keep this file from passing on margin alone,
`test_the_bound_is_load_bearing` scales the bound to zero and requires the
prune to start dropping live candidates — if that does not happen, everything
else here is vacuous.

THE RULE BEING TESTED: a pair whose true score reaches its threshold must
never be pruned. Everything else is throughput.

SENSITIVITY, measured rather than assumed. Four mutants were injected into
`mv_fp16` and this file was re-run:

  * a bound 1% too small                              -> 2 tests fail
  * an empty document scored 0.0 instead of `-inf`    -> 1 test fails
  * a zero-token query scored instead of `-inf`       -> 3 tests fail
  * pruning on `<=` instead of `<`                    -> SURVIVES

The last one survives for a reason worth writing down rather than contorting
a test around: pruning is `upper < threshold`, and `upper >= exact` always, so
`upper == threshold` requires the slack to be exactly zero. The `<=` variant
is therefore unobservable while the bound is non-degenerate. `prune_mask`'s
strictness is pinned directly in `tests/test_mv_fp16.py` instead.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from nova_bf import mv_fp16

DIM = 128          # below 64 `eps_dot` returns +inf and the prune is a no-op


def _emulate_pass_one(scale=1.0):
    """`fused_fp16_token_maxima`'s contract, on the CPU.

    `scale` multiplies the float16 rounding error actually committed, so a
    test can ask what happens when pass one is WORSE than the hardware — the
    bound is supposed to tolerate anything up to its modelled envelope.
    """
    def fn(q_half, c_half, document_offsets, *, out=None, **kw):
        qf, cf = q_half.float(), c_half.float()
        if scale != 1.0:
            # Committed error, amplified: fp16(x) - x, scaled and re-applied.
            qf = qf + (qf - q_half.double().float()) * 0.0  # exact for fp16
            sim_hi = q_half.double() @ c_half.double().T
            sim = (qf @ cf.T).double()
            sim = (sim_hi + (sim - sim_hi) * scale).float()
        else:
            sim = qf @ cf.T
        n_docs = int(document_offsets.numel()) - 1
        res = torch.full((q_half.shape[0], n_docs), float("-inf"),
                         dtype=torch.float32)
        for j in range(n_docs):
            a, b = int(document_offsets[j]), int(document_offsets[j + 1])
            if b > a:
                res[:, j] = sim[:, a:b].max(dim=1).values
        return res
    return fn


def _exact(q, c, q_off, d_off):
    """MaxSim one pair at a time in float64 — no shared code with the prune."""
    n_q, n_d = q_off.numel() - 1, d_off.numel() - 1
    out = torch.full((n_q, n_d), float("-inf"), dtype=torch.float64)
    for i in range(n_q):
        for j in range(n_d):
            q0, q1 = int(q_off[i]), int(q_off[i + 1])
            d0, d1 = int(d_off[j]), int(d_off[j + 1])
            if q1 <= q0 or d1 <= d0:
                continue
            out[i, j] = (q[q0:q1].double() @ c[d0:d1].double().T) \
                .max(dim=1).values.sum()
    return out


def _case(seed, n_q=6, n_d=9, normalize=True, empties=True):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    lo = 0 if empties else 1
    q_len = torch.tensor(rng.integers(lo, 5, n_q).tolist())
    d_len = torch.tensor(rng.integers(lo, 7, n_d).tolist())
    q_off = torch.cat([torch.zeros(1, dtype=torch.int64), q_len.cumsum(0)])
    d_off = torch.cat([torch.zeros(1, dtype=torch.int64), d_len.cumsum(0)])
    q = torch.randn(int(q_off[-1]), DIM, generator=g)
    c = torch.randn(int(d_off[-1]), DIM, generator=g)
    if normalize:
        q = torch.nn.functional.normalize(q, dim=1)
        c = torch.nn.functional.normalize(c, dim=1)
    return q, c, q_off, d_off


def _thresholds(exact, keep_frac):
    """Per-query threshold admitting roughly `keep_frac` of the documents."""
    thr = torch.full((exact.shape[0],), float("-inf"))
    for i in range(exact.shape[0]):
        row = exact[i][torch.isfinite(exact[i])]
        if row.numel():
            thr[i] = float(torch.quantile(row.float(), 1.0 - keep_frac))
    return thr


def _run(monkeypatch, q, c, q_off, d_off, thr, *, scale=1.0):
    from nova_bf import multivector_kernels

    monkeypatch.setattr(multivector_kernels, "fused_fp16_token_maxima",
                        _emulate_pass_one(scale))
    res = mv_fp16._pruned_maxsim_scores(q, c, q_off, d_off, thr,
                                        certified=True, want_upper=True)
    assert res is not None, "the prune declined; the test would be vacuous"
    return res


# --- the contract -------------------------------------------------------------

@pytest.mark.parametrize("keep_frac", [0.1, 0.3, 0.6])
@pytest.mark.parametrize("normalize", [True, False])
def test_no_live_candidate_is_ever_pruned(monkeypatch, keep_frac, normalize):
    """THE rule. A pair whose exact score reaches its threshold must survive."""
    pruned_any = False
    for seed in range(12):
        q, c, q_off, d_off = _case(seed, normalize=normalize)
        exact = _exact(q, c, q_off, d_off)
        thr = _thresholds(exact, keep_frac)
        scores, dead, upper = _run(monkeypatch, q, c, q_off, d_off, thr)

        should_live = torch.isfinite(exact) & (exact >= thr[:, None].double())
        dropped = int((dead & should_live).sum())
        assert dropped == 0, (
            f"seed {seed}: {dropped} pair(s) at or above threshold were pruned")
        pruned_any = pruned_any or bool(dead.any())
    assert pruned_any, "nothing was pruned in any case — the test is vacuous"


@pytest.mark.parametrize("keep_frac", [0.1, 0.5])
def test_the_bound_dominates_the_exact_score(monkeypatch, keep_frac):
    """Admissibility one level below the decision: `upper >= exact` on every
    pair. This fails before a candidate is actually lost."""
    for seed in range(12):
        q, c, q_off, d_off = _case(seed)
        exact = _exact(q, c, q_off, d_off)
        thr = _thresholds(exact, keep_frac)
        _, _, upper = _run(monkeypatch, q, c, q_off, d_off, thr)

        fin = torch.isfinite(exact) & torch.isfinite(upper)
        gap = (upper.double() - exact)[fin]
        assert bool((gap >= 0).all()), (
            f"seed {seed}: bound fell below the truth by "
            f"{float(-gap.min()):.3e}")


def test_survivors_carry_the_exact_score_and_non_candidates_stay_neg_inf(
        monkeypatch):
    """The other half of correctness: a scored pair must get the RIGHT score,
    and a non-candidate must stay `-inf` rather than becoming `0.0`."""
    for seed in range(12):
        q, c, q_off, d_off = _case(seed)
        exact = _exact(q, c, q_off, d_off)
        thr = _thresholds(exact, 0.4)
        scores, dead, _ = _run(monkeypatch, q, c, q_off, d_off, thr)

        live = ~dead
        assert torch.equal(torch.isneginf(scores), torch.isneginf(exact) | dead), (
            f"seed {seed}: wrong pairs carry a score")
        both = live & torch.isfinite(exact)
        if bool(both.any()):
            assert torch.allclose(scores[both].double(), exact[both],
                                  rtol=1e-4, atol=1e-3), f"seed {seed}"


# --- the tests above must be able to fail -------------------------------------

def test_the_bound_is_load_bearing(monkeypatch):
    """Scale the bound to zero and the prune MUST start dropping live
    candidates. Without this, every assertion above could be passing on margin
    the bound never needed — which is exactly how a too-tight bound would ship
    unnoticed."""
    real_slack = mv_fp16.slack
    monkeypatch.setattr(mv_fp16, "slack",
                        lambda *a, **k: real_slack(*a, **k) * 0.0)

    dropped_total = 0
    for seed in range(12):
        q, c, q_off, d_off = _case(seed)
        exact = _exact(q, c, q_off, d_off)
        # Thresholds ON a real score, so the decision is genuinely marginal.
        thr = torch.where(torch.isfinite(exact), exact,
                          torch.full_like(exact, float("-inf"))).float().max(1).values
        _, dead, _ = _run(monkeypatch, q, c, q_off, d_off, thr)
        should_live = torch.isfinite(exact) & (exact >= thr[:, None].double())
        dropped_total += int((dead & should_live).sum())

    assert dropped_total > 0, (
        "a ZERO bound pruned nothing that was live — these fixtures never "
        "exercise the margin, so the admissibility tests above prove nothing")


def test_a_pass_one_worse_than_the_hardware_is_still_admissible(monkeypatch):
    """The bound models a tensor-core error envelope far wider than a CPU
    float16 GEMM commits. Amplify pass one's committed error and the bound
    must still hold — otherwise this file only ever tests the easy case."""
    for scale in (4.0, 16.0):
        for seed in range(6):
            q, c, q_off, d_off = _case(seed)
            exact = _exact(q, c, q_off, d_off)
            thr = _thresholds(exact, 0.3)
            _, dead, _ = _run(monkeypatch, q, c, q_off, d_off, thr, scale=scale)
            should_live = torch.isfinite(exact) & (exact >= thr[:, None].double())
            assert int((dead & should_live).sum()) == 0, (
                f"scale {scale}, seed {seed}: a live candidate was pruned")


def test_a_query_with_no_threshold_yet_keeps_its_non_candidates_at_neg_inf(
        monkeypatch):
    """Warm-up: a query whose top-K is not full carries a `-inf` threshold, so
    nothing of its is pruned (`-inf < -inf` is False) — and the survivor write
    mask becomes the ONLY thing keeping its zero-token queries and empty
    documents at `-inf`.

    That is the one reachable path to the mask, which is why the rest of this
    file cannot see a break in it: everywhere else an empty document has
    `upper = -inf` and is pruned before the mask is consulted."""
    for seed in range(12):
        q, c, q_off, d_off = _case(seed)
        exact = _exact(q, c, q_off, d_off)
        thr = _thresholds(exact, 0.4)
        thr[::2] = float("-inf")               # half the queries still filling
        scores, dead, _ = _run(monkeypatch, q, c, q_off, d_off, thr)

        assert torch.equal(torch.isneginf(scores),
                           torch.isneginf(exact) | dead), (
            f"seed {seed}: a non-candidate carries a score while its query "
            f"has no threshold")


def test_zero_token_queries_are_never_scored(monkeypatch):
    """The other half of the write mask, on the same reachable path."""
    for seed in range(20, 32):
        q, c, q_off, d_off = _case(seed)
        q_len = q_off.diff()
        if not bool((q_len == 0).any()):
            continue
        exact = _exact(q, c, q_off, d_off)
        thr = torch.full((q_off.numel() - 1,), float("-inf"))
        scores, dead, _ = _run(monkeypatch, q, c, q_off, d_off, thr)
        empty = q_len == 0
        assert bool(torch.isneginf(scores[empty]).all()), (
            f"seed {seed}: a zero-token query was scored")
