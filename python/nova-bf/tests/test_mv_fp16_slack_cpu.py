"""`mv_fp16.slack` must dominate float16's ABSOLUTE conversion error.

These run on CPU. That is the point: `slack()` is pure arithmetic, but every
other test of this bound is CUDA-gated, so on a machine without a GPU the bound
went entirely unchecked -- and the one bug that got through was exactly here.

WHY THE CASES LOOK THE WAY THEY DO. An earlier version of this file swept
random gaussians across scales `1e0 .. 1e-6` and was WORTHLESS: measured
against the buggy bound it killed it in 0 of 56 cases. Random signs make the
per-component rounding errors incoherent, and they cancel across a 768-term dot
product. The bug needs COHERENT error, so the sweep here uses constant-valued
tokens at `(k + 0.5) * 2^-24` -- half-way between two float16 subnormals, so
every component rounds to even in the SAME direction and the errors add. That
version kills the buggy bound in 10 of 11 cases, from 55x down to 0.77x as the
value climbs out of the band.

Note also that plain powers of two from `2^-24` up are EXACTLY representable as
float16 subnormals and have no rounding error at all, so a sweep over those
would have been equally empty. The half-grid offset is load-bearing.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nova_bf import closed_form as cf
from nova_bf import mv_fp16

DIM = 768
# float16 subnormals are the multiples of 2^-24, so this lands exactly between
# two of them and rounds to even -- the same direction for every component.
_HALF_GRID = 2.0 ** -24


def _worst_case_error(q: np.ndarray, c: np.ndarray) -> float:
    """|true dot - dot of the float16-rounded operands|, in binary64."""
    qh = q.astype(np.float16).astype(np.float64)
    ch = c.astype(np.float16).astype(np.float64)
    return abs(float(q.astype(np.float64) @ c.astype(np.float64))
               - float(qh @ ch))


def _slack_for(q: np.ndarray, c: np.ndarray, n_tokens: int = 1) -> float:
    qn = float(np.linalg.norm(q.astype(np.float64)))
    cn = float(np.linalg.norm(c.astype(np.float64)))
    return float(mv_fp16.slack(torch.tensor([n_tokens]), qn, cn, len(q))[0])


def test_slack_covers_a_query_token_that_float16_rounds_entirely_to_zero():
    """The deterministic worst case: every component is 2^-25, exactly half of
    float16's smallest subnormal, so round-to-even sends the whole token to
    zero and pass one reports 0.0 against a nonzero true score. A
    relative-only bound cannot see this at all."""
    q = np.full(DIM, 2.0 ** -25, dtype=np.float32)
    c = np.ones(DIM, dtype=np.float32)
    assert not q.astype(np.float16).any(), "fixture no longer rounds to zero"
    assert _slack_for(q, c) >= _worst_case_error(q, c)


def test_slack_covers_a_corpus_token_that_float16_rounds_to_zero():
    """Same hole on the corpus side -- the conversion term is symmetric."""
    q = np.ones(DIM, dtype=np.float32)
    c = np.full(DIM, 2.0 ** -25, dtype=np.float32)
    assert not c.astype(np.float16).any()
    assert _slack_for(q, c) >= _worst_case_error(q, c)


@pytest.mark.parametrize("k", [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_slack_covers_coherent_rounding_across_the_subnormal_band(k):
    """Sweep the band with COHERENT error, which is what actually bites.

    Every component sits half-way between two float16 subnormals, so all of
    them round the same way and the errors accumulate instead of cancelling.
    Against the buggy bound this fails for k <= 256.
    """
    q = np.full(DIM, (k + 0.5) * _HALF_GRID, dtype=np.float32)
    c = np.ones(DIM, dtype=np.float32)
    assert _slack_for(q, c) >= _worst_case_error(q, c)


@pytest.mark.parametrize("k", [0, 4, 64])
def test_slack_covers_coherent_rounding_on_the_corpus_side_too(k):
    q = np.ones(DIM, dtype=np.float32)
    c = np.full(DIM, (k + 0.5) * _HALF_GRID, dtype=np.float32)
    assert _slack_for(q, c) >= _worst_case_error(q, c)


@pytest.mark.parametrize("n_tokens", [1, 7, 64])
def test_slack_covers_the_summed_error_of_a_multi_token_query(n_tokens):
    """MaxSim sums over a query's tokens, so the per-token error accumulates.
    The worst case is every token being the same maximally-bad vector."""
    q = np.full(DIM, 2.0 ** -25, dtype=np.float32)
    c = np.ones(DIM, dtype=np.float32)
    assert _slack_for(q, c, n_tokens) >= n_tokens * _worst_case_error(q, c)


@pytest.mark.parametrize("scale", [1e0, 1e-3, 1e-6])
@pytest.mark.parametrize("seed", range(4))
def test_slack_covers_ordinary_random_tokens(scale, seed):
    """A plain sanity check that the bound holds on unremarkable data.

    Deliberately NOT claimed as coverage of the eta term: random signs make
    the rounding errors incoherent and they cancel, so this passes even with
    the buggy bound. It is here to catch a slack that goes wrong in the
    ordinary regime, which the adversarial cases above would not notice.
    """
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal(DIM) * scale).astype(np.float32)
    c = (rng.standard_normal(DIM) * scale).astype(np.float32)
    assert _slack_for(q, c) >= _worst_case_error(q, c)


def test_the_eta_terms_are_actually_being_passed():
    """Pin the call itself, not just its consequence.

    `eps_dot` silently defaults `eta_q`/`eta_c` to 0, so an edit that drops
    them would still typecheck, still run, and still pass any test whose data
    happens to have well-scaled norms. Compare against both spellings so the
    failure message says which one the code is using.
    """
    qn, cn = 8.259062e-07, 27.712813
    without_eta = float(cf.eps_dot(
        DIM, cf.U16, cf.U16, np.asarray([qn]), cn)[0])
    with_eta = float(cf.eps_dot(
        DIM, cf.FP16.u_t, cf.FP16.u_t, np.asarray([qn]), cn,
        cf.C_HW_ENVELOPE, False, cf.FP16.eta_t, cf.FP16.eta_t,
        cf.FP16.lam_t)[0])
    assert with_eta > 10.0 * without_eta, "fixture no longer separates the two"

    got = float(mv_fp16.slack(torch.tensor([1]), qn, cn, DIM)[0])
    assert got >= with_eta, (
        "slack is below the eta-inclusive allowance — eps_dot is being called "
        "without eta_q/eta_c")


def test_slack_charges_the_float32_summation_terms():
    """The four charges are eps_dot, pass one's float32 sum, the exact path's
    float32 dot, and the exact path's float32 sum. Deleting the last three
    leaves a slack that is still above `eps_dot` alone, so a test comparing
    against `eps_dot` cannot see them. Compare against the composition
    instead, which is what actually pins them.
    """
    qn = cn = 1.0
    n = 30
    fmt = cf.FP16
    eps1 = float(cf.eps_dot(
        DIM, fmt.u_t, fmt.u_t, np.asarray([qn]), cn,
        cf.C_HW_ENVELOPE, False, fmt.eta_t, fmt.eta_t, fmt.lam_t)[0])
    g_dim, g_tok = cf.gamma(DIM), cf.gamma(n)
    prod = qn * cn
    expected_E = n * (eps1 + g_dim * prod + 2.0 * g_tok * (prod + eps1))

    got = float(mv_fp16.slack(torch.tensor([n]), qn, cn, DIM)[0])
    assert got >= expected_E, "slack is below its own stated composition"
    # And strictly above eps_dot alone, so the float32 charges are present.
    assert got > n * eps1
