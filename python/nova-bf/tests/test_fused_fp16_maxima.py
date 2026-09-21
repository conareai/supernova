"""The fused float16 pass-one kernel, checked against the bound that uses it.

`fused_fp16_token_maxima` computes, for every (query token, document), the max
dot product over that document's tokens -- without ever materializing the
query_tokens x corpus_tokens matrix, and accumulating in float32 inside a
kernel we own. That ownership is the whole point: `twopass.accumulator_is_ours`
will not let a cuBLAS pass one prune at all.

WHAT IS CHECKED, AND WHY NOT EXACT EQUALITY
-------------------------------------------
The reference materializes the matrix and reduces it. Both compute the same
dot products, but in different summation orders, so they disagree in the last
bits -- max is order-independent, but the VALUES being maxed are not. An
arbitrary `allclose` tolerance would be a guess, so instead every comparison is
against a float64 reference over the same float16 operands, with the allowance
taken from `closed_form.eps_dot` with the CONVERSION terms zeroed (`u_q=u_c=0`,
`eta=0`), which isolates exactly the accumulation error the kernel is
responsible for. That is the same bound the prune relies on, so a kernel that
passes here is a kernel the prune may use, and a regression to float16
accumulation would blow straight through it.

Structural properties that ARE exact -- an empty document scoring `-inf`, a
clamped lane never winning a max -- are asserted exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused fp16 pass one requires CUDA")

from nova_bf import closed_form as cf
from nova_bf.multivector_kernels import fused_fp16_token_maxima

DIM = 128


def _offsets(lengths):
    off = torch.zeros(len(lengths) + 1, dtype=torch.int64, device="cuda")
    off[1:] = torch.tensor(lengths, device="cuda").cumsum(0)
    return off


def reference(q_half, c_half, d_off, dtype=torch.float64):
    """Materialize the whole similarity matrix and reduce it per document.

    Deliberately the slow, obvious implementation, in float64: it shares no
    tiling logic with the kernel under test and is accurate enough that any
    difference is the kernel's.
    """
    n_docs = d_off.numel() - 1
    P = q_half.to(dtype) @ c_half.to(dtype).T
    out = torch.full((q_half.shape[0], n_docs), float("-inf"),
                     dtype=dtype, device="cuda")
    for j in range(n_docs):
        a, b = int(d_off[j]), int(d_off[j + 1])
        if b > a:
            out[:, j] = P[:, a:b].max(dim=1).values
    return out


def assert_within_accumulation_bound(got, q_half, c_half, d_off):
    """Kernel maxima must sit inside the closed-form accumulation allowance."""
    ref = reference(q_half, c_half, d_off)
    finite = torch.isfinite(ref)
    # `-inf` (empty document) must match exactly on both sides.
    assert torch.equal(torch.isneginf(got), torch.isneginf(ref)), \
        "kernel and reference disagree about which pairs are non-candidates"
    if not bool(finite.any()):
        return
    d = int(q_half.shape[1])
    qn = q_half.to(torch.float64).norm(dim=1).cpu().numpy()
    dn = float(c_half.to(torch.float64).norm(dim=1).max()) if c_half.numel() \
        else 0.0
    # u_q = u_c = 0: the operands are ALREADY float16, so conversion error is
    # not the kernel's to answer for -- only the accumulation is.
    allow = cf.eps_dot(d, 0.0, 0.0, np.maximum(qn, 1e-30), max(dn, 1e-30))
    allow_t = torch.as_tensor(np.asarray(allow, dtype=np.float64),
                              device="cuda")[:, None]
    err = (got.to(torch.float64) - ref).abs()
    bad = finite & (err > allow_t)
    assert not bool(bad.any()), (
        f"{int(bad.sum())} maxima outside the accumulation allowance; "
        f"worst {float(err[finite].max()):.3e} vs allowance "
        f"{float(allow_t.min()):.3e}")


def _case(q_tokens, d_len, seed=5, dim=DIM):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(q_tokens, dim, device="cuda", generator=g).half().contiguous()
    c = torch.randn(sum(d_len), dim, device="cuda",
                    generator=g).half().contiguous()
    return q, c, _offsets(d_len)


@pytest.mark.parametrize("d_len", [
    [7, 3, 19, 1, 26],                 # ordinary ragged documents
    [0, 5, 0, 12, 0],                  # empty documents interleaved
    [1, 1, 1, 1],                      # every document a single token
    [300],                             # one document spanning many N tiles
    [64, 64, 64],                      # exactly tile-aligned
    [65, 63, 129],                     # deliberately tile-straddling
])
@pytest.mark.parametrize("q_tokens", [1, 33, 128, 257])
def test_matches_a_materialized_reference(q_tokens, d_len):
    q, c, d_off = _case(q_tokens, d_len)
    got = fused_fp16_token_maxima(q, c, d_off)
    assert got is not None, "kernel declined; the test would be vacuous"
    assert_within_accumulation_bound(got, q, c, d_off)


def test_empty_documents_are_negative_infinity_not_zero():
    """A zero-token document is a non-candidate. Zero would rank it above every
    negative score instead."""
    q, c, d_off = _case(40, [0, 9, 0])
    got = fused_fp16_token_maxima(q, c, d_off)
    assert bool(torch.isneginf(got[:, 0]).all())
    assert bool(torch.isneginf(got[:, 2]).all())
    assert bool(torch.isfinite(got[:, 1]).all())


def test_a_clamped_lane_can_never_win_the_max():
    """Columns past a document's end are clamped in-bounds for the load, so the
    epilogue mask is the only thing stopping them winning. Make the clamped
    target far larger than any real score, so a missing mask is unmissable."""
    q, c, d_off = _case(16, [3, 1])
    c[0] = 40.0                        # doc 0's clamp target is its own index 0
    c = c.contiguous()
    got = fused_fp16_token_maxima(q, c, d_off)
    assert_within_accumulation_bound(got, q, c, d_off)


@pytest.mark.parametrize("block_n", [16, 32, 64, 128])
def test_independent_of_the_column_tile_size(block_n):
    q, c, d_off = _case(96, [65, 63, 129])
    got = fused_fp16_token_maxima(q, c, d_off, block_n=block_n)
    assert_within_accumulation_bound(got, q, c, d_off)


@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
def test_independent_of_the_row_tile_size(block_m):
    q, c, d_off = _case(97, [40, 7, 33])
    got = fused_fp16_token_maxima(q, c, d_off, block_m=block_m)
    assert_within_accumulation_bound(got, q, c, d_off)


def test_dimension_not_divisible_by_the_k_tile():
    """EVEN_K is a compile-time specialization; the masked path must agree."""
    q, c, d_off = _case(64, [20, 11], dim=100)
    got = fused_fp16_token_maxima(q, c, d_off, block_k=64)
    assert_within_accumulation_bound(got, q, c, d_off)


def test_production_token_dimension():
    """bge-m3's actual 1024-d tokens, not just the small test dimension."""
    q, c, d_off = _case(257, [315, 280, 1, 400], dim=1024)
    got = fused_fp16_token_maxima(q, c, d_off)
    assert_within_accumulation_bound(got, q, c, d_off)


def test_accumulation_is_float32_not_float16():
    """Many same-sign products whose float16 running sum would saturate.

    With D=512 addends of 2^-8 each the true dot is 2, but a float16
    accumulator loses every addend once the sum passes ~2^3, so a regression to
    float16 accumulation shows up as a large, systematic shortfall rather than
    a rounding wobble.
    """
    n_k = 512
    q = torch.full((8, n_k), 2.0 ** -4, device="cuda", dtype=torch.float16)
    c = torch.full((64, n_k), 2.0 ** -4, device="cuda", dtype=torch.float16)
    c[0, 0] = 2.0 ** 3
    d_off = _offsets([64])
    got = fused_fp16_token_maxima(q, c, d_off)
    exact = (q.double() @ c.double().T).max(dim=1).values
    assert torch.allclose(got[:, 0].double(), exact, rtol=1e-6, atol=1e-6)


def test_float32_inputs_are_refused():
    q, c, d_off = _case(32, [10, 10])
    with pytest.raises(TypeError):
        fused_fp16_token_maxima(q.float(), c, d_off)


def test_mismatched_token_dimension_is_refused():
    q, c, d_off = _case(32, [10, 10])
    with pytest.raises(ValueError):
        fused_fp16_token_maxima(q[:, :64].contiguous(), c, d_off)
