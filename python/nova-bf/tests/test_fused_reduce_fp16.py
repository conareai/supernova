"""The fused ragged reducer must accept a float16 similarity tile.

An fp16 pass one exists to use tensor cores, but if the reducer only took
float32 the caller would have to materialize a second, full-size fp32 copy of
the score matrix first -- and that conversion moves more bytes than the tensor
cores save. The kernel accumulates in fp32 regardless of the input dtype
(`row_max` and `score` are declared `tl.float32`), so accepting fp16 changes
nothing about the arithmetic.

Widening float16 to float32 is EXACT -- every fp16 value is representable in
fp32 -- so the reducer fed fp16 must agree BIT FOR BIT with the reducer fed
that same tile widened. Equality, not a tolerance, is the right assertion here,
and a tolerance would hide exactly the bug this guards against.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused ragged reducer requires CUDA"
)

from nova_bf.multivector_kernels import fused_ragged_maxsim_reduce


def _offsets(lengths):
    off = torch.zeros(len(lengths) + 1, dtype=torch.int64, device="cuda")
    off[1:] = torch.tensor(lengths, device="cuda").cumsum(0)
    return off


@pytest.fixture
def tile():
    gen = torch.Generator(device="cuda").manual_seed(11)
    # Zero-token and one-token rows on both axes: the ragged edge cases that a
    # dtype change is most likely to break through a masked `other=-inf` load.
    q_len = [0, 1, 7, 3, 14, 2, 9, 1]
    d_len = [0, 1, 25, 8, 40, 3, 17]
    q_off, d_off = _offsets(q_len), _offsets(d_len)
    n_qtok, n_dtok = int(q_off[-1]), int(d_off[-1])
    # Scaled so the values exercise fp16's range without being denormal.
    sim = (torch.randn(max(n_qtok, 1), max(n_dtok, 1), device="cuda",
                       generator=gen) * 3.0)
    return sim, q_off, d_off, len(q_len), len(d_len)


def _run(sim, q_off, d_off, n_q, n_d):
    out = torch.full((n_q, n_d), float("-inf"), device="cuda")
    fused_ragged_maxsim_reduce(sim, q_off, d_off, query_start=0,
                               query_token_base=0, n_queries=n_q, out=out)
    return out


def test_float16_input_matches_the_widened_float32_input_exactly(tile):
    sim, q_off, d_off, n_q, n_d = tile
    half = sim.half()
    got = _run(half, q_off, d_off, n_q, n_d)
    ref = _run(half.float(), q_off, d_off, n_q, n_d)   # exact widening
    assert torch.equal(got, ref), (
        "fp16 and widened-fp32 inputs disagree; the kernel is not accumulating "
        "in fp32 for one of them")


def test_float16_output_is_still_float32(tile):
    sim, q_off, d_off, n_q, n_d = tile
    got = _run(sim.half(), q_off, d_off, n_q, n_d)
    assert got.dtype == torch.float32


def test_float16_keeps_empty_rows_and_columns_at_negative_infinity(tile):
    """A zero-token query or document is a non-candidate, not a zero score."""
    sim, q_off, d_off, n_q, n_d = tile
    got = _run(sim.half(), q_off, d_off, n_q, n_d)
    q_empty = (q_off.diff() == 0)
    d_empty = (d_off.diff() == 0)
    assert bool(torch.isneginf(got[q_empty]).all())
    assert bool(torch.isneginf(got[:, d_empty]).all())
    assert bool(torch.isfinite(got[~q_empty][:, ~d_empty]).all())


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_unsupported_similarity_dtypes_are_refused(tile, dtype):
    """Only the two dtypes whose fp32 accumulation is verified are allowed."""
    sim, q_off, d_off, n_q, n_d = tile
    with pytest.raises(TypeError):
        _run(sim.to(dtype), q_off, d_off, n_q, n_d)
