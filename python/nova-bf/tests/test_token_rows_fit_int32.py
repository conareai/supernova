"""The fused token kernels' int32 address guard.

`token_rows_fit_int32` decides whether `row_index * dim` stays inside int32 for
every token row. Both operands of that product are int32 inside Triton, and the
product is not promoted, so a shape that overflows does not raise -- it wraps,
reads a different token, and returns a WRONG SCORE. The guard is what turns
that into a decline.

Checking the row COUNT instead of the product is the mistake this pins: at
dim=1024 an `n_rows < 2**31` test passes for shapes whose addresses overflow by
three orders of magnitude.

Deliberately CPU-only: the predicate is pure arithmetic (like the module's
existing `offsets_fit_int32`), so the guard protecting a CUDA-only kernel can
still be tested on a machine with no GPU -- which is exactly when a regression
here would otherwise go unnoticed.
"""
from __future__ import annotations

import pytest

from nova_bf.multivector_kernels import _INT32_MAX, token_rows_fit_int32


def test_a_production_slice_is_accepted():
    """265,521 query tokens and 20,487 corpus tokens at bge-m3's D=1024 --
    the guard must not decline the shape this was built for."""
    assert token_rows_fit_int32(265_521, 1024)
    assert token_rows_fit_int32(20_487, 1024)


def test_the_boundary_is_the_product_not_the_row_count():
    """The whole point: a row count far inside int32 whose ADDRESSES are not."""
    rows = 10_000_000          # 1e7 << 2**31, so a row-count check would pass
    assert rows < _INT32_MAX
    assert not token_rows_fit_int32(rows, 1024)


@pytest.mark.parametrize("dim", [64, 128, 1024])
def test_exactly_at_the_limit_is_accepted_and_one_past_is_not(dim):
    limit = _INT32_MAX // dim
    assert token_rows_fit_int32(limit, dim)
    assert not token_rows_fit_int32(limit + 1, dim)


def test_larger_dimension_lowers_the_row_limit():
    """Doubling the token dimension must halve how many rows are addressable;
    a guard that ignored `dim` would return the same answer for both."""
    rows = _INT32_MAX // 1024
    assert token_rows_fit_int32(rows, 1024)
    assert not token_rows_fit_int32(rows, 2048)


@pytest.mark.parametrize("rows,dim", [(0, 1024), (-1, 1024), (10, 0), (10, -1)])
def test_degenerate_shapes_are_not_declined(rows, dim):
    """An empty or nonsensical shape addresses nothing, so it cannot overflow.
    Declining it here would turn a harmless no-op into a silent fallback."""
    assert token_rows_fit_int32(rows, dim)


def test_no_overflow_in_the_guard_itself():
    """Python ints are arbitrary precision, so the check must stay exact rather
    than computing the product in a width that could itself wrap."""
    assert not token_rows_fit_int32(2 ** 40, 2 ** 20)


def test_the_padded_output_tile_is_what_must_fit():
    """`fused_fp16_token_maxima` stores at unwrapped `offs_m`, which runs to
    the end of the final PADDED tile, and a masked lane still forms its
    address. So the guard has to use `cdiv(M, BLOCK_M) * BLOCK_M`, not `M`.

    Pinned arithmetically because the shape that separates the two is far too
    large to allocate: a row count that fits on its own but whose padded tile
    does not.
    """
    dim, block_m = 1024, 128
    m = _INT32_MAX // dim                      # the largest M that fits
    assert token_rows_fit_int32(m, dim)
    padded = -(-m // block_m) * block_m        # cdiv, then round up
    assert padded > m
    assert not token_rows_fit_int32(padded, dim), (
        "padding pushes it over, which is exactly what the guard must catch")
