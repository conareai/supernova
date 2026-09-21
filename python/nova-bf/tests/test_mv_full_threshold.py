"""`_mv_full_threshold` must never let a threshold broadcast.

A search member may own only a SUBSET of the query rows. The thresholds it
carries are sized to that subset, and they get scattered into a full-length
vector whose unowned rows stay `-inf` so they can never be pruned.

The hazard both branches guard is narrow and easy to miss: a LENGTH-1
threshold broadcasts silently, both through `full_thr[qsel] = thr` and through
`thresholds[:, None]` in `prune_mask`. Every other length mismatch raises on
its own. So length-1 is the case that quietly applies one query's top-K to
every row and marks the wrong pairs dead -- which is a wrong ground truth, not
a slow path.

`qsel` is `None` or a `slice` in this codebase (see `_plain_block`), so the
selected count is computed with `slice.indices()`, which is correct even for a
step the `stop - start` arithmetic used elsewhere would get wrong.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nova_bf.compute import _mv_full_threshold
from nova_bf.tiebreak import pack_score


class _Q:
    def __init__(self, n_q):
        self.n_q = n_q


def _packed(values):
    return pack_score(torch.tensor(values, dtype=torch.float32))


def test_full_selection_expands_unchanged():
    out = _mv_full_threshold(_Q(4), _packed([1.0, 2.0, 3.0, 4.0]), None)
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_a_subset_leaves_unowned_rows_at_negative_infinity():
    """The rows this member does not own must be -inf: a real threshold there
    would prune them against another member's top-K."""
    out = _mv_full_threshold(_Q(5), _packed([1.0, 2.0]), slice(1, 3))
    assert out[1].item() == 1.0 and out[2].item() == 2.0
    assert torch.isneginf(out[[0, 3, 4]]).all()


def test_a_length_one_threshold_on_a_subset_is_refused():
    """THE bug this guards. Without the check, torch broadcasts the single
    value across all three selected rows and prunes them against one query's
    top-K -- silently, with no error."""
    with pytest.raises(ValueError, match="threshold length 1 != 3"):
        _mv_full_threshold(_Q(6), _packed([5.0]), slice(1, 4))


def test_a_length_one_threshold_on_the_full_set_is_refused():
    with pytest.raises(ValueError, match="threshold length 1 != 4"):
        _mv_full_threshold(_Q(4), _packed([5.0]), None)


@pytest.mark.parametrize("sel,n_thr", [(slice(0, 3), 2), (slice(2, 6), 5)])
def test_other_length_mismatches_are_refused_too(sel, n_thr):
    """These would raise from torch anyway; refusing here names the quantity."""
    with pytest.raises(ValueError, match="threshold length"):
        _mv_full_threshold(_Q(8), _packed([1.0] * n_thr), sel)


def test_a_stepped_slice_counts_its_elements_not_its_span():
    """`stop - start` would say 4 for slice(0, 8, 2); the real count is 4 of a
    span of 8. Using `slice.indices()` keeps this right if a non-contiguous
    selection ever reaches here."""
    out = _mv_full_threshold(_Q(8), _packed([1.0, 2.0, 3.0, 4.0]),
                             slice(0, 8, 2))
    assert out[[0, 2, 4, 6]].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert torch.isneginf(out[[1, 3, 5, 7]]).all()


def test_all_negative_infinity_means_nothing_is_prunable_yet():
    """No query has a threshold -- score the slice normally rather than prune
    against -inf."""
    assert _mv_full_threshold(_Q(3), _packed([float("-inf")] * 3), None) is None


def test_a_subset_with_no_finite_threshold_also_declines():
    assert _mv_full_threshold(_Q(5), _packed([float("-inf")] * 2),
                              slice(1, 3)) is None
