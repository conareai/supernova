"""`score_survivors` against an independent pair-at-a-time reference.

So what this pins is the thing nothing else does: **which pairs get a score at
all, and which stay `-inf`**. That invariant was rewritten when the fold moved
out of the loop — three per-document `continue`s and a per-document
`lengths > 0` mask became one global `pair_len > 0 & d_tok[doc] > 0` — and a
pair wrongly given a finite score is worse than a missing one, because `0.0`
outranks every negative MaxSim and can promote a non-candidate into the top-K.

The reference shares no code with the implementation: it loops pairs, slices
the two ragged token blocks, and takes `(Q @ C.T).max(1).sum()` in float64.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nova_bf import mv_fp16


def _reference(dead, q_flat, c_flat, q_off, d_off):
    """MaxSim one pair at a time, in float64. Non-candidates stay `-inf`."""
    n_q, n_d = dead.shape
    out = torch.full((n_q, n_d), float("-inf"), dtype=torch.float64)
    for i in range(n_q):
        for j in range(n_d):
            q0, q1 = int(q_off[i]), int(q_off[i + 1])
            d0, d1 = int(d_off[j]), int(d_off[j + 1])
            if dead[i, j] or q1 <= q0 or d1 <= d0:
                continue                     # a non-candidate stays -inf
            Q = q_flat[q0:q1].double()
            C = c_flat[d0:d1].double()
            out[i, j] = (Q @ C.T).max(dim=1).values.sum()
    return out


def _check(dead, q_flat, c_flat, q_off, d_off):
    # A sentinel-filled `out`, not `-inf`: it makes a stray write visible as a
    # value rather than blending into the non-candidates we are checking for.
    out = torch.full(dead.shape, float("-inf"))
    got = mv_fp16.score_survivors(dead, q_flat, c_flat, q_off, d_off.cpu(), out)
    want = _reference(dead, q_flat, c_flat, q_off, d_off)

    # The -inf STRUCTURE must match exactly — that is the whole point.
    assert torch.equal(torch.isneginf(got), torch.isneginf(want)), (
        f"wrong pairs scored\ngot  {got}\nwant {want}")
    fin = torch.isfinite(want)
    if bool(fin.any()):
        assert torch.allclose(got[fin].double(), want[fin], rtol=1e-4,
                              atol=1e-3), f"\ngot  {got}\nwant {want}"


def _as(d_off, kind):
    """The forms a caller can hand `d_off_cpu` in."""
    import numpy as np

    if kind == "torch":
        return d_off
    return torch.from_numpy(
        d_off.numpy().astype(np.int64 if kind == "numpy64" else np.int32))


def _ragged(rng, n, choices=(0, 0, 1, 2, 3)):
    lens = torch.tensor([choices[int(rng.integers(len(choices)))]
                         for _ in range(n)], dtype=torch.int64)
    return torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)]), lens


@pytest.mark.parametrize("dcpu", ["torch", "numpy64", "numpy32"])
def test_matches_the_reference_on_ragged_input(dcpu):
    """Zero-token queries and empty documents are drawn frequently, because
    they are the cases the global write mask has to reproduce.

    The document lengths now come from `d_off_cpu`, so the parametrization
    covers the forms a caller can supply it in — a torch tensor, and numpy
    arrays of either width. `torch.as_tensor(...).diff()` must agree with all
    three, because it is the only source of the empty-document exclusion."""
    import numpy as np

    rng = np.random.default_rng(11)
    torch.manual_seed(11)
    for trial in range(200):
        n_q = int(rng.integers(1, 7))
        n_d = int(rng.integers(1, 7))
        dim = int(rng.integers(1, 6))
        q_off, _ = _ragged(rng, n_q)
        d_off, _ = _ragged(rng, n_d)
        q = torch.randn(int(q_off[-1]), dim)
        c = torch.randn(int(d_off[-1]), dim)
        dead = torch.rand(n_q, n_d) < float(rng.uniform(0.0, 1.0))
        _check(dead, q, c, q_off, _as(d_off, dcpu))


def test_an_empty_document_is_never_scored():
    """The regression the fold hoist introduced, stated on its own: document 2
    has zero tokens, so no pair on it may carry a score — `0.0` there would
    outrank every negative MaxSim."""
    q_off = torch.tensor([0, 2, 4])
    d_off = torch.tensor([0, 1, 2, 2, 4])          # document 2 is empty
    _check(torch.zeros(2, 4, dtype=torch.bool), torch.randn(4, 3),
           torch.randn(4, 3), q_off, d_off)


def test_a_zero_token_query_is_never_scored():
    q_off = torch.tensor([0, 0, 3])                # query 0 has no tokens
    d_off = torch.tensor([0, 2, 4])
    _check(torch.zeros(2, 2, dtype=torch.bool), torch.randn(3, 4),
           torch.randn(4, 4), q_off, d_off)


def test_every_pair_dead_leaves_the_output_untouched():
    q_off, d_off = torch.tensor([0, 2, 4]), torch.tensor([0, 2, 4])
    _check(torch.ones(2, 2, dtype=torch.bool), torch.randn(4, 3),
           torch.randn(4, 3), q_off, d_off)


def test_a_single_live_pair_among_many_documents():
    """One survivor spread across many documents is the shape the global fold
    has to index correctly."""
    q_off = torch.tensor([0, 3])
    d_off = torch.arange(0, 21, 2, dtype=torch.int64)    # 10 documents
    dead = torch.ones(1, 10, dtype=torch.bool)
    dead[0, 7] = False
    _check(dead, torch.randn(3, 4), torch.randn(20, 4), q_off, d_off)
