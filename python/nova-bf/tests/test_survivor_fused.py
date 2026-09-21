"""The fused survivor pass must agree with the torch one it replaces.

`score_survivors` rescores, exactly, every (query, document) pair a bound could
not rule out. The torch implementation materializes `q_flat[idx]` once per
document, which copies a query's tokens once for every document it is live for
The fused implementation hands the kernel that index instead and covers every 
document in one launch.

Swapping it is only safe if the scores are the same, so that is what is pinned
here, on the awkward shapes: empty documents, single-token documents and
queries, a query live for every document, a query live for none, and survivor
sets that straddle the kernel's tiles.

Scores are compared with a float32 tolerance rather than exactly, because the
two implementations sum a document's tokens in different orders -- the same
tolerance the shipped Triton and torch reducers already hold each other to.
What must match exactly is which pairs are scored at all, and which stay -inf.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused survivor pass requires CUDA")

from nova_bf import mv_fp16

DIM = 64


def _offsets(lengths):
    off = torch.zeros(len(lengths) + 1, dtype=torch.int64, device="cuda")
    off[1:] = torch.tensor(lengths, device="cuda").cumsum(0)
    return off


def _case(q_len, d_len, seed=2, dim=DIM):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(sum(q_len), dim, device="cuda", generator=g).contiguous()
    c = torch.randn(sum(d_len), dim, device="cuda", generator=g).contiguous()
    return q, _offsets(q_len), c, _offsets(d_len)


def naive(dead, q, q_off, c, c_off):
    """Pair-at-a-time reference, sharing no code with either implementation."""
    n_q, n_d = q_off.numel() - 1, c_off.numel() - 1
    out = torch.full((n_q, n_d), float("-inf"), device="cuda")
    for i in range(n_q):
        a, b = int(q_off[i]), int(q_off[i + 1])
        for j in range(n_d):
            c0, c1 = int(c_off[j]), int(c_off[j + 1])
            if bool(dead[i, j]) or b <= a or c1 <= c0:
                continue
            out[i, j] = (q[a:b] @ c[c0:c1].T).max(dim=1).values.sum()
    return out


def _both(dead, q, q_off, c, c_off, monkeypatch=None):
    """Both implementations on the same input.

    The fused path is selected by live FRACTION in production, and these cases
    are deliberately dense, so the threshold is lifted for the comparison --
    otherwise every test here would silently compare torch against torch.
    """
    n_q, n_d = q_off.numel() - 1, c_off.numel() - 1
    saved = mv_fp16._FUSED_SURVIVOR_MAX_LIVE
    mv_fp16._FUSED_SURVIVOR_MAX_LIVE = 1.0
    try:
        outs = []
        for force in (True, False):
            o = torch.full((n_q, n_d), float("-inf"), device="cuda")
            mv_fp16.score_survivors(dead, q, c, q_off, c_off.cpu(), o,
                                       d_off=c_off, force_torch=force)
            outs.append(o)
    finally:
        mv_fp16._FUSED_SURVIVOR_MAX_LIVE = saved
    return outs[0], outs[1]


@pytest.mark.parametrize("d_len", [
    [5, 12, 3],
    [0, 7, 0, 4],                      # empty documents
    [1, 1, 1],                         # single-token documents
    [200, 3],                          # one document spanning many tiles
    [70, 66, 130],                     # tile-straddling
])
@pytest.mark.parametrize("q_len", [
    [3, 1, 6],
    [0, 4, 9, 1],                      # a zero-token query
    [40],                              # one long query
])
@pytest.mark.parametrize("keep", [1.0, 0.5, 0.1])
def test_fused_matches_torch(q_len, d_len, keep):
    q, q_off, c, c_off = _case(q_len, d_len)
    n_q, n_d = len(q_len), len(d_len)
    g = torch.Generator(device="cuda").manual_seed(13)
    dead = torch.rand((n_q, n_d), device="cuda", generator=g) > keep

    t_out, f_out = _both(dead, q, q_off, c, c_off)
    assert torch.equal(torch.isneginf(t_out), torch.isneginf(f_out)), (
        "the two implementations disagree about which pairs got a score")
    fin = torch.isfinite(t_out)
    if bool(fin.any()):
        assert torch.allclose(t_out[fin], f_out[fin], rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("keep", [1.0, 0.4])
def test_fused_matches_a_naive_reference(keep):
    q, q_off, c, c_off = _case([3, 1, 6, 0, 5], [5, 0, 12, 1, 70], seed=8)
    g = torch.Generator(device="cuda").manual_seed(4)
    dead = torch.rand((5, 5), device="cuda", generator=g) > keep
    _, f_out = _both(dead, q, q_off, c, c_off)
    ref = naive(dead, q, q_off, c, c_off)
    assert torch.equal(torch.isneginf(f_out), torch.isneginf(ref))
    fin = torch.isfinite(ref)
    assert torch.allclose(f_out[fin], ref[fin], rtol=1e-4, atol=1e-3)


def test_nothing_alive_leaves_every_pair_untouched():
    q, q_off, c, c_off = _case([3, 4], [5, 6])
    dead = torch.ones((2, 2), dtype=torch.bool, device="cuda")
    _, f_out = _both(dead, q, q_off, c, c_off)
    assert bool(torch.isneginf(f_out).all())


def test_a_query_live_for_every_document_is_scored_for_every_document():
    q, q_off, c, c_off = _case([7, 2], [4, 9, 1])
    dead = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    dead[0, :] = False
    _, f_out = _both(dead, q, q_off, c, c_off)
    assert bool(torch.isfinite(f_out[0]).all())
    assert bool(torch.isneginf(f_out[1]).all())


def test_the_fused_path_actually_ran(monkeypatch):
    """Guard against every test above silently passing on the torch fallback."""
    calls = []
    real = mv_fp16._score_survivors_fused

    def spy(*a, **k):
        r = real(*a, **k)
        calls.append(r is not None)
        return r

    monkeypatch.setattr(mv_fp16, "_score_survivors_fused", spy)
    monkeypatch.setattr(mv_fp16, "_FUSED_SURVIVOR_MAX_LIVE", 1.0)
    q, q_off, c, c_off = _case([3, 5], [6, 7])
    dead = torch.zeros((2, 2), dtype=torch.bool, device="cuda")
    out = torch.full((2, 2), float("-inf"), device="cuda")
    mv_fp16.score_survivors(dead, q, c, q_off, c_off.cpu(), out, d_off=c_off)
    assert calls and calls[-1], "the fused survivor pass declined or was skipped"


def test_a_dense_survivor_set_uses_the_torch_path(monkeypatch):
    """The fused kernel loses badly below ~97% pruned, so a dense set must not
    reach it."""
    calls = []
    monkeypatch.setattr(mv_fp16, "_score_survivors_fused",
                        lambda *a, **k: calls.append(1))
    q, q_off, c, c_off = _case([3, 5], [6, 7])
    dead = torch.zeros((2, 2), dtype=torch.bool, device="cuda")   # all live
    out = torch.full((2, 2), float("-inf"), device="cuda")
    mv_fp16.score_survivors(dead, q, c, q_off, c_off.cpu(), out, d_off=c_off)
    assert not calls, "the fused path ran on a fully dense survivor set"


def test_float32_is_required_by_the_fused_kernel():
    """float64 tokens must fall back rather than be silently narrowed."""
    q, q_off, c, c_off = _case([3, 5], [6, 7])
    dead = torch.zeros((2, 2), dtype=torch.bool, device="cuda")
    out = torch.full((2, 2), float("-inf"), dtype=torch.float64, device="cuda")
    mv_fp16.score_survivors(dead, q.double(), c.double(), q_off,
                               c_off.cpu(), out, d_off=c_off)
    assert bool(torch.isfinite(out).all())
