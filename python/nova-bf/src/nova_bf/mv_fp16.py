"""Admissible float16 pass one for multivector (MaxSim) scoring.

Pass one computes a float16 approximation, adds a certified one-sided error
allowance, and exactly rescoring in float32 any pair the bound cannot prune.
The bound reuses `closed_form.eps_dot`; MaxSim's per-document `max` adds no
factor because it is 1-Lipschitz, while the outer sum contributes one allowance
per query token.

`fused_fp16_token_maxima` performs the float16 GEMM with float32 accumulation
and per-document max without materializing the full similarity matrix.
Certification is delegated to `twopass.certify_closed_form`.

"Exact" means float32 arithmetic, not bitwise agreement between scorers:
different reduction orders can differ by a few low bits. `slack` therefore
bounds both pass-one error and exact-scorer rounding relative to real
arithmetic, so `approx + slack` safely dominates any supported exact scorer.
Any resulting top-k differences are confined to the scorer-roundoff boundary.

Caller invariant: `thresholds` must not exceed the current k-th score under the
semantics being pruned against.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def certify(dim: int, device) -> str | None:
    """Reason this device may not prune with the float16 bound, or `None`.

    Delegates to the existing two-pass certification -- device generation,
    exact-math-mode flags, float16 conversion behaviour and the accumulation
    probes -- with our kernel supplied as the pass-one implementation, so the
    accumulation probe exercises the code that will actually run.
    """
    from nova_bf import twopass

    return twopass.certify_closed_form(dim, device, rowmax_fn=_probe_rowmax)


def _probe_rowmax(Qh, Ch, cs):
    """Adapt the ragged kernel to `twopass.probe_accumulation`.

    Treats the whole corpus as one document. Non-unit column scales are
    rejected because the kernel does not apply column scaling.
    """
    import torch

    from nova_bf.multivector_kernels import fused_fp16_token_maxima

    if cs is not None and not bool(torch.all(cs == 1.0)):
        raise ValueError(
            "the multivector float16 pass one has no column scaling, so a "
            "probe with non-unit column scales would not be testing it")
    d_off = torch.tensor([0, int(Ch.shape[0])], dtype=torch.int64,
                         device=Qh.device)
    out = fused_fp16_token_maxima(Qh, Ch, d_off)
    return None if out is None else out[:, 0].contiguous()


def to_half(x, n_max: float | None = None):
    """Convert `x` to float16, or return `None` if conversion is unsafe.

    Uses the shared closed-form overflow guard. `n_max`, if provided, must be
    an upper bound on the largest row norm; otherwise it is computed with
    `norm_upper`.
    """
    import torch

    from nova_bf import closed_form as cf

    if x.ndim != 2:
        raise ValueError("to_half expects a 2D (tokens, dim) matrix")
    if x.numel() == 0:
        return x.half()
    if n_max is None:
        n_max = norm_upper(x)
    if not cf.overflow_ok(int(x.shape[1]), n_max, cf.FP16):
        return None
    h = x.half()
    # Kept even with the guard above: it also fails safely on a non-finite
    # input, which no norm-based test can catch.
    if not bool(torch.isfinite(h).all()):
        return None
    return h


def norm_upper(x) -> float:
    """Return an upper bound on the largest row norm.

    Norms are computed in binary64 and inflated outward so downward rounding
    cannot weaken the pruning bound. The inflation uses `gamma(dim + 2)` plus
    a small fixed margin for reduction/max implementation details.
    """
    if x.ndim != 2:
        raise ValueError("norm_upper expects a 2D (tokens, dim) matrix")
    if x.numel() == 0:
        return 0.0
    n = float(x.double().norm(dim=1).max())
    # gamma_n = n*u / (1 - n*u) with u = 2^-53, binary64's unit roundoff.
    k = (int(x.shape[1]) + 2) * 2.0 ** -53
    return n * (1.0 + k / (1.0 - k)) * (1.0 + 2.0 ** -40)


def slack(q_tokens, q_norm_max: float, d_norm_max: float, dim: int):
    """Return the per-query allowance making pass-one scores admissible.

    Covers FP16 pass-one dot error, pass-one accumulation, exact FP32 dot
    error, and exact-scorer outer accumulation. The per-document `max` needs no
    extra factor because it is 1-Lipschitz.

    Slice-wide norm bounds are used conservatively, and `eps_final` rounds the
    allowance outward so `score + slack` remains an upper bound. The exact-path
    terms make the bound safe against thresholds produced by any supported
    float32 exact scorer, even when its reduction order differs.
    """
    import numpy as np
    import torch

    from nova_bf import closed_form as cf

    dev = q_tokens.device
    n_tok = np.asarray(q_tokens.detach().cpu(), dtype=np.float64)
    if n_tok.size == 0:
        return torch.zeros(0, dtype=torch.float32, device=dev)

    # The dimension enters the error model directly.
    if dim <= 0:
        raise ValueError(f"invalid token dimension: {dim}")
    # These must be finite, non-negative upper bounds on the token norms.
    qn, dn = float(q_norm_max), float(d_norm_max)
    if not np.isfinite(qn) or qn < 0.0:
        raise ValueError(f"invalid query norm bound: {qn}")
    if not np.isfinite(dn) or dn < 0.0:
        raise ValueError(f"invalid document norm bound: {dn}")
    # Counts scale the allowance directly, so invalid counts break
    # admissibility. NaN would slip a bare `< 0` test, and a fractional count
    # would be truncated by `gamma(int(...))` into too small an allowance.
    if (not np.all(np.isfinite(n_tok)) or np.any(n_tok < 0.0)
            or np.any(n_tok != np.floor(n_tok))):
        raise ValueError(
            "query token counts must be finite, non-negative integers")

    n_max = float(n_tok.max())
    prod = qn * dn
    fmt = cf.FP16

    # Include float16's absolute conversion-error floor: the relative-error
    # model alone is invalid for subnormal inputs.
    eps1 = float(cf.eps_dot(
        dim, fmt.u_t, fmt.u_t, np.asarray([qn], dtype=np.float64), dn,
        cf.C_HW_ENVELOPE, False, fmt.eta_t, fmt.eta_t, fmt.lam_t)[0])
    # Bound each pass's float32 accumulation using the magnitude it actually
    # sums. Written out rather than folded together so it mirrors the four
    # claimed terms, and does not quietly depend on `eps1 >= exact_dot_err`.
    exact_dot_err = cf.gamma(int(dim)) * prod
    g_tok = cf.gamma(int(max(n_max, 1.0)))
    per_token = (eps1
                 + exact_dot_err
                 + g_tok * (prod + eps1)
                 + g_tok * (prod + exact_dot_err))

    E = n_tok * per_token
    # Bound |pass-one score| for the outward rounding of `score + slack`.
    B = n_tok * prod * 2.0 + E
    return torch.as_tensor(np.asarray(cf.eps_final(E, B), dtype=np.float32),
                           device=dev)


def _pruned_maxsim_scores(q_flat, c_flat, q_offsets, doc_offsets, thresholds,
                         *, q_half=None, q_norm_max=None,
                         certified: bool = False):
    """Compute MaxSim with provably dead pairs left at `-inf`.

    Internal fast path used by `Fp16State.score`. Cached `q_half`,
    `q_norm_max`, and `certified=True` are trusted caller invariants; stale
    values can weaken admissibility.

    Returns `(scores, dead)`, or `None` when the FP16 pass cannot safely run.
    """
    import torch

    from nova_bf.multivector_kernels import fused_fp16_token_maxima

    if q_flat.ndim != 2 or c_flat.ndim != 2:
        raise ValueError("multivector prune expects 2D (tokens, dim) matrices")
    # The whole error model is "float16 approximate pass against a float32
    # exact path"; another dtype would not be the thing the bound describes.
    if q_flat.dtype != torch.float32 or c_flat.dtype != torch.float32:
        raise TypeError("multivector prune requires float32 tokens")
    if q_flat.device != c_flat.device:
        raise ValueError("multivector prune requires one device")
    # `dim` for the whole bound is taken from the corpus side, while
    # `norm_upper` measures each side in whatever width it was handed.
    if q_flat.shape[1] != c_flat.shape[1]:
        raise ValueError(
            f"multivector prune: query dimension {q_flat.shape[1]} != corpus "
            f"dimension {c_flat.shape[1]}")

    dev = c_flat.device
    q_off = _as_offsets(q_offsets, dev)
    d_off = _as_offsets(doc_offsets, dev)
    if q_off.numel() == 0 or d_off.numel() == 0:
        raise ValueError("multivector prune requires non-empty offsets")
    # Reuse one host copy for validation and survivor scoring.
    d_off_cpu, q_off_cpu = d_off.cpu(), q_off.cpu()
    n_q = int(q_off.numel()) - 1
    n_rows = max(int(d_off.numel()) - 1, 0)

    # `prune_mask` compares against `thresholds[:, None]`, which BROADCASTS a
    # length-1 vector across every query instead of complaining.
    if thresholds.ndim != 1 or thresholds.shape[0] != n_q:
        raise ValueError(
            f"multivector prune: threshold length "
            f"{tuple(thresholds.shape)} != {n_q} queries")
    if thresholds.device != dev:
        raise ValueError("multivector prune: thresholds are on another device")
    # Contents cannot be checked cheaply, but a mismatched cache would score
    # the wrong queries in pass one while `slack` described `q_flat`.
    if q_half is not None and (q_half.shape != q_flat.shape
                               or q_half.dtype != torch.float16
                               or q_half.device != q_flat.device):
        raise ValueError("cached q_half does not match q_flat")

    out = torch.full((n_q, n_rows), float("-inf"), dtype=c_flat.dtype, device=dev)
    if n_rows == 0 or c_flat.shape[0] == 0 or q_flat.shape[0] == 0:
        return out, torch.ones((n_q, n_rows), dtype=torch.bool, device=dev)

    if not certified:
        reason = certify(int(c_flat.shape[1]), dev)
        if reason:
            logger.warning("float16 multivector prune not certified here: %s",
                           reason)
            return None

    # Neither malformed array would fault: the kernel clamps an out-of-range
    # column back to `d0`, and offsets that merely SPAN the right number of
    # rows — shifted, say — would address the wrong ones while every downstream
    # shape check still passed. Checked on the host copies, so this is free.
    _check_partition(q_off_cpu, int(q_flat.shape[0]), "query")
    _check_partition(d_off_cpu, int(c_flat.shape[0]), "document")

    # Reuse the same binary64 norm bounds for conversion checks and slack.
    qn_max = norm_upper(q_flat) if q_norm_max is None else q_norm_max
    d_norm_max = norm_upper(c_flat)

    Qh = q_half if q_half is not None else to_half(q_flat, n_max=qn_max)
    Ch = to_half(c_flat, n_max=d_norm_max)
    if Qh is None or Ch is None:
        return None

    with _mark("mvfp16_pass_one"):
        # float16 GEMM + per-document max without materializing token
        # similarities.
        per_tok = fused_fp16_token_maxima(Qh, Ch, d_off)
        if per_tok is None:
            return None

    with _mark("mvfp16_fold"):
        # MaxSim's outer reduction: sum each query's own token maxima.
        approx = _sum_rows(per_tok, q_off.diff(), n_rows)

    with _mark("mvfp16_bound"):
        s = slack(q_off.diff(), qn_max, d_norm_max, int(c_flat.shape[1]))
        dead = prune_mask(
            upper_bound(approx, s[:, None]), thresholds)

    score_survivors(dead, q_flat, c_flat, q_off, d_off_cpu, out, d_off=d_off)
    return out, dead


class Fp16State:
    """Run-wide state for float16 multivector pruning.

    Caches certification by (token dimension, device) and the query tensor's
    float16 copy and norm bound. Unsupported configurations decline pruning
    rather than affecting results.
    """

    __slots__ = ("_certified_dims", "_failed", "_q_src", "_q_ver", "_q_half",
                 "_q_norm_max", "_warned", "min_prune_rate", "_prune_ema",
                 "_skipped", "_gate_logged")

    # Last-resort re-engage even if no unpruned slice has been observed
    _REPROBE_EVERY = 256

    # Weight on the newest observation. High enough to follow a rate that is
    # still climbing, low enough that one unusual slice does not flip the gate.
    _EMA_ALPHA = 0.3

    def __init__(self, min_prune_rate: float = 0.0):
        rate = float(min_prune_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"min_prune_rate must be in [0, 1], got {rate!r}")
        self._certified_dims: set[tuple] = set()
        self._failed: dict[tuple, str] = {}
        # Keyed by identity AND torch's in-place version counter:
        self._q_src = None
        self._q_ver = None
        self._q_half = None
        self._q_norm_max: float | None = None
        self._warned = False
        # Engage only while the observed prune rate is at or above this
        self.min_prune_rate = rate
        self._prune_ema: float | None = None
        self._skipped = 0
        self._gate_logged = False

    def _ensure_certified(self, dim: int, device) -> bool:
        key = (dim, str(device))
        if key in self._certified_dims:
            return True
        if key in self._failed:
            return False
        reason = certify(dim, device)
        if reason:
            # Remembered per key, not globally: the accumulation probe is
            # dimension-dependent, so another member's tokens may still
            # certify on this same device.
            self._failed[key] = reason
            logger.warning(
                "float16 multivector prune not certified for dim=%d on %s: "
                "%s. Results are unaffected — the unpruned path produces the "
                "same ground truth, more slowly.", dim, device, reason)
            return False
        self._certified_dims.add(key)
        logger.info(
            "float16 multivector prune enabled for dim=%d on %s: pass one "
            "runs the fused float16 GEMM + per-document max in float32 and "
            "bounds the rounding with closed_form.eps_dot; the exact path "
            "still scores every survivor", dim, device)
        return True

    def note_pruned(self, n_pruned: int, n_pairs: int) -> None:
        """Record what fraction of the last slice's pairs were ruled out.

        Fed from counts the caller has ALREADY materialized for its own tally,
        so maintaining the gate costs no extra device->host sync.
        """
        if n_pairs <= 0:
            return
        rate = n_pruned / n_pairs
        self._prune_ema = (rate if self._prune_ema is None
                           else (1.0 - self._EMA_ALPHA) * self._prune_ema
                           + self._EMA_ALPHA * rate)

    def wants_probe(self) -> bool:
        """Whether an unpruned slice should refresh the prune-rate estimate.

        Useful only while the gate is closed; an open gate already updates the
        estimate from its own pruning results. Probe cadence is managed by the
        caller.
        """
        return (self._prune_ema is not None
                and self._prune_ema < self.min_prune_rate)

    def _gate_open(self) -> bool:
        """Whether the recent prune rate justifies running pass one.

        The gate can reopen as the top-k threshold rises. Periodic speculative
        retries are a backstop when no unpruned probe has refreshed the estimate.
        """
        if self._prune_ema is None or self._prune_ema >= self.min_prune_rate:
            self._skipped = 0
            return True
        self._skipped += 1
        if self._skipped >= self._REPROBE_EVERY:
            self._skipped = 0
            return True                      # periodic re-probe
        if not self._gate_logged:
            self._gate_logged = True
            logger.info(
                "float16 multivector prune gated off: pruning %.1f%% of "
                "pairs against a %.1f%% floor, so pass one costs more than it "
                "saves. It re-engages on its own — the rate is re-read from "
                "unpruned slices as the top-K fills. Set "
                "params.multivector_min_prune_rate=0.0 to disable the gate.",
                100.0 * self._prune_ema, 100.0 * self.min_prune_rate)
        return False

    def score(self, q_flat, c_flat, q_offsets, doc_offsets, thresholds):
        """`(scores, dead)` for this slice, or `None` to score it normally."""
        if c_flat.shape[0] == 0 or q_flat.shape[0] == 0:
            return None
        if not self._gate_open():
            return None
        if not self._ensure_certified(int(c_flat.shape[1]), c_flat.device):
            return None

        ver = getattr(q_flat, "_version", None)
        if self._q_src is not q_flat or self._q_ver != ver:
            qn = norm_upper(q_flat)
            qh = to_half(q_flat, n_max=qn)
            if qh is None:
                # Not fatal for the RUN: another query set or slice may be
                # representable. Warn once so it is not silently slow.
                if not self._warned:
                    self._warned = True
                    logger.warning(
                        "float16 cannot represent these query tokens; scoring "
                        "this slice without the prune")
                return None
            # Everything that can fail has already happened, and the keys the
            # hit test reads are assigned LAST — publishing them first would
            # leave a window where a failure above pairs the new key with the
            # PREVIOUS query set's norm.
            self._q_half, self._q_norm_max = qh, qn
            self._q_src, self._q_ver = q_flat, ver

        return _pruned_maxsim_scores(q_flat, c_flat, q_offsets, doc_offsets,
                                     thresholds, q_half=self._q_half,
                                     q_norm_max=self._q_norm_max,
                                     certified=True)


# ---------------------------------------------------------------------------
# Ragged helpers, the bound primitives, and the exact survivor pass. Only
# `slack` above is float16-specific; the rest is pass-one-agnostic.
# ---------------------------------------------------------------------------


def _check_partition(off_cpu, n_tokens: int, what: str) -> None:
    """Reject offsets that are not an exact CSR partition of `n_tokens` rows.

    Stronger than a range check on purpose: a shifted or truncated array can
    span the right total while pointing at the wrong rows, and nothing further
    down would notice.
    """
    if int(off_cpu[0]) != 0 or int(off_cpu[-1]) != n_tokens:
        raise ValueError(
            f"multivector prune: {what} offsets must partition all "
            f"{n_tokens} token rows, got [{int(off_cpu[0])}, "
            f"{int(off_cpu[-1])}]")
    if bool((off_cpu.diff() < 0).any()):
        raise ValueError(
            f"multivector prune: {what} offsets must be non-decreasing")


def _as_offsets(offsets, device):
    """Normalize numpy or torch offsets to a contiguous int64 tensor on `device`.

    `multivector_to_ragged` hands back numpy while the reductions want torch,
    so both arrive here. int64 and contiguity are not cosmetic: the Triton
    kernels require int64 and read offsets as a raw linear buffer, so a strided
    tensor would silently address the wrong documents.

    A float input is rejected rather than truncated — offsets are exact
    positions, and a value that needed rounding was never a valid one.
    """
    import numpy as np
    import torch

    if isinstance(offsets, torch.Tensor):
        if offsets.dtype.is_floating_point or offsets.dtype.is_complex:
            raise TypeError(f"offsets must be integral, got {offsets.dtype}")
        return offsets.to(device=device, dtype=torch.int64).contiguous()
    arr = np.asarray(offsets)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"offsets must be integral, got {arr.dtype}")
    return torch.as_tensor(arr, dtype=torch.int64, device=device).contiguous()


def upper_bound(approx_scores, slack_):
    """`U = approximate score + slack`, the admissible upper bound."""
    return approx_scores + slack_


def prune_mask(U, thresholds):
    """Return pairs provably below their query's threshold.

    Uses strict `<` so ties remain live. NaN bounds also remain live and are
    scored exactly.
    """
    return U < thresholds[:, None]


def _ragged_gather_index(offsets, ids, lengths, total):
    """Flat token rows for the selected queries, concatenated in `ids` order.

    `lengths` and `total` are supplied rather than derived: this runs once per
    document, and both `int(lengths.sum())` and a `repeat_interleave` without
    `output_size` are device->host syncs. The caller already has the totals on
    the host, so the whole loop can run without one.
    """
    import torch

    starts = offsets[ids]
    # Position within each selected query's own token segment.
    seg_start = lengths.cumsum(0) - lengths
    within = (torch.arange(total, device=offsets.device)
              - torch.repeat_interleave(seg_start, lengths, output_size=total))
    return torch.repeat_interleave(starts, lengths, output_size=total) + within


def _mark(name):
    """`record_function` when profiling, free otherwise.
    """
    from torch.profiler import record_function
    return record_function(name)


# Live fraction below which the fused survivor kernel beats the torch gather.
# See the table in `score_survivors`; the measured crossover is ~3% live.
_FUSED_SURVIVOR_MAX_LIVE = 0.03


def score_survivors(dead, q_flat, c_flat, q_off, d_off_cpu, out, *,
                    d_off=None, force_torch=False):
    """Score every surviving (query, document) pair exactly.

    Survivors are enumerated once for the slice and grouped by document. At low
    live fractions, a fused indexed kernel avoids repeated query-token gathers;
    otherwise the PyTorch/cuBLAS path is faster.

    `force_torch` selects the fallback for equivalence testing.
    """
    import torch

    if dead.shape != out.shape:
        raise ValueError(
            f"survivor mask {tuple(dead.shape)} does not match the output "
            f"{tuple(out.shape)}")

    dev = c_flat.device
    n_rows = out.shape[1]
    pairs = (~dead).nonzero()                       # (n_pairs, 2) = (query, doc)
    if pairs.shape[0] == 0:
        return out
    # Group survivors by document, preserving query order within one: the row
    # reads are then sequential rather than scattered.
    pairs = pairs[torch.argsort(pairs[:, 1], stable=True)]

    # The fused kernel removes the torch path's repeated gather, but its
    # float32 `tl.dot` must run `input_precision="ieee"` to match the exact
    # path.
    fused = None
    live_frac = float(pairs.shape[0]) / max(dead.numel(), 1)
    if not force_torch and dev.type == "cuda" and live_frac <= _FUSED_SURVIVOR_MAX_LIVE:
        fused = _score_survivors_fused(pairs, q_flat, c_flat, q_off,
                                       d_off, out, n_rows)
    if fused is not None:
        return fused

    counts = torch.bincount(pairs[:, 1], minlength=n_rows)
    pair_len = q_off[pairs[:, 0] + 1] - q_off[pairs[:, 0]]
    zero = torch.zeros(1, dtype=torch.int64, device=dev)

    # Where each document's survivors start, and how many token rows they span.
    # Both land on the host HERE, in the only two syncs this function performs,
    # so that nothing inside the loop has to ask the device a question.
    bounds = torch.cat([zero, counts.cumsum(0)]).cpu()
    tok_bounds = torch.cat([zero, pair_len.cumsum(0)]).cpu()
    for j in range(n_rows):
        lo, hi = int(bounds[j]), int(bounds[j + 1])
        if hi <= lo:
            continue
        c0, c1 = int(d_off_cpu[j]), int(d_off_cpu[j + 1])
        if c1 <= c0:
            continue
        total = int(tok_bounds[hi]) - int(tok_bounds[lo])
        if total == 0:
            continue
        live = pairs[lo:hi, 0]
        lengths = pair_len[lo:hi]
        with _mark("mvprune_survivor_gather"):
            Qg = q_flat[_ragged_gather_index(q_off, live, lengths, total)]
        with _mark("mvprune_survivor_gemm"):
            m = (Qg @ c_flat[c0:c1].T).max(dim=1).values
        with _mark("mvprune_survivor_fold"):
            seg = torch.repeat_interleave(
                torch.arange(live.numel(), device=dev), lengths,
                output_size=total)
            acc = torch.zeros(live.numel(), dtype=torch.float64, device=dev)
            acc.index_add_(0, seg, m.double())
            # Zero-token queries are non-candidates: their accumulator is still
            # the initial zero, which would outrank every negative score.
            keep = lengths > 0
            out[live[keep], j] = acc[keep].to(out.dtype)
    return out


def _score_survivors_fused(pairs, q_flat, c_flat, q_off, d_off, out, n_rows):
    """Score all survivors with one indexed GEMM launch.

    Builds the required query-token row indices grouped by document and lets the
    kernel read those rows directly instead of materializing a gathered matrix.
    Returns `None` if the fused kernel cannot run.
    """
    import torch

    try:
        from nova_bf.multivector_kernels import fused_indexed_token_maxima
    except ImportError:
        return None
    if d_off is None or q_flat.dtype != torch.float32 \
            or c_flat.dtype != torch.float32:
        return None

    dev = c_flat.device
    with _mark("mvprune_survivor_index"):
        live_q = pairs[:, 0]
        live_d = pairs[:, 1]
        # Tokens each surviving pair contributes, and where they start.
        lengths = (q_off[live_q + 1] - q_off[live_q])
        total = int(lengths.sum())
        if total == 0:
            return out
        seg = torch.repeat_interleave(
            torch.arange(pairs.shape[0], device=dev), lengths,
            output_size=total)
        starts = torch.zeros(pairs.shape[0] + 1, dtype=torch.int64, device=dev)
        starts[1:] = lengths.cumsum(0)
        row_index = q_off[live_q][seg] + (
            torch.arange(total, device=dev) - starts[seg])
        # Query-token rows assigned to each document.
        tokens_per_doc = torch.zeros(n_rows, dtype=torch.int64, device=dev)
        tokens_per_doc.index_add_(0, live_d, lengths)

    with _mark("mvprune_survivor_gemm_fused"):
        per_tok = fused_indexed_token_maxima(
            q_flat, c_flat, row_index, tokens_per_doc, d_off)
    if per_tok is None:
        return None

    with _mark("mvprune_survivor_fold"):
        # Match the Torch path's float64 outer accumulation.
        acc = torch.zeros(pairs.shape[0], dtype=torch.float64, device=dev)
        acc.index_add_(0, seg, per_tok.double())
        # Keep zero-token queries and empty documents at `-inf`.
        has_tokens = lengths > 0
        empty_doc = (d_off[live_d + 1] - d_off[live_d]) == 0
        keep = has_tokens & ~empty_doc
        out[live_q[keep], live_d[keep]] = acc[keep].to(out.dtype)
    return out


def _sum_rows(per_tok, counts, n_rows):
    """Sum per-query-token maxima into per-query scores, in one `index_add_`.

    Vectorized deliberately: the per-query loop this replaces issued ~440,000
    tiny kernel launches per corpus file, enough to make the pruned path slower
    than the exact one it was meant to beat.

    Zero-token queries stay `-inf`: they are non-candidates, and a 0 from an
    empty sum would outrank every negative score.
    """
    import torch

    dev = per_tok.device
    n_seg = counts.shape[0]
    if per_tok.shape[1] != n_rows:
        raise ValueError(
            f"per-token maxima span {per_tok.shape[1]} documents, expected "
            f"{n_rows}")
    # Must partition `per_tok` exactly. A short count vector would otherwise
    # drop token rows, and an all-zero one would return an all-`-inf` slice --
    # every pair silently a non-candidate. Free: the sync is needed anyway.
    total = int(counts.sum())
    if total != per_tok.shape[0]:
        raise ValueError(
            f"query token counts sum to {total}, but there are "
            f"{per_tok.shape[0]} token rows")

    res = torch.zeros((n_seg, n_rows), dtype=per_tok.dtype, device=dev)
    if total:
        seg = torch.repeat_interleave(torch.arange(n_seg, device=dev), counts,
                                      output_size=total)
        res.index_add_(0, seg, per_tok)
    return res.masked_fill((counts == 0)[:, None], float("-inf"))
