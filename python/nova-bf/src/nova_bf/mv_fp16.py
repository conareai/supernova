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

# --- optional exact audit of prune decisions --------------------------------
#
# Compares the FP16 prune against a full unpruned float32 MaxSim of the same
# slice. The audit checks:
#
#   * STRUCTURE: live pairs are finite exactly where the reference is.
#   * BOUND: `upper >= exact` for every graded pair.
#   * DECISION: no pair reaching its threshold was pruned.
#
# Using the unpruned scorer also verifies that the bound is scorer-agnostic.

_AUDIT = {
    "offered": 0, "graded_slices": 0, "graded_pairs": 0,
    "correct_prune": 0, "false_prune": 0, "correct_live": 0, "wasted_live": 0,
    "graded_bounds": 0, "graded_scores": 0,
    "score_mismatch": 0, "worst_score_gap": 0.0,
    "bound_violations": 0, "worst_bound_gap": 0.0, "worst_false_prune": 0.0,
    "min_headroom": None, "disabled": None,
}

_AUDIT_ENV = "NOVA_BF_MV_PRUNE_AUDIT"


def check_audit_env() -> int:
    """Validate the audit sampling rate ONCE, raising on a malformed value.

    Called once at run start. `audit_rate` stays cheap and non-raising because
    it runs per slice, and aborting a run hours in over an environment
    variable would be worse than the typo.
    """
    import os

    raw = os.environ.get(_AUDIT_ENV)
    if raw is None or raw.strip() == "":
        return 0
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(
            f"{_AUDIT_ENV}={raw!r} is not an integer. It asks for one pruned "
            f"(slice, member) in N to be graded; 0 disables it. Refusing "
            f"rather than running unaudited while looking audited.") from None
    if v < 0:
        raise ValueError(
            f"{_AUDIT_ENV}={v} is negative. Use 0 to disable auditing.")
    return v


def audit_rate() -> int:
    """Audit one pruned (slice, member) in N; 0 disables auditing.

    The per-slice reader: deliberately non-raising, because it runs inside the
    scoring loop. `check_audit_env` does the validation once at run start, so
    a malformed value has already been rejected before this is ever reached.
    """
    import os

    try:
        v = int(os.environ.get(_AUDIT_ENV, "0"))
    except ValueError:
        return 0
    return v if v >= 0 else 0


def audit_stats() -> dict:
    """Return cumulative audit counters for this process."""
    return dict(_AUDIT)


def audit_disabled() -> str | None:
    """Return the audit failure that disabled pruning, if any."""
    return _AUDIT_DISABLED


def reset_audit_counters() -> None:
    """Reset audit counters without clearing a prior audit disable."""
    _AUDIT.update({
        k: (
            None if k in ("min_headroom", "disabled")
            else 0.0 if isinstance(v, float)
            else 0
        )
        for k, v in _AUDIT.items()
    })


def reset_audit() -> None:
    """Clear the counters AND any disable the audit imposed."""
    global _AUDIT_DISABLED

    _AUDIT_DISABLED = None
    _AUDIT.update({k: (None if k in ("min_headroom", "disabled") else
                       (0.0 if isinstance(v, float) else 0))
                   for k, v in _AUDIT.items()})


def _audit_should_grade() -> bool:
    """Whether this slice is the one in N that gets graded."""
    rate = audit_rate()
    if not rate:
        return False
    _AUDIT["offered"] += 1
    return (_AUDIT["offered"] - 1) % rate == 0


# Tolerance for "two exact float32 scorers agree". float32 dots and differ
# only in reduction order, so a real disagreement is orders of magnitude 
# larger than this.
_SCORE_RTOL, _SCORE_ATOL = 1e-4, 1e-3

def audit_decisions(exact, dead, thresholds, upper=None, scores=None) -> dict:
    """Audit one slice against unpruned float32 MaxSim.

    Checks prune decisions, bound admissibility, and survivor scores. Equality
    with the threshold remains live. Any violation disables further pruning.
    """
    import torch

    global _AUDIT_DISABLED

    if exact.shape != dead.shape:
        raise ValueError(
            f"audit: exact scores {tuple(exact.shape)} do not match the prune "
            f"mask {tuple(dead.shape)}")
    if thresholds.ndim != 1 or thresholds.shape[0] != exact.shape[0]:
        raise ValueError(
            f"audit: threshold length {tuple(thresholds.shape)} != "
            f"{exact.shape[0]} queries")

    if upper is not None and upper.shape != exact.shape:
        raise ValueError(
            f"audit: bound {tuple(upper.shape)} does not match the exact "
            f"scores {tuple(exact.shape)}")
    if scores is not None and scores.shape != exact.shape:
        raise ValueError(
            f"audit: kept scores {tuple(scores.shape)} do not match the exact "
            f"scores {tuple(exact.shape)}")

    thr = thresholds[:, None]

    # Decisions require finite exact scores and thresholds.
    ok = torch.isfinite(exact) & torch.isfinite(thr)

    # Audit every bound, including NaN and infinities.
    ok_bound = (
        torch.ones_like(exact, dtype=torch.bool)
        if upper is not None
        else torch.zeros_like(exact, dtype=torch.bool)
    )

    # Survivor structure is meaningful even for non-candidates such as `-inf`.
    gradable = bool(ok.any() or ok_bound.any())
    if scores is not None:
        gradable = gradable or bool((~dead).any())
    if not gradable:
        return {}

    should_live = ok & (exact >= thr)
    live = ~dead

    correct_prune = int((ok & dead & ~should_live).sum())
    false_prune = ok & dead & should_live
    n_false = int(false_prune.sum())
    correct_live = int((ok & live & should_live).sum())
    wasted_live = int((ok & live & ~should_live).sum())
    n_graded = int(ok.sum())

    _AUDIT["graded_slices"] += 1
    _AUDIT["graded_pairs"] += n_graded
    _AUDIT["graded_bounds"] += int(ok_bound.sum()) if upper is not None else 0
    _AUDIT["correct_prune"] += correct_prune
    _AUDIT["correct_live"] += correct_live
    _AUDIT["wasted_live"] += wasted_live

    n_viol, worst_gap = 0, 0.0
    if upper is not None and bool(ok_bound.any()):
        # `~(upper >= exact)` also catches NaN and -inf bounds.
        violated = ok_bound & ~(upper >= exact)
        n_viol = int(violated.sum())

        # Headroom is meaningful only for finite pairs.
        fin_pair = torch.isfinite(exact) & torch.isfinite(upper)
        headroom = (
            float((upper - exact)[fin_pair].min())
            if bool(fin_pair.any())
            else None
        )
        if headroom is not None:
            _AUDIT["min_headroom"] = (
                headroom if _AUDIT["min_headroom"] is None
                else min(_AUDIT["min_headroom"], headroom)
            )

        if n_viol:
            gaps = (exact - upper)[violated]
            gaps = gaps[torch.isfinite(gaps)]
            worst_gap = (
                float(gaps.max()) if gaps.numel() else float("inf")
            )
            _AUDIT["bound_violations"] += n_viol
            _AUDIT["worst_bound_gap"] = max(
                _AUDIT["worst_bound_gap"], worst_gap
            )

    # Compare every survivor against the independent scorer.
    n_bad_score, worst_score = 0, 0.0
    if scores is not None:
        live = ~dead
        if bool(live.any()):
            fin_e = torch.isfinite(exact)
            fin_s = torch.isfinite(scores)

            # Live pairs must match the reference category exactly.
            bad = live & (torch.isnan(exact) | torch.isnan(scores))
            bad = bad | (
                live
                & (torch.isneginf(exact) != torch.isneginf(scores))
            )
            bad = bad | (
                live
                & (torch.isposinf(exact) != torch.isposinf(scores))
            )
            bad = bad | (live & (fin_e != fin_s))

            # Compare values only where both sides are finite.
            both = live & fin_e & fin_s
            diff = (scores - exact).abs()
            bad = bad | (
                both
                & (
                    diff
                    > _SCORE_ATOL + _SCORE_RTOL * exact.abs()
                )
            )

            n_bad_score = int(bad.sum())
            _AUDIT["graded_scores"] += int(live.sum())

            if n_bad_score:
                gaps = diff[bad & torch.isfinite(diff)]
                worst_score = (
                    float(gaps.max()) if gaps.numel() else float("inf")
                )
                _AUDIT["score_mismatch"] += n_bad_score
                _AUDIT["worst_score_gap"] = max(
                    _AUDIT["worst_score_gap"], worst_score
                )

    if n_false:
        worst = float(
            (exact - thr.expand_as(exact))[false_prune].max()
        )
        _AUDIT["false_prune"] += n_false
        _AUDIT["worst_false_prune"] = max(
            _AUDIT["worst_false_prune"], worst
        )

    if n_false or n_viol or n_bad_score:
        reason = (
            f"audit found {n_viol} pairs whose bound fell BELOW their "
            f"exact score (by up to {worst_gap:.3e}), {n_false} "
            f"pruned pairs that reach their threshold, and "
            f"{n_bad_score} SURVIVORS scored wrong (by up to "
            f"{worst_score:.3e}), out of {n_graded} graded"
        )
        _AUDIT["disabled"] = reason
        _AUDIT_DISABLED = reason
        logger.error(
            "float16 multivector prune AUDIT FAILURE: %s. Pruning is disabled "
            "for the rest of this process; treat this run's output as "
            "incomplete.",
            reason,
        )

    return {
        "graded": n_graded,
        "correct_prune": correct_prune,
        "false_prune": n_false,
        "correct_live": correct_live,
        "wasted_live": wasted_live,
        "bound_violations": n_viol,
        "score_mismatch": n_bad_score,
    }

# Set by a failed audit; checked by `Fp16State.score`.
_AUDIT_DISABLED: str | None = None


# How the throughput gate behaved, for the end-of-run line.
_GATE = {"open": 0, "closed": 0, "latched": 0, "unlatched": 0}


def gate_stats() -> dict:
    """Cumulative gate decisions for this process."""
    return dict(_GATE)


def reset_gate_stats() -> None:
    _GATE.update({k: 0 for k in _GATE})


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


# Binary64 working-set budget for `norm_upper`'s chunking. Small enough that
# the widened copy is never the allocation that fails, large enough that the
# per-chunk device->host `max` is not the cost.
_NORM_CHUNK_BYTES = 64 << 20


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
    # Chunked, because `x.double()` on the whole matrix transiently doubles
    # its footprint
    dim = int(x.shape[1])
    per_chunk = max(1, _NORM_CHUNK_BYTES // (dim * 8))
    # Seeded at -inf and compared with an explicit NaN check.
    n = float("-inf")
    for i in range(0, int(x.shape[0]), per_chunk):
        v = float(x[i:i + per_chunk].double().norm(dim=1).max())
        if v != v:
            return v                        # NaN propagates; callers refuse
        n = max(n, v)
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
                         certified: bool = False, want_upper: bool = False):
    """Compute MaxSim with provably dead pairs left at `-inf`.

    Internal fast path used by `Fp16State.score`. Cached `q_half`,
    `q_norm_max`, and `certified=True` are trusted caller invariants; stale
    values can weaken admissibility.

    Returns `(scores, dead, upper)`, or `None` when the FP16 pass cannot
    safely run. `upper` is the bound each pair was pruned with, returned only
    for `want_upper=True` (the audit) and `None` otherwise, so an ordinary
    slice does not keep a second score-sized tensor alive.
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
        return out, torch.ones((n_q, n_rows), dtype=torch.bool, device=dev), None

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
        up = upper_bound(approx, s[:, None])
        dead = prune_mask(up, thresholds)

    score_survivors(dead, q_flat, c_flat, q_off, d_off_cpu, out)
    return out, dead, (up if want_upper else None)


class Fp16State:
    """Run-wide state for float16 multivector pruning.

    Caches certification by (dimension, device) and query float16/norm state.
    Unsupported configurations fall back to unpruned scoring.
    """

    __slots__ = (
        "_certified_dims", "_failed", "_q_cache", "_warned",
        "min_prune_rate", "_latched", "_good", "_bad",
    )

    # Keep both raw and normalized query representations without unbounded
    # GPU-memory growth.
    _Q_CACHE_MAX = 2

    # Engage quickly when pruning appears useful; require more evidence to
    # disengage once the achieved rate falls below the configured floor.
    _LATCH_AFTER = 3
    _UNLATCH_AFTER = 10

    def __init__(self, min_prune_rate: float = 0.0):
        rate = float(min_prune_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"min_prune_rate must be in [0, 1], got {rate!r}")

        self._certified_dims: set[tuple] = set()
        self._failed: dict[tuple, str] = {}

        # Cache by tensor identity and in-place version to detect replacement or mutation.
        self._q_cache: list = []
        self._warned = False

        # Prune-rate floor used by the throughput gate.
        self.min_prune_rate = rate

        # A zero floor means pruning should start enabled.
        self._latched = rate <= 0.0
        self._good = 0
        self._bad = 0

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

    def note_probe(self, n_pruned: int, n_pairs: int) -> None:
        """Record estimated prune effectiveness from an unpruned slice.

        This is optimistic: exact scores below threshold are potentially prunable,
        while the actual bound may remove only a subset.
        """
        if n_pairs <= 0 or self._latched:
            return

        if n_pruned / n_pairs >= self.min_prune_rate:
            self._good += 1
            if self._good >= self._LATCH_AFTER:
                self._latched = True
                self._bad = 0
                _GATE["latched"] += 1
                logger.info(
                    "float16 multivector prune engaged: %d consecutive probes "
                    "found at least %.0f%% of pairs prunable",
                    self._good, 100.0 * self.min_prune_rate)
        else:
            self._good = 0
    
    def note_pruned(self, n_pruned: int, n_pairs: int) -> None:
        """Record the prune rate actually achieved by a pruned slice.

        Disengages after repeated slices fall below the configured prune-rate floor.
        """
        if n_pairs <= 0 or not self._latched:
            return

        if n_pruned / n_pairs >= self.min_prune_rate:
            self._bad = 0
            return

        self._bad += 1
        if self._bad >= self._UNLATCH_AFTER:
            self._latched = False
            self._good = 0
            _GATE["unlatched"] += 1
            logger.info(
                "float16 multivector prune disengaged: %d consecutive slices "
                "pruned less than %.0f%% of pairs, so pass one is costing more "
                "than it saves. Set params.multivector_min_prune_rate=0.0 to "
                "keep it on regardless.",
                self._bad, 100.0 * self.min_prune_rate)


    def wants_probe(self) -> bool:
        """Whether an unpruned slice should refresh the gate estimate."""
        return not self._latched


    def gate_open(self) -> bool:
        """Whether to run the pruning pass on this slice."""
        if self._latched:
            _GATE["open"] += 1
            return True

        _GATE["closed"] += 1
        return False


    def score(self, q_flat, c_flat, q_offsets, doc_offsets, thresholds):
        """Return `(scores, dead, upper)`, or `None` to use unpruned scoring.

        `upper` is returned only when this slice is selected for auditing.
        """
        if _AUDIT_DISABLED is not None:
            return None
        if c_flat.shape[0] == 0 or q_flat.shape[0] == 0:
            return None
        if not self._ensure_certified(int(c_flat.shape[1]), c_flat.device):
            return None

        ver = getattr(q_flat, "_version", None)
        hit = next(
            (e for e in self._q_cache if e[0] is q_flat and e[1] == ver),
            None,
        )

        if hit is None:
            qn = norm_upper(q_flat)
            qh = to_half(q_flat, n_max=qn)

            if qh is None:
                # Fall back safely if this query representation cannot use FP16.
                if not self._warned:
                    self._warned = True
                    logger.warning(
                        "float16 cannot represent these query tokens; scoring "
                        "this slice without the prune"
                    )
                return None

            # Publish the cache entry only after all derived state is valid.
            hit = (q_flat, ver, qh, qn)
            self._q_cache.append(hit)
            del self._q_cache[:-self._Q_CACHE_MAX]

        return _pruned_maxsim_scores(
            q_flat,
            c_flat,
            q_offsets,
            doc_offsets,
            thresholds,
            q_half=hit[2],
            q_norm_max=hit[3],
            certified=True,
            want_upper=_audit_should_grade(),
        )

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


def score_survivors(dead, q_flat, c_flat, q_off, d_off_cpu, out):
    """Score every surviving (query, document) pair exactly.

    Survivors are grouped by document; per-token maxima are written to one
    buffer and folded into MaxSim scores once at the end.
    """
    import torch

    if dead.shape != out.shape:
        raise ValueError(
            f"survivor mask {tuple(dead.shape)} does not match the output "
            f"{tuple(out.shape)}")

    dev = c_flat.device
    n_rows = out.shape[1]
    pairs = (~dead).nonzero()  # (query, document)
    if pairs.shape[0] == 0:
        return out

    # Group by document while preserving query order within each document.
    pairs = pairs[torch.argsort(pairs[:, 1], stable=True)]

    counts = torch.bincount(pairs[:, 1], minlength=n_rows)
    pair_len = q_off[pairs[:, 0] + 1] - q_off[pairs[:, 0]]
    d_tok = torch.as_tensor(d_off_cpu).diff().to(dev)
    zero = torch.zeros(1, dtype=torch.int64, device=dev)

    # Materialize loop bounds once so the per-document loop does not sync.
    bounds = torch.cat([zero, counts.cumsum(0)]).cpu()
    tok_bounds = torch.cat([zero, pair_len.cumsum(0)]).cpu()

    # Store all survivor token maxima in one buffer; fold them once afterward.
    n_tok_total = int(tok_bounds[-1])
    flat_max = torch.zeros(n_tok_total, dtype=torch.float32, device=dev)

    for j in range(n_rows):
        lo, hi = int(bounds[j]), int(bounds[j + 1])
        if hi <= lo:
            continue

        c0, c1 = int(d_off_cpu[j]), int(d_off_cpu[j + 1])
        if c1 <= c0:
            continue

        t_lo, t_hi = int(tok_bounds[lo]), int(tok_bounds[hi])
        total = t_hi - t_lo
        if total == 0:
            continue

        live = pairs[lo:hi, 0]
        lengths = pair_len[lo:hi]

        with _mark("mvprune_survivor_gather"):
            Qg = q_flat[
                _ragged_gather_index(q_off, live, lengths, total)
            ]

        with _mark("mvprune_survivor_gemm"):
            torch.amax(
                Qg @ c_flat[c0:c1].T,
                dim=1,
                out=flat_max[t_lo:t_hi],
            )

    with _mark("mvprune_survivor_fold"):
        # int32 saves memory; pair IDs must fit in int32.
        seg = torch.repeat_interleave(
            torch.arange(
                pairs.shape[0], dtype=torch.int32, device=dev
            ),
            pair_len.to(torch.int32),
            output_size=n_tok_total,
        )

        acc = torch.zeros(
            pairs.shape[0], dtype=torch.float64, device=dev
        )
        acc.index_add_(0, seg, flat_max.double())

        # Zero-token queries and empty documents remain non-candidates.
        keep = (pair_len > 0) & (d_tok[pairs[:, 1]] > 0)
        out[pairs[keep, 0], pairs[keep, 1]] = acc[keep].to(out.dtype)

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
