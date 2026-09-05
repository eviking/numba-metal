"""Correctness verification: CPU vs Metal results must actually agree
before any "use Metal" recommendation can be issued.

Deliberately its own module, separate from comparison.py (timing) --
mixing "is it faster" and "is it correct" into one result would make it
too easy for a caller to act on speed alone, which the governing spec
explicitly forbids: "If correctness fails, display the performance
measurement but do not recommend conversion."

`recommendations.py` enforces this structurally by refusing to emit a
USE_METAL direction without a passing CorrectnessResult attached (see
that module) -- this module only ever produces the CorrectnessResult
itself; it makes no recommendation decisions.
"""

from __future__ import annotations

import numpy as np

from numba_metal.advisor.models import CorrectnessResult

#: Above this many elements, comparison is sampled rather than exhaustive
#: (still clearly reported via CorrectnessResult.sampled) -- keeps a
#: correctness check on a huge output from itself becoming the dominant
#: cost of running `numba-metal advisor compare`.
DEFAULT_FULL_COMPARISON_LIMIT = 5_000_000


def compare_results(
    cpu_result: np.ndarray,
    metal_result: np.ndarray,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    full_comparison_limit: int = DEFAULT_FULL_COMPARISON_LIMIT,
    seed: int | None = None,
) -> CorrectnessResult:
    """Compare a CPU reference array against a Metal result array.

    Integer and boolean dtypes are compared for EXACT equality (no
    tolerance -- a Metal integer/count result that differs at all from
    the CPU reference is a real bug, not floating-point noise).
    Floating-point dtypes use `atol`/`rtol` (matching
    `benchmarks/common.py`'s `assert_allclose` convention, extended here
    to also report the actual max error, not just pass/fail).
    """
    shape_match = cpu_result.shape == metal_result.shape
    dtype_match = cpu_result.dtype == metal_result.dtype

    if not shape_match:
        return CorrectnessResult(
            passed=False,
            elements_compared=0,
            elements_total=int(cpu_result.size),
            sampled=False,
            max_abs_error=None,
            max_rel_error=None,
            atol=atol,
            rtol=rtol,
            cpu_dtype=str(cpu_result.dtype),
            metal_dtype=str(metal_result.dtype),
            shape_match=False,
            dtype_match=dtype_match,
            nan_mismatch=False,
            inf_mismatch=False,
            failure_reason=(
                f"Shape mismatch: CPU result is {cpu_result.shape}, "
                f"Metal result is {metal_result.shape}"
            ),
        )

    total = int(cpu_result.size)
    if total == 0:
        return CorrectnessResult(
            passed=True,
            elements_compared=0,
            elements_total=0,
            sampled=False,
            max_abs_error=0.0,
            max_rel_error=0.0,
            atol=atol,
            rtol=rtol,
            cpu_dtype=str(cpu_result.dtype),
            metal_dtype=str(metal_result.dtype),
            shape_match=True,
            dtype_match=dtype_match,
            nan_mismatch=False,
            inf_mismatch=False,
            failure_reason=None,
        )

    sampled = total > full_comparison_limit
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=full_comparison_limit, replace=False)
        cpu_flat = cpu_result.reshape(-1)[idx]
        metal_flat = metal_result.reshape(-1)[idx]
        elements_compared = full_comparison_limit
    else:
        cpu_flat = cpu_result.reshape(-1)
        metal_flat = metal_result.reshape(-1)
        elements_compared = total

    is_integer_like = np.issubdtype(cpu_flat.dtype, np.integer) or np.issubdtype(
        cpu_flat.dtype, np.bool_
    )

    cpu_nan = np.isnan(cpu_flat) if np.issubdtype(cpu_flat.dtype, np.floating) else None
    metal_nan = (
        np.isnan(metal_flat) if np.issubdtype(metal_flat.dtype, np.floating) else None
    )
    nan_mismatch = False
    if cpu_nan is not None and metal_nan is not None:
        nan_mismatch = bool(np.any(cpu_nan != metal_nan))

    cpu_inf = np.isinf(cpu_flat) if np.issubdtype(cpu_flat.dtype, np.floating) else None
    metal_inf = (
        np.isinf(metal_flat) if np.issubdtype(metal_flat.dtype, np.floating) else None
    )
    inf_mismatch = False
    if cpu_inf is not None and metal_inf is not None:
        # Compare only where both are finite-vs-infinite in agreement;
        # sign of infinity must also match (np.isinf alone conflates +/-inf).
        inf_mismatch = bool(np.any(cpu_inf != metal_inf)) or bool(
            np.any(
                np.sign(cpu_flat[cpu_inf & metal_inf])
                != np.sign(metal_flat[cpu_inf & metal_inf])
            )
        )

    if is_integer_like:
        exact_match = np.array_equal(cpu_flat, metal_flat)
        passed = exact_match and not nan_mismatch and not inf_mismatch
        return CorrectnessResult(
            passed=passed,
            elements_compared=elements_compared,
            elements_total=total,
            sampled=sampled,
            max_abs_error=(
                None
                if exact_match
                else float(
                    np.max(
                        np.abs(cpu_flat.astype(np.int64) - metal_flat.astype(np.int64))
                    )
                )
            ),
            max_rel_error=None,
            atol=None,
            rtol=None,
            cpu_dtype=str(cpu_result.dtype),
            metal_dtype=str(metal_result.dtype),
            shape_match=True,
            dtype_match=dtype_match,
            nan_mismatch=nan_mismatch,
            inf_mismatch=inf_mismatch,
            failure_reason=(
                None
                if passed
                else "Integer/boolean result differs from the CPU reference "
                "(exact equality required for this dtype)"
            ),
        )

    # Floating-point: exclude NaN positions from the error computation
    # (NaN handling is reported separately via nan_mismatch), and exclude
    # positions where either side is infinite (handled via inf_mismatch).
    valid_mask = np.ones(cpu_flat.shape, dtype=bool)
    if cpu_nan is not None:
        valid_mask &= ~cpu_nan & ~metal_nan
    if cpu_inf is not None:
        valid_mask &= ~cpu_inf & ~metal_inf

    if np.any(valid_mask):
        cpu_valid = cpu_flat[valid_mask].astype(np.float64)
        metal_valid = metal_flat[valid_mask].astype(np.float64)
        abs_err = np.abs(cpu_valid - metal_valid)
        max_abs_error = float(np.max(abs_err))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(
                cpu_valid != 0,
                abs_err / np.abs(cpu_valid),
                np.where(abs_err == 0, 0, np.inf),
            )
        max_rel_error = float(np.max(rel_err))
        within_tolerance = bool(np.all(abs_err <= atol + rtol * np.abs(cpu_valid)))
    else:
        max_abs_error = 0.0
        max_rel_error = 0.0
        within_tolerance = True

    passed = within_tolerance and not nan_mismatch and not inf_mismatch
    failure_reason = None
    if not passed:
        reasons = []
        if not within_tolerance:
            reasons.append(
                f"max abs error {max_abs_error:.6g} / max rel error "
                f"{max_rel_error:.6g} exceeds atol={atol:.3g}, rtol={rtol:.3g}"
            )
        if nan_mismatch:
            reasons.append("NaN positions differ between CPU and Metal results")
        if inf_mismatch:
            reasons.append(
                "Infinity positions/signs differ between CPU and Metal results"
            )
        failure_reason = "; ".join(reasons)

    return CorrectnessResult(
        passed=passed,
        elements_compared=elements_compared,
        elements_total=total,
        sampled=sampled,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        atol=atol,
        rtol=rtol,
        cpu_dtype=str(cpu_result.dtype),
        metal_dtype=str(metal_result.dtype),
        shape_match=True,
        dtype_match=dtype_match,
        nan_mismatch=nan_mismatch,
        inf_mismatch=inf_mismatch,
        failure_reason=failure_reason,
    )
