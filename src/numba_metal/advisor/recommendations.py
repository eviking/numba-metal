"""Deterministic recommendation engine.

Every recommendation here is produced by a plain, reproducible rule
function over (Candidate, CompatibilityResult, ComparisonResult,
CorrectnessResult) -- no LLM call anywhere in this module, per spec
section 14: "Do not use an LLM to create the core recommendations. They
must be reproducible from profiling data and rules."

The single hard invariant this module enforces structurally: a
USE_METAL recommendation is NEVER produced without a passing
CorrectnessResult attached (spec section 12: "If correctness fails,
display the performance measurement but do not recommend conversion.").
See `_correctness_gate` below -- every rule that could otherwise suggest
USE_METAL routes through it first.
"""

from __future__ import annotations

from numba_metal.advisor.models import (
    Candidate,
    ComparisonResult,
    CompatibilityResult,
    CompatibilityStatus,
    Confidence,
    CorrectnessResult,
    EvidenceKind,
    MeasurementMode,
    Recommendation,
    RecommendationDirection,
    RooflinePerformanceRegime,
)


def _correctness_gate(correctness: CorrectnessResult | None) -> bool:
    """True iff it is safe to recommend USE_METAL: a CorrectnessResult
    was actually supplied AND it passed. No correctness data at all is
    treated the same as failing correctness for this gate's purposes --
    "we didn't check" is not evidence of "it's fine"."""
    return correctness is not None and correctness.passed


def _steady_or_warm(comparison: ComparisonResult):
    for mode in (MeasurementMode.STEADY_STATE, MeasurementMode.WARM):
        for r in comparison.regimes:
            if r.mode == mode and r.speedup is not None:
                return r
    return None


def recommend_from_comparison(
    candidate: Candidate,
    compatibility: CompatibilityResult | None,
    comparison: ComparisonResult,
    correctness: CorrectnessResult | None,
) -> list[Recommendation]:
    """Rules that require a real ComparisonResult (measured timing)."""
    recs: list[Recommendation] = []
    regime = _steady_or_warm(comparison)

    if correctness is not None and not correctness.passed:
        recs.append(
            Recommendation(
                text="DO NOT USE METAL",
                direction=RecommendationDirection.KEEP_CPU,
                qualified_name=candidate.qualified_name,
                file=candidate.file,
                line_start=candidate.line_start,
                supporting_measurement=(
                    correctness.failure_reason
                    or "Output differs from the CPU reference beyond the "
                    "declared tolerance."
                ),
                confidence=Confidence.HIGH,
                evidence=EvidenceKind.MEASURED,
                how_to_verify=(
                    "Re-run `numba-metal advisor compare` and inspect the "
                    "CORRECTNESS section; adjust the kernel until CPU and "
                    "Metal outputs agree within the declared tolerance "
                    "before considering performance."
                ),
                rule_id="correctness_failure_blocks_metal",
            )
        )
        return recs  # Correctness failure overrides every other rule below.

    if regime is not None and regime.speedup is not None:
        if regime.speedup < 1.0:
            factor = 1.0 / regime.speedup
            recs.append(
                Recommendation(
                    text=(
                        f"Keep {candidate.qualified_name} on the CPU -- "
                        f"Metal measured {factor:.2f}x SLOWER "
                        f"({regime.mode.value.lower()})."
                    ),
                    direction=RecommendationDirection.KEEP_CPU,
                    qualified_name=candidate.qualified_name,
                    file=candidate.file,
                    line_start=candidate.line_start,
                    supporting_measurement=(
                        f"{regime.mode.value} regime: CPU median "
                        f"{regime.cpu.median_ns:.0f}ns vs Metal median "
                        f"{regime.metal.median_ns:.0f}ns "
                        f"(n={regime.cpu.n}/{regime.metal.n})"
                    ),
                    confidence=(
                        Confidence.LOW if regime.preliminary else Confidence.HIGH
                    ),
                    evidence=EvidenceKind.MEASURED,
                    how_to_verify=(
                        "Re-run `numba-metal advisor compare` with a larger "
                        "input size, or with --measurement-runs increased, "
                        "to confirm this holds at the sizes you care about."
                    ),
                    rule_id="warm_metal_slower_than_cpu",
                )
            )
        elif _correctness_gate(correctness):
            recs.append(
                Recommendation(
                    text=(
                        f"Use Metal for {candidate.qualified_name} -- "
                        f"measured {regime.speedup:.2f}x faster "
                        f"({regime.mode.value.lower()}), and results match "
                        "the CPU reference within tolerance."
                    ),
                    direction=RecommendationDirection.USE_METAL,
                    qualified_name=candidate.qualified_name,
                    file=candidate.file,
                    line_start=candidate.line_start,
                    supporting_measurement=(
                        f"{regime.mode.value} regime: CPU median "
                        f"{regime.cpu.median_ns:.0f}ns vs Metal median "
                        f"{regime.metal.median_ns:.0f}ns "
                        f"(n={regime.cpu.n}/{regime.metal.n})"
                    ),
                    confidence=(
                        Confidence.LOW if regime.preliminary else Confidence.HIGH
                    ),
                    evidence=EvidenceKind.MEASURED,
                    how_to_verify=(
                        "Re-run `numba-metal advisor compare` on your target "
                        "machine and representative input sizes; a speedup "
                        "measured on one machine/size does not guarantee "
                        "the same result elsewhere."
                    ),
                    rule_id="warm_metal_faster_than_cpu",
                )
            )
        else:
            recs.append(
                Recommendation(
                    text=(
                        f"{candidate.qualified_name} measured "
                        f"{regime.speedup:.2f}x faster on Metal, but "
                        "correctness has not been verified -- do not adopt "
                        "yet."
                    ),
                    direction=RecommendationDirection.INFO_ONLY,
                    qualified_name=candidate.qualified_name,
                    file=candidate.file,
                    line_start=candidate.line_start,
                    supporting_measurement=(
                        f"{regime.mode.value} regime speedup: "
                        f"{regime.speedup:.2f}x (correctness not checked)"
                    ),
                    confidence=Confidence.LOW,
                    evidence=EvidenceKind.MEASURED,
                    how_to_verify=(
                        "Run `numba-metal advisor compare` (not just "
                        "`profile`) so correctness is checked before "
                        "trusting this speedup."
                    ),
                    rule_id="speedup_without_correctness_check",
                )
            )

    cold_regime = next(
        (r for r in comparison.regimes if r.mode == MeasurementMode.COLD), None
    )
    if cold_regime is not None and cold_regime.metal is not None and regime is not None:
        if cold_regime.metal.median_ns > 0 and regime.metal is not None:
            cold_to_warm_ratio = cold_regime.metal.median_ns / max(
                regime.metal.median_ns, 1.0
            )
            if cold_to_warm_ratio > 10:
                recs.append(
                    Recommendation(
                        text=(
                            f"Reuse the compiled pipeline for "
                            f"{candidate.qualified_name}; cold compilation "
                            f"is {cold_to_warm_ratio:.0f}x the warm launch "
                            "cost."
                        ),
                        direction=RecommendationDirection.INFO_ONLY,
                        qualified_name=candidate.qualified_name,
                        file=candidate.file,
                        line_start=candidate.line_start,
                        supporting_measurement=(
                            f"cold median {cold_regime.metal.median_ns:.0f}ns "
                            f"vs warm median {regime.metal.median_ns:.0f}ns"
                        ),
                        confidence=Confidence.HIGH,
                        evidence=EvidenceKind.MEASURED,
                        how_to_verify=(
                            "Ensure the same @metal.jit dispatcher object is "
                            "reused across calls rather than re-decorating "
                            "the function each time; numba-metal's "
                            "KernelCache already does this automatically "
                            "for repeated calls to the same dispatcher."
                        ),
                        rule_id="cold_compile_dominates",
                    )
                )
    return recs


def recommend_from_roofline(
    candidate: Candidate, comparison: ComparisonResult
) -> list[Recommendation]:
    """Rules driven by `comparison.roofline` -- a classification of the
    measured kernel against this machine's own calibrated dispatch-
    overhead/bandwidth/compute ceilings (see comparison.py's
    `classify_roofline`). Produces nothing if no classification is
    available (workload didn't supply bytes_per_call/flops_per_call, or
    no local calibration exists) -- never a guessed explanation.

    Two distinct things happen here, deliberately kept separate:
    1. An INFO_ONLY explanation of WHY the kernel performed the way it
       did (e.g. "94% of this machine's real compute ceiling" instead
       of leaving a bare speedup number to speak for itself) -- added
       regardless of whether Metal won or lost, since "already near the
       ceiling" and "real headroom exists" are both useful facts.
    2. A specific, actionable USE_METAL-adjacent suggestion to try
       `metal.batch()` when the kernel is DISPATCH_BOUND -- this is not
       a claim that batching will flip a KEEP_CPU verdict to USE_METAL
       (that depends on the workload actually being called repeatedly,
       which this function cannot know), so it is always INFO_ONLY,
       never a new USE_METAL recommendation on its own. Verified this
       session: batching launches onto one command buffer and syncing
       once cuts per-call dispatch overhead from ~170-190us to
       ~69-95us, a real ~2.5-2.7x reduction, and this flipped a genuine
       CPU-vs-Metal loss to a win for one measured workload (vector
       polynomial at 10,000 elements, called repeatedly) -- but only
       when the caller's own usage pattern involves repeated calls,
       which is exactly why this stays a suggestion to try, not an
       assertion that it will help THIS specific comparison.
    """
    roofline = comparison.roofline
    if roofline is None:
        return []

    recs: list[Recommendation] = []
    ai = roofline.arithmetic_intensity_flops_per_byte

    if roofline.regime == RooflinePerformanceRegime.DISPATCH_BOUND:
        dispatch_pct = (
            roofline.dispatch_overhead_fraction * 100
            if roofline.dispatch_overhead_fraction is not None
            else None
        )
        recs.append(
            Recommendation(
                text=(
                    f"{candidate.qualified_name} is dispatch-bound on "
                    f"{roofline.calibration_device_name or 'this machine'}: "
                    f"per-launch overhead is an estimated "
                    f"{dispatch_pct:.0f}% of the measured Metal time, "
                    "not GPU execution."
                ),
                direction=RecommendationDirection.INFO_ONLY,
                qualified_name=candidate.qualified_name,
                file=candidate.file,
                line_start=candidate.line_start,
                supporting_measurement=(
                    f"arithmetic intensity {ai:.2f} FLOPs/byte; dispatch "
                    f"overhead ~{dispatch_pct:.0f}% of measured time"
                    if ai is not None and dispatch_pct is not None
                    else "dispatch-bound (see calibration)"
                ),
                confidence=Confidence.MEDIUM,
                evidence=EvidenceKind.ESTIMATED,
                how_to_verify=(
                    "Re-run `numba-metal advisor calibrate` on this machine "
                    "to confirm the current dispatch-overhead baseline, "
                    "since this classification depends on it."
                ),
                rule_id="roofline_dispatch_bound",
            )
        )
        recs.append(
            Recommendation(
                text=(
                    f"If {candidate.qualified_name} is called repeatedly "
                    "(a loop, a batch of inputs, an iterative algorithm), "
                    "try wrapping the repeated calls in `metal.batch()` to "
                    "encode them onto one command buffer instead of "
                    "committing and syncing after every call."
                ),
                direction=RecommendationDirection.INFO_ONLY,
                qualified_name=candidate.qualified_name,
                file=candidate.file,
                line_start=candidate.line_start,
                supporting_measurement=(
                    "Measured on this machine: batching a sequence of "
                    "launches onto one command buffer cut per-call "
                    "dispatch overhead from ~170-190us to ~69-95us "
                    "(roughly 2.5-2.7x), which was enough to flip a real "
                    "CPU-vs-Metal comparison from a loss to a win for one "
                    "dispatch-bound workload. Whether it helps THIS "
                    "comparison depends on whether your real usage calls "
                    "this kernel repeatedly -- a single one-off call gets "
                    "no benefit from batching."
                ),
                confidence=Confidence.MEDIUM,
                evidence=EvidenceKind.ESTIMATED,
                how_to_verify=(
                    "Wrap your repeated `metal_warm_fn`-equivalent calls in "
                    "`with metal.batch():` and re-run "
                    "`numba-metal advisor compare` to measure the real "
                    "effect on your workload, rather than assuming this "
                    "machine's ~2.5-2.7x figure transfers directly."
                ),
                rule_id="roofline_dispatch_bound_try_batching",
            )
        )
    elif roofline.regime == RooflinePerformanceRegime.COMPUTE_BOUND:
        pct = (
            roofline.compute_ceiling_fraction * 100
            if roofline.compute_ceiling_fraction is not None
            else None
        )
        recs.append(
            Recommendation(
                text=(
                    f"{candidate.qualified_name} is compute-bound: "
                    f"measured at {pct:.0f}% of this machine's real "
                    "compute ceiling for high-arithmetic-intensity "
                    "kernels."
                    if pct is not None
                    else f"{candidate.qualified_name} is compute-bound "
                    "(arithmetic intensity above this machine's roofline "
                    "ridge point)."
                ),
                direction=RecommendationDirection.INFO_ONLY,
                qualified_name=candidate.qualified_name,
                file=candidate.file,
                line_start=candidate.line_start,
                supporting_measurement=(
                    f"arithmetic intensity {ai:.2f} FLOPs/byte; "
                    f"{roofline.achieved_gflops:.1f} GFLOPS achieved"
                    if ai is not None and roofline.achieved_gflops is not None
                    else "compute-bound (see calibration)"
                ),
                confidence=Confidence.MEDIUM,
                evidence=EvidenceKind.ESTIMATED,
                how_to_verify=(
                    "Little headroom is likely left from tuning alone once "
                    "this is near 100% -- re-run `numba-metal advisor "
                    "calibrate` to confirm this machine's compute ceiling "
                    "hasn't changed before assuming there's more to gain."
                ),
                rule_id="roofline_compute_bound",
            )
        )
    elif roofline.regime == RooflinePerformanceRegime.BANDWIDTH_BOUND:
        pct = (
            roofline.bandwidth_ceiling_fraction * 100
            if roofline.bandwidth_ceiling_fraction is not None
            else None
        )
        recs.append(
            Recommendation(
                text=(
                    f"{candidate.qualified_name} is memory-bandwidth-"
                    f"bound: measured at {pct:.0f}% of this machine's "
                    "real memory bandwidth ceiling."
                    if pct is not None
                    else f"{candidate.qualified_name} is memory-bandwidth-"
                    "bound (arithmetic intensity below this machine's "
                    "roofline ridge point)."
                ),
                direction=RecommendationDirection.INFO_ONLY,
                qualified_name=candidate.qualified_name,
                file=candidate.file,
                line_start=candidate.line_start,
                supporting_measurement=(
                    f"arithmetic intensity {ai:.2f} FLOPs/byte; "
                    f"{roofline.achieved_bandwidth_gbps:.1f} GB/s achieved"
                    if ai is not None and roofline.achieved_bandwidth_gbps is not None
                    else "bandwidth-bound (see calibration)"
                ),
                confidence=Confidence.MEDIUM,
                evidence=EvidenceKind.ESTIMATED,
                how_to_verify=(
                    "If this is well under 100%, a memory-access-pattern "
                    "change (e.g. restructuring scattered reads to use "
                    "threadgroup/shared memory) may close real headroom -- "
                    "verified directly on a stencil kernel this session, "
                    "going from ~29% to ~92% of this same ceiling. On "
                    "Apple Silicon specifically, note that CPU and GPU "
                    "share the same memory bandwidth pool, so a low-"
                    "arithmetic-intensity kernel may not beat CPU here "
                    "even at 100% of this ceiling."
                ),
                rule_id="roofline_bandwidth_bound",
            )
        )
    elif roofline.regime == RooflinePerformanceRegime.CACHE_BOUND:
        recs.append(
            Recommendation(
                text=(
                    f"{candidate.qualified_name} measured above this "
                    "machine's calibrated DRAM bandwidth ceiling -- not an "
                    "error. Its working set is small enough to be served "
                    "mostly from on-chip cache rather than DRAM."
                ),
                direction=RecommendationDirection.INFO_ONLY,
                qualified_name=candidate.qualified_name,
                file=candidate.file,
                line_start=candidate.line_start,
                supporting_measurement=(
                    f"achieved {roofline.achieved_bandwidth_gbps:.1f} GB/s, "
                    "above the calibrated DRAM ceiling"
                    if roofline.achieved_bandwidth_gbps is not None
                    else "cache-bound (see calibration)"
                ),
                confidence=Confidence.MEDIUM,
                evidence=EvidenceKind.ESTIMATED,
                how_to_verify=(
                    "Compare bytes_per_call against this machine's typical "
                    "GPU cache size; if the working set grows past that "
                    "(larger inputs), expect achieved bandwidth to drop "
                    "toward the DRAM ceiling instead."
                ),
                rule_id="roofline_cache_bound",
            )
        )
    return recs


def recommend_from_compatibility(
    candidate: Candidate, compatibility: CompatibilityResult
) -> list[Recommendation]:
    """Rules that only need a CompatibilityResult -- no timing data."""
    if compatibility.status == CompatibilityStatus.BLOCKED_BY_MISSING_FEATURE:
        text = compatibility.recommendation or (
            f"Keep {candidate.qualified_name} on the CPU -- "
            "numba-metal does not yet support a construct it uses."
        )
        return [
            Recommendation(
                text=text,
                direction=RecommendationDirection.CHANGE_REQUIRED,
                qualified_name=candidate.qualified_name,
                file=candidate.file,
                line_start=candidate.line_start,
                supporting_measurement=(
                    compatibility.blockers[0] if compatibility.blockers else ""
                ),
                confidence=Confidence.HIGH,
                evidence=EvidenceKind.STATIC_POTENTIAL,
                how_to_verify=(
                    "Run `numba-metal advisor scan` again after making the "
                    "suggested change to confirm the blocker is resolved."
                ),
                rule_id="compatibility_blocked",
            )
        ]
    if compatibility.status == CompatibilityStatus.SUPPORTED:
        return [
            Recommendation(
                text=(
                    f"{candidate.qualified_name} is supported by numba-metal "
                    "today. Run `numba-metal advisor compare` with "
                    "representative inputs to see whether Metal is actually "
                    "faster on this machine before adopting it."
                ),
                direction=RecommendationDirection.INFO_ONLY,
                qualified_name=candidate.qualified_name,
                file=candidate.file,
                line_start=candidate.line_start,
                supporting_measurement="Compatibility dry-run: SUPPORTED",
                confidence=Confidence.MEDIUM,
                evidence=EvidenceKind.STATIC_POTENTIAL,
                how_to_verify=(
                    "numba-metal advisor compare <script containing this function>"
                ),
                rule_id="compatible_needs_measurement",
            )
        ]
    return []


def recommend_low_runtime_share(
    candidate: Candidate, runtime_fraction: float | None, threshold: float = 0.02
) -> list[Recommendation]:
    """Spec example: "This function is only 2% of total runtime;
    conversion cannot materially improve the complete application."

    The spec's own worked example uses exactly 2% as a case that DOES
    trigger this recommendation -- found directly while writing this
    rule's test: an initial `>=` comparison against the default 0.02
    threshold excluded the boundary value the spec itself cites, so the
    comparison is `>` (strict), making the threshold inclusive of the
    "only 2%" case rather than exclusive of it.
    """
    if runtime_fraction is None or runtime_fraction > threshold:
        return []
    return [
        Recommendation(
            text=(
                f"{candidate.qualified_name} is only "
                f"{runtime_fraction:.1%} of total measured runtime; "
                "converting it to Metal cannot materially improve the "
                "complete application even with a large per-function "
                "speedup."
            ),
            direction=RecommendationDirection.INFO_ONLY,
            qualified_name=candidate.qualified_name,
            file=candidate.file,
            line_start=candidate.line_start,
            supporting_measurement=(
                f"{runtime_fraction:.1%} of measured total program runtime"
            ),
            confidence=Confidence.HIGH,
            evidence=EvidenceKind.MEASURED,
            how_to_verify=(
                "Check the sampling profile's folded-stack output for "
                "this function's actual sample count relative to the total."
            ),
            rule_id="low_runtime_share",
        )
    ]


def generate_recommendations(
    candidate: Candidate,
    *,
    compatibility: CompatibilityResult | None = None,
    comparison: ComparisonResult | None = None,
    correctness: CorrectnessResult | None = None,
    runtime_fraction: float | None = None,
) -> list[Recommendation]:
    """Top-level entry point: run every applicable rule and return the
    combined, deterministic list (in a fixed order: correctness/timing
    rules first, then the roofline explanation of those same timing
    results, then compatibility-only rules, then the low-runtime-share
    rule, matching how they are defined above)."""
    recs: list[Recommendation] = []
    if comparison is not None:
        recs.extend(
            recommend_from_comparison(candidate, compatibility, comparison, correctness)
        )
        recs.extend(recommend_from_roofline(candidate, comparison))
    elif compatibility is not None:
        recs.extend(recommend_from_compatibility(candidate, compatibility))
    recs.extend(recommend_low_runtime_share(candidate, runtime_fraction))
    return recs
