"""Opportunity scoring: an explainable, component-by-component ranking of
how worthwhile it is to move a candidate function to numba-metal.

Every component is independently visible (`ScoreComponent.explanation`),
never hidden inside one opaque number -- see models.OpportunityScore.
When only static evidence exists (no ComparisonResult), the result is
explicitly `EvidenceKind.STATIC_POTENTIAL`; callers/renderers must label
this "POTENTIAL", never "OPPORTUNITY SCORE" (see spec section 13: "When
only static evidence exists, call the result POTENTIAL, not an estimated
speedup.").
"""

from __future__ import annotations

from numba_metal.advisor.models import (
    Candidate,
    ComparisonResult,
    CompatibilityResult,
    CompatibilityStatus,
    Confidence,
    EvidenceKind,
    MeasurementMode,
    OpportunityScore,
    ScoreComponent,
)

MAX_HOTSPOT = 30.0
MAX_PARALLELISM = 20.0
MAX_ARITHMETIC_INTENSITY = 20.0
MAX_COMPATIBILITY = 20.0
MAX_CALL_FREQUENCY = 10.0
MAX_TRANSFER_PENALTY = 10.0  # subtracted, not added
MAX_TOTAL = MAX_HOTSPOT + MAX_PARALLELISM + MAX_ARITHMETIC_INTENSITY + MAX_COMPATIBILITY


def _hotspot_component(
    runtime_fraction: float | None,
) -> ScoreComponent:
    """`runtime_fraction`: this function's share of total measured
    program runtime, in [0, 1], or None if unknown (no profiling data)."""
    if runtime_fraction is None:
        return ScoreComponent(
            name="Runtime importance",
            value=MAX_HOTSPOT * 0.3,
            max_value=MAX_HOTSPOT,
            explanation=(
                "No runtime profile available -- assumed moderate "
                "importance; run `numba-metal advisor profile` for a "
                "measured value."
            ),
        )
    value = MAX_HOTSPOT * min(1.0, max(0.0, runtime_fraction))
    return ScoreComponent(
        name="Runtime importance",
        value=value,
        max_value=MAX_HOTSPOT,
        explanation=f"{runtime_fraction:.1%} of measured total program runtime",
    )


def _parallelism_component(candidate: Candidate) -> ScoreComponent:
    reasons = []
    score = 0.0
    if candidate.parallel_dimension is not None:
        score += MAX_PARALLELISM * 0.6
        reasons.append(f"explicit parallel dimension ({candidate.parallel_dimension})")
    if "nested_loop" in candidate.patterns:
        score += MAX_PARALLELISM * 0.3
        reasons.append("nested numerical loop over array indices")
    if "reduction" in candidate.patterns:
        score += MAX_PARALLELISM * 0.1
        reasons.append("reduction pattern (partial parallelism, needs care)")
    score = min(score, MAX_PARALLELISM)
    explanation = "; ".join(reasons) if reasons else "no parallel structure detected"
    return ScoreComponent(
        name="Parallel structure",
        value=score,
        max_value=MAX_PARALLELISM,
        explanation=explanation,
    )


def _arithmetic_intensity_component(candidate: Candidate) -> ScoreComponent:
    score = 0.0
    reasons = []
    if "elementwise_math" in candidate.patterns:
        score += MAX_ARITHMETIC_INTENSITY * 0.5
        reasons.append("element-wise math function calls")
    if "pairwise_or_stencil" in candidate.patterns:
        score += MAX_ARITHMETIC_INTENSITY * 0.5
        reasons.append("pairwise/stencil-shaped computation (high arithmetic density)")
    if "monte_carlo" in candidate.patterns:
        score += MAX_ARITHMETIC_INTENSITY * 0.3
        reasons.append("repeated random sampling (Monte Carlo)")
    score = min(score, MAX_ARITHMETIC_INTENSITY)
    explanation = (
        "; ".join(reasons) if reasons else "no strong arithmetic-intensity signal"
    )
    return ScoreComponent(
        name="Arithmetic intensity",
        value=score,
        max_value=MAX_ARITHMETIC_INTENSITY,
        explanation=explanation,
    )


def _compatibility_component(
    compat: CompatibilityResult | None,
) -> ScoreComponent:
    if compat is None:
        return ScoreComponent(
            name="Current compatibility",
            value=0.0,
            max_value=MAX_COMPATIBILITY,
            explanation="Not analyzed for numba-metal compatibility yet",
        )
    mapping = {
        CompatibilityStatus.SUPPORTED: 1.0,
        CompatibilityStatus.SUPPORTED_WITH_CHANGES: 0.6,
        CompatibilityStatus.BLOCKED_BY_MISSING_FEATURE: 0.1,
        CompatibilityStatus.POOR_GPU_CANDIDATE: 0.0,
        CompatibilityStatus.UNABLE_TO_ANALYZE: 0.2,
    }
    fraction = mapping[compat.status]
    return ScoreComponent(
        name="Current compatibility",
        value=MAX_COMPATIBILITY * fraction,
        max_value=MAX_COMPATIBILITY,
        explanation=f"numba-metal dry-run result: {compat.status.value}",
    )


def _call_frequency_component(call_count: int | None) -> ScoreComponent:
    if call_count is None:
        return ScoreComponent(
            name="Call frequency",
            value=0.0,
            max_value=MAX_CALL_FREQUENCY,
            explanation="No call-count data available",
        )
    # Log-scaled: 1 call -> ~0, 1000+ calls -> full score. Deliberately
    # coarse (this is a minor component relative to hotspot/parallelism).
    import math

    scaled = min(1.0, math.log10(max(1, call_count) + 1) / 3.0)
    return ScoreComponent(
        name="Call frequency",
        value=MAX_CALL_FREQUENCY * scaled,
        max_value=MAX_CALL_FREQUENCY,
        explanation=f"called {call_count} time(s) in the profiled run",
    )


def _transfer_penalty_component(comparison: ComparisonResult | None) -> ScoreComponent:
    if comparison is None:
        return ScoreComponent(
            name="Transfer/sync penalty",
            value=0.0,
            max_value=MAX_TRANSFER_PENALTY,
            explanation="No timing data available to assess transfer/sync overhead",
        )
    warm = next((r for r in comparison.regimes if r.mode == MeasurementMode.WARM), None)
    if warm is None or warm.metal is None or warm.cpu is None:
        return ScoreComponent(
            name="Transfer/sync penalty",
            value=0.0,
            max_value=MAX_TRANSFER_PENALTY,
            explanation="No warm Metal timing available",
        )
    if warm.speedup is not None and warm.speedup < 1.0:
        # Metal is slower warm -- penalize proportionally to how much
        # slower, capped at MAX_TRANSFER_PENALTY.
        deficit = 1.0 - warm.speedup
        penalty = min(MAX_TRANSFER_PENALTY, MAX_TRANSFER_PENALTY * deficit)
        return ScoreComponent(
            name="Transfer/sync penalty",
            value=-penalty,
            max_value=MAX_TRANSFER_PENALTY,
            explanation=(
                f"Warm Metal is {1/warm.speedup:.2f}x SLOWER than CPU "
                "(dispatch/transfer/sync overhead likely dominates at this size)"
            ),
        )
    return ScoreComponent(
        name="Transfer/sync penalty",
        value=0.0,
        max_value=MAX_TRANSFER_PENALTY,
        explanation="Warm Metal timing does not show a transfer/sync penalty",
    )


def score_candidate(
    candidate: Candidate,
    *,
    compatibility: CompatibilityResult | None = None,
    comparison: ComparisonResult | None = None,
    runtime_fraction: float | None = None,
    call_count: int | None = None,
) -> OpportunityScore:
    """Build an explainable OpportunityScore. `evidence` is MEASURED only
    when `comparison` (real timing data) is supplied; otherwise
    STATIC_POTENTIAL, per spec section 13."""
    components = [
        _hotspot_component(runtime_fraction),
        _parallelism_component(candidate),
        _arithmetic_intensity_component(candidate),
        _compatibility_component(compatibility),
    ]
    total = sum(c.value for c in components)
    max_total = MAX_TOTAL

    if comparison is not None:
        components.append(_call_frequency_component(call_count))
        transfer_penalty = _transfer_penalty_component(comparison)
        components.append(transfer_penalty)
        total += transfer_penalty.value
        max_total += MAX_CALL_FREQUENCY

    if candidate.blockers:
        confidence = Confidence.HIGH  # confident it's NOT a good candidate yet
    elif comparison is not None:
        confidence = Confidence.HIGH
    elif compatibility is not None and compatibility.status in (
        CompatibilityStatus.SUPPORTED,
        CompatibilityStatus.BLOCKED_BY_MISSING_FEATURE,
    ):
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    evidence = (
        EvidenceKind.MEASURED
        if comparison is not None
        else EvidenceKind.STATIC_POTENTIAL
    )

    return OpportunityScore(
        qualified_name=candidate.qualified_name,
        components=tuple(components),
        total=total,
        max_total=max_total,
        confidence=confidence,
        evidence=evidence,
    )
