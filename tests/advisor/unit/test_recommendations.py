"""Test 4 (small workload loses), Test 5 (large workload wins), Test 6
(correctness failure blocks a USE_METAL recommendation) from the
governing spec.

All synthetic TimingStats/ComparisonResult/CorrectnessResult -- no
Metal hardware or real timing involved, matching the spec's explicit
"do not make timing-sensitive unit tests depend on real machine
performance" instruction.
"""

from __future__ import annotations

from numba_metal.advisor.models import (
    Candidate,
    ComparisonResult,
    Confidence,
    CorrectnessResult,
    MeasurementMode,
    RecommendationDirection,
    RegimeComparison,
    TimingStats,
)
from numba_metal.advisor.recommendations import generate_recommendations


def _candidate(name: str = "simulate") -> Candidate:
    return Candidate(
        file="sim.py",
        line_start=10,
        line_end=20,
        qualified_name=name,
        decorator="metal.jit",
        reasons=(),
        unknowns=(),
        blockers=(),
        parallel_dimension=None,
        inferred_dtypes=(),
        confidence=Confidence.HIGH,
        is_numba_decorated=False,
        is_metal_decorated=True,
    )


def _passing_correctness() -> CorrectnessResult:
    return CorrectnessResult(
        passed=True,
        elements_compared=1000,
        elements_total=1000,
        sampled=False,
        max_abs_error=1e-7,
        max_rel_error=1e-7,
        atol=1e-5,
        rtol=1e-5,
        cpu_dtype="float32",
        metal_dtype="float32",
        shape_match=True,
        dtype_match=True,
        nan_mismatch=False,
        inf_mismatch=False,
        failure_reason=None,
    )


def _failing_correctness() -> CorrectnessResult:
    return CorrectnessResult(
        passed=False,
        elements_compared=1000,
        elements_total=1000,
        sampled=False,
        max_abs_error=5.0,
        max_rel_error=5.0,
        atol=1e-5,
        rtol=1e-5,
        cpu_dtype="float32",
        metal_dtype="float32",
        shape_match=True,
        dtype_match=True,
        nan_mismatch=False,
        inf_mismatch=False,
        failure_reason="max abs error 5.0 exceeds atol/rtol",
    )


def _comparison(*, cpu_ns: int, metal_ns: int, n: int = 7) -> ComparisonResult:
    cpu = TimingStats.from_samples([cpu_ns] * n)
    metal = TimingStats.from_samples([metal_ns] * n)
    speedup = cpu_ns / metal_ns
    regime = RegimeComparison(
        mode=MeasurementMode.WARM,
        cpu=cpu,
        metal=metal,
        speedup=speedup,
        preliminary=False,
    )
    return ComparisonResult(
        qualified_name="simulate",
        file="sim.py",
        line_start=10,
        regimes=(regime,),
        whole_program_speedup=speedup,
        whole_program_preliminary=False,
        warmup_runs=2,
        measurement_runs=n,
        profiler_overhead_ns=1000,
    )


def test_small_workload_loses_recommends_keep_cpu():
    """Test 4: Synthetic data shows dispatch overhead making Metal
    slower. The advisor recommends remaining on the CPU or batching the
    work."""
    candidate = _candidate()
    # Metal is 10x SLOWER (dispatch overhead dominates at small size).
    comparison = _comparison(cpu_ns=1_000, metal_ns=10_000)
    recs = generate_recommendations(
        candidate, comparison=comparison, correctness=_passing_correctness()
    )
    directions = [r.direction for r in recs]
    assert RecommendationDirection.KEEP_CPU in directions
    assert RecommendationDirection.USE_METAL not in directions
    keep_cpu_rec = next(
        r for r in recs if r.direction == RecommendationDirection.KEEP_CPU
    )
    assert "SLOWER" in keep_cpu_rec.text
    assert keep_cpu_rec.evidence.value == "MEASURED"


def test_large_workload_wins_reports_function_and_whole_program_speedup():
    """Test 5: Synthetic data shows a compute-heavy kernel winning on
    Metal. The advisor reports both function and whole-program
    speedup."""
    candidate = _candidate()
    # Metal is 8x faster.
    comparison = _comparison(cpu_ns=800_000, metal_ns=100_000)
    recs = generate_recommendations(
        candidate, comparison=comparison, correctness=_passing_correctness()
    )
    directions = [r.direction for r in recs]
    assert RecommendationDirection.USE_METAL in directions
    use_metal_rec = next(
        r for r in recs if r.direction == RecommendationDirection.USE_METAL
    )
    assert "8.00x faster" in use_metal_rec.text
    assert comparison.whole_program_speedup == 8.0
    assert use_metal_rec.evidence.value == "MEASURED"


def test_correctness_failure_blocks_use_metal_even_with_huge_speedup():
    """Test 6: Metal is faster but produces an incorrect result. The
    tool must clearly recommend against using the Metal version."""
    candidate = _candidate()
    # Metal is 10x faster -- but WRONG.
    comparison = _comparison(cpu_ns=1_000_000, metal_ns=100_000)
    recs = generate_recommendations(
        candidate, comparison=comparison, correctness=_failing_correctness()
    )
    directions = [r.direction for r in recs]
    assert RecommendationDirection.USE_METAL not in directions
    assert RecommendationDirection.KEEP_CPU in directions
    rec = next(r for r in recs if r.direction == RecommendationDirection.KEEP_CPU)
    assert "DO NOT USE METAL" in rec.text
    assert "tolerance" in rec.supporting_measurement.lower() or "exceeds" in (
        rec.supporting_measurement.lower()
    )


def test_no_correctness_data_at_all_also_blocks_use_metal():
    """Absence of a correctness check must be treated the same as a
    failure for the purpose of gating USE_METAL -- "we didn't check" is
    not evidence of "it's fine"."""
    candidate = _candidate()
    comparison = _comparison(cpu_ns=1_000_000, metal_ns=100_000)
    recs = generate_recommendations(candidate, comparison=comparison, correctness=None)
    directions = [r.direction for r in recs]
    assert RecommendationDirection.USE_METAL not in directions


def test_preliminary_single_run_speedup_is_labeled_low_confidence():
    """Do not calculate a speedup from a single run unless explicitly
    requested. Label single-run output as preliminary."""
    candidate = _candidate()
    cpu = TimingStats.from_samples([1000])
    metal = TimingStats.from_samples([100])
    regime = RegimeComparison(
        mode=MeasurementMode.WARM, cpu=cpu, metal=metal, speedup=10.0, preliminary=True
    )
    comparison = ComparisonResult(
        qualified_name="simulate",
        file="sim.py",
        line_start=10,
        regimes=(regime,),
        whole_program_speedup=10.0,
        whole_program_preliminary=True,
        warmup_runs=0,
        measurement_runs=1,
        profiler_overhead_ns=None,
    )
    recs = generate_recommendations(
        candidate, comparison=comparison, correctness=_passing_correctness()
    )
    use_metal_rec = next(
        r for r in recs if r.direction == RecommendationDirection.USE_METAL
    )
    assert use_metal_rec.confidence == Confidence.LOW


def test_low_runtime_share_recommendation():
    candidate = _candidate()
    recs = generate_recommendations(candidate, runtime_fraction=0.02)
    assert any(r.rule_id == "low_runtime_share" for r in recs)
    rec = next(r for r in recs if r.rule_id == "low_runtime_share")
    assert "2.0%" in rec.text
    assert "cannot materially improve" in rec.text


def test_recommendations_never_call_an_llm():
    """Structural check: recommendations.py must be pure/deterministic
    over its inputs -- calling it twice with identical inputs must
    produce byte-identical output (an LLM call would not guarantee
    this)."""
    candidate = _candidate()
    comparison = _comparison(cpu_ns=800_000, metal_ns=100_000)
    recs1 = generate_recommendations(
        candidate, comparison=comparison, correctness=_passing_correctness()
    )
    recs2 = generate_recommendations(
        candidate, comparison=comparison, correctness=_passing_correctness()
    )
    assert [r.to_json_dict() for r in recs1] == [r.to_json_dict() for r in recs2]
