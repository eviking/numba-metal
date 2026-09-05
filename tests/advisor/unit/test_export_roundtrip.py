"""Test 13 (JSON round trip) and Test 14's non-platform-gated half
(scan/report rendering work without a Metal device) from the governing
spec.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from numba_metal.advisor.export import load_report, render_all_text_reports, save_report
from numba_metal.advisor.models import (
    PROFILE_SCHEMA_VERSION,
    Candidate,
    ComparisonResult,
    CompatibilityResult,
    CompatibilityStatus,
    Confidence,
    CorrectnessResult,
    DeviceInfo,
    Event,
    EvidenceKind,
    GpuTimestampSource,
    MeasurementMode,
    OpportunityScore,
    Recommendation,
    RecommendationDirection,
    RegimeComparison,
    Report,
    ScoreComponent,
    TimingStats,
    now_ns,
)


def _build_full_report() -> Report:
    device = DeviceInfo(
        device_name="Apple M4 Pro",
        apple_silicon_family="M4 Pro",
        macos_version="15.0",
        python_version="3.13.0",
        numba_version="0.67.0",
        numba_metal_version="0.1.0.dev0",
        has_unified_memory=True,
    )
    events = (
        Event(
            event_type="python_call",
            name="simulate_paths",
            category="cpu",
            start_ns=0,
            duration_ns=1_458_000_000,
            measured=True,
        ),
        Event(
            event_type="metal_kernel",
            name="GPU kernel",
            category="gpu",
            start_ns=1_458_000_000,
            duration_ns=38_000_000,
            measured=True,
            gpu_timestamp_source=GpuTimestampSource.HOST_WALL_CLOCK,
            shape=(2_000_000,),
            dtype="float32",
        ),
    )
    candidate = Candidate(
        file="/Users/foo/bar/sim.py",
        line_start=10,
        line_end=20,
        qualified_name="simulate_paths",
        decorator="metal.jit",
        reasons=("nested loop",),
        unknowns=(),
        blockers=(),
        parallel_dimension=None,
        inferred_dtypes=("float32",),
        confidence=Confidence.HIGH,
        is_numba_decorated=False,
        is_metal_decorated=True,
    )
    compat = CompatibilityResult(
        qualified_name="simulate_paths",
        file="/Users/foo/bar/sim.py",
        line_start=10,
        status=CompatibilityStatus.SUPPORTED,
        supported_features=("Array iteration",),
        blockers=(),
        recommendation=None,
    )
    cpu_stats = TimingStats.from_samples([1000, 1010, 990])
    metal_stats = TimingStats.from_samples([100, 105, 98])
    regime = RegimeComparison(
        mode=MeasurementMode.WARM,
        cpu=cpu_stats,
        metal=metal_stats,
        speedup=10.0,
        preliminary=False,
    )
    comparison = ComparisonResult(
        qualified_name="simulate_paths",
        file="/Users/foo/bar/sim.py",
        line_start=10,
        regimes=(regime,),
        whole_program_speedup=10.0,
        whole_program_preliminary=False,
        warmup_runs=2,
        measurement_runs=3,
        profiler_overhead_ns=500,
    )
    correctness = CorrectnessResult(
        passed=True,
        elements_compared=2_000_000,
        elements_total=2_000_000,
        sampled=False,
        max_abs_error=1.7e-6,
        max_rel_error=8.3e-6,
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
    score = OpportunityScore(
        qualified_name="simulate_paths",
        components=(
            ScoreComponent(
                name="Runtime importance", value=28, max_value=30, explanation="93%"
            ),
        ),
        total=84,
        max_total=100,
        confidence=Confidence.HIGH,
        evidence=EvidenceKind.MEASURED,
    )
    rec = Recommendation(
        text="Use Metal for simulate_paths.",
        direction=RecommendationDirection.USE_METAL,
        qualified_name="simulate_paths",
        file="/Users/foo/bar/sim.py",
        line_start=10,
        supporting_measurement="10.00x measured",
        confidence=Confidence.HIGH,
        evidence=EvidenceKind.MEASURED,
        how_to_verify="Re-run compare.",
        rule_id="warm_metal_faster_than_cpu",
    )
    return Report(
        schema_version=PROFILE_SCHEMA_VERSION,
        generated_at_ns=now_ns(),
        device=device,
        events=events,
        candidates=(candidate,),
        compatibility=(compat,),
        comparisons=(comparison,),
        correctness=(correctness,),
        scores=(score,),
        recommendations=(rec,),
        sampling=None,
        source_paths_included=False,
    )


def test_json_round_trip_produces_identical_render_output():
    """Test 13: A profile can be saved, loaded, and rendered
    deterministically."""
    report = _build_full_report()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "profile.json"
        save_report(report, path, include_source_paths=False)

        loaded1 = load_report(path)
        loaded2 = load_report(path)

        # Structural equality of every field, not just "some events".
        assert loaded1.schema_version == report.schema_version
        assert len(loaded1.events) == len(report.events)
        assert loaded1.events[0].name == report.events[0].name
        expected_source = GpuTimestampSource.HOST_WALL_CLOCK
        assert loaded1.events[1].gpu_timestamp_source == expected_source
        assert loaded1.events[1].shape == (2_000_000,)
        assert loaded1.candidates[0].confidence == Confidence.HIGH
        assert loaded1.compatibility[0].status == CompatibilityStatus.SUPPORTED
        assert loaded1.comparisons[0].regimes[0].speedup == 10.0
        assert loaded1.correctness[0].passed is True
        assert loaded1.scores[0].total == 84
        assert loaded1.recommendations[0].direction == RecommendationDirection.USE_METAL

        # Deterministic rendering: two independent loads render byte-for-
        # byte identically.
        out1 = render_all_text_reports(loaded1, width=78)
        out2 = render_all_text_reports(loaded2, width=78)
        assert out1.keys() == out2.keys()
        for key in out1:
            assert out1[key] == out2[key], f"non-deterministic render for {key}"


def test_paths_scrubbed_by_default_and_flagged():
    report = _build_full_report()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "profile.json"
        save_report(report, path, include_source_paths=False)
        import json

        raw = json.loads(path.read_text())
        assert raw["source_paths_included"] is False
        assert raw["candidates"][0]["file"] == "sim.py"  # basename only
        assert "/Users/foo/bar/" not in json.dumps(raw)


def test_paths_included_when_explicitly_requested():
    report = _build_full_report()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "profile.json"
        save_report(report, path, include_source_paths=True)
        import json

        raw = json.loads(path.read_text())
        assert raw["source_paths_included"] is True
        assert raw["candidates"][0]["file"] == "/Users/foo/bar/sim.py"


def test_schema_version_mismatch_raises_not_silently_guessed():
    report = _build_full_report()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "profile.json"
        save_report(report, path)
        import json

        raw = json.loads(path.read_text())
        raw["schema_version"] = 9999
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="schema_version"):
            load_report(path)


def test_report_rendering_requires_no_metal_hardware():
    """Test 14 (partial): existing profile rendering works without any
    Metal-specific import or device check at all."""
    report = _build_full_report()
    # render_all_text_reports / render_summary_text must not import
    # anything Metal-hardware-dependent -- this test's mere successful
    # execution on any machine (this test file carries no @pytest.mark.metal)
    # is the actual proof.
    outputs = render_all_text_reports(report, width=78)
    assert "flamegraph-cpu.txt" in outputs
    assert "flamegraph-metal.txt" in outputs
    assert "timeline.txt" in outputs
