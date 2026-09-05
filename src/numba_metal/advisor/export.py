"""Report file writers: profile.json (versioned, source-path-scrubbed by
default) and the plain-text/ASCII report files
(summary.txt/flamegraph-*.txt/timeline.txt/folded-*.txt).

Rendering from a `Report` loaded back from JSON must produce byte-
identical text output to rendering from the freshly-built `Report` that
was saved (Test 13: JSON round trip) -- this module's `load_report`/
`save_report` pair is the only place a Report ever crosses the JSON
boundary, and neither function does anything render.py doesn't already
do deterministically from the same in-memory model.
"""

from __future__ import annotations

import json
from pathlib import Path

from numba_metal.advisor.flamegraph import (
    build_tree_from_events,
    build_tree_from_folded_stacks,
)
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
    FoldedStack,
    MeasurementMode,
    OpportunityScore,
    Recommendation,
    RecommendationDirection,
    RegimeComparison,
    Report,
    SamplingResult,
    ScoreComponent,
    TimingStats,
)
from numba_metal.advisor.render import (
    render_comparison_summary,
    render_correctness,
    render_differential_flame_graph,
    render_flame_graph,
    render_opportunity_score,
    render_recommendation,
    render_timeline,
)
from numba_metal.advisor.timeline import build_timeline


def _scrub_path(path: str, *, include_source_paths: bool) -> str:
    if include_source_paths:
        return path
    return Path(path).name


def save_report(
    report: Report, path: str | Path, *, include_source_paths: bool = False
) -> None:
    """Write `report` as versioned JSON. By default, every file path is
    reduced to its basename (spec: "Avoid absolute source paths by
    default when exporting, with an option to include them.");
    `source_paths_included` in the JSON records which mode was used, so a
    loader never has to guess."""
    payload = report.to_json_dict()
    if not include_source_paths:
        for section in ("candidates", "compatibility", "comparisons", "correctness"):
            for item in payload.get(section, []):
                if "file" in item:
                    item["file"] = _scrub_path(item["file"], include_source_paths=False)
        for rec in payload.get("recommendations", []):
            rec["file"] = _scrub_path(rec["file"], include_source_paths=False)
    payload["source_paths_included"] = include_source_paths
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False))


def _timing_stats_from_dict(d: dict | None) -> TimingStats | None:
    if d is None:
        return None
    return TimingStats(
        samples_ns=tuple(d["samples_ns"]),
        median_ns=d["median_ns"],
        mean_ns=d["mean_ns"],
        min_ns=d["min_ns"],
        max_ns=d["max_ns"],
        stdev_ns=d["stdev_ns"],
        p90_ns=d["p90_ns"],
        p95_ns=d["p95_ns"],
        n=d["n"],
        outliers_ns=tuple(d["outliers_ns"]),
    )


def load_report(path: str | Path) -> Report:
    """Load a Report from JSON saved by `save_report`. Raises a plain
    `ValueError` (not a silent best-effort partial load) if
    `schema_version` does not match what this version of the advisor
    understands -- a schema mismatch must be reported, never guessed
    around."""
    data = json.loads(Path(path).read_text())
    version = data.get("schema_version")
    if version != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"profile.json schema_version {version!r} is not supported by "
            f"this version of numba-metal advisor (expects "
            f"{PROFILE_SCHEMA_VERSION!r}). Re-run the profile with a "
            "matching numba-metal version, or upgrade/downgrade to render it."
        )
    device_d = data["device"]
    device = DeviceInfo(
        device_name=device_d.get("device_name"),
        apple_silicon_family=device_d.get("apple_silicon_family"),
        macos_version=device_d.get("macos_version"),
        python_version=device_d["python_version"],
        numba_version=device_d["numba_version"],
        numba_metal_version=device_d["numba_metal_version"],
        has_unified_memory=device_d.get("has_unified_memory"),
    )
    events = tuple(Event.from_json_dict(e) for e in data.get("events", []))

    candidates = tuple(
        Candidate(
            file=c["file"],
            line_start=c["line_start"],
            line_end=c["line_end"],
            qualified_name=c["qualified_name"],
            decorator=c["decorator"],
            reasons=tuple(c["reasons"]),
            unknowns=tuple(c["unknowns"]),
            blockers=tuple(c["blockers"]),
            parallel_dimension=c["parallel_dimension"],
            inferred_dtypes=tuple(c["inferred_dtypes"]),
            confidence=Confidence(c["confidence"]),
            is_numba_decorated=c["is_numba_decorated"],
            is_metal_decorated=c["is_metal_decorated"],
            patterns=tuple(c.get("patterns", ())),
        )
        for c in data.get("candidates", [])
    )

    compatibility = tuple(
        CompatibilityResult(
            qualified_name=c["qualified_name"],
            file=c["file"],
            line_start=c["line_start"],
            status=CompatibilityStatus(c["status"]),
            supported_features=tuple(c["supported_features"]),
            blockers=tuple(c["blockers"]),
            recommendation=c["recommendation"],
            raw_error=c.get("raw_error"),
        )
        for c in data.get("compatibility", [])
    )

    def _regime_from_dict(r: dict) -> RegimeComparison:
        return RegimeComparison(
            mode=MeasurementMode(r["mode"]),
            cpu=_timing_stats_from_dict(r["cpu"]),
            metal=_timing_stats_from_dict(r["metal"]),
            speedup=r["speedup"],
            preliminary=r["preliminary"],
        )

    comparisons = tuple(
        ComparisonResult(
            qualified_name=c["qualified_name"],
            file=c["file"],
            line_start=c["line_start"],
            regimes=tuple(_regime_from_dict(r) for r in c["regimes"]),
            whole_program_speedup=c["whole_program_speedup"],
            whole_program_preliminary=c["whole_program_preliminary"],
            warmup_runs=c["warmup_runs"],
            measurement_runs=c["measurement_runs"],
            profiler_overhead_ns=c["profiler_overhead_ns"],
        )
        for c in data.get("comparisons", [])
    )

    correctness = tuple(
        CorrectnessResult(
            passed=c["passed"],
            elements_compared=c["elements_compared"],
            elements_total=c["elements_total"],
            sampled=c["sampled"],
            max_abs_error=c["max_abs_error"],
            max_rel_error=c["max_rel_error"],
            atol=c["atol"],
            rtol=c["rtol"],
            cpu_dtype=c["cpu_dtype"],
            metal_dtype=c["metal_dtype"],
            shape_match=c["shape_match"],
            dtype_match=c["dtype_match"],
            nan_mismatch=c["nan_mismatch"],
            inf_mismatch=c["inf_mismatch"],
            failure_reason=c.get("failure_reason"),
        )
        for c in data.get("correctness", [])
    )

    scores = tuple(
        OpportunityScore(
            qualified_name=s["qualified_name"],
            components=tuple(
                ScoreComponent(
                    name=comp["name"],
                    value=comp["value"],
                    max_value=comp["max_value"],
                    explanation=comp["explanation"],
                )
                for comp in s["components"]
            ),
            total=s["total"],
            max_total=s["max_total"],
            confidence=Confidence(s["confidence"]),
            evidence=EvidenceKind(s["evidence"]),
        )
        for s in data.get("scores", [])
    )

    recommendations = tuple(
        Recommendation(
            text=r["text"],
            direction=RecommendationDirection(r["direction"]),
            qualified_name=r["qualified_name"],
            file=r["file"],
            line_start=r["line_start"],
            supporting_measurement=r["supporting_measurement"],
            confidence=Confidence(r["confidence"]),
            evidence=EvidenceKind(r["evidence"]),
            how_to_verify=r["how_to_verify"],
            rule_id=r["rule_id"],
        )
        for r in data.get("recommendations", [])
    )

    sampling_d = data.get("sampling")
    sampling = None
    if sampling_d is not None:
        sampling = SamplingResult(
            stacks=tuple(
                FoldedStack(frames=tuple(s["frames"]), count=s["count"])
                for s in sampling_d["stacks"]
            ),
            total_samples=sampling_d["total_samples"],
            interval_ms=sampling_d["interval_ms"],
            dropped_samples=sampling_d["dropped_samples"],
            profiler_overhead_ns=sampling_d["profiler_overhead_ns"],
        )

    return Report(
        schema_version=data["schema_version"],
        generated_at_ns=data["generated_at_ns"],
        device=device,
        events=events,
        candidates=candidates,
        compatibility=compatibility,
        comparisons=comparisons,
        correctness=correctness,
        scores=scores,
        recommendations=recommendations,
        sampling=sampling,
        source_paths_included=data.get("source_paths_included", False),
        dropped_event_count=data.get("dropped_event_count", 0),
    )


def render_summary_text(report: Report) -> str:
    lines = ["NUMBA-METAL ADVISOR SUMMARY", "=" * 28]
    lines.append(f"Schema version: {report.schema_version}")
    lines.append(f"Device: {report.device.device_name or 'unknown'}")
    lines.append(f"macOS: {report.device.macos_version or 'unknown'}")
    lines.append(
        f"Python {report.device.python_version}, Numba "
        f"{report.device.numba_version}, numba-metal "
        f"{report.device.numba_metal_version}"
    )
    if report.dropped_event_count:
        lines.append(
            f"WARNING: {report.dropped_event_count} events were dropped "
            "(event limit reached during profiling)"
        )
    lines.append("")
    for c in report.candidates:
        lines.append(f"CANDIDATE: {c.qualified_name} ({c.file}:{c.line_start})")
    lines.append("")
    for comp in report.compatibility:
        lines.append(f"COMPATIBILITY: {comp.qualified_name} -> {comp.status.value}")
    lines.append("")
    for cmp_result in report.comparisons:
        lines.append(render_comparison_summary(cmp_result))
        lines.append("")
    for corr in report.correctness:
        lines.append(render_correctness(corr))
        lines.append("")
    for score in report.scores:
        lines.append(render_opportunity_score(score))
        lines.append("")
    for rec in report.recommendations:
        lines.append(render_recommendation(rec))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_all_text_reports(
    report: Report, *, width: int = 78, color: bool = False
) -> dict[str, str]:
    """Return {filename: content} for every text report file this
    package can produce from `report` alone (no re-execution of any
    workload) -- exactly what `numba-metal advisor report PROFILE.json`
    and the `--output` directory writer both call."""
    outputs: dict[str, str] = {"summary.txt": render_summary_text(report)}

    cpu_events = tuple(e for e in report.events if e.category in ("cpu", "python"))
    metal_events = tuple(
        e for e in report.events if e.category in ("gpu", "sync", "compile", "transfer")
    )
    if report.sampling is not None and report.sampling.stacks:
        cpu_tree = build_tree_from_folded_stacks(
            report.sampling.stacks, unit_ns_per_sample=report.sampling.interval_ms * 1e6
        )
        outputs["folded-cpu.txt"] = report.sampling.to_folded_text()
        outputs["flamegraph-cpu.txt"] = render_flame_graph(
            cpu_tree,
            "CPU BASELINE",
            cpu_tree.total_ns,
            width=width,
            color=color,
            call_style_labels=True,
        )
    elif cpu_events:
        cpu_tree = build_tree_from_events(cpu_events)
        outputs["flamegraph-cpu.txt"] = render_flame_graph(
            cpu_tree, "CPU BASELINE", cpu_tree.total_ns, width=width, color=color
        )
    else:
        cpu_tree = build_tree_from_events(())

    if metal_events:
        metal_tree = build_tree_from_events(metal_events)
        outputs["flamegraph-metal.txt"] = render_flame_graph(
            metal_tree, "METAL RUN", metal_tree.total_ns, width=width, color=color
        )
        if cpu_tree.total_ns > 0:
            outputs["flamegraph-diff.txt"] = render_differential_flame_graph(
                cpu_tree,
                metal_tree,
                cpu_total_ns=cpu_tree.total_ns,
                metal_total_ns=metal_tree.total_ns,
                width=width,
                color=color,
            )

    if report.events:
        timeline = build_timeline(report.events)
        outputs["timeline.txt"] = render_timeline(timeline, width=width)

    return outputs
