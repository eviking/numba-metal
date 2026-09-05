"""Normalized data model shared by every advisor stage.

These are plain, explicit dataclasses (per the task's requirement to avoid
implicit dict-shaped data): collection stages (scanner, compatibility,
profiler, sampling) produce them; analysis stages (comparison, correctness,
scoring, recommendations) consume and combine them; presentation stages
(render, flamegraph, timeline, export) only ever read them, never compute a
new measurement.

Every field that carries a number a user could mistake for ground truth
carries an explicit `measured: bool` (or is nested inside a class that
does) so renderers can never blur "we timed this" with "we estimated this"
with "this is a static guess" -- the central, non-negotiable requirement
of the whole advisor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Confidence / classification enums
# ---------------------------------------------------------------------------


class Confidence(StrEnum):
    """How much a Candidate/Recommendation/OpportunityScore should be
    trusted. Never a numeric probability -- there is no calibrated model
    producing one; this is an explicit, three-level qualitative label."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CompatibilityStatus(StrEnum):
    """Result of a numba-metal compatibility dry-run (see compatibility.py).

    Exactly the five classes required by the governing spec -- no
    percentage, no score, just which bucket a function falls into and why
    (the `blockers`/`supported_features` lists on CompatibilityResult carry
    the "why").
    """

    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_CHANGES = "SUPPORTED_WITH_CHANGES"
    BLOCKED_BY_MISSING_FEATURE = "BLOCKED_BY_MISSING_FEATURE"
    POOR_GPU_CANDIDATE = "POOR_GPU_CANDIDATE"
    UNABLE_TO_ANALYZE = "UNABLE_TO_ANALYZE"


class MeasurementMode(StrEnum):
    """Which of the three CPU/Metal comparison regimes a ComparisonResult
    describes. Never merged into one number -- see comparison.py."""

    COLD = "COLD"
    WARM = "WARM"
    STEADY_STATE = "STEADY_STATE"


class EvidenceKind(StrEnum):
    """Tags every number that reaches a renderer with exactly how it was
    obtained. This is the mechanism satisfying the spec's central
    principle: "Never claim that Metal is faster based only on static
    analysis. Clearly separate measured results, model-based estimates,
    and unsupported speculation." """

    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    STATIC_POTENTIAL = "STATIC_POTENTIAL"


class RecommendationDirection(StrEnum):
    USE_METAL = "USE_METAL"
    KEEP_CPU = "KEEP_CPU"
    CHANGE_REQUIRED = "CHANGE_REQUIRED"
    INFO_ONLY = "INFO_ONLY"


def now_ns() -> int:
    """Monotonic-clock timestamp in nanoseconds. All Event.start_ns /
    duration_ns values in this package are derived from
    time.monotonic_ns() exclusively -- never time.time() / wall-clock --
    so that durations are immune to system clock adjustments. See
    metal_events.py's module docstring for how CPU-side and GPU
    submission/completion timestamps are aligned (they share this same
    clock: everything is host-side wall time, since numba-metal exposes no
    device-side clock -- see Event.gpu_timestamp_source)."""
    return time.monotonic_ns()


# ---------------------------------------------------------------------------
# Event model (runtime profiling)
# ---------------------------------------------------------------------------


class GpuTimestampSource(StrEnum):
    """Where a GPU-related Event's timing actually came from. numba-metal
    exposes no MTLCommandBuffer.GPUStartTime/GPUEndTime reads and no
    addCompletedHandler_-based device timestamp anywhere in its runtime
    (verified directly against runtime/dispatcher.py and
    runtime/context.py) -- so any "kernel" event this package reports is,
    at best, a host-side wall-clock span between command-buffer commit()
    and the corresponding waitUntilCompleted() returning. This enum makes
    that distinction explicit and unskippable in the data model itself,
    rather than a docstring promise a renderer could quietly violate."""

    #: No timestamp at all (e.g. this event is a pure CPU-side span).
    NOT_APPLICABLE = "NOT_APPLICABLE"
    #: Host wall-clock time between submission and completion confirmation.
    #: This is what numba-metal can measure today for "GPU kernel" events.
    #: It is measured, but it is NOT isolated GPU-busy time -- it also
    #: includes queue wait and any driver-side scheduling latency.
    HOST_WALL_CLOCK = "HOST_WALL_CLOCK"
    #: A true on-device timestamp (GPUStartTime/GPUEndTime or equivalent).
    #: No code path in this package can produce this today; it exists so
    #: the schema does not need to change if/when numba-metal gains one
    #: (see docs/advisor.md's "Known limitations").
    DEVICE_TIMESTAMP = "DEVICE_TIMESTAMP"


@dataclass(frozen=True, slots=True)
class Event:
    """One normalized profiling event, matching the JSON shape in the
    governing spec section 6. Every Event is either `measured=True` (a real
    wall-clock span this process actually observed) or `measured=False`
    (a modeled/estimated duration -- see comparison.py's crossover
    modeling); there is no third option smuggled in via a missing field.
    """

    event_type: str
    """e.g. "python_call", "numba_cpu_call", "metal_wrapper_call",
    "metal_compile_source", "metal_compile_shader", "metal_pipeline_create",
    "metal_cache_hit", "metal_cache_miss", "buffer_alloc", "buffer_reuse",
    "command_encode", "command_submit", "metal_kernel", "synchronize",
    "cpu_fallback", "result_materialize"."""

    name: str
    category: str  # "cpu" | "gpu" | "transfer" | "sync" | "compile" | "python"
    start_ns: int
    duration_ns: int
    measured: bool
    thread_id: int = 0
    device: str | None = None
    kernel: str | None = None
    input_bytes: int | None = None
    output_bytes: int | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    cold_start: bool = False
    gpu_timestamp_source: GpuTimestampSource = GpuTimestampSource.NOT_APPLICABLE
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        d = {
            "event_type": self.event_type,
            "name": self.name,
            "category": self.category,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "measured": self.measured,
            "thread_id": self.thread_id,
            "device": self.device,
            "kernel": self.kernel,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "cold_start": self.cold_start,
            "gpu_timestamp_source": self.gpu_timestamp_source.value,
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @staticmethod
    def from_json_dict(d: dict) -> Event:
        return Event(
            event_type=d["event_type"],
            name=d["name"],
            category=d["category"],
            start_ns=d["start_ns"],
            duration_ns=d["duration_ns"],
            measured=d["measured"],
            thread_id=d.get("thread_id", 0),
            device=d.get("device"),
            kernel=d.get("kernel"),
            input_bytes=d.get("input_bytes"),
            output_bytes=d.get("output_bytes"),
            shape=tuple(d["shape"]) if d.get("shape") is not None else None,
            dtype=d.get("dtype"),
            cold_start=d.get("cold_start", False),
            gpu_timestamp_source=GpuTimestampSource(
                d.get("gpu_timestamp_source", GpuTimestampSource.NOT_APPLICABLE.value)
            ),
            extra=dict(d.get("extra", {})),
        )


# ---------------------------------------------------------------------------
# Static scanning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One function the static scanner flagged as worth a closer look.

    No numeric speedup estimate is ever attached here -- `confidence` is
    qualitative and `reasons`/`unknowns`/`blockers` are the actual content,
    matching the spec's example format ("[HIGH POTENTIAL] ... Why: ...
    Unknown: ... Next step: ...")."""

    file: str
    line_start: int
    line_end: int
    qualified_name: str
    decorator: str | None  # e.g. "numba.njit", "metal.jit", None for plain Python
    reasons: tuple[str, ...]
    unknowns: tuple[str, ...]
    blockers: tuple[str, ...]
    parallel_dimension: str | None
    inferred_dtypes: tuple[str, ...]
    confidence: Confidence
    is_numba_decorated: bool
    is_metal_decorated: bool
    patterns: tuple[str, ...] = ()
    """Detected pattern tags, e.g. "nested_loop", "reduction", "stencil",
    "monte_carlo", "pairwise_distance", "image_op", "elementwise_chain"."""

    def to_json_dict(self) -> dict:
        return {
            "file": self.file,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "qualified_name": self.qualified_name,
            "decorator": self.decorator,
            "reasons": list(self.reasons),
            "unknowns": list(self.unknowns),
            "blockers": list(self.blockers),
            "parallel_dimension": self.parallel_dimension,
            "inferred_dtypes": list(self.inferred_dtypes),
            "confidence": self.confidence.value,
            "is_numba_decorated": self.is_numba_decorated,
            "is_metal_decorated": self.is_metal_decorated,
            "patterns": list(self.patterns),
        }


@dataclass(frozen=True, slots=True)
class ScanError:
    """Records that one file could not be fully analyzed, without aborting
    the rest of the scan (spec: "Unsupported syntax must never crash the
    complete scan")."""

    file: str
    message: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    root: str
    candidates: tuple[Candidate, ...]
    errors: tuple[ScanError, ...]
    files_scanned: int


# ---------------------------------------------------------------------------
# Compatibility analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """Result of running the real numba-metal typing+lowering dry-run (or,
    for undecorated Python, a static-only judgment) against one candidate.
    """

    qualified_name: str
    file: str
    line_start: int
    status: CompatibilityStatus
    supported_features: tuple[str, ...]
    blockers: tuple[str, ...]
    recommendation: str | None
    raw_error: str | None = None

    def to_json_dict(self) -> dict:
        return {
            "qualified_name": self.qualified_name,
            "file": self.file,
            "line_start": self.line_start,
            "status": self.status.value,
            "supported_features": list(self.supported_features),
            "blockers": list(self.blockers),
            "recommendation": self.recommendation,
            "raw_error": self.raw_error,
        }


# ---------------------------------------------------------------------------
# Timing statistics / comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimingStats:
    """Descriptive statistics over a set of repeated-run durations, all in
    nanoseconds. Never collapses to a single number -- median/min/max/mean/
    stdev/p90/p95 are all kept, and `n` plus `outliers_ns` make it possible
    to see that outliers exist without them being silently removed."""

    samples_ns: tuple[int, ...]
    median_ns: float
    mean_ns: float
    min_ns: int
    max_ns: int
    stdev_ns: float
    p90_ns: float | None
    p95_ns: float | None
    n: int
    outliers_ns: tuple[int, ...] = ()

    @staticmethod
    def from_samples(samples_ns: list[int] | tuple[int, ...]) -> TimingStats:
        import statistics

        samples = tuple(samples_ns)
        if not samples:
            raise ValueError("TimingStats.from_samples requires at least one sample")
        n = len(samples)
        median = float(statistics.median(samples))
        mean = float(statistics.mean(samples))
        stdev = float(statistics.stdev(samples)) if n > 1 else 0.0
        sorted_samples = sorted(samples)

        def _percentile(p: float) -> float | None:
            if n < 2:
                return None
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            if f == c:
                return float(sorted_samples[f])
            d0 = sorted_samples[f] * (c - k)
            d1 = sorted_samples[c] * (k - f)
            return float(d0 + d1)

        # Outliers: samples more than 3 stdev from the mean, reported (not
        # removed) per the spec's explicit "without silently removing them".
        outliers = tuple(s for s in samples if stdev > 0 and abs(s - mean) > 3 * stdev)
        return TimingStats(
            samples_ns=samples,
            median_ns=median,
            mean_ns=mean,
            min_ns=min(samples),
            max_ns=max(samples),
            stdev_ns=stdev,
            p90_ns=_percentile(0.90),
            p95_ns=_percentile(0.95),
            n=n,
            outliers_ns=outliers,
        )

    def to_json_dict(self) -> dict:
        return {
            "samples_ns": list(self.samples_ns),
            "median_ns": self.median_ns,
            "mean_ns": self.mean_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "stdev_ns": self.stdev_ns,
            "p90_ns": self.p90_ns,
            "p95_ns": self.p95_ns,
            "n": self.n,
            "outliers_ns": list(self.outliers_ns),
        }


@dataclass(frozen=True, slots=True)
class RegimeComparison:
    """CPU vs Metal timing for exactly one of COLD/WARM/STEADY_STATE. Never
    merged with the other two regimes -- see ComparisonResult."""

    mode: MeasurementMode
    cpu: TimingStats | None
    metal: TimingStats | None
    speedup: float | None  # cpu.median_ns / metal.median_ns, if both present
    preliminary: bool
    """True when n < some minimum (e.g. a single run) -- spec: "Do not
    calculate a speedup from a single run unless explicitly requested.
    Label single-run output as preliminary." """

    def to_json_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "cpu": self.cpu.to_json_dict() if self.cpu else None,
            "metal": self.metal.to_json_dict() if self.metal else None,
            "speedup": self.speedup,
            "preliminary": self.preliminary,
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    qualified_name: str
    file: str
    line_start: int
    regimes: tuple[RegimeComparison, ...]
    whole_program_speedup: float | None
    whole_program_preliminary: bool
    warmup_runs: int
    measurement_runs: int
    profiler_overhead_ns: int | None
    """Estimated overhead this profiler itself added while measuring, per
    spec section 18's "The profiler must report its own estimated
    overhead." None when not measured (e.g. a pure static comparison)."""

    def to_json_dict(self) -> dict:
        return {
            "qualified_name": self.qualified_name,
            "file": self.file,
            "line_start": self.line_start,
            "regimes": [r.to_json_dict() for r in self.regimes],
            "whole_program_speedup": self.whole_program_speedup,
            "whole_program_preliminary": self.whole_program_preliminary,
            "warmup_runs": self.warmup_runs,
            "measurement_runs": self.measurement_runs,
            "profiler_overhead_ns": self.profiler_overhead_ns,
        }


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    passed: bool
    elements_compared: int
    elements_total: int
    sampled: bool
    """True when elements_compared < elements_total (spec: "Sampled
    comparison for extremely large outputs, with the sampling clearly
    reported")."""
    max_abs_error: float | None
    max_rel_error: float | None
    atol: float | None
    rtol: float | None
    cpu_dtype: str
    metal_dtype: str
    shape_match: bool
    dtype_match: bool
    nan_mismatch: bool
    inf_mismatch: bool
    failure_reason: str | None = None

    def to_json_dict(self) -> dict:
        return {
            "passed": self.passed,
            "elements_compared": self.elements_compared,
            "elements_total": self.elements_total,
            "sampled": self.sampled,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "atol": self.atol,
            "rtol": self.rtol,
            "cpu_dtype": self.cpu_dtype,
            "metal_dtype": self.metal_dtype,
            "shape_match": self.shape_match,
            "dtype_match": self.dtype_match,
            "nan_mismatch": self.nan_mismatch,
            "inf_mismatch": self.inf_mismatch,
            "failure_reason": self.failure_reason,
        }


# ---------------------------------------------------------------------------
# Opportunity scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    name: str
    value: float
    max_value: float
    explanation: str


@dataclass(frozen=True, slots=True)
class OpportunityScore:
    qualified_name: str
    components: tuple[ScoreComponent, ...]
    total: float
    max_total: float
    confidence: Confidence
    evidence: EvidenceKind
    """MEASURED when built from a ComparisonResult; STATIC_POTENTIAL when
    built from scanner/compatibility evidence alone -- rendered as
    "OPPORTUNITY SCORE" only in the MEASURED case, "POTENTIAL" otherwise,
    per spec section 13's "call the result POTENTIAL, not an estimated
    speedup" instruction."""

    def to_json_dict(self) -> dict:
        return {
            "qualified_name": self.qualified_name,
            "components": [
                {
                    "name": c.name,
                    "value": c.value,
                    "max_value": c.max_value,
                    "explanation": c.explanation,
                }
                for c in self.components
            ],
            "total": self.total,
            "max_total": self.max_total,
            "confidence": self.confidence.value,
            "evidence": self.evidence.value,
        }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Recommendation:
    text: str
    direction: RecommendationDirection
    qualified_name: str
    file: str
    line_start: int
    supporting_measurement: str
    confidence: Confidence
    evidence: EvidenceKind
    how_to_verify: str
    rule_id: str
    """Which deterministic rule in recommendations.py produced this, so
    output is reproducible/traceable back to source, never an opaque LLM
    guess (spec: "Do not use an LLM to create the core recommendations")."""

    def to_json_dict(self) -> dict:
        return {
            "text": self.text,
            "direction": self.direction.value,
            "qualified_name": self.qualified_name,
            "file": self.file,
            "line_start": self.line_start,
            "supporting_measurement": self.supporting_measurement,
            "confidence": self.confidence.value,
            "evidence": self.evidence.value,
            "how_to_verify": self.how_to_verify,
            "rule_id": self.rule_id,
        }


# ---------------------------------------------------------------------------
# Sampling profiler (folded stacks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldedStack:
    """One aggregated call-stack path from the statistical sampler, in the
    conventional folded-stack format ("main;caller;callee")."""

    frames: tuple[str, ...]
    count: int

    def folded_line(self) -> str:
        return f"{';'.join(self.frames)} {self.count}"


@dataclass(frozen=True, slots=True)
class SamplingResult:
    stacks: tuple[FoldedStack, ...]
    total_samples: int
    interval_ms: float
    dropped_samples: int
    profiler_overhead_ns: int | None

    def to_folded_text(self) -> str:
        return "\n".join(s.folded_line() for s in self.stacks)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Advisor-level device/software metadata snapshot -- distinct from
    (but sourced from) numba_metal.runtime.device.DeviceInfo /
    benchmarks/common.py's get_environment_info(); kept as its own
    dataclass here so profile.json's schema does not depend on internal
    numba-metal dataclass shapes remaining stable."""

    device_name: str | None
    apple_silicon_family: str | None
    macos_version: str | None
    python_version: str
    numba_version: str
    numba_metal_version: str
    has_unified_memory: bool | None

    def to_json_dict(self) -> dict:
        return {
            "device_name": self.device_name,
            "apple_silicon_family": self.apple_silicon_family,
            "macos_version": self.macos_version,
            "python_version": self.python_version,
            "numba_version": self.numba_version,
            "numba_metal_version": self.numba_metal_version,
            "has_unified_memory": self.has_unified_memory,
        }


PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Report:
    """The complete, self-contained unit saved as profile.json and
    reloaded by `numba-metal advisor report`. Rendering from this object
    must be deterministic (Test 13: JSON round trip)."""

    schema_version: int
    generated_at_ns: int
    device: DeviceInfo
    events: tuple[Event, ...]
    candidates: tuple[Candidate, ...]
    compatibility: tuple[CompatibilityResult, ...]
    comparisons: tuple[ComparisonResult, ...]
    correctness: tuple[CorrectnessResult, ...]
    scores: tuple[OpportunityScore, ...]
    recommendations: tuple[Recommendation, ...]
    sampling: SamplingResult | None
    source_paths_included: bool
    dropped_event_count: int = 0

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "generated_at_ns": self.generated_at_ns,
            "device": self.device.to_json_dict(),
            "events": [e.to_json_dict() for e in self.events],
            "candidates": [c.to_json_dict() for c in self.candidates],
            "compatibility": [c.to_json_dict() for c in self.compatibility],
            "comparisons": [c.to_json_dict() for c in self.comparisons],
            "correctness": [c.to_json_dict() for c in self.correctness],
            "scores": [s.to_json_dict() for s in self.scores],
            "recommendations": [r.to_json_dict() for r in self.recommendations],
            "sampling": (
                {
                    "stacks": [
                        {"frames": list(s.frames), "count": s.count}
                        for s in self.sampling.stacks
                    ],
                    "total_samples": self.sampling.total_samples,
                    "interval_ms": self.sampling.interval_ms,
                    "dropped_samples": self.sampling.dropped_samples,
                    "profiler_overhead_ns": self.sampling.profiler_overhead_ns,
                }
                if self.sampling is not None
                else None
            ),
            "source_paths_included": self.source_paths_included,
            "dropped_event_count": self.dropped_event_count,
        }
