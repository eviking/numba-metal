"""CPU vs Metal timing comparison, split into COLD / WARM / STEADY_STATE
regimes that are never merged into a single number.

`benchmarks/common.py` (this repo's own benchmark timing infrastructure)
is NOT importable from an installed `numba-metal` package -- `benchmarks/`
is a top-level repo directory, not shipped inside the `numba_metal`
package, so advisor code (which must work from a plain `pip install
numba-metal`) cannot depend on it at runtime. This module therefore
reimplements the small amount of cold-compile timing logic it needs
(`_measure_cold_metal_pipeline` below) directly against
`numba_metal.compiler.frontend`/`msl_backend`/`pipeline`'s own public/
semi-public functions -- the same functions `benchmarks/common.py` itself
calls, so the methodology stays identical even though the code is not
literally shared. See docs/advisor.md's "Known limitations" for this
note.

COLD includes source generation, Metal compilation, and pipeline
creation (measured with no `KernelCache` involved at all, so "cold" is
unambiguous). WARM reuses an already-compiled `@metal.jit` dispatcher but
may still allocate/synchronize buffers per call. STEADY STATE repeats
warm execution many times (the same warmup+repeats loop as WARM, at a
higher repeat count), matching this repo's own median-of-repeats
methodology (`benchmarks/common.py`'s `time_repeated`).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from numba_metal.advisor.models import (
    ComparisonResult,
    MeasurementMode,
    RegimeComparison,
    RooflineClassification,
    RooflinePerformanceRegime,
    TimingStats,
)

DEFAULT_WARMUP_RUNS = 2
DEFAULT_MEASUREMENT_RUNS = 7
_MIN_RUNS_FOR_SPEEDUP = 2
"""Below this many measurement runs, a speedup number is marked
preliminary rather than asserted -- spec: "Do not calculate a speedup
from a single run unless explicitly requested. Label single-run output
as preliminary." A single run (n=1) always sets preliminary=True; 2+ is
treated as a real (if still noisy) sample, since TimingStats itself
already reports stdev/spread so the caller can judge reliability."""


def _time_ns(fn: Callable[[], None], *, warmup: int, repeats: int) -> list[int]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    return samples


def time_cpu_warm(
    fn: Callable[[], None],
    *,
    warmup: int = DEFAULT_WARMUP_RUNS,
    repeats: int = DEFAULT_MEASUREMENT_RUNS,
) -> TimingStats:
    """Time a plain CPU (Python or already-JIT-compiled Numba) callable
    across `repeats` repetitions after `warmup` discarded runs -- there is
    no separate "cold" CPU regime here: Numba's own JIT warmup is the
    "cold" cost for CPU, folded into the discarded warmup runs exactly as
    `benchmarks/*.py` already does, since the spec's COLD/WARM/STEADY_STATE
    distinction is specifically about numba-metal's compilation pipeline,
    not Numba's own CPU JIT."""
    samples = _time_ns(fn, warmup=warmup, repeats=repeats)
    return TimingStats.from_samples(samples)


def time_metal_warm(
    fn: Callable[[], None],
    *,
    warmup: int = DEFAULT_WARMUP_RUNS,
    repeats: int = DEFAULT_MEASUREMENT_RUNS,
) -> TimingStats:
    """Time an already-compiled `@metal.jit` kernel launch (`fn` must
    call `metal.synchronize()` itself before returning, matching every
    other timing helper in this codebase's convention)."""
    samples = _time_ns(fn, warmup=warmup, repeats=repeats)
    return TimingStats.from_samples(samples)


def _measure_cold_metal_pipeline_ns(py_func, arg_types: tuple) -> int:
    """One guaranteed-cold Metal compilation: Numba typed-IR frontend +
    numba-metal MSL lowering + on-device MTLLibrary/MTLComputePipelineState
    creation, timed as a single wall-clock span with no `KernelCache`
    involved at all (so there is no cache to hit, unlike a `@metal.jit`
    dispatcher's normal launch path) -- methodology matches
    `benchmarks/common.py`'s `measure_cold_metal_pipeline`, reimplemented
    here since that module is not importable from an installed package
    (see this module's docstring)."""
    from numba_metal.compiler.frontend import compile_to_typed_ir
    from numba_metal.compiler.msl_backend import MSLKernelLowerer
    from numba_metal.compiler.pipeline import _MSL_PRELUDE, _next_kernel_name
    from numba_metal.runtime.context import get_context

    t0 = time.perf_counter_ns()
    typed = compile_to_typed_ir(py_func, arg_types)
    kernel_name = _next_kernel_name(getattr(py_func, "__name__", "kernel"))
    lowerer = MSLKernelLowerer(kernel_name, typed)
    body_src = lowerer.lower()
    full_src = _MSL_PRELUDE + body_src

    import Metal

    ctx = get_context()
    device = ctx.device
    opts = Metal.MTLCompileOptions.alloc().init()
    opts.setFastMathEnabled_(False)

    library, err = device.newLibraryWithSource_options_error_(full_src, opts, None)
    if library is None:
        raise RuntimeError(f"Metal shader compilation failed: {err}")
    function = library.newFunctionWithName_(kernel_name)
    pipeline_state, perr = device.newComputePipelineStateWithFunction_error_(
        function, None
    )
    if pipeline_state is None:
        raise RuntimeError(f"Metal pipeline creation failed: {perr}")
    t1 = time.perf_counter_ns()
    return t1 - t0


def time_metal_cold(
    py_func, arg_types: tuple, *, repeats: int = DEFAULT_MEASUREMENT_RUNS
) -> TimingStats:
    """Time `repeats` independent cold compilations of the SAME kernel --
    there is deliberately no "warmup" here, since warming up a cold-
    compile measurement would defeat its purpose."""
    samples = [
        _measure_cold_metal_pipeline_ns(py_func, arg_types) for _ in range(repeats)
    ]
    return TimingStats.from_samples(samples)


def classify_roofline(
    *,
    metal_median_ns: float,
    bytes_per_call: int | None,
    flops_per_call: int | None,
) -> RooflineClassification | None:
    """Classify one measured Metal launch against this machine's own
    calibrated ceilings (`numba-metal advisor calibrate`), so a
    recommendation can explain WHY a kernel is fast or slow instead of
    only reporting a bare speedup number.

    Returns None (never a guessed classification) when:
    - bytes_per_call or flops_per_call was not supplied by the workload
      (this project does not attempt to infer memory traffic or FLOP
      counts from source -- see AdvisorWorkload's docstring), or
    - no local calibration file exists yet.

    The four regimes and the thresholds between them are not arbitrary:
    - DISPATCH_BOUND fires when the calibrated dispatch-overhead floor
      is itself a large share (>30%) of the measured time, regardless
      of arithmetic intensity -- verified this session that this
      overhead is almost entirely the commit()+waitUntilCompleted()
      round-trip, not GPU execution, and that batching launches onto
      one command buffer cuts it by roughly 2.5-2.7x (see
      recommendations.py's batching rule, which this classification
      feeds).
    - Otherwise, arithmetic intensity (flops_per_call / bytes_per_call)
      is compared against the calibrated roofline ridge point
      (compute ceiling / bandwidth ceiling) to decide whether bandwidth
      or compute is the binding constraint -- standard roofline-model
      reasoning, using this machine's own measured ceilings rather than
      published/theoretical hardware specs.
    - CACHE_BOUND is reported instead of an impossible >100%-of-DRAM-
      bandwidth figure when achieved bandwidth exceeds the calibrated
      DRAM ceiling -- a real, legitimate result when a kernel's working
      set is small enough to be served mostly from on-chip cache
      (verified directly this session: pairwise distance with inputs
      small enough to fit on-chip measured above the DRAM ceiling, and
      tracing it back confirmed genuine cache reuse, not a measurement
      error).
    """
    from numba_metal.advisor.calibration import calibration_path

    if bytes_per_call is None or flops_per_call is None or bytes_per_call <= 0:
        return None
    if not calibration_path().exists():
        return None

    import json

    calib = json.loads(calibration_path().read_text())
    dispatch_overhead_ns = calib.get("dispatch_overhead_ns")
    bw_ceiling_gbps = calib.get("memory_bandwidth_gbps")
    compute_ceiling_gflops = calib.get("float32_gflops_compute_bound")
    ridge = calib.get("roofline_ridge_flops_per_byte")
    device_name = calib.get("device_name")

    if metal_median_ns <= 0:
        return None

    arithmetic_intensity = flops_per_call / bytes_per_call
    achieved_bw_gbps = bytes_per_call / (metal_median_ns / 1e9) / 1e9
    achieved_gflops = flops_per_call / (metal_median_ns / 1e9) / 1e9

    dispatch_frac = (
        dispatch_overhead_ns / metal_median_ns
        if dispatch_overhead_ns is not None
        else None
    )
    bw_frac = achieved_bw_gbps / bw_ceiling_gbps if bw_ceiling_gbps else None
    gflops_frac = (
        achieved_gflops / compute_ceiling_gflops if compute_ceiling_gflops else None
    )

    if dispatch_frac is not None and dispatch_frac > 0.3:
        regime = RooflinePerformanceRegime.DISPATCH_BOUND
    elif ridge is not None and arithmetic_intensity >= ridge:
        regime = RooflinePerformanceRegime.COMPUTE_BOUND
    elif bw_frac is not None and bw_frac > 1.0:
        regime = RooflinePerformanceRegime.CACHE_BOUND
    elif bw_frac is not None:
        regime = RooflinePerformanceRegime.BANDWIDTH_BOUND
    else:
        regime = RooflinePerformanceRegime.UNKNOWN

    return RooflineClassification(
        regime=regime,
        arithmetic_intensity_flops_per_byte=arithmetic_intensity,
        dispatch_overhead_fraction=dispatch_frac,
        achieved_bandwidth_gbps=achieved_bw_gbps,
        bandwidth_ceiling_fraction=bw_frac,
        achieved_gflops=achieved_gflops,
        compute_ceiling_fraction=gflops_frac,
        calibration_device_name=device_name,
    )


def compare(
    *,
    qualified_name: str,
    file: str,
    line_start: int,
    cpu_fn: Callable[[], None] | None,
    metal_warm_fn: Callable[[], None] | None,
    py_func_for_cold=None,
    arg_types_for_cold: tuple | None = None,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    measurement_runs: int = DEFAULT_MEASUREMENT_RUNS,
    steady_state_runs: int | None = None,
    bytes_per_call: int | None = None,
    flops_per_call: int | None = None,
) -> ComparisonResult:
    """Run whichever of COLD/WARM/STEADY_STATE regimes have the inputs
    needed to measure them (a caller with no `py_func_for_cold` simply
    gets no COLD regime in the result, rather than a fabricated one).

    STEADY_STATE is a SEPARATE, independently-measured run at a higher
    repeat count (`steady_state_runs`, default `measurement_runs * 5`) --
    never merely WARM's own samples relabeled -- so a caller can see
    whether the distribution actually stabilizes over many more
    iterations (e.g. reuse effects, thermal/frequency scaling) rather
    than assuming WARM's smaller sample already represents steady state.
    Only measured when both `cpu_fn` and `metal_warm_fn` are given (a
    steady-state comparison with only one side present would have
    nothing to compare against, so it is skipped rather than reported
    with a missing half).

    Never merges regimes: `ComparisonResult.regimes` always keeps them as
    separate `RegimeComparison` entries.
    """
    if steady_state_runs is None:
        steady_state_runs = measurement_runs * 5
    profiler_overhead_start = time.perf_counter_ns()
    regimes: list[RegimeComparison] = []

    if py_func_for_cold is not None and arg_types_for_cold is not None:
        cold_metal = time_metal_cold(
            py_func_for_cold, arg_types_for_cold, repeats=measurement_runs
        )
        regimes.append(
            RegimeComparison(
                mode=MeasurementMode.COLD,
                cpu=None,
                metal=cold_metal,
                speedup=None,
                preliminary=cold_metal.n < _MIN_RUNS_FOR_SPEEDUP,
            )
        )

    warm_cpu_stats: TimingStats | None = None
    warm_metal_stats: TimingStats | None = None
    if cpu_fn is not None:
        warm_cpu_stats = time_cpu_warm(
            cpu_fn, warmup=warmup_runs, repeats=measurement_runs
        )
    if metal_warm_fn is not None:
        warm_metal_stats = time_metal_warm(
            metal_warm_fn, warmup=warmup_runs, repeats=measurement_runs
        )
    if warm_cpu_stats is not None or warm_metal_stats is not None:
        speedup = None
        if warm_cpu_stats is not None and warm_metal_stats is not None:
            if warm_metal_stats.median_ns > 0:
                speedup = warm_cpu_stats.median_ns / warm_metal_stats.median_ns
        n = min(
            (warm_cpu_stats.n if warm_cpu_stats else measurement_runs),
            (warm_metal_stats.n if warm_metal_stats else measurement_runs),
        )
        regimes.append(
            RegimeComparison(
                mode=MeasurementMode.WARM,
                cpu=warm_cpu_stats,
                metal=warm_metal_stats,
                speedup=speedup,
                preliminary=n < _MIN_RUNS_FOR_SPEEDUP,
            )
        )

    if cpu_fn is not None and metal_warm_fn is not None:
        steady_cpu_stats = time_cpu_warm(
            cpu_fn, warmup=warmup_runs, repeats=steady_state_runs
        )
        steady_metal_stats = time_metal_warm(
            metal_warm_fn, warmup=warmup_runs, repeats=steady_state_runs
        )
        steady_speedup = None
        if steady_metal_stats.median_ns > 0:
            steady_speedup = steady_cpu_stats.median_ns / steady_metal_stats.median_ns
        regimes.append(
            RegimeComparison(
                mode=MeasurementMode.STEADY_STATE,
                cpu=steady_cpu_stats,
                metal=steady_metal_stats,
                speedup=steady_speedup,
                preliminary=steady_state_runs < _MIN_RUNS_FOR_SPEEDUP,
            )
        )

    # Prefer STEADY_STATE for the headline whole-program number when
    # available (more repeats, more representative of long-run behavior);
    # fall back to WARM otherwise. Never COLD -- a cold compile is a
    # one-time cost, not representative of "the" speedup.
    whole_program_speedup = None
    whole_program_preliminary = True
    for mode in (MeasurementMode.STEADY_STATE, MeasurementMode.WARM):
        for r in regimes:
            if r.mode == mode and r.speedup is not None:
                whole_program_speedup = r.speedup
                whole_program_preliminary = r.preliminary
        if whole_program_speedup is not None:
            break

    profiler_overhead_ns = time.perf_counter_ns() - profiler_overhead_start
    # This "overhead" figure is deliberately a rough upper bound (the
    # entire time compare() itself took minus the actual measured
    # samples would be a tighter number, but the whole point of this
    # field is to give the caller SOME visibility into profiler cost,
    # not a precisely isolated one -- see spec section 18).
    measured_ns = sum(
        sum(r.cpu.samples_ns) if r.cpu else 0
        for r in regimes
        if r.mode != MeasurementMode.COLD
    ) + sum(sum(r.metal.samples_ns) if r.metal else 0 for r in regimes)
    profiler_overhead_ns = max(0, profiler_overhead_ns - measured_ns)

    # Same regime-preference order as whole_program_speedup above
    # (STEADY_STATE, then WARM, never COLD) -- the roofline
    # classification describes ongoing/repeated execution, and a cold
    # compile's timing would misrepresent that.
    roofline = None
    for mode in (MeasurementMode.STEADY_STATE, MeasurementMode.WARM):
        for r in regimes:
            if r.mode == mode and r.metal is not None:
                roofline = classify_roofline(
                    metal_median_ns=r.metal.median_ns,
                    bytes_per_call=bytes_per_call,
                    flops_per_call=flops_per_call,
                )
                break
        if roofline is not None:
            break

    return ComparisonResult(
        qualified_name=qualified_name,
        file=file,
        line_start=line_start,
        regimes=tuple(regimes),
        whole_program_speedup=whole_program_speedup,
        whole_program_preliminary=whole_program_preliminary,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        profiler_overhead_ns=profiler_overhead_ns,
        bytes_per_call=bytes_per_call,
        flops_per_call=flops_per_call,
        roofline=roofline,
    )
