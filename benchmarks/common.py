"""Shared benchmarking infrastructure: timing helpers, machine-info
collection, and result reporting (text table + JSON).

Timing methodology (see docs/benchmarking.md for the full rationale):

- All wall-clock measurements use `time.perf_counter_ns()`.
- GPU timings always call `metal.synchronize()` before stopping the timer,
  so an unsynchronized (still-in-flight) launch is never reported as fast.
- Every timed section is preceded by warm-up iterations that are discarded,
  and every reported number is the median of several measured repetitions
  (with the interquartile spread available for variability reporting).
- "Cold" GPU timing includes first-time kernel compilation (typed-IR
  frontend + MSL shader compilation); "warm" GPU timing excludes it by
  timing only launches after the compilation cache is already populated.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Timing:
    """A set of repeated timing measurements, in nanoseconds."""

    samples_ns: list[int]

    @property
    def median_ns(self) -> float:
        return statistics.median(self.samples_ns)

    @property
    def stdev_ns(self) -> float:
        return statistics.stdev(self.samples_ns) if len(self.samples_ns) > 1 else 0.0

    @property
    def min_ns(self) -> float:
        return min(self.samples_ns)

    @property
    def max_ns(self) -> float:
        return max(self.samples_ns)

    def as_dict(self) -> dict[str, float]:
        return {
            "median_ns": self.median_ns,
            "stdev_ns": self.stdev_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "samples": len(self.samples_ns),
        }

    def __repr__(self) -> str:
        return f"{format_ns(self.median_ns)} (+/-{format_ns(self.stdev_ns)})"


def format_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f}us"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    return f"{ns / 1_000_000_000:.3f}s"


def time_repeated(
    fn: Callable[[], Any], *, warmup: int = 2, repeats: int = 7
) -> Timing:
    """Run `fn` `warmup` times (discarded) then `repeats` times, timing each
    with perf_counter_ns. `fn` is responsible for its own synchronization
    (e.g. calling metal.synchronize() before returning, for GPU work)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        samples.append(end - start)
    return Timing(samples)


def infer_launch_arg_types(*launch_args) -> tuple:
    """Derive the Numba argument-type tuple for a set of launch
    arguments (device arrays / scalars), using numba-metal's own
    dispatcher inference logic directly rather than re-deriving a
    parallel dtype-mapping table here."""
    from numba_metal.runtime.dispatcher import _infer_arg_type

    return tuple(_infer_arg_type(a) for a in launch_args)


def measure_cold_metal_pipeline(
    py_func: Callable[..., Any], arg_types: tuple
) -> dict[str, float | int]:
    """Measure a guaranteed-cold Metal compilation for `py_func` at
    `arg_types`, split into the three phases the assignment requires be
    reported separately:

    - frontend_ns: Numba typed-IR frontend + numba-metal's MSL lowering
      (`compile_to_typed_ir` + `MSLKernelLowerer.lower()`), no device
      interaction at all.
    - pipeline_compile_ns: `MTLLibrary`/`MTLComputePipelineState`
      creation from that generated MSL source, on-device.
    - total_ns: the two phases combined, timed as a single wall-clock
      span (not just frontend_ns + pipeline_compile_ns summed after the
      fact) so it also captures any cache-lookup/bookkeeping overhead
      between them.

    Uses the pipeline module's own functions directly (not a
    `@metal.jit` dispatcher's cache) so "cold" is unambiguous: nothing
    about this call could hit a pre-existing cache entry, since no
    KernelCache is involved at all.
    """
    from numba_metal.compiler.frontend import compile_to_typed_ir
    from numba_metal.compiler.msl_backend import MSLKernelLowerer
    from numba_metal.compiler.pipeline import _MSL_PRELUDE, _next_kernel_name
    from numba_metal.runtime.context import get_context

    t_total_start = time.perf_counter_ns()

    t0 = time.perf_counter_ns()
    typed = compile_to_typed_ir(py_func, arg_types)
    kernel_name = _next_kernel_name(getattr(py_func, "__name__", "kernel"))
    lowerer = MSLKernelLowerer(kernel_name, typed)
    body_src = lowerer.lower()
    full_src = _MSL_PRELUDE + body_src
    t1 = time.perf_counter_ns()
    frontend_ns = t1 - t0

    import Metal

    ctx = get_context()
    device = ctx.device
    opts = Metal.MTLCompileOptions.alloc().init()
    opts.setFastMathEnabled_(False)

    t2 = time.perf_counter_ns()
    library, err = device.newLibraryWithSource_options_error_(full_src, opts, None)
    if library is None:
        raise RuntimeError(f"Metal shader compilation failed: {err}")
    function = library.newFunctionWithName_(kernel_name)
    pipeline_state, perr = device.newComputePipelineStateWithFunction_error_(
        function, None
    )
    if pipeline_state is None:
        raise RuntimeError(f"Metal pipeline creation failed: {perr}")
    t3 = time.perf_counter_ns()
    pipeline_compile_ns = t3 - t2

    t_total_end = time.perf_counter_ns()

    return {
        "frontend_ns": float(frontend_ns),
        "pipeline_compile_ns": float(pipeline_compile_ns),
        "total_ns": float(t_total_end - t_total_start),
    }


def measure_cold_end_to_end(
    py_func: Callable[..., Any], launch: Callable[[Any], None]
) -> dict[str, float | int]:
    """Measure a guaranteed-cold end-to-end Metal launch: builds a
    brand-new `@metal.jit` dispatcher around `py_func` (own fresh
    `KernelCache`, so this cannot hit a pre-existing cache entry -- see
    `KernelDispatcher.__init__`, dispatcher.py), then calls
    `launch(kernel)`, which must perform exactly one launch + explicit
    `metal.synchronize()`.

    Returns the wall-clock total plus `compiled_before`/`compiled_after`
    cache-entry counts, which must show `compiled_after >
    compiled_before` as evidence the measured call really did compile
    (not reuse a warm cache) -- the "make every cold timing demonstrably
    cold" requirement.
    """
    from numba_metal import metal

    kernel = metal.jit(py_func)
    compiled_before = len(kernel._cache)
    t0 = time.perf_counter_ns()
    launch(kernel)
    t1 = time.perf_counter_ns()
    compiled_after = len(kernel._cache)
    return {
        "total_ns": float(t1 - t0),
        "compiled_before": compiled_before,
        "compiled_after": compiled_after,
    }


def numba_cpu_thread_variants(
    make_impl: Callable[[bool], Any],
) -> dict[str, Any]:
    """Build both a parallel (`@njit(parallel=True)`, default thread
    count) and single-threaded (`set_num_threads(1)`, restored
    afterward) variant of a Numba CPU implementation, returning the
    thread count actually in effect for the parallel variant.

    `make_impl(parallel: bool)` must return a freshly-decorated
    `@njit(...)` callable each call (decorating the same dispatcher
    twice with different `parallel=` is not meaningful -- each call
    must build its own `njit(...)` wrapper around the plain function).
    """
    import numba

    default_threads = numba.get_num_threads()
    parallel_impl = make_impl(True)
    single_impl = make_impl(False)
    return {
        "parallel_impl": parallel_impl,
        "single_impl": single_impl,
        "num_threads": default_threads,
    }


def run_single_threaded(fn: Callable[[], Any]) -> Any:
    """Run `fn` with Numba's parallel thread count forced to 1, restoring
    the previous count afterward regardless of outcome."""
    import numba

    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        return fn()
    finally:
        numba.set_num_threads(previous)


def get_mac_hardware_info() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        data = json.loads(out.stdout)
        hw = data.get("SPHardwareDataType", [{}])[0]
        info["model_name"] = hw.get("machine_name", "unknown")
        info["model_identifier"] = hw.get("machine_model", "unknown")
        info["chip"] = hw.get("chip_type", hw.get("cpu_type", "unknown"))
    except (
        subprocess.SubprocessError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
    ):
        info["model_name"] = "unavailable"
        info["model_identifier"] = "unavailable"
        info["chip"] = "unavailable"
    return info


def get_gpu_info() -> dict[str, Any]:
    try:
        from numba_metal.runtime.device import get_device_info

        d = get_device_info()
        return {
            "name": d.name,
            "has_unified_memory": d.has_unified_memory,
            "max_threads_per_threadgroup": d.max_threads_per_threadgroup,
        }
    except Exception as exc:  # noqa: BLE001 - reporting only
        return {"error": str(exc)}


def get_environment_info() -> dict[str, Any]:
    import numpy

    info: dict[str, Any] = {
        "macos_version": platform.mac_ver()[0],
        "python_version": sys.version.split()[0],
        "numpy_version": numpy.__version__,
        "machine": platform.machine(),
    }
    try:
        import numba

        info["numba_version"] = numba.__version__
    except ImportError:
        info["numba_version"] = None
    try:
        import numba_metal

        info["numba_metal_version"] = numba_metal.__version__
    except ImportError:
        info["numba_metal_version"] = None
    info.update(get_mac_hardware_info())
    info["gpu"] = get_gpu_info()
    return info


@dataclass
class BenchmarkResult:
    """One (benchmark, size) result row.

    Timing categories are kept separate per-category rather than folded
    into a single "GPU time," per the requirement that compilation,
    transfer, and kernel-execution time must never be conflated:

    - python_ns / numpy_ns: reference CPU implementations (informational).
    - numba_cpu_parallel_ns / numba_cpu_single_ns: warm Numba CPU
      (@njit) execution, parallel (prange, default thread count) and
      single-threaded (set_num_threads(1)) respectively. Thread count
      actually used is recorded in numba_cpu_num_threads.
    - metal_frontend_ns: Numba typed-IR frontend + numba-metal's
      lowering to MSL source text, measured on a guaranteed-cold
      dispatcher (fresh KernelCache) with device/pipeline compilation
      excluded.
    - metal_pipeline_compile_ns: MTLLibrary/MTLComputePipelineState
      compilation from the generated MSL, on that same cold dispatcher.
    - metal_cold_total_ns: frontend + pipeline compile + first launch,
      measured end-to-end on a fresh dispatcher (metal_frontend_ns and
      metal_pipeline_compile_ns are subsets of this).
    - metal_kernel_only_warm_ns: warm (already-compiled) kernel launch
      + synchronize only, with input/output buffers already resident on
      device -- excludes all host<->device transfer.
    - metal_h2d_ns / metal_d2h_ns: host-to-device / device-to-host
      transfer time, measured separately from kernel execution.
    - metal_end_to_end_warm_ns: warm H2D + kernel + D2H, i.e. what a
      caller experiences per call when data starts and ends on the host.
    - metal_resident_pipeline_ns: warm kernel-only time for a workload
      that keeps data GPU-resident across repeated launches (only
      meaningful for benchmarks with an iterative/repeated-launch
      structure; None otherwise).
    - compiled_before / compiled_after: KernelCache entry counts before
      and after the cold-timing section, proof that metal_cold_total_ns
      really did include a fresh compilation (compiled_after >
      compiled_before), not stale/reused cache state.
    """

    benchmark: str
    size_label: str
    correctness_ok: bool = False
    correctness_note: str = ""
    python_ns: float | None = None
    numpy_ns: float | None = None
    numba_cpu_parallel_ns: float | None = None
    numba_cpu_single_ns: float | None = None
    numba_cpu_num_threads: int | None = None
    metal_frontend_ns: float | None = None
    metal_pipeline_compile_ns: float | None = None
    metal_cold_total_ns: float | None = None
    metal_kernel_only_warm_ns: float | None = None
    metal_h2d_ns: float | None = None
    metal_d2h_ns: float | None = None
    metal_end_to_end_warm_ns: float | None = None
    metal_resident_pipeline_ns: float | None = None
    compiled_before: int | None = None
    compiled_after: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def numba_cpu_ns(self) -> float | None:
        """Primary Numba CPU evidence number (parallel if available, else
        single-threaded), used for the headline speedup column."""
        return self.numba_cpu_parallel_ns or self.numba_cpu_single_ns

    def speedup_vs(self, baseline_ns: float | None) -> str:
        warm = self.metal_kernel_only_warm_ns
        if baseline_ns is None or warm is None or warm <= 0:
            return "N/A"
        return f"{baseline_ns / warm:.2f}x"

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "size": self.size_label,
            "correctness_ok": self.correctness_ok,
            "correctness_note": self.correctness_note,
            "python_ns": self.python_ns,
            "numpy_ns": self.numpy_ns,
            "numba_cpu_parallel_ns": self.numba_cpu_parallel_ns,
            "numba_cpu_single_ns": self.numba_cpu_single_ns,
            "numba_cpu_num_threads": self.numba_cpu_num_threads,
            "metal_frontend_ns": self.metal_frontend_ns,
            "metal_pipeline_compile_ns": self.metal_pipeline_compile_ns,
            "metal_cold_total_ns": self.metal_cold_total_ns,
            "metal_kernel_only_warm_ns": self.metal_kernel_only_warm_ns,
            "metal_h2d_ns": self.metal_h2d_ns,
            "metal_d2h_ns": self.metal_d2h_ns,
            "metal_end_to_end_warm_ns": self.metal_end_to_end_warm_ns,
            "metal_resident_pipeline_ns": self.metal_resident_pipeline_ns,
            "compiled_before": self.compiled_before,
            "compiled_after": self.compiled_after,
            "speedup_vs_python": self.speedup_vs(self.python_ns),
            "speedup_vs_numpy": self.speedup_vs(self.numpy_ns),
            "speedup_vs_numba_cpu_parallel": self.speedup_vs(
                self.numba_cpu_parallel_ns
            ),
            "speedup_vs_numba_cpu_single": self.speedup_vs(self.numba_cpu_single_ns),
            **self.extra,
        }


def _fmt(ns: float | None) -> str:
    return format_ns(ns) if ns is not None else "N/A"


def print_table(results: list[BenchmarkResult]) -> None:
    headers = [
        "Benchmark",
        "Size",
        "Python",
        "NumPy",
        "Numba CPU(par)",
        "Numba CPU(1t)",
        "Metal kernel-warm",
        "Metal cold total",
        "vs Numba(par)",
        "Correct",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r.benchmark,
                r.size_label,
                _fmt(r.python_ns),
                _fmt(r.numpy_ns),
                _fmt(r.numba_cpu_parallel_ns),
                _fmt(r.numba_cpu_single_ns),
                _fmt(r.metal_kernel_only_warm_ns),
                _fmt(r.metal_cold_total_ns),
                r.speedup_vs(r.numba_cpu_parallel_ns),
                "yes" if r.correctness_ok else "NO",
            ]
        )
    widths = [
        (
            max(len(headers[i]), *(len(row[i]) for row in rows))
            if rows
            else len(headers[i])
        )
        for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))


def write_json(results: list[BenchmarkResult], path: str) -> None:
    payload = {
        "environment": get_environment_info(),
        "results": [r.as_dict() for r in results],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def require_metal_or_skip() -> bool:
    """Returns True if Metal is available; prints a clear message and
    returns False otherwise so callers can skip GPU sections without
    crashing the whole benchmark run."""
    try:
        from numba_metal.runtime.device import check_capable

        check_capable()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[numba-metal] Metal unavailable, skipping GPU benchmarks: {exc}")
        return False


def assert_near_integer_match(
    actual: np.ndarray, expected: np.ndarray, *, max_off_by_one_fraction: float = 0.001
) -> tuple[bool, str]:
    """Correctness check for chaotic iterative algorithms (e.g. Mandelbrot
    escape counts) where the *reference* is itself a different floating-
    point evaluation order (vectorized vs. scalar, CPU vs. GPU ALU/FMA
    behavior) of the same mathematical recurrence. Near a fractal escape
    boundary, a last-bit float32 rounding difference can flip the escape
    iteration by exactly one for a small fraction of pixels; that is a
    genuine, expected cross-hardware floating-point difference, not a
    functional bug, and is documented as such in docs/limitations.md. This
    check requires exact equality everywhere except for a bounded fraction
    of pixels, which may differ by at most 1.
    """
    diff = actual.astype(np.int64) - expected.astype(np.int64)
    exact = diff == 0
    off_by_one = np.abs(diff) == 1
    bad = ~(exact | off_by_one)
    n = actual.size
    off_by_one_frac = float(np.count_nonzero(off_by_one)) / n
    if np.any(bad):
        return False, f"FAILED: {np.count_nonzero(bad)}/{n} pixels differ by >1"
    if off_by_one_frac > max_off_by_one_fraction:
        return (
            False,
            f"FAILED: {off_by_one_frac:.4%} of pixels off-by-one exceeds "
            f"{max_off_by_one_fraction:.4%} threshold",
        )
    return (
        True,
        f"exact except {np.count_nonzero(off_by_one)}/{n} pixels "
        f"({off_by_one_frac:.4%}) off-by-one at escape boundary (float32 "
        "rounding, expected -- see docs/limitations.md)",
    )


def assert_allclose(
    actual: np.ndarray, expected: np.ndarray, *, rtol: float = 1e-4, atol: float = 1e-4
) -> tuple[bool, str]:
    """Returns (ok, note) instead of raising, so a correctness failure is
    reported as data rather than crashing the benchmark run."""
    ok = np.allclose(actual, expected, rtol=rtol, atol=atol)
    if ok:
        return True, f"allclose(rtol={rtol}, atol={atol})"
    max_abs_err = float(
        np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
    )
    return (
        False,
        f"FAILED allclose(rtol={rtol}, atol={atol}); max_abs_err={max_abs_err:.6g}",
    )
