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
    """One (benchmark, size) result row."""

    benchmark: str
    size_label: str
    correctness_ok: bool = False
    correctness_note: str = ""
    python_ns: float | None = None
    numpy_ns: float | None = None
    numba_cpu_ns: float | None = None
    metal_warm_ns: float | None = None
    metal_cold_ns: float | None = None
    metal_compile_ns: float | None = None
    metal_h2d_ns: float | None = None
    metal_d2h_ns: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def speedup_vs(self, baseline_ns: float | None) -> str:
        if baseline_ns is None or self.metal_warm_ns is None or self.metal_warm_ns <= 0:
            return "N/A"
        return f"{baseline_ns / self.metal_warm_ns:.2f}x"

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "size": self.size_label,
            "correctness_ok": self.correctness_ok,
            "correctness_note": self.correctness_note,
            "python_ns": self.python_ns,
            "numpy_ns": self.numpy_ns,
            "numba_cpu_ns": self.numba_cpu_ns,
            "metal_warm_ns": self.metal_warm_ns,
            "metal_cold_ns": self.metal_cold_ns,
            "metal_compile_ns": self.metal_compile_ns,
            "metal_h2d_ns": self.metal_h2d_ns,
            "metal_d2h_ns": self.metal_d2h_ns,
            "speedup_vs_python": self.speedup_vs(self.python_ns),
            "speedup_vs_numpy": self.speedup_vs(self.numpy_ns),
            "speedup_vs_numba_cpu": self.speedup_vs(self.numba_cpu_ns),
            **self.extra,
        }


def print_table(results: list[BenchmarkResult]) -> None:
    headers = [
        "Benchmark",
        "Size",
        "Python",
        "NumPy",
        "Numba CPU",
        "Metal warm",
        "Metal total",
        "vs Numba",
        "Correct",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r.benchmark,
                r.size_label,
                format_ns(r.python_ns) if r.python_ns is not None else "N/A",
                format_ns(r.numpy_ns) if r.numpy_ns is not None else "N/A",
                format_ns(r.numba_cpu_ns) if r.numba_cpu_ns is not None else "N/A",
                format_ns(r.metal_warm_ns) if r.metal_warm_ns is not None else "N/A",
                format_ns(r.metal_cold_ns) if r.metal_cold_ns is not None else "N/A",
                r.speedup_vs(r.numba_cpu_ns),
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
