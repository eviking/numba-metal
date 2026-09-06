"""Local device calibration: `numba-metal advisor calibrate`.

Measures dispatch overhead, cold-compile cost, buffer allocation
overhead, synchronization overhead, and rough float32 throughput/memory
bandwidth using the same timing methodology as comparison.py, storing
results as versioned JSON in a user cache directory.

Calibration results are hints, not promises (spec section 15): they are
never consulted by comparison.py/correctness.py, which always measure
project-specific numbers directly. `calibration.py`'s only consumer is
the advisor's own reporting (`numba-metal advisor calibrate` printing
its own results) and, optionally, a renderer that wants to show "this
machine's baseline dispatch overhead was Xus" as background context.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CALIBRATION_SCHEMA_VERSION = 1


def _cache_dir() -> Path:
    """`~/.cache/numba-metal/` on all platforms this runs on (macOS).
    This repo has no existing convention for a user cache directory
    (verified: no `platformdirs`/`appdirs` dependency, no prior
    `~/.cache` or `~/Library/Caches` usage anywhere in the codebase) --
    `~/.cache/numba-metal/` is a documented assumption, not a discovered
    convention; see docs/advisor.md's "Known limitations"."""
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "numba-metal"
    )


def calibration_path() -> Path:
    return _cache_dir() / "calibration.json"


@dataclass
class CalibrationResult:
    schema_version: int
    timestamp_ns: int
    device_name: str | None
    apple_silicon_family: str | None
    macos_version: str
    python_version: str
    numba_version: str | None
    numba_metal_version: str | None
    dispatch_overhead_ns: float
    cold_compile_ns: float
    buffer_alloc_ns: float
    sync_overhead_ns: float
    float32_gflops_memory_bound: float | None
    memory_bandwidth_gbps: float | None
    float32_gflops_compute_bound: float | None
    roofline_ridge_flops_per_byte: float | None
    crossover_elements_estimate: int | None

    def to_json_dict(self) -> dict:
        return asdict(self)


def _get_mac_hardware_info() -> dict[str, str]:
    try:
        out = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        data = json.loads(out.stdout)
        hw = data.get("SPHardwareDataType", [{}])[0]
        return {"chip": hw.get("chip_type", hw.get("cpu_type", "unknown"))}
    except (
        subprocess.SubprocessError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
    ):
        return {"chip": "unknown"}


def run_calibration() -> CalibrationResult:
    """Run every calibration measurement against the real Metal device.
    Raises `numba_metal.errors.UnsupportedPlatformError` (propagated
    unchanged, not caught here) if this machine cannot run Metal at all
    -- calibration has nothing meaningful to measure on such a machine,
    so failing loudly is correct, not a bug to work around."""
    from numba_metal import metal
    from numba_metal.runtime.context import get_context
    from numba_metal.runtime.device import get_device_info

    device_info = get_device_info()
    hw = _get_mac_hardware_info()

    import numpy as np

    n = 1_000_000
    x = np.random.default_rng(0).random(n).astype(np.float32)
    y = np.random.default_rng(1).random(n).astype(np.float32)

    @metal.jit
    def _calib_add(a, b, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] + b[i] * 2.0

    d_x = metal.to_device(x)
    d_y = metal.to_device(y)
    d_out = metal.to_device(np.zeros(n, dtype=np.float32))
    blocks = (n + 255) // 256

    # Cold compile: first launch of a never-before-seen dispatcher.
    t0 = time.perf_counter_ns()
    _calib_add[blocks, 256](d_x, d_y, d_out)
    metal.synchronize()
    cold_compile_ns = time.perf_counter_ns() - t0

    # Dispatch overhead: warm launches of a TINY kernel (1 element), so
    # the measured time is dominated by dispatch/encode/submit/sync
    # machinery rather than actual GPU compute.
    tiny_x = metal.to_device(np.zeros(1, dtype=np.float32))
    tiny_out = metal.to_device(np.zeros(1, dtype=np.float32))

    @metal.jit
    def _calib_tiny(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i]

    _calib_tiny[1, 1](tiny_x, tiny_out)
    metal.synchronize()
    warmup, repeats = 3, 20
    for _ in range(warmup):
        _calib_tiny[1, 1](tiny_x, tiny_out)
        metal.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        _calib_tiny[1, 1](tiny_x, tiny_out)
        metal.synchronize()
        samples.append(time.perf_counter_ns() - t0)
    dispatch_overhead_ns = float(sorted(samples)[len(samples) // 2])

    # Buffer allocation overhead: to_device() of a moderate-size array.
    alloc_samples = []
    for _ in range(10):
        t0 = time.perf_counter_ns()
        metal.to_device(np.zeros(100_000, dtype=np.float32))
        alloc_samples.append(time.perf_counter_ns() - t0)
    buffer_alloc_ns = float(sorted(alloc_samples)[len(alloc_samples) // 2])

    # Synchronization overhead: synchronize() with nothing outstanding.
    ctx = get_context()
    sync_samples = []
    for _ in range(10):
        t0 = time.perf_counter_ns()
        ctx.synchronize()
        sync_samples.append(time.perf_counter_ns() - t0)
    sync_overhead_ns = float(sorted(sync_samples)[len(sync_samples) // 2])

    # Rough float32 throughput: a large elementwise op, warm.
    #
    # Deliberately a SEPARATE, much larger problem size than the
    # dispatch-overhead measurement above (n=1,000,000 there is exactly
    # right for isolating per-launch overhead, since it's meant to stay
    # small). Reusing that same size here was a real bug: at
    # n=1,000,000 this kernel's actual GPU time is only slightly larger
    # than the ~148us dispatch-overhead floor already measured above, so
    # the computed GFLOPS/bandwidth numbers were mostly re-measuring
    # dispatch overhead, not sustained throughput -- verified directly
    # by re-running this same kernel at increasing sizes on real M4 Pro
    # hardware: n=1e6 => ~48 GB/s, n=1e7 => ~151 GB/s, n=5e7 => ~212
    # GB/s, n=1e8 => ~222 GB/s (approaching the M4 Pro's real published
    # unified-memory bandwidth, ~273 GB/s) -- i.e. the smaller size was
    # underreporting real achievable bandwidth by roughly 4-5x. Using a
    # 100,000,000-element problem here keeps dispatch overhead to a
    # small fraction of the measured time so the reported number
    # reflects genuine sustained throughput rather than overhead.
    n_throughput = 100_000_000
    x_t = np.random.default_rng(2).random(n_throughput).astype(np.float32)
    y_t = np.random.default_rng(3).random(n_throughput).astype(np.float32)
    d_x_t = metal.to_device(x_t)
    d_y_t = metal.to_device(y_t)
    d_out_t = metal.to_device(np.zeros(n_throughput, dtype=np.float32))
    blocks_t = (n_throughput + 255) // 256

    for _ in range(3):
        _calib_add[blocks_t, 256](d_x_t, d_y_t, d_out_t)
        metal.synchronize()
    throughput_samples = []
    for _ in range(10):
        t0 = time.perf_counter_ns()
        _calib_add[blocks_t, 256](d_x_t, d_y_t, d_out_t)
        metal.synchronize()
        throughput_samples.append(time.perf_counter_ns() - t0)
    median_ns = sorted(throughput_samples)[len(throughput_samples) // 2]
    # 2 flops per element (multiply-add), n_throughput elements. This
    # number is named `float32_gflops_memory_bound` (not just
    # `float32_gflops`) deliberately: `_calib_add` has arithmetic
    # intensity of only 2 flops / 12 bytes ~= 0.17 flops/byte, far below
    # this hardware's real roofline ridge point (~3 flops/byte, per the
    # compute-bound measurement below) -- so this kernel is memory-
    # bandwidth-bound, and the GFLOPS figure it produces reflects
    # "throughput achievable when also bandwidth-limited," NOT the
    # chip's real peak arithmetic rate. Using this number as a general
    # "compute ceiling" for a compute-bound kernel (high arithmetic
    # intensity, little memory traffic -- e.g. an escape-time fractal
    # with many loop iterations per pixel and one write) understates the
    # real achievable throughput by more than an order of magnitude:
    # verified directly on this M4 Pro, this kernel measures ~37 GFLOPS
    # while a genuinely compute-bound kernel (high arithmetic intensity,
    # ~0.5 bytes/flop) sustains ~680 GFLOPS on the same hardware.
    float32_gflops_memory_bound = (
        (2.0 * n_throughput) / (median_ns / 1e9) / 1e9 if median_ns > 0 else None
    )
    # 3 arrays of n_throughput float32 touched (2 read + 1 write) per launch.
    bytes_moved = 3 * n_throughput * 4
    memory_bandwidth_gbps = (
        bytes_moved / (median_ns / 1e9) / 1e9 if median_ns > 0 else None
    )

    # Compute-bound float32 throughput: a kernel with high arithmetic
    # intensity (many sequential FMA-shaped operations per element, only
    # one array read and one write per thread total) so memory traffic
    # is negligible and the measured time reflects the GPU's actual
    # sustained arithmetic rate instead of bandwidth. `compute_iters` is
    # large enough that dispatch overhead and the single read/write are
    # both a small fraction of total time.
    compute_iters = 10_000
    n_compute = 4_000_000
    x_c = np.random.default_rng(4).random(n_compute).astype(np.float32)
    d_x_c = metal.to_device(x_c)
    d_out_c = metal.to_device(np.zeros(n_compute, dtype=np.float32))
    blocks_c = (n_compute + 255) // 256

    @metal.jit
    def _calib_compute(a, out, iters):
        i = metal.grid(1)
        if i < out.size:
            x = a[i]
            y = 1.0000001
            for _ in range(iters):
                x = x * y + 0.0000001
            out[i] = x

    for _ in range(3):
        _calib_compute[blocks_c, 256](d_x_c, d_out_c, compute_iters)
        metal.synchronize()
    compute_samples = []
    for _ in range(10):
        t0 = time.perf_counter_ns()
        _calib_compute[blocks_c, 256](d_x_c, d_out_c, compute_iters)
        metal.synchronize()
        compute_samples.append(time.perf_counter_ns() - t0)
    compute_median_ns = sorted(compute_samples)[len(compute_samples) // 2]
    # 2 flops per loop iteration (multiply + add), n_compute elements,
    # compute_iters iterations each.
    float32_gflops_compute_bound = (
        (2.0 * n_compute * compute_iters) / (compute_median_ns / 1e9) / 1e9
        if compute_median_ns > 0
        else None
    )
    roofline_ridge_flops_per_byte = (
        (float32_gflops_compute_bound * 1e9) / (memory_bandwidth_gbps * 1e9)
        if float32_gflops_compute_bound and memory_bandwidth_gbps
        else None
    )

    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = None
    try:
        import numba_metal

        numba_metal_version = numba_metal.__version__
    except (ImportError, AttributeError):
        numba_metal_version = None

    return CalibrationResult(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        timestamp_ns=time.time_ns(),
        device_name=device_info.name,
        apple_silicon_family=hw.get("chip"),
        macos_version=platform.mac_ver()[0],
        python_version=sys.version.split()[0],
        numba_version=numba_version,
        numba_metal_version=numba_metal_version,
        dispatch_overhead_ns=dispatch_overhead_ns,
        cold_compile_ns=float(cold_compile_ns),
        buffer_alloc_ns=buffer_alloc_ns,
        sync_overhead_ns=sync_overhead_ns,
        float32_gflops_memory_bound=float32_gflops_memory_bound,
        float32_gflops_compute_bound=float32_gflops_compute_bound,
        roofline_ridge_flops_per_byte=roofline_ridge_flops_per_byte,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        crossover_elements_estimate=None,
    )


def save_calibration(result: CalibrationResult) -> Path:
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json_dict(), indent=2))
    return path


def load_calibration() -> CalibrationResult | None:
    path = calibration_path()
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        return None
    return CalibrationResult(**data)


def delete_calibration() -> bool:
    path = calibration_path()
    if path.exists():
        path.unlink()
        return True
    return False
