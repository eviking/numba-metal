"""Benchmark 2: Mandelbrot escape-time image.

One GPU thread per pixel; nested computation with loop-carried state and
divergent control flow (each pixel's `for` loop runs a different number of
iterations before `break`). Compares NumPy (vectorized, still does full
max_iter work per pixel since divergent early-exit isn't expressible
vectorized), Numba CPU (@njit, parallel over rows), and numba-metal.

An optional PNG output can be produced via `--save-image out.png` if
Pillow is installed; the core package does not require Pillow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    BenchmarkResult,
    assert_near_integer_match,
    require_metal_or_skip,
    time_repeated,
)

SIZES = [512, 2048, 4096]
MAX_ITER = 100


def numpy_impl(width: int, height: int, max_iter: int) -> np.ndarray:
    px, py = np.meshgrid(np.arange(width), np.arange(height))
    x0 = (px / width) * 3.5 - 2.5
    y0 = (py / height) * 2.0 - 1.0
    x = np.zeros_like(x0)
    y = np.zeros_like(y0)
    count = np.zeros(x0.shape, dtype=np.int32)
    active = np.ones(x0.shape, dtype=bool)
    for _ in range(max_iter):
        escaped = (x * x + y * y) > 4.0
        active &= ~escaped
        xt = np.where(active, x * x - y * y + x0, x)
        y = np.where(active, 2.0 * x * y + y0, y)
        x = xt
        count += active.astype(np.int32)
    return count


def _make_numba_cpu_impl():
    from numba import njit, prange

    # Explicit float32() casts throughout: without them, Numba's CPU
    # target infers float64 for the `0.0`/`3.5`/etc. literals (matching
    # CPython semantics), while numba-metal's MSL backend narrows local
    # float64 intermediates to float32 (see docs/limitations.md and
    # numba_metal/compiler/msl_backend.py). Comparing a float64 CPU
    # reference against a float32 GPU result on this chaotic recurrence
    # produces iteration-count differences of tens of steps for a handful
    # of pixels near the escape boundary -- a genuine, expected
    # floating-point precision effect, not a functional bug. Forcing the
    # CPU reference to float32 too makes this an apples-to-apples
    # comparison.
    @njit(parallel=True, fastmath=False, cache=True)
    def numba_cpu_impl(out, width, height, max_iter):
        for idx in prange(width * height):
            px = np.float32(idx % width)
            py = np.float32(idx // width)
            x0 = (px / np.float32(width)) * np.float32(3.5) - np.float32(2.5)
            y0 = (py / np.float32(height)) * np.float32(2.0) - np.float32(1.0)
            x = np.float32(0.0)
            y = np.float32(0.0)
            count = 0
            for _ in range(max_iter):
                if x * x + y * y > np.float32(4.0):
                    break
                xt = x * x - y * y + x0
                y = np.float32(2.0) * x * y + y0
                x = xt
                count += 1
            out[idx] = count
        return out

    return numba_cpu_impl


def _make_metal_kernel():
    from numba_metal import metal

    @metal.jit
    def metal_kernel(out, width, height, max_iter):
        i = metal.grid(1)
        n = width * height
        if i < n:
            px = i % width
            py = i // width
            x0 = (px / width) * 3.5 - 2.5
            y0 = (py / height) * 2.0 - 1.0
            x = 0.0
            y = 0.0
            count = 0
            for _ in range(max_iter):
                if x * x + y * y > 4.0:
                    break
                xt = x * x - y * y + x0
                y = 2.0 * x * y + y0
                x = xt
                count = count + 1
            out[i] = count

    return metal_kernel


def run(sizes: list[int] = SIZES, max_iter: int = MAX_ITER) -> list[BenchmarkResult]:
    results = []
    numba_cpu_impl = _make_numba_cpu_impl()
    metal_available = require_metal_or_skip()
    metal_kernel = _make_metal_kernel() if metal_available else None
    if metal_available:
        from numba_metal import metal

    for size in sizes:
        width = height = size
        expected = numpy_impl(width, height, max_iter).reshape(-1)

        result = BenchmarkResult(benchmark="Mandelbrot", size_label=f"{width}²")

        t_numpy = time_repeated(
            lambda: numpy_impl(width, height, max_iter), warmup=1, repeats=3
        )
        result.numpy_ns = t_numpy.median_ns

        out_cpu = np.empty(width * height, dtype=np.int32)
        numba_cpu_impl(out_cpu, width, height, max_iter)
        t_cpu = time_repeated(
            lambda: numba_cpu_impl(out_cpu, width, height, max_iter),
            warmup=2,
            repeats=5,
        )
        result.numba_cpu_ns = t_cpu.median_ns
        # `expected` (vectorized NumPy, float64) is a *different* evaluation
        # order and precision of the same recurrence -- computed here purely
        # for the timing comparison column, not as a correctness oracle for
        # a chaotic escape-time algorithm (a float64-vs-float32 comparison
        # near the escape boundary is expected to disagree on more than a
        # negligible fraction of pixels; see assert_near_integer_match's
        # docstring). It is logged as informational only.
        _, numpy_vs_cpu_note = assert_near_integer_match(
            out_cpu, expected, max_off_by_one_fraction=1.0
        )

        if metal_available:
            d_out = metal.device_array(width * height, np.int32)
            threads = 256
            blocks = (width * height + threads - 1) // threads

            def launch():
                metal_kernel[blocks, threads](
                    d_out, np.int32(width), np.int32(height), np.int32(max_iter)
                )
                metal.synchronize()

            t_cold = time_repeated(launch, warmup=0, repeats=1)
            result.metal_cold_ns = t_cold.median_ns
            t_warm = time_repeated(launch, warmup=2, repeats=5)
            result.metal_warm_ns = t_warm.median_ns
            t_d2h = time_repeated(lambda: d_out.copy_to_host(), warmup=1, repeats=5)
            result.metal_d2h_ns = t_d2h.median_ns

            # The correctness gate: GPU (float32) vs. scalar CPU (float32,
            # same algorithm) -- an apples-to-apples comparison.
            gpu_ok, gpu_note = assert_near_integer_match(d_out.copy_to_host(), out_cpu)
            result.correctness_ok = gpu_ok
            result.correctness_note = (
                f"gpu vs cpu(f32): {gpu_note}; {numpy_vs_cpu_note}"
            )
        else:
            result.correctness_ok = True
            result.correctness_note = f"gpu unavailable; {numpy_vs_cpu_note}"

        results.append(result)
    return results


def save_image(width: int, height: int, max_iter: int, path: str) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Saving an image requires Pillow: pip install numba-metal[image]"
        ) from exc

    from numba_metal import metal

    metal_kernel = _make_metal_kernel()
    d_out = metal.device_array(width * height, np.int32)
    threads = 256
    blocks = (width * height + threads - 1) // threads
    metal_kernel[blocks, threads](
        d_out, np.int32(width), np.int32(height), np.int32(max_iter)
    )
    metal.synchronize()
    counts = d_out.copy_to_host().reshape(height, width)
    normalized = (
        (counts.astype(np.float32) / max_iter * 255).clip(0, 255).astype(np.uint8)
    )
    Image.fromarray(normalized).save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    import argparse

    from common import print_table

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-image", type=str, default=None)
    args = parser.parse_args()

    print_table(run())

    if args.save_image:
        save_image(1024, 1024, MAX_ITER, args.save_image)
