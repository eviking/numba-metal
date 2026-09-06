"""Benchmark 2: Mandelbrot escape-time image.

One GPU thread per pixel; nested computation with loop-carried state and
divergent control flow (each pixel's `for` loop runs a different number of
iterations before `break`). Compares NumPy (vectorized, still does full
max_iter work per pixel since divergent early-exit isn't expressible
vectorized), Numba CPU (@njit, parallel over rows), and numba-metal.

Launched as a genuine 2D grid (`metal.grid(2)`) over a real 2D output
array (`out[py, px]`), not a flattened 1D thread index recovering (px,
py) via `i % width` / `i // width` -- both the Metal kernel AND the
Numba CPU reference are written this way, matching each other exactly.
This is deliberate, learned from a real mistake made on a different
benchmark (heat_diffusion.py): fixing only ONE side of a CPU-vs-Metal
comparison to remove manual index-flattening measures the launch
-configuration effect confounded with an unfair baseline, and can
produce a speedup number that evaporates once the CPU side is fixed
too. See docs/performance-guidance.md's bandwidth-bound section for
the full account of that mistake and correction.

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
    infer_launch_arg_types,
    measure_cold_end_to_end,
    measure_cold_metal_pipeline,
    require_metal_or_skip,
    run_single_threaded,
    time_repeated,
)

SIZES = [512, 2048, 4096]
MAX_ITER = 100

#: 2D threadgroup shape, matching heat_diffusion.py's TILE convention --
#: TILE x TILE = 256 threads per threadgroup, the same total occupancy as
#: the previous 1D launch's 256-thread threadgroup.
TILE = 16


def _launch_config(
    width: int, height: int, tile: int = TILE
) -> tuple[tuple[int, int], tuple[int, int]]:
    blocks = ((width + tile - 1) // tile, (height + tile - 1) // tile)
    threads = (tile, tile)
    return blocks, threads


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


def _make_numba_cpu_impl(*, parallel: bool):
    from numba import njit, prange

    loop_range = prange if parallel else range

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
    @njit(parallel=parallel, fastmath=False, cache=True)
    def numba_cpu_impl(out, width, height, max_iter):
        for py_i in loop_range(height):
            py = np.float32(py_i)
            for px_i in range(width):
                px = np.float32(px_i)
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
                out[py_i, px_i] = count
        return out

    return numba_cpu_impl


def _make_metal_kernel():
    from numba_metal import metal

    @metal.jit
    def metal_kernel(out, width, height, max_iter):
        px, py = metal.grid(2)
        if px < width and py < height:
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
            out[py, px] = count

    return metal_kernel


def run(sizes: list[int] = SIZES, max_iter: int = MAX_ITER) -> list[BenchmarkResult]:
    results = []
    metal_available = require_metal_or_skip()
    if metal_available:
        from numba_metal import metal

    for size in sizes:
        width = height = size
        expected = numpy_impl(width, height, max_iter)

        result = BenchmarkResult(benchmark="Mandelbrot", size_label=f"{width}²")

        t_numpy = time_repeated(
            lambda: numpy_impl(width, height, max_iter), warmup=1, repeats=3
        )
        result.numpy_ns = t_numpy.median_ns

        cpu_parallel = _make_numba_cpu_impl(parallel=True)
        out_cpu = np.empty((height, width), dtype=np.int32)
        cpu_parallel(out_cpu, width, height, max_iter)
        t_cpu_par = time_repeated(
            lambda: cpu_parallel(out_cpu, width, height, max_iter),
            warmup=2,
            repeats=5,
        )
        result.numba_cpu_parallel_ns = t_cpu_par.median_ns
        import numba

        result.numba_cpu_num_threads = numba.get_num_threads()

        cpu_single = _make_numba_cpu_impl(parallel=False)
        out_cpu_single = np.empty((height, width), dtype=np.int32)
        cpu_single(out_cpu_single, width, height, max_iter)
        t_cpu_single = run_single_threaded(
            lambda: time_repeated(
                lambda: cpu_single(out_cpu_single, width, height, max_iter),
                warmup=2,
                repeats=5,
            )
        )
        result.numba_cpu_single_ns = t_cpu_single.median_ns

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
            metal_kernel = _make_metal_kernel()
            d_out = metal.device_array((height, width), np.int32)
            blocks, threads = _launch_config(width, height)

            def launch(kernel=None):
                k = kernel or metal_kernel
                k[blocks, threads](
                    d_out, np.int32(width), np.int32(height), np.int32(max_iter)
                )
                metal.synchronize()

            cold = measure_cold_end_to_end(metal_kernel.py_func, launch)
            result.metal_cold_total_ns = cold["total_ns"]
            result.compiled_before = cold["compiled_before"]
            result.compiled_after = cold["compiled_after"]

            arg_types = infer_launch_arg_types(
                d_out, np.int32(width), np.int32(height), np.int32(max_iter)
            )
            phases = measure_cold_metal_pipeline(metal_kernel.py_func, arg_types)
            result.metal_frontend_ns = phases["frontend_ns"]
            result.metal_pipeline_compile_ns = phases["pipeline_compile_ns"]

            launch()
            t_kernel_warm = time_repeated(launch, warmup=2, repeats=5)
            result.metal_kernel_only_warm_ns = t_kernel_warm.median_ns
            t_d2h = time_repeated(lambda: d_out.copy_to_host(), warmup=1, repeats=5)
            result.metal_d2h_ns = t_d2h.median_ns
            # No host input data for this kernel (output-only); end-to-end
            # warm time is kernel + D2H (no H2D input to transfer).
            result.metal_end_to_end_warm_ns = (
                result.metal_kernel_only_warm_ns + result.metal_d2h_ns
            )

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
    d_out = metal.device_array((height, width), np.int32)
    blocks, threads = _launch_config(width, height)
    metal_kernel[blocks, threads](
        d_out, np.int32(width), np.int32(height), np.int32(max_iter)
    )
    metal.synchronize()
    counts = d_out.copy_to_host()
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
