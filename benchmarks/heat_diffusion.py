"""Benchmark 3: 2D heat diffusion (Jacobi-style stencil).

    next[x,y] = 0.25 * (cur[x-1,y] + cur[x+1,y] + cur[x,y-1] + cur[x,y+1])

Ping-pongs between two GPU-resident buffers for many iterations, comparing
"copy around every launch" (transfer grid back to host and re-upload each
step) against "keep data resident" (only transfer the initial and final
grids).

Launched as a genuine 2D grid (`metal.grid(2)`, 2D blocks/threads) AND
indexed with real 2D arrays (`cur[x, y]`), not a flattened 1D index
recovering (x, y) via `i // n` / `i % n` with manual `arr[x*n+y]`
indexing on EITHER the GPU or CPU side.

This benchmark went through two rounds of investigation worth being
explicit about, because the first round's own conclusion turned out to
be wrong once measured more carefully:

Round 1: switching the METAL kernel from a 1D-flat launch (with manual
`x*n+y` flattening) to a genuine 2D launch and 2D array indexing, while
leaving the Numba CPU implementation on its ORIGINAL flattened 1D
indexing, measured a ~5x improvement at 1024x1024 on an Apple M4 Pro
and looked like a clear win against CPU (0.85x -> 4.2x+).

Round 2: that comparison was not apples-to-apples. Manual flattened
indexing turned out to slow down Numba's OWN CPU codegen too --
directly measured in isolation: an identical stencil, 2D-indexed vs.
manually flattened, ran ~1.6-2x FASTER on CPU alone once flattening was
removed, negligible at 128x128, real by 512x512, largest at 1024x1024
-- the same size-dependent shape as the GPU-side effect. Once BOTH
sides were rewritten to use real 2D indexing (this file, as it stands
now), the honest comparison at 1024x1024 is roughly PARITY (~1.0x-1.1x
across repeated runs), not a clear Metal win -- Metal still loses at
128x128 and 512x512. The real, reproducible lesson is narrower than
originally concluded: manual index-flattening hides array structure
from BOTH Numba's LLVM backend and Metal's memory system, so removing
it is worth doing regardless of which side you're optimizing, but it
is not, on its own, evidence that this specific stencil is a good fit
for the GPU on Apple Silicon at these problem sizes. See
docs/performance-guidance.md's bandwidth-bound section for the full,
corrected writeup, including why an added threadgroup-memory
(shared-memory tiling) layer on top of the 2D launch was tried and
found to add pure overhead rather than help on this hardware at these
sizes.

Real 2D array indexing (`cur[x, y]` instead of manually flattening to
`cur[x*n+y]`) is itself a direct consequence of this investigation: it
was exactly the class of kernel that manual flattening makes easy to
get subtly wrong (a transposed stride, an off-by-one in the row width)
-- and, as it turned out, easy to accidentally under-optimize on either
side of the comparison -- that motivated adding native
multi-dimensional array support to numba-metal in the first place. See
tests/integration/test_multidim_arrays.py and docs/limitations.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    BenchmarkResult,
    assert_allclose,
    infer_launch_arg_types,
    measure_cold_end_to_end,
    measure_cold_metal_pipeline,
    require_metal_or_skip,
    run_single_threaded,
    time_repeated,
)

SIZES = [128, 512, 1024]
ITERATIONS = 200


def _initial_grid(n: int) -> np.ndarray:
    grid = np.zeros((n, n), dtype=np.float32)
    grid[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 100.0
    return grid


def numpy_impl(grid: np.ndarray, iterations: int) -> np.ndarray:
    cur = grid.copy()
    for _ in range(iterations):
        nxt = cur.copy()
        nxt[1:-1, 1:-1] = 0.25 * (
            cur[:-2, 1:-1] + cur[2:, 1:-1] + cur[1:-1, :-2] + cur[1:-1, 2:]
        )
        cur = nxt
    return cur


def _make_numba_cpu_impl(*, parallel: bool):
    from numba import njit, prange

    # Each iteration of the outer x loop writes only row x of `nxt` and
    # only reads `cur` (never `nxt`), so rows are fully independent
    # within one step -- safe to parallelize over x with prange.
    loop_range = prange if parallel else range

    @njit(parallel=parallel, cache=True, fastmath=False)
    def step(cur, nxt, n, m):
        for x in loop_range(1, n - 1):
            for y in range(1, m - 1):
                nxt[x, y] = 0.25 * (
                    cur[x - 1, y] + cur[x + 1, y] + cur[x, y - 1] + cur[x, y + 1]
                )

    def numba_cpu_impl(grid: np.ndarray, iterations: int) -> np.ndarray:
        n, m = grid.shape
        cur = grid.copy()
        nxt = cur.copy()
        for _ in range(iterations):
            step(cur, nxt, n, m)
            cur, nxt = nxt, cur
        return cur

    return numba_cpu_impl


#: 2D threadgroup shape (TILE x TILE = 256 threads per threadgroup,
#: matching the previous 1D launch's 256-thread threadgroup size for a
#: fair, apples-to-apples comparison against Numba CPU -- only the
#: launch dimensionality changed, not the total occupancy).
TILE = 16


def _make_metal_kernel():
    from numba_metal import metal

    @metal.jit
    def stencil_step(cur, nxt, n, m):
        x, y = metal.grid(2)
        if x < n and y < m:
            if x >= 1 and x < n - 1 and y >= 1 and y < m - 1:
                nxt[x, y] = 0.25 * (
                    cur[x - 1, y] + cur[x + 1, y] + cur[x, y - 1] + cur[x, y + 1]
                )
            else:
                nxt[x, y] = cur[x, y]

    return stencil_step


def _launch_config(
    n: int, m: int, tile: int = TILE
) -> tuple[tuple[int, int], tuple[int, int]]:
    blocks = ((n + tile - 1) // tile, (m + tile - 1) // tile)
    threads = (tile, tile)
    return blocks, threads


def _metal_resident(grid: np.ndarray, iterations: int):
    """Keep both buffers GPU-resident for the whole simulation; only
    transfer the initial grid in and the final grid out."""
    from numba_metal import metal

    n, m = grid.shape
    stencil_step = _make_metal_kernel()
    blocks, threads = _launch_config(n, m)
    d_cur = metal.to_device(grid)
    d_nxt = metal.device_array_like(grid)
    for _ in range(iterations):
        stencil_step[blocks, threads](d_cur, d_nxt, np.int32(n), np.int32(m))
        d_cur, d_nxt = d_nxt, d_cur
    metal.synchronize()
    return d_cur.copy_to_host()


def _metal_copy_every_launch(grid: np.ndarray, iterations: int):
    """Worst-case comparison: re-upload the current grid and download the
    result on every single iteration, to make the cost of NOT keeping data
    resident explicit."""
    from numba_metal import metal

    n, m = grid.shape
    stencil_step = _make_metal_kernel()
    blocks, threads = _launch_config(n, m)
    cur_host = grid.copy()
    for _ in range(iterations):
        d_cur = metal.to_device(cur_host)
        d_nxt = metal.device_array_like(cur_host)
        stencil_step[blocks, threads](d_cur, d_nxt, np.int32(n), np.int32(m))
        metal.synchronize()
        cur_host = d_nxt.copy_to_host()
    return cur_host


def run(
    sizes: list[int] = SIZES, iterations: int = ITERATIONS
) -> list[BenchmarkResult]:
    results = []
    metal_available = require_metal_or_skip()

    for size in sizes:
        grid = _initial_grid(size)
        expected = numpy_impl(grid, iterations)

        result = BenchmarkResult(benchmark="Heat diffusion", size_label=f"{size}²")

        t_numpy = time_repeated(
            lambda: numpy_impl(grid, iterations), warmup=1, repeats=3
        )
        result.numpy_ns = t_numpy.median_ns

        cpu_parallel = _make_numba_cpu_impl(parallel=True)
        out_cpu = cpu_parallel(grid, iterations)  # warm up / compile
        t_cpu_par = time_repeated(
            lambda: cpu_parallel(grid, iterations), warmup=1, repeats=3
        )
        result.numba_cpu_parallel_ns = t_cpu_par.median_ns
        import numba

        result.numba_cpu_num_threads = numba.get_num_threads()
        cpu_ok, cpu_note = assert_allclose(out_cpu, expected, rtol=1e-3, atol=1e-3)

        cpu_single = _make_numba_cpu_impl(parallel=False)
        cpu_single(grid, iterations)  # warm up / compile
        t_cpu_single = run_single_threaded(
            lambda: time_repeated(
                lambda: cpu_single(grid, iterations), warmup=1, repeats=3
            )
        )
        result.numba_cpu_single_ns = t_cpu_single.median_ns

        if metal_available:
            from numba_metal import metal

            metal_kernel = _make_metal_kernel()
            n, m = grid.shape
            blocks, threads = _launch_config(n, m)

            def cold_launch(kernel=None):
                k = kernel or metal_kernel
                d_cur = metal.to_device(grid)
                d_nxt = metal.device_array_like(grid)
                k[blocks, threads](d_cur, d_nxt, np.int32(n), np.int32(m))
                metal.synchronize()

            cold = measure_cold_end_to_end(metal_kernel.py_func, cold_launch)
            result.metal_cold_total_ns = cold["total_ns"]
            result.compiled_before = cold["compiled_before"]
            result.compiled_after = cold["compiled_after"]

            d_cur_probe = metal.to_device(grid)
            arg_types = infer_launch_arg_types(
                d_cur_probe, d_cur_probe, np.int32(n), np.int32(m)
            )
            phases = measure_cold_metal_pipeline(metal_kernel.py_func, arg_types)
            result.metal_frontend_ns = phases["frontend_ns"]
            result.metal_pipeline_compile_ns = phases["pipeline_compile_ns"]

            # "Resident": keep both buffers GPU-resident for the whole
            # simulation; only transfer the initial grid in and the final
            # grid out. This is the kernel-only + minimal-transfer number.
            t_resident = time_repeated(
                lambda: _metal_resident(grid, iterations), warmup=1, repeats=3
            )
            result.metal_resident_pipeline_ns = t_resident.median_ns
            # `metal_kernel_only_warm_ns` feeds `speedup_vs()`, which divides
            # a FULL-RUN CPU time (all `iterations` steps) by this value --
            # so it must also be a full-run time, not a per-iteration one.
            # This previously divided by `iterations` here, which silently
            # compared one Metal iteration against `iterations` CPU
            # iterations and inflated the reported speedup by roughly that
            # factor (a real bug: the printed "vs Numba(par)" column for
            # this benchmark was off by ~200x at iterations=200). The
            # correct per-iteration figure is reported separately below via
            # `extra["metal_resident_per_iter_ns"]`.
            result.metal_kernel_only_warm_ns = t_resident.median_ns
            result.metal_end_to_end_warm_ns = t_resident.median_ns
            gpu_resident = _metal_resident(grid, iterations)
            gpu_ok, gpu_note = assert_allclose(
                gpu_resident, expected, rtol=1e-2, atol=1e-2
            )

            # "Copy every launch": re-upload the current grid and download
            # the result on every single iteration, at the SAME iteration
            # count as the resident variant, so the two are a fair,
            # apples-to-apples per-iteration comparison (not different
            # iteration counts dressed up as comparable numbers). This is
            # O(iterations) host<->device round trips and is deliberately
            # the worst case.
            t_copy = time_repeated(
                lambda: _metal_copy_every_launch(grid, iterations),
                warmup=0,
                repeats=1,
            )
            copy_every_launch_ns = t_copy.median_ns

            result.correctness_ok = cpu_ok and gpu_ok
            result.correctness_note = f"cpu: {cpu_note}; gpu(resident): {gpu_note}"
            result.extra["metal_copy_every_launch_total_ns"] = copy_every_launch_ns
            result.extra["metal_copy_every_launch_per_iter_ns"] = (
                copy_every_launch_ns / iterations
            )
            result.extra["metal_resident_per_iter_ns"] = (
                result.metal_resident_pipeline_ns / iterations
            )
            result.extra["iterations"] = iterations
        else:
            result.correctness_ok = cpu_ok
            result.correctness_note = f"cpu: {cpu_note}; gpu: unavailable"

        results.append(result)
    return results


if __name__ == "__main__":
    from common import format_ns, print_table

    results = run()
    print_table(results)
    print()
    for r in results:
        per_iter_copy = r.extra.get("metal_copy_every_launch_per_iter_ns")
        per_iter_resident = r.extra.get("metal_resident_per_iter_ns")
        if per_iter_copy is not None:
            print(
                f"{r.benchmark} {r.size_label}, {r.extra['iterations']} iters "
                f"(same count for both): resident={format_ns(per_iter_resident)}/iter "
                f"vs copy-every-launch={format_ns(per_iter_copy)}/iter "
                "(per-iteration transfer cost dominates when data is not resident)"
            )
