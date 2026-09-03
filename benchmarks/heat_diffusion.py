"""Benchmark 3: 2D heat diffusion (Jacobi-style stencil).

    next[x,y] = 0.25 * (cur[x-1,y] + cur[x+1,y] + cur[x,y-1] + cur[x,y+1])

Ping-pongs between two GPU-resident buffers for many iterations, comparing
"copy around every launch" (transfer grid back to host and re-upload each
step) against "keep data resident" (only transfer the initial and final
grids). Uses flattened 1D array storage with manual 2D index arithmetic,
per numba-metal's 1D-array kernel-argument restriction.
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
    def step(cur, nxt, n):
        for x in loop_range(1, n - 1):
            for y in range(1, n - 1):
                nxt[x * n + y] = 0.25 * (
                    cur[(x - 1) * n + y]
                    + cur[(x + 1) * n + y]
                    + cur[x * n + (y - 1)]
                    + cur[x * n + (y + 1)]
                )

    def numba_cpu_impl(grid: np.ndarray, iterations: int) -> np.ndarray:
        n = grid.shape[0]
        cur = grid.reshape(-1).copy()
        nxt = cur.copy()
        for _ in range(iterations):
            step(cur, nxt, n)
            cur, nxt = nxt, cur
        return cur.reshape(n, n)

    return numba_cpu_impl


def _make_metal_kernel():
    from numba_metal import metal

    @metal.jit
    def stencil_step(cur, nxt, n):
        i = metal.grid(1)
        total = n * n
        if i < total:
            x = i // n
            y = i % n
            if x >= 1 and x < n - 1 and y >= 1 and y < n - 1:
                nxt[i] = 0.25 * (
                    cur[(x - 1) * n + y]
                    + cur[(x + 1) * n + y]
                    + cur[x * n + (y - 1)]
                    + cur[x * n + (y + 1)]
                )
            else:
                nxt[i] = cur[i]

    return stencil_step


def _metal_resident(grid: np.ndarray, iterations: int, threads: int = 256):
    """Keep both buffers GPU-resident for the whole simulation; only
    transfer the initial grid in and the final grid out."""
    from numba_metal import metal

    n = grid.shape[0]
    stencil_step = _make_metal_kernel()
    d_cur = metal.to_device(grid.reshape(-1))
    d_nxt = metal.device_array_like(grid.reshape(-1))
    blocks = (n * n + threads - 1) // threads
    for _ in range(iterations):
        stencil_step[blocks, threads](d_cur, d_nxt, np.int32(n))
        d_cur, d_nxt = d_nxt, d_cur
    metal.synchronize()
    return d_cur.copy_to_host().reshape(n, n)


def _metal_copy_every_launch(grid: np.ndarray, iterations: int, threads: int = 256):
    """Worst-case comparison: re-upload the current grid and download the
    result on every single iteration, to make the cost of NOT keeping data
    resident explicit."""
    from numba_metal import metal

    n = grid.shape[0]
    stencil_step = _make_metal_kernel()
    cur_host = grid.reshape(-1).copy()
    blocks = (n * n + threads - 1) // threads
    for _ in range(iterations):
        d_cur = metal.to_device(cur_host)
        d_nxt = metal.device_array_like(cur_host)
        stencil_step[blocks, threads](d_cur, d_nxt, np.int32(n))
        metal.synchronize()
        cur_host = d_nxt.copy_to_host()
    return cur_host.reshape(n, n)


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
            threads = 256
            n = grid.shape[0]
            blocks = (n * n + threads - 1) // threads

            def cold_launch(kernel=None):
                k = kernel or metal_kernel
                d_cur = metal.to_device(grid.reshape(-1))
                d_nxt = metal.device_array_like(grid.reshape(-1))
                k[blocks, threads](d_cur, d_nxt, np.int32(n))
                metal.synchronize()

            cold = measure_cold_end_to_end(metal_kernel.py_func, cold_launch)
            result.metal_cold_total_ns = cold["total_ns"]
            result.compiled_before = cold["compiled_before"]
            result.compiled_after = cold["compiled_after"]

            d_cur_probe = metal.to_device(grid.reshape(-1))
            arg_types = infer_launch_arg_types(d_cur_probe, d_cur_probe, np.int32(n))
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
            result.metal_kernel_only_warm_ns = t_resident.median_ns / iterations
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
