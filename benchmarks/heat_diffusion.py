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
    require_metal_or_skip,
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


def _make_numba_cpu_impl():
    from numba import njit

    @njit(cache=True, fastmath=False)
    def step(cur, nxt, n):
        for x in range(1, n - 1):
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
    numba_cpu_impl = _make_numba_cpu_impl()
    metal_available = require_metal_or_skip()

    for size in sizes:
        grid = _initial_grid(size)
        expected = numpy_impl(grid, iterations)

        result = BenchmarkResult(benchmark="Heat diffusion", size_label=f"{size}²")

        t_numpy = time_repeated(
            lambda: numpy_impl(grid, iterations), warmup=1, repeats=3
        )
        result.numpy_ns = t_numpy.median_ns

        out_cpu = numba_cpu_impl(grid, iterations)  # warm up / compile
        t_cpu = time_repeated(
            lambda: numba_cpu_impl(grid, iterations), warmup=1, repeats=3
        )
        result.numba_cpu_ns = t_cpu.median_ns
        cpu_ok, cpu_note = assert_allclose(out_cpu, expected, rtol=1e-3, atol=1e-3)

        if metal_available:
            t_resident = time_repeated(
                lambda: _metal_resident(grid, iterations), warmup=1, repeats=3
            )
            result.metal_warm_ns = t_resident.median_ns
            gpu_resident = _metal_resident(grid, iterations)
            gpu_ok, gpu_note = assert_allclose(
                gpu_resident, expected, rtol=1e-2, atol=1e-2
            )

            # Only measure the copy-every-launch variant at a small
            # iteration count for the smallest size -- it is O(iterations)
            # host<->device round trips and is illustratively slow, not
            # something worth waiting on at full scale.
            copy_every_launch_ns = None
            if size == sizes[0]:
                small_iters = min(iterations, 50)
                t_copy = time_repeated(
                    lambda: _metal_copy_every_launch(grid, small_iters),
                    warmup=0,
                    repeats=1,
                )
                copy_every_launch_ns = t_copy.median_ns

            result.correctness_ok = cpu_ok and gpu_ok
            result.correctness_note = f"cpu: {cpu_note}; gpu(resident): {gpu_note}"
            result.extra["metal_copy_every_launch_ns"] = copy_every_launch_ns
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
        copy_ns = r.extra.get("metal_copy_every_launch_ns")
        if copy_ns is not None:
            print(
                f"{r.benchmark} {r.size_label}: resident="
                f"{format_ns(r.metal_warm_ns)} for {r.extra['iterations']} iters vs "
                f"copy-every-launch={format_ns(copy_ns)} for "
                f"{min(r.extra['iterations'], 50)} iters "
                "(per-iteration transfer cost dominates when data is not resident)"
            )
