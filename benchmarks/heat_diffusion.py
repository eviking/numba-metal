"""Benchmark 3: 2D nonlinear (Perona-Malik) anisotropic heat diffusion.

    next[x,y] = cur[x,y] + dt * sum over 4 neighbors of:
                    exp(-(gradient/kappa)^2) * gradient

Unlike simple linear diffusion (`next = 0.25 * sum of 4 neighbors`,
constant weight everywhere), the diffusion coefficient here depends on
the LOCAL gradient at each point -- Perona-Malik anisotropic diffusion
(P. Perona & J. Malik, "Scale-space and edge detection using
anisotropic diffusion," IEEE PAMI 1990), the standard nonlinear
diffusion model used in image denoising and edge-preserving smoothing,
not a synthetic stand-in. Same 4-neighbor memory-access pattern as
linear diffusion, but 4 `math.exp` calls per grid point per iteration
instead of a single constant multiply -- real arithmetic intensity, not
padding.

REPLACES an earlier version of this benchmark that used simple linear
diffusion. That version is preserved in git history; the reason for
the replacement is itself the most important finding of this file:

Linear diffusion is BANDWIDTH-bound on Apple Silicon's unified memory,
where Metal has no bandwidth advantage over parallel Numba CPU to
exploit. Measured directly (all sizes on an Apple M4 Pro, 200
iterations, GPU-resident buffers): 128^2 -> 0.72x, 512^2 -> 0.91x,
1024^2 -> 1.00x (parity) -- and, critically, tested at 10x and 100x
the grid-point count of 1024^2 (3238^2 and 10240^2), Metal got WORSE,
not better, as the grid grew: 0.52x and 0.60x. Bandwidth pressure
grows with grid size; the arithmetic per point does not, for a
constant-coefficient stencil -- so there is no reason to expect a
bandwidth-bound kernel to favor the GPU at any scale on this hardware.

This nonlinear version was built specifically to test the other lever
available for moving a workload from bandwidth-bound toward
compute-bound (the same lever benchmarks/asian_option_pricing.py's own
docstring describes using successfully for Monte Carlo pricing): add
real per-point arithmetic without changing the memory-access pattern.
Measured result, same sizes, same machine, same methodology: 128^2 ->
0.53x, 512^2 -> 1.88x, 1024^2 -> 4.57x, 3238^2 (10x) -> 12.80x,
10240^2 (100x) -> 13.57x. Not only does the nonlinear version clear a
real speedup at 1024^2 where linear diffusion only reached parity, it
gets BETTER as the grid grows, the opposite trend from linear
diffusion -- exactly the signature of a workload that has crossed from
bandwidth-bound into compute-bound territory. Correctness (GPU vs.
scalar-CPU reference, and CPU-parallel vs. NumPy-vectorized reference)
was verified at every size tested, including 10x/100x, before trusting
any of these numbers.

This file went through three earlier rounds of investigation on the
LINEAR-diffusion version, worth preserving here since the lessons
still apply directly to the nonlinear kernel below (same 2D-indexing
and device-function code, only the per-point math changed):

Round 1: switching the METAL kernel from a 1D-flat launch (with manual
`x*n+y` flattening) to a genuine 2D launch and 2D array indexing, while
leaving the Numba CPU implementation on its ORIGINAL flattened 1D
indexing, measured a ~5x improvement at 1024x1024 and looked like a
clear win against CPU (0.85x -> 4.2x+).

Round 2: that comparison was not apples-to-apples. Manual flattened
indexing turned out to slow down Numba's OWN CPU codegen too --
directly measured in isolation: an identical stencil, 2D-indexed vs.
manually flattened, ran ~1.6-2x FASTER on CPU alone once flattening was
removed. Once BOTH sides were rewritten to use real 2D indexing, the
honest linear-diffusion comparison at 1024x1024 was roughly PARITY, not
a clear Metal win. The lesson: manual index-flattening hides array
structure from BOTH Numba's LLVM backend and Metal's memory system, so
removing it is worth doing regardless of which side you're optimizing,
but it is not, on its own, evidence that a given stencil favors the
GPU. See docs/performance-guidance.md's bandwidth-bound section, and
tests/integration/test_multidim_arrays.py/docs/limitations.md for why
this motivated adding native multi-dimensional array support at all.

Round 3: `_make_metal_kernel_device_func` factors the per-point update
into a `@metal.device_func` taking the 2D `cur` array directly --
exercising multi-dimensional `@metal.device_func` array arguments.
Measured separately from the inline version, on the ORIGINAL linear
kernel: the device-function call cost a real, consistent 2.3-2.6x
per-iteration slowdown across all three sizes -- a non-inlined MSL
function call is not free, and that stencil's per-point work was small
enough that call overhead dominated. Kept in this file (now ported to
the nonlinear math) as an honest data point about when to reach for
`@metal.device_func`. A DIFFERENT, genuine win from `@metal.device_func`
(compile-time, not runtime, requiring reuse across multiple different
kernels) is measured in `benchmarks/device_function_compile_cache.py`.

Also compares "copy around every launch" (transfer grid back to host
and re-upload each step) against "keep data resident" (only transfer
the initial and final grids), ping-ponging between two GPU-resident
buffers for many iterations.
"""

from __future__ import annotations

import math
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
#: Perona-Malik edge-sensitivity constant -- controls how strongly the
#: diffusion coefficient falls off with gradient magnitude. Not tuned
#: for any particular visual result; picked to keep conductances in a
#: numerically well-behaved range (0, 1] for this benchmark's initial
#: condition's gradient scale.
KAPPA = 20.0
#: Explicit-scheme stability requires a time step small enough that the
#: scheme doesn't blow up; conservative fixed value, not adaptive
#: (adaptive time-stepping would be a further, separate change to the
#: numerical method, out of scope for an arithmetic-intensity test).
DT = 0.2


def _initial_grid(n: int) -> np.ndarray:
    grid = np.zeros((n, n), dtype=np.float32)
    grid[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 100.0
    return grid


def numpy_impl(grid: np.ndarray, iterations: int) -> np.ndarray:
    cur = grid.copy()
    for _ in range(iterations):
        dn = np.zeros_like(cur)
        ds = np.zeros_like(cur)
        de = np.zeros_like(cur)
        dw = np.zeros_like(cur)
        dn[1:, :] = cur[:-1, :] - cur[1:, :]
        ds[:-1, :] = cur[1:, :] - cur[:-1, :]
        de[:, :-1] = cur[:, 1:] - cur[:, :-1]
        dw[:, 1:] = cur[:, :-1] - cur[:, 1:]
        cn = np.exp(-((dn / KAPPA) ** 2))
        cs = np.exp(-((ds / KAPPA) ** 2))
        ce = np.exp(-((de / KAPPA) ** 2))
        cw = np.exp(-((dw / KAPPA) ** 2))
        cur = cur + DT * (cn * dn + cs * ds + ce * de + cw * dw)
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
                center = cur[x, y]
                dn = cur[x - 1, y] - center
                ds = cur[x + 1, y] - center
                de = cur[x, y + 1] - center
                dw = cur[x, y - 1] - center
                cn = np.exp(-((dn / KAPPA) ** 2))
                cs = np.exp(-((ds / KAPPA) ** 2))
                ce = np.exp(-((de / KAPPA) ** 2))
                cw = np.exp(-((dw / KAPPA) ** 2))
                nxt[x, y] = center + DT * (cn * dn + cs * ds + ce * de + cw * dw)

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
                center = cur[x, y]
                dn = cur[x - 1, y] - center
                ds = cur[x + 1, y] - center
                de = cur[x, y + 1] - center
                dw = cur[x, y - 1] - center
                cn = math.exp(-((dn / 20.0) ** 2))
                cs = math.exp(-((ds / 20.0) ** 2))
                ce = math.exp(-((de / 20.0) ** 2))
                cw = math.exp(-((dw / 20.0) ** 2))
                nxt[x, y] = center + 0.2 * (cn * dn + cs * ds + ce * de + cw * dw)
            else:
                nxt[x, y] = cur[x, y]

    return stencil_step


def _make_metal_kernel_device_func():
    """Same nonlinear stencil as `_make_metal_kernel`, but the per-point
    update is factored into a `@metal.device_func` taking the 2D `cur`
    array directly -- exercises multi-dimensional `@metal.device_func`
    array arguments (see `docs/limitations.md`'s entry on this). Kept as
    a SEPARATE kernel (not a replacement for `_make_metal_kernel`) so
    `run()` can measure both and report the device-function call
    overhead honestly, rather than assuming factoring it out is free --
    see this module's docstring "Round 3" for the measured result on
    the original linear-diffusion version of this stencil."""
    from numba_metal import metal

    @metal.device_func
    def updated_value(grid, x, y):
        center = grid[x, y]
        dn = grid[x - 1, y] - center
        ds = grid[x + 1, y] - center
        de = grid[x, y + 1] - center
        dw = grid[x, y - 1] - center
        cn = math.exp(-((dn / 20.0) ** 2))
        cs = math.exp(-((ds / 20.0) ** 2))
        ce = math.exp(-((de / 20.0) ** 2))
        cw = math.exp(-((dw / 20.0) ** 2))
        return center + 0.2 * (cn * dn + cs * ds + ce * de + cw * dw)

    @metal.jit
    def stencil_step(cur, nxt, n, m):
        x, y = metal.grid(2)
        if x < n and y < m:
            if x >= 1 and x < n - 1 and y >= 1 and y < m - 1:
                nxt[x, y] = updated_value(cur, x, y)
            else:
                nxt[x, y] = cur[x, y]

    return stencil_step


def _launch_config(
    n: int, m: int, tile: int = TILE
) -> tuple[tuple[int, int], tuple[int, int]]:
    blocks = ((n + tile - 1) // tile, (m + tile - 1) // tile)
    threads = (tile, tile)
    return blocks, threads


def _metal_resident(grid: np.ndarray, iterations: int, make_kernel=_make_metal_kernel):
    """Keep both buffers GPU-resident for the whole simulation; only
    transfer the initial grid in and the final grid out."""
    from numba_metal import metal

    n, m = grid.shape
    stencil_step = make_kernel()
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
            result.metal_kernel_only_warm_ns = t_resident.median_ns
            result.metal_end_to_end_warm_ns = t_resident.median_ns
            gpu_resident = _metal_resident(grid, iterations)
            gpu_ok, gpu_note = assert_allclose(
                gpu_resident, expected, rtol=1e-2, atol=1e-2
            )

            # "Copy every launch": re-upload the current grid and download
            # the result on every single iteration, at the SAME iteration
            # count as the resident variant, so the two are a fair,
            # apples-to-apples per-iteration comparison. O(iterations)
            # host<->device round trips, deliberately the worst case.
            t_copy = time_repeated(
                lambda: _metal_copy_every_launch(grid, iterations),
                warmup=0,
                repeats=1,
            )
            copy_every_launch_ns = t_copy.median_ns

            # Device-function variant: the same per-point update, but
            # factored into a @metal.device_func taking the 2D `cur`
            # array directly (see `_make_metal_kernel_device_func`) --
            # measured separately, at the same iteration count and
            # launch config, to report the real cost (or lack thereof)
            # of a non-inlined device-function call honestly rather than
            # assuming factoring code out is free.
            t_resident_device_func = time_repeated(
                lambda: _metal_resident(
                    grid, iterations, make_kernel=_make_metal_kernel_device_func
                ),
                warmup=1,
                repeats=3,
            )
            gpu_resident_device_func = _metal_resident(
                grid, iterations, make_kernel=_make_metal_kernel_device_func
            )
            gpu_device_func_ok, gpu_device_func_note = assert_allclose(
                gpu_resident_device_func, expected, rtol=1e-2, atol=1e-2
            )

            result.correctness_ok = cpu_ok and gpu_ok and gpu_device_func_ok
            result.correctness_note = (
                f"cpu: {cpu_note}; gpu(resident): {gpu_note}; "
                f"gpu(device_func): {gpu_device_func_note}"
            )
            result.extra["metal_copy_every_launch_total_ns"] = copy_every_launch_ns
            result.extra["metal_copy_every_launch_per_iter_ns"] = (
                copy_every_launch_ns / iterations
            )
            result.extra["metal_resident_per_iter_ns"] = (
                result.metal_resident_pipeline_ns / iterations
            )
            result.extra["metal_resident_device_func_ns"] = (
                t_resident_device_func.median_ns
            )
            result.extra["metal_resident_device_func_per_iter_ns"] = (
                t_resident_device_func.median_ns / iterations
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
        per_iter_device_func = r.extra.get("metal_resident_device_func_per_iter_ns")
        if per_iter_copy is not None:
            print(
                f"{r.benchmark} {r.size_label}, {r.extra['iterations']} iters "
                f"(same count for both): resident={format_ns(per_iter_resident)}/iter "
                f"vs copy-every-launch={format_ns(per_iter_copy)}/iter "
                "(per-iteration transfer cost dominates when data is not resident)"
            )
        if per_iter_device_func is not None:
            print(
                f"{r.benchmark} {r.size_label}: inline update="
                f"{format_ns(per_iter_resident)}/iter vs factored into "
                f"@metal.device_func={format_ns(per_iter_device_func)}/iter "
                "(cost of a non-inlined 2D-array device-function call)"
            )
