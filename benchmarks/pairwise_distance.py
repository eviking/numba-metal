"""Benchmark 5: pairwise squared Euclidean distance between two sets of
low-dimensional vectors.

    distance[i,j] = sum_k (a[i,k] - b[j,k])**2

One GPU thread per (i, j) output element (a flattened 1D grid), each doing
a small inner loop over the K dimensions -- exercises nested loops,
multidimensional indexing (flattened), and moderate compute intensity per
thread. Arrays are stored flattened (row-major) since numba-metal kernel
arguments are 1D-only in this MVP.
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

# (n_a, n_b, k) problem sizes: two vector-set sizes and the dimensionality.
SIZES = [(200, 200, 8), (2_000, 2_000, 8), (5_000, 5_000, 16)]


def numpy_impl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # NOTE: this vectorized form materializes a full (n_a, n_b, k) `diff`
    # broadcast temporary (and a second same-shape `diff*diff` temporary)
    # -- at the largest configured size (5,000 x 5,000 x 16) that is
    # 5000*5000*16*4 bytes ~= 1.6 GiB per temporary, ~3.2 GiB total. This
    # is a genuine algorithmic difference from both the CPU (@njit) and
    # GPU kernels below, which never materialize more than O(n_a*n_b)
    # output plus O(1) per-thread scratch -- not a fairness bug to "fix,"
    # but a real memory-footprint difference worth calling out rather
    # than treating NumPy's wall-clock number as a clean apples-to-apples
    # comparison.
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def python_impl(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n_a, n_b, k = len(a), len(b), len(a[0])
    out = [[0.0] * n_b for _ in range(n_a)]
    for i in range(n_a):
        for j in range(n_b):
            s = 0.0
            for d in range(k):
                diff = a[i][d] - b[j][d]
                s += diff * diff
            out[i][j] = s
    return out


def _make_numba_cpu_impl(*, parallel: bool):
    from numba import njit, prange

    loop_range = prange if parallel else range

    @njit(parallel=parallel, fastmath=False, cache=True)
    def numba_cpu_impl(a, b, out, n_a, n_b, k):
        for i in loop_range(n_a):
            for j in range(n_b):
                s = 0.0
                for d in range(k):
                    diff = a[i * k + d] - b[j * k + d]
                    s += diff * diff
                out[i * n_b + j] = s

    return numba_cpu_impl


def _make_metal_kernel():
    from numba_metal import metal

    @metal.jit
    def metal_kernel(a, b, out, n_a, n_b, k):
        idx = metal.grid(1)
        total = n_a * n_b
        if idx < total:
            i = idx // n_b
            j = idx % n_b
            s = 0.0
            for d in range(k):
                diff = a[i * k + d] - b[j * k + d]
                s = s + diff * diff
            out[idx] = s

    return metal_kernel


def run(
    sizes: list[tuple[int, int, int]] = SIZES, run_python: bool = True
) -> list[BenchmarkResult]:
    results = []
    metal_available = require_metal_or_skip()
    if metal_available:
        from numba_metal import metal

    for n_a, n_b, k in sizes:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((n_a, k)).astype(np.float32)
        b = rng.standard_normal((n_b, k)).astype(np.float32)
        expected = numpy_impl(a, b)

        size_label = f"{n_a}x{n_b}x{k}"
        result = BenchmarkResult(benchmark="Pairwise distance", size_label=size_label)

        if run_python and n_a * n_b <= 40_000:
            a_list, b_list = a.tolist(), b.tolist()
            t_py = time_repeated(
                lambda: python_impl(a_list, b_list), warmup=0, repeats=1
            )
            result.python_ns = t_py.median_ns

        t_numpy = time_repeated(lambda: numpy_impl(a, b), warmup=2, repeats=5)
        result.numpy_ns = t_numpy.median_ns

        a_flat, b_flat = a.reshape(-1), b.reshape(-1)

        cpu_parallel = _make_numba_cpu_impl(parallel=True)
        out_cpu = np.empty(n_a * n_b, dtype=np.float32)
        cpu_parallel(a_flat, b_flat, out_cpu, n_a, n_b, k)
        t_cpu_par = time_repeated(
            lambda: cpu_parallel(a_flat, b_flat, out_cpu, n_a, n_b, k),
            warmup=2,
            repeats=5,
        )
        result.numba_cpu_parallel_ns = t_cpu_par.median_ns
        import numba

        result.numba_cpu_num_threads = numba.get_num_threads()
        cpu_ok, cpu_note = assert_allclose(
            out_cpu.reshape(n_a, n_b), expected, rtol=1e-3, atol=1e-3
        )

        cpu_single = _make_numba_cpu_impl(parallel=False)
        out_cpu_single = np.empty(n_a * n_b, dtype=np.float32)
        cpu_single(a_flat, b_flat, out_cpu_single, n_a, n_b, k)
        t_cpu_single = run_single_threaded(
            lambda: time_repeated(
                lambda: cpu_single(a_flat, b_flat, out_cpu_single, n_a, n_b, k),
                warmup=2,
                repeats=5,
            )
        )
        result.numba_cpu_single_ns = t_cpu_single.median_ns

        if metal_available:
            metal_kernel = _make_metal_kernel()
            # Consistent output allocation: `d_out` is allocated once and
            # reused across every timed section below (cold, warm,
            # transfer) rather than freshly allocated per call, matching
            # how the CPU baselines reuse `out_cpu`/`out_cpu_single`.
            d_a = metal.to_device(a_flat)
            d_b = metal.to_device(b_flat)
            d_out = metal.device_array(n_a * n_b, np.float32)
            threads = 256
            blocks = (n_a * n_b + threads - 1) // threads

            def launch(kernel=None):
                k_ = kernel or metal_kernel
                k_[blocks, threads](
                    d_a, d_b, d_out, np.int32(n_a), np.int32(n_b), np.int32(k)
                )
                metal.synchronize()

            cold = measure_cold_end_to_end(metal_kernel.py_func, launch)
            result.metal_cold_total_ns = cold["total_ns"]
            result.compiled_before = cold["compiled_before"]
            result.compiled_after = cold["compiled_after"]

            arg_types = infer_launch_arg_types(
                d_a, d_b, d_out, np.int32(n_a), np.int32(n_b), np.int32(k)
            )
            phases = measure_cold_metal_pipeline(metal_kernel.py_func, arg_types)
            result.metal_frontend_ns = phases["frontend_ns"]
            result.metal_pipeline_compile_ns = phases["pipeline_compile_ns"]

            launch()
            t_kernel_warm = time_repeated(launch, warmup=2, repeats=5)
            result.metal_kernel_only_warm_ns = t_kernel_warm.median_ns
            # Separate D2H transfer (of the full n_a*n_b output) from
            # kernel execution above.
            t_d2h = time_repeated(lambda: d_out.copy_to_host(), warmup=1, repeats=5)
            result.metal_d2h_ns = t_d2h.median_ns
            result.metal_end_to_end_warm_ns = (
                result.metal_kernel_only_warm_ns + result.metal_d2h_ns
            )

            gpu_result = d_out.copy_to_host().reshape(n_a, n_b)
            gpu_ok, gpu_note = assert_allclose(
                gpu_result, expected, rtol=1e-3, atol=1e-3
            )
            result.correctness_ok = cpu_ok and gpu_ok
            result.correctness_note = f"cpu: {cpu_note}; gpu: {gpu_note}"
        else:
            result.correctness_ok = cpu_ok
            result.correctness_note = f"cpu: {cpu_note}; gpu: unavailable"

        results.append(result)
    return results


if __name__ == "__main__":
    from common import print_table

    print_table(run())
