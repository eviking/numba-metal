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
    require_metal_or_skip,
    time_repeated,
)

# (n_a, n_b, k) problem sizes: two vector-set sizes and the dimensionality.
SIZES = [(200, 200, 8), (2_000, 2_000, 8), (5_000, 5_000, 16)]


def numpy_impl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def _make_numba_cpu_impl():
    from numba import njit, prange

    @njit(parallel=True, fastmath=False, cache=True)
    def numba_cpu_impl(a, b, out, n_a, n_b, k):
        for i in prange(n_a):
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
    numba_cpu_impl = _make_numba_cpu_impl()
    metal_available = require_metal_or_skip()
    metal_kernel = _make_metal_kernel() if metal_available else None
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
        out_cpu = np.empty(n_a * n_b, dtype=np.float32)
        numba_cpu_impl(a_flat, b_flat, out_cpu, n_a, n_b, k)
        t_cpu = time_repeated(
            lambda: numba_cpu_impl(a_flat, b_flat, out_cpu, n_a, n_b, k),
            warmup=2,
            repeats=5,
        )
        result.numba_cpu_ns = t_cpu.median_ns
        cpu_ok, cpu_note = assert_allclose(
            out_cpu.reshape(n_a, n_b), expected, rtol=1e-3, atol=1e-3
        )

        if metal_available:
            d_a = metal.to_device(a_flat)
            d_b = metal.to_device(b_flat)
            d_out = metal.device_array(n_a * n_b, np.float32)
            threads = 256
            blocks = (n_a * n_b + threads - 1) // threads

            def launch():
                metal_kernel[blocks, threads](
                    d_a, d_b, d_out, np.int32(n_a), np.int32(n_b), np.int32(k)
                )
                metal.synchronize()

            t_cold = time_repeated(launch, warmup=0, repeats=1)
            result.metal_cold_ns = t_cold.median_ns
            t_warm = time_repeated(launch, warmup=2, repeats=5)
            result.metal_warm_ns = t_warm.median_ns
            t_d2h = time_repeated(lambda: d_out.copy_to_host(), warmup=1, repeats=5)
            result.metal_d2h_ns = t_d2h.median_ns

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
