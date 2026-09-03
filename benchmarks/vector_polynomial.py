"""Benchmark 1: fused vector polynomial.

    out[i] = ((a[i]*1.75 + b[i]*0.25) ** 2 - a[i]*b[i]) / (abs(b[i]) + 1.0)

Compares plain Python, NumPy (vectorized, with intermediate temporaries),
Numba CPU (@njit), and numba-metal, at several problem sizes. Demonstrates
that small arrays may not benefit from GPU dispatch overhead while larger
arrays can.
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

SIZES = [10_000, 1_000_000, 10_000_000]


def python_impl(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * len(a)
    for i in range(len(a)):
        out[i] = ((a[i] * 1.75 + b[i] * 0.25) ** 2 - a[i] * b[i]) / (abs(b[i]) + 1.0)
    return out


def numpy_impl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a * 1.75 + b * 0.25) ** 2 - a * b) / (np.abs(b) + 1.0)


def _make_numba_cpu_impl():
    from numba import njit

    @njit(parallel=True, fastmath=True, cache=True)
    def numba_cpu_impl(a, b, out):
        for i in range(a.shape[0]):
            out[i] = ((a[i] * 1.75 + b[i] * 0.25) ** 2 - a[i] * b[i]) / (
                abs(b[i]) + 1.0
            )
        return out

    return numba_cpu_impl


def _make_metal_kernel():
    from numba_metal import metal

    @metal.jit
    def metal_kernel(a, b, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = ((a[i] * 1.75 + b[i] * 0.25) ** 2 - a[i] * b[i]) / (
                abs(b[i]) + 1.0
            )

    return metal_kernel


def run(sizes: list[int] = SIZES, run_python: bool = True) -> list[BenchmarkResult]:
    results = []
    numba_cpu_impl = _make_numba_cpu_impl()
    metal_available = require_metal_or_skip()
    metal_kernel = _make_metal_kernel() if metal_available else None
    if metal_available:
        from numba_metal import metal

    for size in sizes:
        rng = np.random.default_rng(0)
        a = rng.standard_normal(size).astype(np.float32)
        b = rng.standard_normal(size).astype(np.float32)
        expected = numpy_impl(a, b)

        result = BenchmarkResult(benchmark="Vector polynomial", size_label=f"{size:,}")

        if run_python and size <= 20_000:
            a_list, b_list = a.tolist(), b.tolist()
            t = time_repeated(lambda: python_impl(a_list, b_list), warmup=1, repeats=3)
            result.python_ns = t.median_ns

        t_numpy = time_repeated(lambda: numpy_impl(a, b), warmup=2, repeats=7)
        result.numpy_ns = t_numpy.median_ns

        out_cpu = np.empty_like(a)
        numba_cpu_impl(a, b, out_cpu)  # warm up / compile
        t_cpu = time_repeated(
            lambda: numba_cpu_impl(a, b, out_cpu), warmup=2, repeats=7
        )
        result.numba_cpu_ns = t_cpu.median_ns
        cpu_ok, cpu_note = assert_allclose(out_cpu, expected)

        if metal_available:
            d_a = metal.to_device(a)
            d_b = metal.to_device(b)
            d_out = metal.device_array_like(a)
            threads = 256
            blocks = (size + threads - 1) // threads

            def launch():
                metal_kernel[blocks, threads](d_a, d_b, d_out)
                metal.synchronize()

            t_cold = time_repeated(launch, warmup=0, repeats=1)
            result.metal_cold_ns = t_cold.median_ns
            t_warm = time_repeated(launch, warmup=2, repeats=7)
            result.metal_warm_ns = t_warm.median_ns

            t_h2d = time_repeated(
                lambda: (metal.to_device(a), metal.to_device(b)), warmup=1, repeats=5
            )
            result.metal_h2d_ns = t_h2d.median_ns
            t_d2h = time_repeated(lambda: d_out.copy_to_host(), warmup=1, repeats=5)
            result.metal_d2h_ns = t_d2h.median_ns

            gpu_ok, gpu_note = assert_allclose(d_out.copy_to_host(), expected)
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
