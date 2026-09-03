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
    infer_launch_arg_types,
    measure_cold_end_to_end,
    measure_cold_metal_pipeline,
    require_metal_or_skip,
    run_single_threaded,
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


def _make_numba_cpu_impl(*, parallel: bool):
    # fastmath=True here (unlike mandelbrot.py's fastmath=False) is
    # deliberate and benign for this benchmark: it is a single
    # non-iterative elementwise expression with no chaotic recurrence, so
    # FMA fusion/reassociation cannot compound into a large divergence --
    # assert_allclose's rtol/atol=1e-4 tolerance already accounts for
    # ordinary float32 rounding-order differences.
    from numba import njit, prange

    loop_range = prange if parallel else range

    @njit(parallel=parallel, fastmath=True, cache=True)
    def numba_cpu_impl(a, b, out):
        for i in loop_range(a.shape[0]):
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
    metal_available = require_metal_or_skip()
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

        # Parallel (default thread count, via prange) and single-threaded
        # (set_num_threads(1)) Numba CPU variants are separate compiled
        # dispatchers -- see _make_numba_cpu_impl's loop_range switch.
        cpu_parallel = _make_numba_cpu_impl(parallel=True)
        out_cpu = np.empty_like(a)
        cpu_parallel(a, b, out_cpu)  # warm up / compile
        t_cpu_par = time_repeated(
            lambda: cpu_parallel(a, b, out_cpu), warmup=2, repeats=7
        )
        result.numba_cpu_parallel_ns = t_cpu_par.median_ns
        import numba

        result.numba_cpu_num_threads = numba.get_num_threads()
        cpu_ok, cpu_note = assert_allclose(out_cpu, expected)

        cpu_single = _make_numba_cpu_impl(parallel=False)
        out_cpu_single = np.empty_like(a)
        cpu_single(a, b, out_cpu_single)  # warm up / compile
        t_cpu_single = run_single_threaded(
            lambda: time_repeated(
                lambda: cpu_single(a, b, out_cpu_single), warmup=2, repeats=7
            )
        )
        result.numba_cpu_single_ns = t_cpu_single.median_ns

        if metal_available:
            metal_kernel = _make_metal_kernel()
            d_a = metal.to_device(a)
            d_b = metal.to_device(b)
            d_out = metal.device_array_like(a)
            threads = 256
            blocks = (size + threads - 1) // threads

            def launch(kernel=None):
                k = kernel or metal_kernel
                k[blocks, threads](d_a, d_b, d_out)
                metal.synchronize()

            cold = measure_cold_end_to_end(metal_kernel.py_func, launch)
            result.metal_cold_total_ns = cold["total_ns"]
            result.compiled_before = cold["compiled_before"]
            result.compiled_after = cold["compiled_after"]

            arg_types = infer_launch_arg_types(d_a, d_b, d_out)
            phases = measure_cold_metal_pipeline(metal_kernel.py_func, arg_types)
            result.metal_frontend_ns = phases["frontend_ns"]
            result.metal_pipeline_compile_ns = phases["pipeline_compile_ns"]

            launch()  # ensure warm cache on the persistent kernel used below
            t_kernel_warm = time_repeated(launch, warmup=2, repeats=7)
            result.metal_kernel_only_warm_ns = t_kernel_warm.median_ns

            t_h2d = time_repeated(
                lambda: (metal.to_device(a), metal.to_device(b)), warmup=1, repeats=5
            )
            result.metal_h2d_ns = t_h2d.median_ns
            t_d2h = time_repeated(lambda: d_out.copy_to_host(), warmup=1, repeats=5)
            result.metal_d2h_ns = t_d2h.median_ns

            def end_to_end():
                d_a2 = metal.to_device(a)
                d_b2 = metal.to_device(b)
                d_out2 = metal.device_array_like(a)
                metal_kernel[blocks, threads](d_a2, d_b2, d_out2)
                metal.synchronize()
                return d_out2.copy_to_host()

            t_e2e = time_repeated(end_to_end, warmup=2, repeats=5)
            result.metal_end_to_end_warm_ns = t_e2e.median_ns

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
