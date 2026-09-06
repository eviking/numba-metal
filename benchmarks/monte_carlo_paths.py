"""Benchmark 4: Monte Carlo European call option pricing via GBM paths.

Simulates many independent geometric Brownian motion price paths, each
processed by one GPU thread across many time steps, then computes the
discounted call-option payoff. Random draws (standard normals) are
generated on the CPU/host with NumPy for the MVP (documented boundary --
no GPU RNG is implemented); the terminal-value aggregation (mean payoff,
discounting) is also done on the CPU since numba-metal does not yet
implement GPU-side reductions (see docs/limitations.md and
docs/roadmap.md).

`z` is a genuine 2D array (n_paths, n_steps) and is indexed with real 2D
indexing (`z[p, step]`) on both the Metal kernel and the Numba-CPU
reference, not a flattened 1D index (`z[p*n_steps+step]`) -- applied to
both sides together, per the fair-comparison methodology used elsewhere
in this benchmark suite (see heat_diffusion.py/mandelbrot.py). Measured
result: this access pattern is a sequential, stride-1 sweep down one
row per thread either way (row-major offset math `p*n_steps+step` is
identical whether written as `z[p,step]` or by hand), so unlike
Mandelbrot's spatial pixel grid or pairwise_distance's transposed-ish
access, native 2D indexing here produced no measurable difference --
confirmed by direct before/after timing, not assumed.
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

SIZES = [10_000, 200_000, 2_000_000]
N_STEPS = 100
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0


def _black_scholes_call(s0, k, r, sigma, t) -> float:
    from math import erf, exp, log, sqrt

    d1 = (log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)

    def norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    return s0 * norm_cdf(d1) - k * exp(-r * t) * norm_cdf(d2)


def numpy_impl(z: np.ndarray, n_steps: int) -> np.ndarray:
    """z: (n_paths, n_steps) standard normal draws. Returns terminal prices."""
    dt = T / n_steps
    drift = (R - 0.5 * SIGMA * SIGMA) * dt
    diffusion = SIGMA * np.sqrt(dt)
    log_returns = drift + diffusion * z
    log_s = np.log(S0) + np.cumsum(log_returns, axis=1)
    return np.exp(log_s[:, -1])


def _make_numba_cpu_impl(*, parallel: bool):
    from numba import njit, prange

    loop_range = prange if parallel else range

    @njit(parallel=parallel, fastmath=False, cache=True)
    def numba_cpu_impl(z, out, n_paths, n_steps, s0, k, r, sigma, t):
        dt = t / n_steps
        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * np.sqrt(dt)
        for p in loop_range(n_paths):
            log_s = np.log(s0)
            for step in range(n_steps):
                log_s = log_s + drift + diffusion * z[p, step]
            out[p] = np.exp(log_s)

    return numba_cpu_impl


def _make_metal_kernel():
    from numba_metal import metal

    @metal.jit
    def metal_kernel(z, out, n_steps, s0, drift, diffusion):
        p = metal.grid(1)
        if p < out.size:
            log_s = math.log(s0)
            for step in range(n_steps):
                log_s = log_s + drift + diffusion * z[p, step]
            out[p] = math.exp(log_s)

    return metal_kernel


def _make_metal_payoff_kernel():
    """Like `_make_metal_kernel`, but writes each path's call-option
    payoff (`max(terminal - K, 0)`) directly, instead of the raw terminal
    price -- so the entire "terminal value -> payoff" step that would
    otherwise run on the CPU after `copy_to_host()` happens on the GPU,
    leaving only a single scalar `metal.reduce_sum()` + a tiny host-side
    mean/discount computation before the estimate is done. This is what
    `metal.reduce_sum` (see numba_metal/reductions.py) is for: replacing
    the current "download the whole per-path array, reduce on the CPU"
    pattern documented as a boundary in this file's own module docstring
    and in docs/roadmap.md's Phase 2 reductions entry.
    """
    from numba_metal import metal

    @metal.jit
    def metal_payoff_kernel(z, out, n_steps, s0, k, drift, diffusion):
        p = metal.grid(1)
        if p < out.size:
            log_s = math.log(s0)
            for step in range(n_steps):
                log_s = log_s + drift + diffusion * z[p, step]
            terminal = math.exp(log_s)
            payoff = terminal - k
            if payoff < 0.0:
                payoff = 0.0
            out[p] = payoff

    return metal_payoff_kernel


def run(sizes: list[int] = SIZES, n_steps: int = N_STEPS) -> list[BenchmarkResult]:
    results = []
    metal_available = require_metal_or_skip()
    if metal_available:
        from numba_metal import metal

    bs_price = _black_scholes_call(S0, K, R, SIGMA, T)

    for n_paths in sizes:
        # RNG (standard-normal draws) is generated once, outside every
        # timed section below -- none of the timed callables regenerate
        # random numbers, so "computation time" never includes RNG cost.
        rng = np.random.default_rng(42)
        z = rng.standard_normal((n_paths, n_steps)).astype(np.float32)

        result = BenchmarkResult(
            benchmark="Monte Carlo paths", size_label=f"{n_paths:,}"
        )

        def numpy_price():
            terminal = numpy_impl(z, n_steps)
            payoff = np.maximum(terminal - K, 0.0)
            return np.exp(-R * T) * payoff.mean()

        t_numpy = time_repeated(numpy_price, warmup=1, repeats=3)
        result.numpy_ns = t_numpy.median_ns
        numpy_estimate = numpy_price()

        out_cpu = np.empty(n_paths, dtype=np.float32)

        cpu_parallel = _make_numba_cpu_impl(parallel=True)

        def cpu_kernel_only():
            cpu_parallel(z, out_cpu, n_paths, n_steps, S0, K, R, SIGMA, T)

        cpu_kernel_only()  # warm up / compile
        t_cpu_par = time_repeated(cpu_kernel_only, warmup=1, repeats=3)
        result.numba_cpu_parallel_ns = t_cpu_par.median_ns
        import numba

        result.numba_cpu_num_threads = numba.get_num_threads()
        payoff = np.maximum(out_cpu - K, 0.0)
        cpu_estimate = float(np.exp(-R * T) * payoff.mean())

        cpu_single = _make_numba_cpu_impl(parallel=False)
        out_cpu_single = np.empty(n_paths, dtype=np.float32)

        def cpu_kernel_only_single():
            cpu_single(z, out_cpu_single, n_paths, n_steps, S0, K, R, SIGMA, T)

        cpu_kernel_only_single()  # warm up / compile
        t_cpu_single = run_single_threaded(
            lambda: time_repeated(cpu_kernel_only_single, warmup=1, repeats=3)
        )
        result.numba_cpu_single_ns = t_cpu_single.median_ns

        # Separate CPU reduction (mean payoff / discounting, a tiny NumPy
        # pass over the terminal-price array) from kernel execution time
        # above -- reduction cost is reported here, not folded silently
        # into numba_cpu_parallel_ns/numba_cpu_single_ns.
        t_cpu_reduction = time_repeated(
            lambda: float(np.exp(-R * T) * np.maximum(out_cpu - K, 0.0).mean()),
            warmup=1,
            repeats=5,
        )
        result.extra["numba_cpu_reduction_ns"] = t_cpu_reduction.median_ns

        cpu_ok, cpu_note = assert_allclose(
            np.array([cpu_estimate]), np.array([bs_price]), rtol=0.05, atol=0.5
        )

        if metal_available:
            dt = T / n_steps
            drift = np.float32((R - 0.5 * SIGMA * SIGMA) * dt)
            diffusion = np.float32(SIGMA * np.sqrt(dt))

            metal_kernel = _make_metal_kernel()
            d_z = metal.to_device(z)
            d_out = metal.device_array(n_paths, np.float32)
            threads = 256
            blocks = (n_paths + threads - 1) // threads

            def kernel_launch(kernel=None):
                k = kernel or metal_kernel
                k[blocks, threads](
                    d_z, d_out, np.int32(n_steps), np.float32(S0), drift, diffusion
                )
                metal.synchronize()

            cold = measure_cold_end_to_end(metal_kernel.py_func, kernel_launch)
            result.metal_cold_total_ns = cold["total_ns"]
            result.compiled_before = cold["compiled_before"]
            result.compiled_after = cold["compiled_after"]

            arg_types = infer_launch_arg_types(
                d_z, d_out, np.int32(n_steps), np.float32(S0), drift, diffusion
            )
            phases = measure_cold_metal_pipeline(metal_kernel.py_func, arg_types)
            result.metal_frontend_ns = phases["frontend_ns"]
            result.metal_pipeline_compile_ns = phases["pipeline_compile_ns"]

            # Kernel-only warm time: buffers already resident, no
            # transfer, no reduction -- pure GPU compute.
            kernel_launch()
            t_kernel_warm = time_repeated(kernel_launch, warmup=2, repeats=5)
            result.metal_kernel_only_warm_ns = t_kernel_warm.median_ns

            t_h2d = time_repeated(lambda: metal.to_device(z), warmup=1, repeats=5)
            result.metal_h2d_ns = t_h2d.median_ns
            t_d2h = time_repeated(lambda: d_out.copy_to_host(), warmup=1, repeats=5)
            result.metal_d2h_ns = t_d2h.median_ns

            def launch_and_reduce():
                kernel_launch()
                terminal = d_out.copy_to_host()
                payoff = np.maximum(terminal - K, 0.0)
                return float(np.exp(-R * T) * payoff.mean())

            t_e2e = time_repeated(launch_and_reduce, warmup=2, repeats=5)
            result.metal_end_to_end_warm_ns = t_e2e.median_ns

            gpu_estimate = launch_and_reduce()
            gpu_ok, gpu_note = assert_allclose(
                np.array([gpu_estimate]), np.array([bs_price]), rtol=0.05, atol=0.5
            )

            # GPU-reduced variant: the payoff kernel writes max(terminal-K,0)
            # directly, and metal.reduce_sum() (numba_metal/reductions.py)
            # sums it on the GPU -- only a 1-element buffer round-trips to
            # the host, instead of the full n_paths-element array. Compares
            # directly against `launch_and_reduce` above (identical work,
            # different reduction strategy) for an honest, apples-to-apples
            # before/after of this specific primitive.
            payoff_kernel = _make_metal_payoff_kernel()
            d_payoff = metal.device_array(n_paths, np.float32)

            def payoff_launch():
                payoff_kernel[blocks, threads](
                    d_z,
                    d_payoff,
                    np.int32(n_steps),
                    np.float32(S0),
                    np.float32(K),
                    drift,
                    diffusion,
                )

            def launch_and_reduce_on_gpu():
                payoff_launch()
                total = metal.reduce_sum(d_payoff)
                return float(np.exp(-R * T) * total.copy_to_host()[0] / n_paths)

            launch_and_reduce_on_gpu()  # warm up / compile
            t_e2e_gpu_reduced = time_repeated(
                launch_and_reduce_on_gpu, warmup=2, repeats=5
            )
            result.extra["metal_gpu_reduced_end_to_end_ns"] = (
                t_e2e_gpu_reduced.median_ns
            )

            gpu_reduced_estimate = launch_and_reduce_on_gpu()
            gpu_reduced_ok, gpu_reduced_note = assert_allclose(
                np.array([gpu_reduced_estimate]),
                np.array([bs_price]),
                rtol=0.05,
                atol=0.5,
            )

            result.correctness_ok = cpu_ok and gpu_ok and gpu_reduced_ok
            result.correctness_note = (
                f"BS analytic={bs_price:.4f} cpu={cpu_estimate:.4f} "
                f"gpu={gpu_estimate:.4f} gpu_reduced={gpu_reduced_estimate:.4f} "
                f"numpy={numpy_estimate:.4f}; gpu_reduced: {gpu_reduced_note}"
            )
            result.extra["black_scholes_price"] = bs_price
            result.extra["gpu_estimate"] = gpu_estimate
            result.extra["gpu_reduced_estimate"] = gpu_reduced_estimate
        else:
            result.correctness_ok = cpu_ok
            result.correctness_note = (
                f"BS analytic={bs_price:.4f} cpu={cpu_estimate:.4f}; gpu: unavailable"
            )

        results.append(result)
    return results


if __name__ == "__main__":
    from common import format_ns, print_table

    results = run()
    print_table(results)
    print()
    for r in results:
        e2e_cpu_reduced = r.metal_end_to_end_warm_ns
        e2e_gpu_reduced = r.extra.get("metal_gpu_reduced_end_to_end_ns")
        if e2e_gpu_reduced is not None and e2e_cpu_reduced:
            print(
                f"{r.benchmark} {r.size_label}: end-to-end with CPU-side "
                f"reduction={format_ns(e2e_cpu_reduced)} vs. GPU-side "
                f"reduction (metal.reduce_sum)={format_ns(e2e_gpu_reduced)}"
            )
