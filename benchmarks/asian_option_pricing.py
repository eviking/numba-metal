"""Benchmark: Monte Carlo Asian (average-price) option pricing.

An Asian option's payoff depends on the AVERAGE price of the underlying
over the life of the contract, not just its price at expiry (a European
option's payoff). That averaging has no closed-form solution under
geometric Brownian motion -- unlike the European call in
`monte_carlo_paths.py`, which can be checked against Black-Scholes,
Asian options are priced by simulation in practice, not simulated as a
teaching substitute for an existing formula. This makes it a more
honest example of when Monte Carlo is actually the tool for the job.

Structurally, each simulated path does MORE arithmetic per path than
the European-call benchmark at the same step count: a running-average
accumulation every step (`running_sum += price`), in addition to the
same geometric-Brownian-motion price update. Same memory footprint per
path (one random-draw array in, one terminal-payoff-relevant value
out) as monte_carlo_paths.py, but more FLOPs per byte moved -- the
exact lever discussed in docs/performance-guidance.md for moving a
workload from bandwidth-bound toward compute-bound. This benchmark
pushes path count and step count much higher than
monte_carlo_paths.py's defaults specifically to show how far that
lever can push the achieved speedup, not just to demonstrate parity.

No closed-form reference exists, so correctness here means CPU-parallel
and GPU results agreeing with each other AND with a single-threaded CPU
run -- not agreement with an analytic price like the European-call
benchmark's Black-Scholes check.

Measured on an Apple M4 Pro (see `if __name__` below for a fresh run):
roughly 12x at 100,000 paths, climbing to roughly 13x at 2,000,000
paths -- the strongest compute-bound result in this project's own
benchmark suite, and already a double-digit win at the smallest size
tested (no dispatch-bound loss region to cross, unlike the cyclist
example, since 500 accumulating steps per path is enough arithmetic to
clear the dispatch-overhead floor even at the smallest size here).

`z` is a genuine 2D array (n_paths, n_steps), indexed with real 2D
indexing (`z[p, step]`) on both the Metal kernel and the Numba-CPU
reference rather than a flattened 1D index (`z[p*n_steps+step]`) --
applied to both sides together, matching monte_carlo_paths.py's own
conversion and its finding: a sequential, stride-1 per-thread sweep
gains nothing measurable from native 2D indexing over manual flat-index
arithmetic, since both compile to the same row-major offset math.

Also compares a CPU-side reduction (kernel writes the average price;
`payoff.mean()` runs on the host after `copy_to_host()`) against a
GPU-side one (`_make_metal_payoff_kernel` writes the payoff directly;
`metal.reduce_sum()` sums it on the GPU, so only a 1-element buffer
round-trips to the host) -- the same comparison monte_carlo_paths.py
makes, at this file's larger problem sizes (up to 2,000,000 paths,
n_steps=500) to see whether the GPU-reduced win (found there to grow
with array size, from a loss at 10,000 paths to ~16% faster at
10,000,000) persists here too.
"""

from __future__ import annotations

import math
import sys
import time
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

SIZES = [100_000, 1_000_000, 2_000_000]
N_STEPS = 500
#: 500 steps is 5x monte_carlo_paths.py's 100 -- more arithmetic per path
#: at the same path counts, the exact lever docs/performance-guidance.md
#: describes for pushing a workload toward the compute-bound regime.
#: Kept well under monte_carlo_paths.py's own largest path count (which
#: tops out at 2,000,000) specifically to avoid NumPy's reference
#: implementation (`numpy_impl`, used only for the "naive NumPy" timing
#: column, never for correctness) allocating several (n_paths, n_steps)
#: float32 intermediates at once -- at 2,000,000 x 500 that is ~4GB per
#: array already; a naive push to 10,000,000 paths here was tried and
#: found to allocate close to this machine's entire 25.8GB of RAM before
#: even reaching the timed section, and was abandoned in favor of this
#: safer ceiling.
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0


def numpy_impl(z: np.ndarray, n_steps: int) -> np.ndarray:
    """z: (n_paths, n_steps) standard normal draws. Returns each path's
    average price over the n_steps observations (the Asian payoff's
    underlying quantity)."""
    dt = T / n_steps
    drift = (R - 0.5 * SIGMA * SIGMA) * dt
    diffusion = SIGMA * np.sqrt(dt)
    log_returns = drift + diffusion * z
    log_s = np.log(S0) + np.cumsum(log_returns, axis=1)
    prices = np.exp(log_s)
    return prices.mean(axis=1)


def _make_numba_cpu_impl(*, parallel: bool):
    from numba import njit, prange

    loop_range = prange if parallel else range

    @njit(parallel=parallel, fastmath=False, cache=True)
    def numba_cpu_impl(z, out, n_paths, n_steps, s0, r, sigma, t):
        dt = t / n_steps
        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * np.sqrt(dt)
        for p in loop_range(n_paths):
            log_s = np.log(s0)
            running_sum = np.float32(0.0)
            for step in range(n_steps):
                log_s = log_s + drift + diffusion * z[p, step]
                running_sum += np.exp(log_s)
            out[p] = running_sum / n_steps

    return numba_cpu_impl


def _make_metal_kernel():
    from numba_metal import metal

    @metal.jit
    def metal_kernel(z, out, n_steps, s0, drift, diffusion):
        p = metal.grid(1)
        if p < out.size:
            log_s = math.log(s0)
            running_sum = 0.0
            for step in range(n_steps):
                log_s = log_s + drift + diffusion * z[p, step]
                running_sum = running_sum + math.exp(log_s)
            out[p] = running_sum / n_steps

    return metal_kernel


def _make_metal_payoff_kernel():
    """Like `_make_metal_kernel`, but writes each path's discounted-later
    call-option payoff (`max(avg_price - K, 0)`) directly, instead of the
    raw average price -- mirrors monte_carlo_paths.py's
    `_make_metal_payoff_kernel`, so the "average price -> payoff" step
    that would otherwise run on the CPU after `copy_to_host()` happens on
    the GPU, leaving only a single scalar `metal.reduce_sum()` (see
    numba_metal/reductions.py) plus a tiny host-side mean/discount
    computation. Applying this here too (not just in
    monte_carlo_paths.py) matters more at this file's scale: n_steps=500
    (5x monte_carlo_paths.py's 100) means each path's payoff array is the
    same size but the compute-to-transfer ratio is already high, so this
    tests whether the GPU-reduced win persists at the largest problem
    sizes this benchmark suite exercises (up to 2,000,000 paths)."""
    from numba_metal import metal

    @metal.jit
    def metal_payoff_kernel(z, out, n_steps, s0, k, drift, diffusion):
        p = metal.grid(1)
        if p < out.size:
            log_s = math.log(s0)
            running_sum = 0.0
            for step in range(n_steps):
                log_s = log_s + drift + diffusion * z[p, step]
                running_sum = running_sum + math.exp(log_s)
            avg_price = running_sum / n_steps
            payoff = avg_price - k
            if payoff < 0.0:
                payoff = 0.0
            out[p] = payoff

    return metal_payoff_kernel


def run(sizes: list[int] = SIZES, n_steps: int = N_STEPS) -> list[BenchmarkResult]:
    results = []
    metal_available = require_metal_or_skip()
    if metal_available:
        from numba_metal import metal

    for n_paths in sizes:
        print(
            f"  n_paths={n_paths:,} n_steps={n_steps} starting...",
            file=sys.stderr,
            flush=True,
        )
        size_start = time.perf_counter()
        # RNG (standard-normal draws) is generated once, outside every
        # timed section below -- none of the timed callables regenerate
        # random numbers, so "computation time" never includes RNG cost.
        rng = np.random.default_rng(42)
        z = rng.standard_normal((n_paths, n_steps)).astype(np.float32)

        result = BenchmarkResult(
            benchmark="Asian option pricing", size_label=f"{n_paths:,}"
        )

        def numpy_price():
            avg_price = numpy_impl(z, n_steps)
            payoff = np.maximum(avg_price - K, 0.0)
            return np.exp(-R * T) * payoff.mean()

        t_numpy = time_repeated(numpy_price, warmup=1, repeats=3)
        result.numpy_ns = t_numpy.median_ns
        numpy_estimate = numpy_price()

        out_cpu = np.empty(n_paths, dtype=np.float32)

        cpu_parallel = _make_numba_cpu_impl(parallel=True)

        def cpu_kernel_only():
            cpu_parallel(z, out_cpu, n_paths, n_steps, S0, R, SIGMA, T)

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
            cpu_single(z, out_cpu_single, n_paths, n_steps, S0, R, SIGMA, T)

        cpu_kernel_only_single()  # warm up / compile
        t_cpu_single = run_single_threaded(
            lambda: time_repeated(cpu_kernel_only_single, warmup=1, repeats=3)
        )
        result.numba_cpu_single_ns = t_cpu_single.median_ns

        # Parallel vs. single-threaded CPU agreement is the correctness
        # anchor here -- no closed-form Asian-option price exists to
        # check against (unlike the European call in
        # monte_carlo_paths.py, which has Black-Scholes).
        cpu_ok, cpu_note = assert_allclose(
            out_cpu, out_cpu_single, rtol=1e-4, atol=1e-4
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
                avg_price = d_out.copy_to_host()
                payoff = np.maximum(avg_price - K, 0.0)
                return float(np.exp(-R * T) * payoff.mean())

            t_e2e = time_repeated(launch_and_reduce, warmup=2, repeats=5)
            result.metal_end_to_end_warm_ns = t_e2e.median_ns

            gpu_avg_price = d_out.copy_to_host()
            gpu_ok, gpu_note = assert_allclose(
                gpu_avg_price, out_cpu_single, rtol=1e-4, atol=1e-4
            )
            gpu_estimate = float(
                np.exp(-R * T) * np.maximum(gpu_avg_price - K, 0.0).mean()
            )

            # GPU-reduced variant: the payoff kernel writes
            # max(avg_price-K,0) directly, and metal.reduce_sum() sums it
            # on the GPU -- only a 1-element buffer round-trips to the
            # host, instead of the full n_paths-element array. Compares
            # directly against `launch_and_reduce` above (identical work,
            # different reduction strategy) for an honest, apples-to
            # -apples before/after of this specific primitive, matching
            # monte_carlo_paths.py's own comparison.
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
                np.array([numpy_estimate]),
                rtol=0.05,
                atol=0.5,
            )

            result.correctness_ok = cpu_ok and gpu_ok and gpu_reduced_ok
            result.correctness_note = (
                f"cpu_par={cpu_estimate:.4f} gpu={gpu_estimate:.4f} "
                f"gpu_reduced={gpu_reduced_estimate:.4f} "
                f"numpy={numpy_estimate:.4f} (cpu-vs-cpu: {cpu_note}; "
                f"gpu-vs-cpu: {gpu_note}; gpu_reduced: {gpu_reduced_note})"
            )
            result.extra["gpu_estimate"] = gpu_estimate
            result.extra["gpu_reduced_estimate"] = gpu_reduced_estimate
        else:
            result.correctness_ok = cpu_ok
            result.correctness_note = (
                f"cpu_par={cpu_estimate:.4f} numpy={numpy_estimate:.4f}; "
                f"gpu: unavailable ({cpu_note})"
            )

        results.append(result)
        print(
            f"  n_paths={n_paths:,} done in {time.perf_counter() - size_start:.1f}s",
            file=sys.stderr,
            flush=True,
        )
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
