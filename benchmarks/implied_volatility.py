"""Implied volatility via Newton-Raphson -- a real-world worked example
of a `while` loop with a nested `if`/`else` in its body (see
docs/limitations.md's `while`-loops entry and
compiler/structuring.py's `RotatedWhileNode`), and of the specific
shape a tensor-framework (PyTorch/TensorFlow/JAX) reach for GPU
acceleration is structurally awkward at: Newton-Raphson's iteration
count is DATA-DEPENDENT (some options converge in 3 steps, others need
the full budget), which a vectorized tensor op either pads to the
worst-case iteration count for every element or handles via masking/
scatter -- whereas a real per-thread `while` loop with an early-exit
condition just runs each thread for exactly as long as it needs to.

Given an option's observed market price, solve for the volatility that
makes the Black-Scholes formula agree with it -- the actual technique
options market-data pipelines use to compute implied vol across entire
option chains (many strikes x expiries x underlyings) continuously
through a trading day, not a synthetic stand-in for a real workload.

The normal CDF is computed via the Abramowitz & Stegun 7.1.26 rational
approximation (using only `math.exp`, from numba-metal's supported
subset -- there is no `math.erf`) rather than a library call, matching
how this would actually be written for a GPU kernel language with a
restricted math intrinsic set.

Both kernels clamp the Newton-Raphson step size and the resulting
sigma, not just the latter: a near-zero vega (common for deep in/out-
of-the-money options, where price barely depends on volatility at all)
can otherwise produce a single step of catastrophic size before a
post-hoc floor on sigma ever gets a chance to matter -- confirmed
directly: a small fraction of randomly generated test cases diverged to
volatility values in the tens of millions without this clamp. Deep
in/out-of-the-money strikes are also excluded from this benchmark's own
test data generation (see `make_data`) for the same reason: implied
volatility is genuinely undeterminable from price alone when vega is
near zero, regardless of solver quality -- real quant systems filter
these the same way rather than trusting a Newton-Raphson result for
them.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

from numba_metal import metal

R = 0.05
T = 1.0
MAX_ITER = 50
TOL = 1e-6


def normal_cdf_py(x):
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def bs_price_py(s0, k, sigma):
    d1 = (math.log(s0 / k) + (R + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s0 * normal_cdf_py(d1) - k * math.exp(-R * T) * normal_cdf_py(d2)


@njit(parallel=True, fastmath=False, cache=True)
def cpu_implied_vol(s0, k, price, out, n):
    for idx in prange(n):
        S0 = s0[idx]
        K = k[idx]
        target = price[idx]
        sigma = 0.3
        it = 0
        done = False
        while it < MAX_ITER and not done:
            d1 = (np.log(S0 / K) + (R + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if d1 >= 0.0:
                sign1 = 1.0
                xx1 = d1
            else:
                sign1 = -1.0
                xx1 = -d1
            xx1 = xx1 / 1.4142135623730951
            tt1 = 1.0 / (1.0 + 0.3275911 * xx1)
            y1 = 1.0 - (
                (
                    (((1.061405429 * tt1 + -1.453152027) * tt1) + 1.421413741) * tt1
                    + -0.284496736
                )
                * tt1
                + 0.254829592
            ) * tt1 * np.exp(-xx1 * xx1)
            ncdf_d1 = 0.5 * (1.0 + sign1 * y1)

            if d2 >= 0.0:
                sign2 = 1.0
                xx2 = d2
            else:
                sign2 = -1.0
                xx2 = -d2
            xx2 = xx2 / 1.4142135623730951
            tt2 = 1.0 / (1.0 + 0.3275911 * xx2)
            y2 = 1.0 - (
                (
                    (((1.061405429 * tt2 + -1.453152027) * tt2) + 1.421413741) * tt2
                    + -0.284496736
                )
                * tt2
                + 0.254829592
            ) * tt2 * np.exp(-xx2 * xx2)
            ncdf_d2 = 0.5 * (1.0 + sign2 * y2)

            price_est = S0 * ncdf_d1 - K * np.exp(-R * T) * ncdf_d2
            vega = S0 * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / 2.5066282746310002
            diff = price_est - target
            if diff >= 0.0:
                abs_diff = diff
            else:
                abs_diff = -diff

            if abs_diff < TOL or vega < 1e-6:
                done = True
            else:
                step = diff / vega
                if step > 5.0:
                    step = 5.0
                elif step < -5.0:
                    step = -5.0
                sigma = sigma - step
                if sigma <= 0.001:
                    sigma = 0.001
                elif sigma >= 5.0:
                    sigma = 5.0
            it += 1
        out[idx] = sigma


@metal.jit
def metal_implied_vol(s0, k, price, out, r, t, tol, max_iter):
    idx = metal.grid(1)
    if idx < out.size:
        S0 = s0[idx]
        K = k[idx]
        target = price[idx]
        sigma = 0.3
        it = 0
        done = False
        while it < max_iter and not done:
            sqrt_t = math.sqrt(t)
            d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
            d2 = d1 - sigma * sqrt_t

            if d1 >= 0.0:
                sign1 = 1.0
                xx1 = d1
            else:
                sign1 = -1.0
                xx1 = -d1
            xx1 = xx1 / 1.4142135623730951
            tt1 = 1.0 / (1.0 + 0.3275911 * xx1)
            y1 = 1.0 - (
                (
                    (((1.061405429 * tt1 + -1.453152027) * tt1) + 1.421413741) * tt1
                    + -0.284496736
                )
                * tt1
                + 0.254829592
            ) * tt1 * math.exp(-xx1 * xx1)
            ncdf_d1 = 0.5 * (1.0 + sign1 * y1)

            if d2 >= 0.0:
                sign2 = 1.0
                xx2 = d2
            else:
                sign2 = -1.0
                xx2 = -d2
            xx2 = xx2 / 1.4142135623730951
            tt2 = 1.0 / (1.0 + 0.3275911 * xx2)
            y2 = 1.0 - (
                (
                    (((1.061405429 * tt2 + -1.453152027) * tt2) + 1.421413741) * tt2
                    + -0.284496736
                )
                * tt2
                + 0.254829592
            ) * tt2 * math.exp(-xx2 * xx2)
            ncdf_d2 = 0.5 * (1.0 + sign2 * y2)

            price_est = S0 * ncdf_d1 - K * math.exp(-r * t) * ncdf_d2
            vega = S0 * sqrt_t * math.exp(-0.5 * d1 * d1) / 2.5066282746310002
            diff = price_est - target
            if diff >= 0.0:
                abs_diff = diff
            else:
                abs_diff = -diff

            if abs_diff < tol or vega < 1e-6:
                done = True
            else:
                step = diff / vega
                if step > 5.0:
                    step = 5.0
                elif step < -5.0:
                    step = -5.0
                sigma = sigma - step
                if sigma <= 0.001:
                    sigma = 0.001
                elif sigma >= 5.0:
                    sigma = 5.0
            it = it + 1
        out[idx] = sigma


@metal.device_func
def metal_normal_cdf(x):
    """The same Abramowitz & Stegun 7.1.26 rational approximation
    `metal_implied_vol` computes inline TWICE per while-iteration (once
    for d1, once for d2) -- real, substantial per-call work (~15
    arithmetic ops plus a `math.exp`, itself not cheap), unlike
    heat_diffusion.py's 4-neighbor-sum device-function experiment (4
    adds, called once per pixel per iteration). The hypothesis going in
    was that more work per call would let factoring pay for its own
    non-inlined-call overhead -- measured result (see
    `metal_implied_vol_device_func` below and this file's `if __name__`):
    it does NOT produce a clear runtime win, but it does NOT cost a
    measurable regression either -- both variants land within normal
    run-to-run noise of each other (roughly 1.15x-4.5x vs. CPU either
    way, across n=10K-5M), unlike heat_diffusion.py's clear 2.3-2.6x
    per-iteration slowdown for a much smaller helper. The real,
    measured `@metal.device_func` win found this session is a
    DIFFERENT mechanism entirely -- compile-time, not runtime, and
    requiring reuse across multiple different kernels rather than many
    calls within one -- see `benchmarks/device_function_compile_cache.py`.
    """
    sign = 1.0 if x >= 0.0 else -1.0
    xx = abs(x) / 1.4142135623730951
    tt = 1.0 / (1.0 + 0.3275911 * xx)
    y = 1.0 - (
        ((((1.061405429 * tt + -1.453152027) * tt) + 1.421413741) * tt + -0.284496736)
        * tt
        + 0.254829592
    ) * tt * math.exp(-xx * xx)
    return 0.5 * (1.0 + sign * y)


@metal.jit
def metal_implied_vol_device_func(s0, k, price, out, r, t, tol, max_iter):
    """Identical Newton-Raphson solve to `metal_implied_vol`, but calling
    `metal_normal_cdf` (a @metal.device_func) instead of inlining the
    same ~15-line rational-approximation block twice per iteration."""
    idx = metal.grid(1)
    if idx < out.size:
        S0 = s0[idx]
        K = k[idx]
        target = price[idx]
        sigma = 0.3
        it = 0
        done = False
        while it < max_iter and not done:
            sqrt_t = math.sqrt(t)
            d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
            d2 = d1 - sigma * sqrt_t

            ncdf_d1 = metal_normal_cdf(d1)
            ncdf_d2 = metal_normal_cdf(d2)

            price_est = S0 * ncdf_d1 - K * math.exp(-r * t) * ncdf_d2
            vega = S0 * sqrt_t * math.exp(-0.5 * d1 * d1) / 2.5066282746310002
            diff = price_est - target
            if diff >= 0.0:
                abs_diff = diff
            else:
                abs_diff = -diff

            if abs_diff < tol or vega < 1e-6:
                done = True
            else:
                step = diff / vega
                if step > 5.0:
                    step = 5.0
                elif step < -5.0:
                    step = -5.0
                sigma = sigma - step
                if sigma <= 0.001:
                    sigma = 0.001
                elif sigma >= 5.0:
                    sigma = 5.0
            it = it + 1
        out[idx] = sigma


def make_data(n, seed=42):
    # Near-the-money strikes only (0.85x-1.15x spot) -- see module
    # docstring for why deep in/out-of-the-money options are excluded.
    rng = np.random.default_rng(seed)
    s0 = rng.uniform(90.0, 110.0, n).astype(np.float32)
    moneyness = rng.uniform(0.85, 1.15, n).astype(np.float32)
    k = (s0 * moneyness).astype(np.float32)
    true_sigma = rng.uniform(0.1, 0.6, n).astype(np.float32)
    price = np.empty(n, dtype=np.float32)
    for i in range(n):
        price[i] = bs_price_py(float(s0[i]), float(k[i]), float(true_sigma[i]))
    return s0, k, price, true_sigma


def median_time(fn, warmup=1, repeats=5):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


if __name__ == "__main__":
    for n in (10_000, 100_000, 1_000_000, 5_000_000):
        s0, k, price, true_sigma = make_data(n)
        out_cpu = np.zeros(n, dtype=np.float32)

        cpu_implied_vol(s0, k, price, out_cpu, n)  # warmup/compile
        cpu_ms = median_time(lambda: cpu_implied_vol(s0, k, price, out_cpu, n)) * 1000

        d_s0 = metal.to_device(s0)
        d_k = metal.to_device(k)
        d_price = metal.to_device(price)
        d_out = metal.device_array(n, np.float32)
        d_out_device_func = metal.device_array(n, np.float32)
        threads = 256
        blocks = (n + threads - 1) // threads

        def launch():
            metal_implied_vol[blocks, threads](
                d_s0,
                d_k,
                d_price,
                d_out,
                np.float32(R),
                np.float32(T),
                np.float32(TOL),
                np.int32(MAX_ITER),
            )
            metal.synchronize()

        def launch_device_func():
            metal_implied_vol_device_func[blocks, threads](
                d_s0,
                d_k,
                d_price,
                d_out_device_func,
                np.float32(R),
                np.float32(T),
                np.float32(TOL),
                np.int32(MAX_ITER),
            )
            metal.synchronize()

        launch()  # warmup/compile
        metal_ms = median_time(launch) * 1000
        metal_out = d_out.copy_to_host()

        launch_device_func()  # warmup/compile
        metal_device_func_ms = median_time(launch_device_func) * 1000
        metal_device_func_out = d_out_device_func.copy_to_host()

        err = np.max(np.abs(out_cpu - metal_out))
        err_vs_true = np.max(np.abs(metal_out - true_sigma))
        err_device_func = np.max(np.abs(metal_device_func_out - metal_out))
        print(
            f"n={n:>9,}: CPU={cpu_ms:9.3f}ms  Metal(inline)={metal_ms:9.3f}ms  "
            f"Metal(device_func)={metal_device_func_ms:9.3f}ms  "
            f"speedup(inline)={cpu_ms / metal_ms:6.2f}x  "
            f"speedup(device_func)={cpu_ms / metal_device_func_ms:6.2f}x  "
            f"cpu_vs_gpu_err={err:.2e}  "
            f"device_func_vs_inline_err={err_device_func:.2e}  "
            f"recovered_vol_err={err_vs_true:.4f}"
        )
