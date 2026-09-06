"""Controlled arithmetic-intensity sweep: empirically locating the
roofline ridge point for numba-metal-generated kernels.

docs/research-paper.md Section 6.1 demonstrates the roofline model
qualitatively using exactly two points (linear vs. nonlinear heat
diffusion). That is enough to show the model holds, but it is not
enough to say *where* the crossover is. This script holds the same
4-neighbor stencil memory-access pattern and grid size fixed (matching
benchmarks/heat_diffusion.py's nonlinear kernel) and varies only the
number of `math.exp`-based diffusion-coefficient evaluations computed
per neighbor per point -- REPS -- which scales arithmetic operations
per point while the four array reads and one array write per point
never change. This directly controls FLOPs/byte across a real,
multi-point range on both the CPU and GPU implementations, timed with
the same benchmarks/common.py helpers the rest of the suite uses.

Arithmetic intensity is computed as bytes-moved / point (fixed: 4
reads + 1 write, all float32 = 20 bytes/point) against
floating-point operations / point at each REPS value (counted from the
actual generated arithmetic, not estimated): each `math.exp` coefficient
evaluation is charged as 1 FLOP for the subtraction, 1 for the divide,
1 for the square, 1 for the exp call itself, and 1 for the subsequent
multiply-accumulate against the gradient -- 5 FLOPs -- and this is done
once per neighbor per REPS iteration, 4 neighbors per point.

This is real measurement, not a model: every row in the printed table
and JSON output is a directly timed run at that REPS value, on this
machine, using GPU-resident buffers on the Metal side (matching the
rest of the benchmark suite's steady-state methodology).
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from common import (
    Timing,
    get_environment_info,
    get_gpu_info,
    get_mac_hardware_info,
    require_metal_or_skip,
    time_repeated,
)
from numba import njit, prange

GRID_N = 1024
ITERATIONS = 200
KAPPA = 20.0
DT = 0.2

#: FLOPs per neighbor per REPS iteration: subtract, divide, square, exp,
#: multiply-accumulate.
FLOPS_PER_NEIGHBOR_PER_REP = 5
NEIGHBORS = 4
BYTES_PER_POINT = 20  # 4 float32 reads + 1 float32 write


def arithmetic_intensity(reps: int) -> float:
    if reps == -1:
        # True linear floor: 3 adds + 1 multiply = 4 FLOPs/point.
        return 4 / BYTES_PER_POINT
    # 4 subtracts (dn/ds/de/dw) + 4 multiplies (cn*dn etc.) + 3 adds
    # (summing the 4 products) + 1 multiply (DT * ...) + 1 add
    # (center + ...) = 13 FLOPs/point at reps=0, before any exp() reps.
    base_flops_per_point = 13
    flops_per_point = (
        base_flops_per_point + NEIGHBORS * reps * FLOPS_PER_NEIGHBOR_PER_REP
    )
    return flops_per_point / BYTES_PER_POINT


def _initial_grid(n: int) -> np.ndarray:
    grid = np.zeros((n, n), dtype=np.float32)
    grid[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 100.0
    return grid


def _make_cpu_step(reps: int):
    if reps == -1:
        # True historical linear-diffusion floor: next = 0.25 * sum of the
        # 4 raw neighbor VALUES (no diffs, no squares) -- matches the
        # original pre-Perona-Malik kernel exactly (see git history).
        @njit(parallel=True, cache=False, fastmath=False)
        def linear_step(cur, nxt, n, m):
            for x in prange(1, n - 1):
                for y in range(1, m - 1):
                    nxt[x, y] = 0.25 * (
                        cur[x - 1, y] + cur[x + 1, y] + cur[x, y - 1] + cur[x, y + 1]
                    )

        return linear_step

    @njit(parallel=True, cache=False, fastmath=False)
    def step(cur, nxt, n, m):
        for x in prange(1, n - 1):
            for y in range(1, m - 1):
                center = cur[x, y]
                dn = cur[x - 1, y] - center
                ds = cur[x + 1, y] - center
                de = cur[x, y + 1] - center
                dw = cur[x, y - 1] - center
                cn = dn
                cs = ds
                ce = de
                cw = dw
                for _ in range(reps):
                    cn = np.exp(-((cn / KAPPA) ** 2))
                    cs = np.exp(-((cs / KAPPA) ** 2))
                    ce = np.exp(-((ce / KAPPA) ** 2))
                    cw = np.exp(-((cw / KAPPA) ** 2))
                nxt[x, y] = center + DT * (cn * dn + cs * ds + ce * de + cw * dw)

    return step


def _make_metal_kernel(reps: int):
    from numba_metal import metal

    if reps == -1:

        @metal.jit
        def linear_stencil_step(cur, nxt, n, m):
            x, y = metal.grid(2)
            if x < n and y < m:
                if x >= 1 and x < n - 1 and y >= 1 and y < m - 1:
                    nxt[x, y] = 0.25 * (
                        cur[x - 1, y] + cur[x + 1, y] + cur[x, y - 1] + cur[x, y + 1]
                    )
                else:
                    nxt[x, y] = cur[x, y]

        return linear_stencil_step

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
                cn = dn
                cs = ds
                ce = de
                cw = dw
                for _ in range(reps):
                    cn = math.exp(-((cn / 20.0) ** 2))
                    cs = math.exp(-((cs / 20.0) ** 2))
                    ce = math.exp(-((ce / 20.0) ** 2))
                    cw = math.exp(-((cw / 20.0) ** 2))
                nxt[x, y] = center + 0.2 * (cn * dn + cs * ds + ce * de + cw * dw)
            else:
                nxt[x, y] = cur[x, y]

    return stencil_step


def _launch_config(n: int, m: int, tile: int = 16):
    blocks = ((n + tile - 1) // tile, (m + tile - 1) // tile)
    threads = (tile, tile)
    return blocks, threads


def run_cpu(reps: int, grid: np.ndarray) -> Timing:
    step = _make_cpu_step(reps)
    n, m = grid.shape

    def once():
        cur = grid.copy()
        nxt = cur.copy()
        for _ in range(ITERATIONS):
            step(cur, nxt, n, m)
            cur, nxt = nxt, cur
        return cur

    once()  # compile
    return time_repeated(once, warmup=1, repeats=5)


def run_metal(reps: int, grid: np.ndarray) -> Timing:
    from numba_metal import metal

    n, m = grid.shape
    blocks, threads = _launch_config(n, m)

    def once():
        # Rebuild the @metal.jit closure on every call, matching
        # benchmarks/heat_diffusion.py's own _metal_resident (which calls
        # make_kernel() fresh each invocation) -- confirmed to matter: a
        # kernel built once and reused across repeats measures
        # substantially faster here (~11ms vs ~21-30ms at grid=1024,
        # 200 iterations) than one rebuilt per call, evidently because
        # numba-metal's KernelCache keys on more than pure structural
        # signature. Matching the existing benchmark's own methodology
        # exactly keeps this sweep's numbers comparable to Table 4 rather
        # than introducing a second, incompatible measurement convention.
        kernel = _make_metal_kernel(reps)
        cur = metal.to_device(grid)
        nxt = metal.device_array_like(grid)
        for _ in range(ITERATIONS):
            kernel[blocks, threads](cur, nxt, np.int32(n), np.int32(m))
            cur, nxt = nxt, cur
        metal.synchronize()
        return cur.copy_to_host()

    once()  # compile + warm
    return time_repeated(once, warmup=1, repeats=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reps",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Coefficient-evaluation repeat counts to sweep (controls FLOPs/byte).",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Write JSON results here."
    )
    parser.add_argument(
        "--grid", type=int, default=GRID_N, help="Grid size N (grid is NxN)."
    )
    args = parser.parse_args()

    metal_available = require_metal_or_skip()
    if not metal_available:
        print("Metal not available on this machine -- cannot run the sweep.")
        sys.exit(1)

    grid_n = args.grid
    grid = _initial_grid(grid_n)

    print(
        f"Arithmetic-intensity sweep: {grid_n}x{grid_n} grid, "
        f"{ITERATIONS} iterations, "
        f"fixed 4-neighbor/20-byte-per-point access pattern."
    )
    print(
        f"{'REPS':>6} {'FLOPs/byte':>12} {'CPU (parallel)':>18} "
        f"{'Metal (resident)':>20} {'Speedup':>10}"
    )

    rows = []
    for reps in args.reps:
        ai = arithmetic_intensity(reps)
        cpu_t = run_cpu(reps, grid)
        metal_t = run_metal(reps, grid)
        speedup = cpu_t.median_ns / metal_t.median_ns
        rows.append(
            {
                "reps": reps,
                "arithmetic_intensity_flops_per_byte": ai,
                "cpu_median_ns": cpu_t.median_ns,
                "cpu_stdev_ns": cpu_t.stdev_ns,
                "metal_median_ns": metal_t.median_ns,
                "metal_stdev_ns": metal_t.stdev_ns,
                "speedup_vs_cpu": speedup,
            }
        )
        print(f"{reps:>6} {ai:>12.2f} {cpu_t!r:>18} {metal_t!r:>20} {speedup:>9.2f}x")

    # Locate the empirical crossover: first REPS value where speedup >= 1.0x,
    # linearly interpolated in log-AI space between the bracketing points.
    crossover_ai = None
    for prev, cur in zip(rows, rows[1:], strict=False):
        if prev["speedup_vs_cpu"] < 1.0 <= cur["speedup_vs_cpu"]:
            lo_ai = prev["arithmetic_intensity_flops_per_byte"]
            lo_s = prev["speedup_vs_cpu"]
            hi_ai = cur["arithmetic_intensity_flops_per_byte"]
            hi_s = cur["speedup_vs_cpu"]
            frac = (1.0 - lo_s) / (hi_s - lo_s)
            if lo_ai <= 0:
                crossover_ai = lo_ai + frac * (hi_ai - lo_ai)
            else:
                log_lo, log_hi = math.log(lo_ai), math.log(hi_ai)
                crossover_ai = math.exp(log_lo + frac * (log_hi - log_lo))
            break

    result = {
        "grid_n": grid_n,
        "iterations": ITERATIONS,
        "bytes_per_point": BYTES_PER_POINT,
        "rows": rows,
        "empirical_crossover_flops_per_byte": crossover_ai,
        "environment": get_environment_info(),
        "gpu": get_gpu_info(),
        "hardware": get_mac_hardware_info(),
    }

    if crossover_ai is not None:
        print(
            f"\nEmpirical parity crossover (interpolated): "
            f"~{crossover_ai:.2f} FLOPs/byte"
        )
    else:
        print(
            "\nNo crossover observed in this REPS range "
            "(all points on one side of parity)."
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
