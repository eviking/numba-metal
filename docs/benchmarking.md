# Benchmarking

## Methodology

All timing uses `time.perf_counter_ns()` (see `benchmarks/common.py`).
Every measured section:

1. Runs a small number of **warm-up** iterations first (discarded), so
   the first-call JIT/compile cost of Python-level wrappers, Numba's own
   `@njit` compilation, and numba-metal's kernel compilation don't pollute
   steady-state numbers (except where cold-start *is* explicitly what's
   being measured -- see below).
2. Runs several **repeated** measured iterations and reports the
   **median** (via `Timing.median_ns`), with standard deviation and
   min/max also recorded, to characterize variability rather than report
   a single noisy sample.
3. For GPU work, **always calls `metal.synchronize()` before stopping the
   clock**. An unsynchronized launch returns almost immediately regardless
   of actual GPU execution time (it's asynchronous), so timing without a
   synchronize call would silently measure encode/submit overhead instead
   of real work -- this is called out explicitly in `docs/quickstart.md`
   and `docs/troubleshooting.md` because it is the single easiest way to
   produce a meaningless benchmark number.

## Timing categories

Every result row (`BenchmarkResult` in `benchmarks/common.py`) reports
these as **separate, never-conflated** fields -- compilation, transfer,
and kernel execution are never folded into one "GPU time" number:

| Field | What it measures |
|---|---|
| `python_ns` / `numpy_ns` | Reference CPU implementations (informational, not the primary evidence). |
| `numba_cpu_parallel_ns` | Warm `@njit(parallel=True)` execution using `prange`, at Numba's default thread count (recorded in `numba_cpu_num_threads`). |
| `numba_cpu_single_ns` | The same kernel, forced to one thread via `numba.set_num_threads(1)` (restored afterward). A separate compiled dispatcher, not the same object timed twice. |
| `metal_frontend_ns` | Numba typed-IR frontend + numba-metal's lowering to MSL source text. No device interaction. Measured via `compile_to_typed_ir` + `MSLKernelLowerer.lower()` called directly, so nothing about the call could hit a pre-existing cache. |
| `metal_pipeline_compile_ns` | `MTLLibrary`/`MTLComputePipelineState` creation from that generated MSL, on-device. |
| `metal_cold_total_ns` | Frontend + pipeline compile + first launch, timed end-to-end on a **brand-new** `@metal.jit` dispatcher (its own fresh `KernelCache`). |
| `compiled_before` / `compiled_after` | `len(dispatcher._cache)` before and after the cold-timing call. `compiled_after > compiled_before` is the evidence that the cold measurement really did compile, not reuse a warm cache -- this is checked, not assumed, for every cold number in this document. |
| `metal_kernel_only_warm_ns` | Warm (already-compiled) launch + synchronize, with all buffers already device-resident. Excludes every host&harr;device transfer. |
| `metal_h2d_ns` / `metal_d2h_ns` | Host-to-device / device-to-host transfer time, measured separately from kernel execution. |
| `metal_end_to_end_warm_ns` | What a caller actually experiences per call when data starts and ends on the host: warm H2D + kernel + D2H (or the closest equivalent for kernels with no host input, e.g. Mandelbrot). |
| `metal_resident_pipeline_ns` | Only populated for the iterative heat-diffusion benchmark: warm kernel-only time across the *whole* multi-iteration run with data kept GPU-resident throughout. |

`speedup_vs_numba_cpu_parallel` / `speedup_vs_numba_cpu_single` in the
JSON output are always computed against `metal_kernel_only_warm_ns` (not
a transfer-inclusive number), so the "vs Numba" column is a genuine
compute-vs-compute comparison; the separate `metal_h2d_ns`/`metal_d2h_ns`
fields let you add transfer cost back in for the caller's actual
end-to-end scenario.

## Cold vs. warm execution, made demonstrably cold

Earlier revisions of this benchmark suite reused the *same* persistent
`@metal.jit` dispatcher for "cold" and "warm" timing within one problem
size, meaning only the very first size measured in a run was ever
actually uncompiled -- every subsequent size's "cold" number silently hit
an already-warm process-wide state for that kernel function. That bug is
fixed: every cold measurement (`measure_cold_end_to_end` /
`measure_cold_metal_pipeline` in `benchmarks/common.py`) now constructs
its own fresh `KernelDispatcher`/`KernelCache` (or calls the frontend/MSL
compilation functions directly, bypassing any cache entirely), and
records `compiled_before`/`compiled_after` cache-entry counts as
machine-checkable proof.

The gap between cold and warm is often large (tens of milliseconds for
cold vs. sub-millisecond to low-milliseconds for warm on the workloads in
this repo) and is the actual cost of numba-metal's JIT model -- a fair
comparison against Numba's own `@njit` (which has an equivalent,
separately-measured cold-compile cost, not reported here since Numba's
own compilation-caching behavior is out of scope for this project) should
account for this rather than only reporting whichever number looks
better.

## Numba CPU: parallel and single-threaded, with thread count recorded

Every benchmark builds **two independent** Numba CPU dispatchers per
problem size: `_make_numba_cpu_impl(parallel=True)` (using `prange` for
the outer loop) and `_make_numba_cpu_impl(parallel=False)` (the same
source, with the outer loop's `prange` replaced by plain `range` via a
`loop_range` closure variable, and `set_num_threads(1)` in effect for the
duration of the measurement, restored afterward). `numba_cpu_num_threads`
records the thread count actually in effect for the parallel run
(`numba.get_num_threads()`), so a reader never has to guess how many
cores the "Numba CPU (par)" column represents. The single-threaded number
is reported as primary supporting evidence alongside the parallel one,
not omitted -- for several small-grid workloads in this repo (e.g. Heat
diffusion at 64²/128², vector polynomial at 10,000 elements), the
single-threaded number is *faster* than the parallel one, because
`prange`'s thread-pool dispatch overhead exceeds the actual per-thread
work at that size. That is reported plainly, not smoothed over.

## Transfer-inclusive vs. resident-data timing

`vector_polynomial.py`, `mandelbrot.py`, `monte_carlo_paths.py`, and
`pairwise_distance.py` separately measure `metal_h2d_ns` (host-to-device
upload) and `metal_d2h_ns` (device-to-host download) so transfer cost is
visible independent of kernel execution time. `metal_kernel_only_warm_ns`
measures only the launch+sync with data already resident; it never
includes a fresh upload/download.

`heat_diffusion.py` makes this distinction the actual subject of the
benchmark: it compares keeping both ping-pong buffers GPU-resident for an
entire multi-iteration simulation (`_metal_resident`, only uploading the
initial grid and downloading the final one, reported as
`metal_resident_pipeline_ns`) against re-uploading and re-downloading on
every single iteration (`_metal_copy_every_launch`). Both variants now
run at the **same iteration count** (previously the copy-every-launch
variant was capped at a smaller iteration count "for illustration,"
which made the two numbers not directly comparable); the per-iteration
cost of each (`extra["metal_resident_per_iter_ns"]` /
`extra["metal_copy_every_launch_per_iter_ns"]`) is the fair, apples-to-
apples comparison, and the difference between them is the actual
measured cost of not keeping data resident.

## Synchronization

Every GPU-timed closure passed to `time_repeated()` in every benchmark
calls `metal.synchronize()` (or `copy_to_host()`, which calls it
internally) before returning, so no benchmark number in this repository
was produced by timing an unsynchronized launch.

## Problem sizes

Each benchmark uses at least two, usually three, problem sizes chosen to
span "GPU dispatch overhead dominates" (small) to "compute/bandwidth
dominates" (large):

| Benchmark | Sizes used |
|---|---|
| Vector polynomial | 10,000 / 1,000,000 / 10,000,000 elements |
| Mandelbrot | 512² / 2048² / 4096² pixels |
| Heat diffusion (nonlinear) | 128² / 512² / 1024² grid, 200 explicit-scheme iterations |
| Monte Carlo paths | 10,000 / 200,000 / 2,000,000 paths x 100 time steps |
| Pairwise distance | 200x200x8 / 2000x2000x8 / 5000x5000x16 |

`--quick` (in `run_all.py`) substitutes smaller sizes for a fast
development/CI sanity check; it is not used to produce the numbers below.

## Baselines

- **Python**: a plain interpreted-loop implementation, included only
  where it completes in reasonable time (skipped above a per-benchmark
  size threshold, since an O(n) or worse pure-Python loop over millions
  of elements is prohibitively slow and adds no useful signal).
- **NumPy**: a vectorized implementation using standard array operations.
  This is a genuinely different algorithm in two of the five benchmarks,
  documented explicitly rather than treated as a like-for-like
  comparison: Mandelbrot's vectorized form does the *same* fixed amount
  of work per pixel every iteration (masked rather than early-exited,
  since per-element early exit isn't expressible vectorized), and
  pairwise distance's broadcast form materializes a full `(n_a, n_b, k)`
  temporary array (~3.2 GiB at the largest configured size) that neither
  the CPU nor GPU kernel ever allocates.
- **Numba CPU**: `@njit`, both `parallel=True` (via `prange`, default
  thread count) and `parallel=False` (single-threaded), the same scalar,
  early-exiting algorithm as the GPU kernel. This is the primary
  comparison evidence; Python/NumPy numbers are secondary/informational.
- **Metal**: this project's backend.

## Numerical tolerance

- Purely arithmetic benchmarks (vector polynomial, heat diffusion,
  pairwise distance, Monte Carlo) use `numpy.allclose` with
  `rtol=atol=1e-3` or `1e-4` depending on how much floating-point
  reduction/accumulation the algorithm does (looser tolerance where more
  operations compound rounding error, e.g. Monte Carlo's summed log
  returns).
- Mandelbrot (an integer escape-iteration count, produced by a *chaotic*
  recurrence) uses a dedicated comparator,
  `assert_near_integer_match` in `benchmarks/common.py`: exact match is
  required everywhere except a small, bounded fraction of pixels
  (default 0.1%) that may differ by exactly 1 iteration. This is not
  tolerance for a bug -- see "Why some workloads can be slower / differ on
  Metal" below for the concrete, measured root cause. Its `correctness_ok`
  gate always compares GPU (float32) against a float32 scalar CPU
  reference built specifically for this comparison; the vectorized NumPy
  float64 number is logged as a separate, explicitly-labeled
  informational note (`numpy_vs_cpu_note`) and never participates in the
  pass/fail decision.
- Every correctness check reports which comparison it used and the
  measured deviation (or pass/fail reason) in `correctness_note`, both in
  the printed table and the JSON output -- tolerances are never silently
  widened without being visible in the result, and `correctness_ok` is
  always derived directly from the same boolean the note text describes
  (never a case where the note says "FAILED" but `correctness_ok` is
  `True`, or vice versa).

## How to reproduce these results

```bash
python benchmarks/run_all.py --json results.json
```

on the machine you want numbers for. The environment block of the JSON
output (`get_environment_info()` in `benchmarks/common.py`) records the
Mac model, chip, macOS version, Python/Numba/numba-metal versions, and
GPU name/capabilities alongside every run, so results are traceable to
the exact machine and software versions that produced them.

## Measured results (this machine, this run)

These numbers were produced by `python benchmarks/run_all.py` (full,
non-`--quick` sizes) on the machine used to build this package: **Apple
M4 Pro, macOS 26.5.1, Python 3.13.5, Numba 0.67.0**. They are one sample
run, not a guarantee -- rerun `run_all.py` on your own machine, and treat
any number below as illustrative of *what was actually measured once*,
not a promised speedup. All 15 correctness checks passed on this run.
Every `metal_cold_total_ns` figure below was confirmed via
`compiled_before`/`compiled_after` to reflect a genuine fresh compilation
(`compiled_before=0, compiled_after=1` in every row). Refreshed
2026-09-06 after replacing the Heat diffusion benchmark's linear
diffusion model with nonlinear (Perona-Malik) diffusion -- see
`docs/performance-guidance.md`'s bandwidth-bound section for why.

| Benchmark | Size | Numba CPU (par, N threads) | Numba CPU (1 thread) | Metal kernel-only (warm) | Metal cold (total) | vs Numba CPU (par) |
|---|---|---|---|---|---|---|
| Vector polynomial | 10,000 | 107.0us (12t) | 9.33us | 164.2us | 14.6ms | 0.65x |
| Vector polynomial | 1,000,000 | 185.8us (12t) | 964.2us | 258.2us | 8.69ms | 0.72x |
| Vector polynomial | 10,000,000 | 674.8us (12t) | 10.0ms | 1.27ms | 11.6ms | 0.53x |
| Mandelbrot | 512² | 2.04ms (12t) | 11.5ms | 236.5us | 14.2ms | 8.62x |
| Mandelbrot | 2048² | 30.8ms (12t) | 178.5ms | 1.07ms | 15.5ms | 28.86x |
| Mandelbrot | 4096² | 123.6ms (12t) | 743.6ms | 3.30ms | 18.6ms | 37.42x |
| Heat diffusion (nonlinear) | 128² (200 iters) | 23.3ms (12t) | 27.1ms | 131.0us/iter | 18.8ms | 0.89x |
| Heat diffusion (nonlinear) | 512² (200 iters) | 88.3ms (12t) | 438.0ms | 133.0us/iter | 20.1ms | 3.32x |
| Heat diffusion (nonlinear) | 1024² (200 iters) | 297.8ms (12t) | 1.75s | 208.2us/iter | 20.9ms | 7.15x |
| Monte Carlo paths | 10,000 | 255.4us (12t) | 688.2us | 221.2us | 6.70ms | 1.15x |
| Monte Carlo paths | 200,000 | 2.44ms (12t) | 14.1ms | 742.1us | 8.24ms | 3.29x |
| Monte Carlo paths | 2,000,000 | 20.3ms (12t) | 140.7ms | 4.60ms | 28.6ms | 4.41x |
| Pairwise distance | 200x200x8 | 101.0us (12t) | 67.5us | 189.6us | 9.85ms | 0.53x |
| Pairwise distance | 2000x2000x8 | 871.5us (12t) | 5.95ms | 525.2us | 8.24ms | 1.66x |
| Pairwise distance | 5000x5000x16 | 9.95ms (12t) | 65.0ms | 4.28ms | 15.0ms | 2.32x |

Heat diffusion's speedup varies more run-to-run than the other
benchmarks at 1024² specifically (4.6x-7.2x observed across repeated
runs, all with correctness verified) -- likely GPU clock/thermal
state variance on a 200-iteration, many-small-launches workload; take
it as a range, not the single value above. Also measured, at 10x and
100x this benchmark's largest grid-point count (3238² and 10240²,
outside `run_all.py`'s default sizes): **13.05x and 13.66x**, climbing
rather than degrading as the grid grows -- see
`docs/performance-guidance.md` for the full comparison against the
retired linear-diffusion model, which did the opposite (0.52x and
0.60x at the same two larger sizes).

Note the smallest Vector polynomial row and Heat diffusion at 128²
where single-threaded Numba CPU beats or nearly matches the "parallel"
(12-thread) variant -- `prange`'s thread-pool dispatch overhead can
exceed the actual per-thread work at small sizes, a real and expected
effect, not an error. `metal_kernel_only_warm_ns` for Heat diffusion is
reported per-iteration (`.../iter`) since it is measured across the
full multi-iteration resident run, not a single launch.

A machine-readable copy of this exact run is saved at
`benchmarks/results/benchmarking_doc_refresh_<date>.json`. Earlier,
differently-structured results files (`benchmarks/results/corrected_
m4pro_<date>.json`, `example_m4pro_<date>.json`, predating both the
timing-category corrections described in this document and the
heat-diffusion linear-to-nonlinear replacement) are kept for historical
reference only and should not be used for comparison.

## Why some workloads can be slower on Metal

- **Small problem sizes.** Every kernel launch has fixed overhead:
  encoding a command buffer, binding buffers, dispatching, and (in these
  benchmarks) synchronizing before returning. For workloads that complete
  in single-digit microseconds on CPU (e.g. vector polynomial at 10,000
  elements, pairwise distance at 200x200), this fixed overhead exceeds the
  actual compute time by 10-40x. This is expected and is exactly why the
  benchmark suite includes small sizes deliberately -- to show the
  crossover point, not to hide it.
- **`prange` thread-pool overhead at small sizes.** The parallel Numba
  CPU number can be close to, or slower than, the single-threaded one
  (Heat diffusion at 128² in the table above) -- spinning up and
  synchronizing 12 worker threads costs more than the serial work saves
  when each thread's share of the loop is tiny relative to thread-pool
  dispatch. This is reported directly via the separate
  `numba_cpu_parallel_ns`/`numba_cpu_single_ns` columns rather than only
  showing whichever is faster.
- **Many small launches instead of one large one.** `heat_diffusion.py`
  at small grid sizes does 200 separate kernel launches (one diffusion
  step each); per-launch dispatch overhead can dominate over the
  actual stencil computation per launch, especially at 128² where
  arithmetic intensity is real but the per-launch work is still small.
  The resident-vs-copy-every-launch comparison (see above) isolates
  exactly how much of that cost is host&harr;device transfer versus
  launch overhead itself.
- **Mandelbrot's dramatic scaling** (8.62x at 512² up to 37.42x at
  4096²) demonstrates the opposite regime clearly: per-pixel divergent
  control flow (variable iteration counts) is exactly the kind of
  workload GPUs are built for once there's enough parallelism to hide
  the fixed launch cost, and the M4 Pro's GPU core count dominates as
  pixel count grows.
- **The Mandelbrot correctness comparison** (see `docs/limitations.md`)
  required using a float32 CPU reference instead of Numba's default
  float64 inference, because a chaotic escape-time recurrence amplifies
  even single-ULP floating-point differences into iteration-count
  differences of tens of steps near the escape boundary. This was
  discovered and diagnosed concretely while building this benchmark (not
  a theoretical caveat) -- see the git history / `docs/architecture.md`
  for the investigation.
