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

## Cold vs. warm execution

- **Cold**: the very first launch of a given kernel+signature, which
  includes Numba's typed-IR frontend compilation *and* Apple's Metal
  shader compilation (both triggered by `KernelCache.get_or_compile` on a
  cache miss). Reported as `metal_cold_ns` / "Metal total" in the table
  and JSON output.
- **Warm**: launches after the kernel is already in the in-process
  compilation cache -- pure dispatch + GPU execution + (implicit)
  synchronization cost, no compilation. Reported as `metal_warm_ns` /
  "Metal warm".

The gap between these two numbers is often large (milliseconds to tens of
milliseconds for cold vs. sub-millisecond to low-milliseconds for warm on
the workloads in this repo) and is the actual cost of numba-metal's JIT
model -- a fair comparison against Numba's own `@njit` (which has an
equivalent, separately-measured cold-compile cost) accounts for this
rather than only reporting whichever number looks better.

## Transfer-inclusive vs. resident-data timing

`benchmarks/vector_polynomial.py`, `pairwise_distance.py`, and
`monte_carlo_paths.py` separately measure `metal_h2d_ns` (host-to-device
upload) and `metal_d2h_ns` (device-to-host download) so the transfer cost
can be seen independent of kernel execution time. `metal_warm_ns` itself
measures only the launch+sync (data already resident); it does not
include a fresh upload/download on every iteration.

`benchmarks/heat_diffusion.py` makes this distinction the actual subject
of the benchmark: it compares keeping both ping-pong buffers GPU-resident
for an entire multi-iteration simulation (`_metal_resident`, only
uploading the initial grid and downloading the final one) against
re-uploading and re-downloading on every single iteration
(`_metal_copy_every_launch`). The difference between these two numbers is
the actual cost of not keeping data resident -- and it is substantial,
because per-iteration `MTLBuffer` allocation and host<->device memcpy
dominate over a tiny per-iteration compute kernel at small grid sizes.

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
| Heat diffusion | 128² / 512² / 1024² grid, 200 Jacobi iterations |
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
  For Mandelbrot specifically, the "vectorized" NumPy implementation
  still does the *same* fixed amount of work per pixel every iteration
  (masked rather than early-exited, since per-element early exit isn't
  expressible in vectorized NumPy) -- this makes it a legitimate but
  structurally different algorithm from the scalar CPU/GPU versions, which
  matters for the correctness discussion below.
- **Numba CPU**: `@njit` (with `parallel=True, fastmath=False, cache=True`
  where a `prange`-parallelizable loop exists), the same scalar,
  early-exiting algorithm as the GPU kernel.
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
  Metal" below for the concrete, measured root cause.
- Every correctness check reports which comparison it used and the
  measured deviation (or pass/fail reason) in `correctness_note`, both in
  the printed table and the JSON output -- tolerances are never silently
  widened without being visible in the result.

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

| Benchmark | Size | Numba CPU (warm) | Metal (warm) | Metal vs Numba CPU |
|---|---|---|---|---|
| Vector polynomial | 10,000 | 10.5us | 483us | 0.02x (GPU slower: dispatch overhead dominates at this size) |
| Vector polynomial | 1,000,000 | 969us | 898us | 1.08x |
| Vector polynomial | 10,000,000 | 9.86ms | 1.62ms | 6.10x |
| Mandelbrot | 512² | 2.06ms | 1.07ms | 1.92x |
| Mandelbrot | 2048² | 30.5ms | 2.00ms | 15.27x |
| Mandelbrot | 4096² | 126.2ms | 7.12ms | 17.72x |
| Heat diffusion | 128² (200 iters) | 3.26ms | 40.4ms | 0.08x (GPU slower: per-launch overhead over 200 tiny kernels dominates at this grid size) |
| Heat diffusion | 512² (200 iters) | 52.5ms | 40.8ms | 1.29x |
| Heat diffusion | 1024² (200 iters) | 210.0ms | 66.0ms | 3.18x |
| Monte Carlo paths | 10,000 | 219us | 320us | 0.69x |
| Monte Carlo paths | 200,000 | 2.19ms | 1.48ms | 1.48x |
| Monte Carlo paths | 2,000,000 | 20.4ms | 6.34ms | 3.22x |
| Pairwise distance | 200x200x8 | 112us | 663us | 0.17x |
| Pairwise distance | 2000x2000x8 | 1.63ms | 2.02ms | 0.81x |
| Pairwise distance | 5000x5000x16 | 17.5ms | 10.7ms | 1.63x |

A machine-readable copy of this exact run is saved at
`benchmarks/results/example_m4pro_<date>.json` for reference; it is
example data from one machine, not a target or a claim about other Macs.

## Why some workloads can be slower on Metal

- **Small problem sizes.** Every kernel launch has fixed overhead:
  encoding a command buffer, binding buffers, dispatching, and (in these
  benchmarks) synchronizing before returning. For workloads that complete
  in single-digit microseconds on CPU (e.g. vector polynomial at 10,000
  elements, pairwise distance at 200x200), this fixed overhead exceeds the
  actual compute time by 10-40x. This is expected and is exactly why the
  benchmark suite includes small sizes deliberately -- to show the
  crossover point, not to hide it.
- **Many small launches instead of one large one.** `heat_diffusion.py`
  at 128² does 200 separate kernel launches (one Jacobi iteration each) of
  a very small grid (16,384 elements); at that size, per-launch dispatch
  overhead dominates over the tiny amount of actual stencil computation
  per launch, and the CPU (which has no equivalent per-call dispatch
  overhead for a tight `@njit` loop) wins. The same algorithm at 1024²
  (1M+ elements per launch) flips decisively in the GPU's favor because
  the fixed per-launch overhead becomes negligible relative to the work.
- **Mandelbrot's dramatic scaling** (1.92x at 512² up to 17.72x at 4096²)
  demonstrates the opposite regime clearly: per-pixel divergent control
  flow (variable iteration counts) is exactly the kind of workload GPUs
  are built for once there's enough parallelism to hide the fixed launch
  cost, and the M4 Pro's GPU core count dominates as pixel count grows.
- **The Mandelbrot correctness comparison** (see `docs/limitations.md`)
  required using a float32 CPU reference instead of Numba's default
  float64 inference, because a chaotic escape-time recurrence amplifies
  even single-ULP floating-point differences into iteration-count
  differences of tens of steps near the escape boundary. This was
  discovered and diagnosed concretely while building this benchmark (not
  a theoretical caveat) -- see the git history / `docs/architecture.md`
  for the investigation.
