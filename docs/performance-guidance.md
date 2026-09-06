# When to expect numba-metal to actually help

Every number below was measured on this project's own benchmark suite and
advisor calibration, on an Apple M4 Pro. They are not universal constants --
different Apple Silicon generations will calibrate differently -- but the
*shape* of the guidance (which regime a workload falls into, and why) is
architectural, not machine-specific, and should transfer.

The short version: **numba-metal wins when a kernel does a lot of arithmetic
per byte of memory it touches, and loses when per-launch dispatch overhead or
memory bandwidth dominates instead.** Nothing here is a guess -- every regime
below was produced by the roofline classification in `numba-metal advisor
compare` (see `docs/advisor.md`), reproduced independently in this project's
own benchmark suite and the two worked examples,
`benchmarks/cyclist_aerodynamics.py` and `benchmarks/asian_option_pricing.py`.

## The three questions to ask before porting a function

1. **How much arithmetic happens per element, relative to how much memory
   that element touches?** This is arithmetic intensity (FLOPs per byte). It
   is the single strongest predictor of whether Metal wins, and it is
   something you can usually estimate from the algorithm itself before
   writing a single line of Metal code.
2. **How many times will this kernel actually be launched?** A single
   one-off call pays the full ~170-190us dispatch-and-sync round-trip with
   nothing to amortize it against. A kernel called in a loop, over a batch,
   or across many repeated invocations can amortize that cost -- or hide it
   entirely behind `metal.batch()`.
3. **Is the CPU-side reference already fast?** `@njit(parallel=True)` on
   twelve real CPU cores is a genuinely strong baseline. numba-metal is not
   competing against unoptimized Python -- it is competing against Numba's
   own parallel CPU backend, which already wins a meaningful share of the
   comparisons in this project's own benchmark suite.

## The three regimes, with real numbers

### Dispatch-bound: the workload is too small or too infrequent

Per-launch overhead (command buffer encode, commit, `waitUntilCompleted()`
round-trip) measures at roughly **170-190us per unbatched launch** on this
machine -- almost entirely OS/GPU-scheduler round-trip, not GPU execution
time. If a kernel's real work is smaller than that, the launch overhead *is*
the runtime.

This project's own vector-polynomial benchmark makes the point directly: at
10,000 elements, Metal measured 0.56x-0.67x against Numba's parallel CPU
implementation across every size tested, 10,000 through 10,000,000 elements
-- a loss at every size, because the elementwise polynomial itself is only a
handful of FLOPs per element, and more elements doesn't change that ratio.
The cyclist `sweep` kernel (one instantaneous drag calculation per rider)
shows the same plateau all the way out to 50,000,000 elements: 0.58x-1.04x,
never a clear win, because the amount of arithmetic per element never
changes as the problem grows either.

**The fix, when it applies:** if the same kernel is genuinely called
repeatedly -- a loop, a batch of inputs, an iterative solver -- wrap the
repeated launches in `metal.batch()` to encode them onto one shared command
buffer with a single synchronize at the end, instead of committing and
waiting after every call:

```python
with metal.batch():
    for _ in range(iterations):
        kernel[blocks, threads](data)
metal.synchronize()
```

Measured directly on this machine: batching a sequence of small launches cut
per-call dispatch overhead from ~170-190us to ~69-95us -- a real **2.5-2.7x**
reduction -- and this was enough to flip a real CPU-vs-Metal loss into a win
for one dispatch-bound workload. It does nothing for a kernel called exactly
once; the benefit is proportional to how many launches you can fuse.

### Bandwidth-bound: the workload has real headroom, or none

Below this machine's roofline ridge point (~3.09 FLOPs per byte), a kernel is
limited by how fast data can move, not by how fast the GPU can compute --
and on Apple Silicon specifically, **CPU and GPU share the same memory
bandwidth pool** (measured at ~221 GB/s on this machine). A low-arithmetic-
intensity kernel does not get a dedicated, separate bandwidth budget by
moving to the GPU; it competes for the same one the CPU was already using.
This is the single biggest reason a "just port it to the GPU" instinct fails
for elementwise or lightly-reducing kernels on unified memory hardware, in a
way that would not be true on a discrete-GPU machine with its own VRAM.

Two outcomes are both real and both were measured on this machine. The
second one below also contains a real correction to an earlier version of
this document's own conclusion -- worth reading in full even if you already
believe your kernel is bandwidth-bound for good reason, since the mistake
made here (an unfair before/after comparison) is an easy one to repeat.

- **No real headroom**, because the CPU is already using most of the shared
  bandwidth efficiently and the kernel's own memory-access pattern is not
  the problem. This is the harder case to fix, and this project has not yet
  found a clean example of it in its own benchmark suite -- every
  bandwidth-bound loss investigated so far turned out to have a fixable
  access-pattern or launch-configuration cause instead (see below). Suspect
  this case when `numba-metal advisor compare` reports `BANDWIDTH_BOUND`
  already close to 100% of the calibrated ceiling and the fixes below don't
  move it.
- **Real, fixable structure, but not necessarily a GPU win** -- the
  heat-diffusion benchmark (`benchmarks/heat_diffusion.py`) is the concrete
  example, and it took two rounds of measurement to get right. Its Metal
  kernel originally launched with a FLATTENED 1D thread index
  (`i = metal.grid(1)`, recovering `x = i // n`, `y = i % n` by hand) even
  though the problem and its memory layout are genuinely 2D, and manually
  flattened every array access to match (`cur[x*n+y]`).

  **Round 1** switched only the Metal side to a genuine 2D launch and real
  2D array indexing (`x, y = metal.grid(2)`; `cur[x, y]` directly, using
  numba-metal's native multi-dimensional array support -- see
  `tests/integration/test_multidim_arrays.py`), leaving the Numba CPU
  implementation on its original flattened indexing. That measured a
  **~5x improvement at 1024x1024** and looked like a clear win against CPU
  (0.85x -> 4.2x+) -- worth being honest that this was the number first
  written into this document.

  **Round 2** caught the mistake: the comparison wasn't apples-to-apples,
  because manual index-flattening turns out to hurt Numba's own CPU
  codegen too, not just Metal's memory system. Measured directly in
  isolation (same stencil, same correctness, CPU only): 2D-indexed vs.
  flattened-1D-indexed ran **~1.6-2x FASTER on the CPU alone** once
  flattening was removed -- negligible at 128x128, real by 512x512,
  largest at 1024x1024, the same size-dependent shape as the GPU-side
  effect. Once BOTH sides were rewritten to real 2D indexing (the
  benchmark's current, actual state), the honest comparison at 1024x1024
  is **roughly parity (~1.0x-1.1x across repeated runs)** -- not a clear
  Metal win. Metal still loses at 128x128 and 512x512.

  The real, narrower, still-useful lesson: manual index-flattening hides
  real array structure from BOTH Numba's LLVM backend and Metal's own
  codegen/memory system, so removing it is worth doing on general
  principle whichever side you're optimizing -- but a before/after
  comparison that only fixes ONE side is not evidence about which
  hardware wins; fix both sides before drawing that conclusion, or you
  will overstate the win exactly as this document originally did.

  Threadgroup memory (explicit tiling, loading a shared tile once per
  threadgroup instead of relying on repeated global reads) was also tried
  on top of the 2D-launch fix, expecting a further improvement -- and
  measured slightly SLOWER at every size tested, from 128x128 up to
  8192x8192 (roughly 1.0x-1.5x overhead, never a win, until working sets
  reached multi-gigabyte sizes far larger than this stencil is normally run
  at, where it finally reached rough parity). The likely explanation:
  Apple Silicon's GPU cache was already serving this stencil's redundant
  neighbor reads efficiently once the launch matched the data's real
  locality -- adding explicit shared-memory management on top added real
  overhead (barrier synchronization, halo-load branch divergence) for a
  problem (redundant global reads) the cache had already solved. This does
  not mean threadgroup memory is never worth it on Apple Silicon -- only
  that it was not the right fix for THIS access pattern at THESE sizes on
  THIS hardware.

**How to tell which one you have:** if `numba-metal advisor compare` reports
`BANDWIDTH_BOUND` at well under 100% of the calibrated ceiling, check the
launch configuration and indexing FIRST -- does a flattened 1D grid index
actually represent multi-dimensional data? If so, numba-metal supports real
2D/3D array arguments and a genuine multi-dimensional launch
(`metal.grid(2)`/`metal.grid(3)`) directly; try rewriting BOTH the Metal
kernel and the CPU reference this way before concluding anything about
which one wins, since fixing only one side (as this document's own first
draft did) inflates whichever side you fixed. Only after that still leaves
headroom on the Metal side specifically is threadgroup memory worth trying,
and even then, measure it against the fairly-compared 2D/3D baseline -- it
can make things worse, not better, once the launch dimensionality already
matches the data. If the workload is already near 100% of the ceiling after
all of this and still losing to a fairly-optimized CPU implementation, the
workload is likely fundamentally memory-bound on this hardware; the fix, if
one exists, is algorithmic (reduce memory traffic per output), not a
Metal-side optimization.

### Compute-bound: this is where numba-metal actually wins

Above the ridge point, this machine's real, measured compute-bound ceiling
(~683 GFLOPS float32, measured on a kernel with high arithmetic intensity and
negligible memory traffic) applies, and the wins here are large and
consistent:

- **Mandelbrot** (many escape-time iterations per pixel, minimal memory
  traffic): already a clear win at the smallest tested size, 512x512
  (5.96x-7.44x across independent runs), climbing to **15-18x** at 2048x2048
  and 4096x4096 as more independent work accumulates per thread against the
  same fixed per-launch cost.
- **European-call Monte Carlo** (`benchmarks/monte_carlo_paths.py`, 100
  timesteps per path): 3.25-4.64x at 200,000-2,000,000 paths, following the
  same more-work-per-thread scaling shape.
- **The cyclist `course_energy` kernel** (`benchmarks/cyclist_aerodynamics.py`):
  identical physical model and memory footprint to the `sweep` kernel above,
  but integrating over a 200-segment course instead of one instantaneous
  condition -- 200x more arithmetic per output value, same memory traffic per
  output value. Measured: 0.86x at 10,000 riders, climbing to **4.7-5.2x** by
  1,000,000-10,000,000 riders. This is a clean demonstration of the general
  lesson: the fix for a dispatch/bandwidth-bound kernel is very often *more
  arithmetic per byte already being moved*, not a different memory layout or
  a bigger problem size alone.
- **Asian-option Monte Carlo** (`benchmarks/asian_option_pricing.py`, 500
  timesteps per path, a running-average payoff accumulated every step): the
  strongest compute-bound result in this project's own suite --
  **10.73x-12.73x** across 100,000-2,000,000 paths, already double digits at
  the smallest size tested. An Asian option's payoff has no closed-form
  solution under geometric Brownian motion (unlike the European call above,
  which can be checked against Black-Scholes), so this is also a case where
  Monte Carlo simulation is the actual pricing technique used in practice,
  not a stand-in for a formula that already exists. Structurally, it is the
  European-call kernel with 5x the steps and one extra accumulation per step
  (`running_sum += price`) -- the same "more arithmetic per byte already
  moved" lever as the cyclist example, applied to a workload whose real-world
  version genuinely needs that much simulation.

**The pattern across all these:** the win grows with problem size, because
larger inputs mean more independent, embarrassingly-parallel work amortized
against the same fixed per-launch overhead. The cyclist `course_energy`
kernel shows the low end of this curve directly: 0.86x (a loss) at 10,000
riders, before crossing over to a clear win by 100,000. A compute-bound
kernel can still lose to CPU at a small enough input, because dispatch
overhead still has to be cleared first, regardless of arithmetic intensity
above it -- high arithmetic intensity determines the *ceiling* a kernel can
reach, not whether it clears the dispatch-overhead floor at any given size.
The Asian-option kernel shows what happens once a workload has enough
arithmetic per byte to matter from the smallest tested size onward: no loss
region to cross at all, because 500 accumulating steps per path is already
well clear of the dispatch-overhead floor even at 100,000 paths.

## A practical checklist

Before porting a CPU function to `@metal.jit`, in order:

1. **Estimate FLOPs per byte moved.** Count real arithmetic operations per
   output element, divide by bytes read plus written per output element. A
   number below ~3 (on this machine's calibration; run `numba-metal advisor
   calibrate` for your own) means bandwidth is the likely constraint, not
   compute. If your kernel is a single arithmetic expression per array
   element, you are almost certainly here.
2. **Check whether the kernel is called once or many times.** If it's called
   in a loop or over a batch, `metal.batch()` is worth trying regardless of
   the other two answers -- it's a real, ~2.5-2.7x reduction in the part of
   the cost that a single-call kernel can't avoid.
3. **Compare against `@njit(parallel=True)`, not against plain Python or
   single-threaded Numba.** This project's own benchmarks show Numba's
   parallel CPU backend winning outright against Metal on several real
   workloads (vector polynomial, small heat-diffusion grids, small pairwise
   distance). That is the honest baseline on Apple Silicon with a many-core
   CPU; a comparison against unparallelized code overstates what porting to
   Metal actually buys you.
4. **If the estimate says bandwidth-bound and you still want to try it,
   check the launch configuration and indexing before anything else -- and
   fix BOTH the Metal kernel and the CPU reference the same way before
   comparing them.** Does a flattened 1D grid index actually represent 2D
   or 3D data? numba-metal supports real 2D/3D array arguments directly
   (`arr[x, y]`, no manual `arr[x*n+y]` flattening needed); rewrite both
   sides to use `metal.grid(2)`/`metal.grid(3)` and real multi-dimensional
   indexing before drawing any conclusion. This project's own
   heat-diffusion benchmark got this wrong on the first pass -- fixing only
   the Metal side looked like a ~5x win, but manual flattening was also
   slowing down the Numba CPU reference by a similar margin; fixing both
   sides revealed the honest result was closer to parity. Only reach for
   threadgroup memory after a FAIR 2D/3D-vs-2D/3D comparison still shows
   real headroom on the Metal side, and measure it against that fair
   baseline specifically -- it made this project's own stencil slower, not
   faster, at every size tested short of multi-gigabyte working sets.
5. **Measure, don't extrapolate.** Every number in this document came from
   `numba-metal advisor compare` or a real benchmark run, never from a
   theoretical estimate alone. Run the comparison on your actual workload and
   your actual machine before trusting any of the figures above as your own.

## What this doesn't cover

- **Zero-copy memory** (`newBufferWithBytesNoCopy_length_options_deallocator_`
  wrapping a NumPy array's own memory directly, instead of `to_device`'s
  current `memcpy`). Prototyped in earlier investigation on this project but
  not built into numba-metal's public API (see `docs/architecture.md`/
  `docs/limitations.md`). It addresses host-device transfer cost
  specifically, not the shared-bandwidth ceiling discussed above -- it would
  not be expected to flip a fundamentally bandwidth-bound-vs-CPU comparison
  into a win on its own. No production-API benchmark exists yet to cite a
  number from; treat any pre-API-support figure as preliminary until
  re-measured against the real interface.
- **Multi-queue concurrency** (issuing independent kernels across more than
  one Metal command queue concurrently). Investigated in earlier work on
  this project; not re-verified in this pass, so not given a number here.
- **Hardware other than Apple M4 Pro.** Every number here is this machine,
  this week, per this project's own stated methodology (see
  `docs/benchmarking.md`). The regimes and the reasoning transfer; the exact
  thresholds do not.
