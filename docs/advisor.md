# Numba-Metal Advisor

The advisor is a terminal-only profiler and static analyzer that answers
five questions about your own code, on your own Apple Silicon machine:

1. Which functions might benefit from numba-metal?
2. Which of those are actually supported by numba-metal today?
3. What would need to change to make an unsupported function compatible?
4. Does Metal actually run faster than the CPU here, measured, not guessed?
5. Where does the time go -- Python, Numba CPU, compilation, dispatch,
   data transfer/sync, or the kernel itself?

**The one rule everything below follows:** a number in advisor output is
always one of three things, and the output always says which:

- **MEASURED** -- a real wall-clock timing from an actual run on this
  machine.
- **ESTIMATED** -- derived from a model or from statistical sampling
  (e.g. the CPU call-stack sampler), not a direct instrumentation event.
- **STATIC POTENTIAL** -- from source-code analysis alone, with no
  runtime data at all. The advisor never turns this into a speedup
  number; it says "HIGH POTENTIAL" or "LOW POTENTIAL", never "2.3x
  faster", because static analysis cannot know that.

If you only remember one thing from this document, remember that
`numba-metal advisor scan` never runs your code and never claims a
speedup, and `numba-metal advisor compare` always measures on real
hardware before recommending anything.

## Installation

The advisor ships as part of `numba-metal` itself -- no separate
package.

```bash
pip install -e .          # installs the `numba-metal` command
pip install -e ".[tui]"   # optional: adds the interactive terminal UI
```

`pip install numba-metal` alone (without `[tui]`) is enough for `scan`,
`profile`, `compare`, `report`, and `calibrate` in non-interactive mode.
The interactive TUI (`blessed`) is an optional extra; without it, every
command still works, falling back to plain ASCII output.

## Quick start

```bash
# 1. Find candidate functions in your project, without running any of it.
numba-metal advisor scan .

# 2. See where a script's time actually goes (Python, Numba, Metal).
numba-metal advisor profile my_script.py

# 3. Measure whether Metal is actually faster for a specific workload,
#    with correctness verification.
numba-metal advisor compare my_workload.py

# 4. Re-render a saved report later, without rerunning anything.
numba-metal advisor report numba-metal-profile-20260101-120000/profile.json

# 5. Measure this machine's baseline overhead numbers once.
numba-metal advisor calibrate
```

## Commands and flags

### `numba-metal advisor scan PATH`

Static-only. Walks the Python AST under `PATH` (a file or directory) and
never imports or executes anything it finds. Reports candidate functions
by confidence (HIGH / MEDIUM / LOW potential), existing `@numba.njit`/
`@metal.jit` usage, and precise blockers for code that would need
changes before it could even be attempted on Metal.

```
numba-metal advisor scan benchmarks/mandelbrot.py
```

```
Scanned 1 file(s) under benchmarks/mandelbrot.py
Found 7 candidate(s), 0 error(s)

[LOW POTENTIAL] mandelbrot.py:36 numpy_impl
Why:
  - Accumulator pattern detected (looks like a reduction)
Unknown:
  - Runtime array sizes
  - Actual data types
  - Call frequency
  - Transfer cost

[HIGH POTENTIAL] mandelbrot.py:71 _make_numba_cpu_impl.<locals>.numba_cpu_impl
Why:
  - Already decorated with @njit
  - Accumulator pattern detected (looks like a reduction)

[HIGH POTENTIAL] mandelbrot.py:97 _make_metal_kernel.<locals>.metal_kernel
Why:
  - Already running on numba-metal (@metal.jit)

[LOW POTENTIAL] mandelbrot.py:120 run
Blockers:
  - line 121: list literal -- Python lists are not a supported Metal kernel type (use a fixed-size array instead)
  - line 130: f-string -- string formatting is not supported inside a Metal kernel
  ...
```

(Captured verbatim from a real run against this repository's own
`benchmarks/mandelbrot.py` -- not hand-typed. Note that a function
already decorated with `@metal.jit` still shows up, labeled "Already
running on numba-metal" -- the scanner doesn't skip functions just
because they're already converted; it tells you what it found either
way.)

### `numba-metal advisor profile SCRIPT.py [-- SCRIPT_ARGS]`

Runs `SCRIPT.py` (via `runpy`, exactly as `python SCRIPT.py` would) with
instrumentation active: Metal kernel launches/syncs/compiles are
recorded as timed events, and a background statistical sampler takes
periodic snapshots of the Python call stack. Produces flame graphs and a
CPU/GPU timeline from what actually happened during that one run.

```
numba-metal advisor profile --pytest TEST_PATH [-- PYTEST_ARGS]
```

does the same thing but runs a pytest target instead of a script (via
`pytest.main(...)`), useful for profiling an existing test suite without
writing a separate script.

### `numba-metal advisor compare SCRIPT.py [-- SCRIPT_ARGS]`

The command that actually answers "is Metal faster, measured?" It
imports `SCRIPT.py` and calls a function you write, `advisor_workload()`,
which returns an `AdvisorWorkload` describing how to run the CPU
version, how to run the (already-compiled) Metal version, and how to
fetch each one's result for correctness checking:

```python
# my_workload.py
import numpy as np
from numba import njit
from numba_metal import metal
from numba_metal.advisor.workload import AdvisorWorkload


@njit(parallel=True)
def cpu_impl(out, n):
    for i in range(n):
        out[i] = i * i


@metal.jit
def metal_kernel(out, n):
    i = metal.grid(1)
    if i < n:
        out[i] = i * i


def advisor_workload() -> AdvisorWorkload:
    n = 1_000_000
    cpu_out = np.zeros(n, dtype=np.int64)
    metal_out = metal.to_device(np.zeros(n, dtype=np.int64))
    threads = 256
    blocks = (n + threads - 1) // threads

    def run_cpu():
        cpu_impl(cpu_out, n)

    def run_metal_warm():
        metal_kernel[blocks, threads](metal_out, n)
        metal.synchronize()

    return AdvisorWorkload(
        qualified_name="my_workload.metal_kernel",
        cpu_fn=run_cpu,
        metal_warm_fn=run_metal_warm,
        get_cpu_result=lambda: cpu_out,
        get_metal_result=lambda: metal_out.copy_to_host(),
    )
```

`compare` measures WARM and STEADY_STATE regimes (see "Cold, warm, and
steady state" below), runs a correctness check between
`get_cpu_result()`/`get_metal_result()`, and only then produces a
recommendation. Real, captured output from this exact pattern against
`benchmarks/mandelbrot.py`'s actual kernels (2048x2048, 100 max
iterations, on an Apple M4 Pro):

```
COMPARISON: mandelbrot.metal_kernel (mandelbrot_advisor_workload.py:1)
-- WARM --
  CPU:   median=30.948ms min=30.838ms max=34.957ms mean=31.542ms stdev=1.510ms n=7
  Metal: median=2.033ms min=1.928ms max=2.079ms mean=2.015ms stdev=0.063ms n=7
  Speedup: 15.22x
-- STEADY_STATE --
  CPU:   median=31.042ms min=30.695ms max=33.954ms mean=31.318ms stdev=0.780ms n=35
    outliers (not removed): 1
  Metal: median=2.018ms min=1.931ms max=2.323ms mean=2.042ms stdev=0.088ms n=35
    outliers (not removed): 1
  Speedup: 15.38x
Whole-program speedup: 15.38x
Profiler overhead (estimated): 256.398ms

CORRECTNESS
Result:                 PASS
Elements compared:      4,194,304
Dtype:
  CPU:                  int32
  Metal:                int32

Recommendation (USE_METAL): Use Metal for mandelbrot.metal_kernel -- measured 15.38x faster (steady_state), and results match the CPU reference within tolerance.
```

### `numba-metal advisor report PROFILE.json`

Re-renders a previously saved `profile.json` -- no re-execution, no
Metal device required, works on any machine. This is how you view a
report generated on your Mac from a CI log or a colleague's machine.

### `numba-metal advisor calibrate [--delete]`

Measures this machine's own baseline numbers once (dispatch overhead,
cold-compile cost, buffer allocation, sync overhead, float32 throughput,
memory bandwidth) and caches them in `~/.cache/numba-metal/`. See
"Device calibration" below.

### Common flags

| Flag | Meaning |
|---|---|
| `--output DIR` | Write `profile.json` + text reports to `DIR` |
| `--format terminal\|text\|json` | Output format for stdout |
| `--no-color` | Disable ANSI color (also respects `NO_COLOR` env var) |
| `--width N` | Render width, 60-240 columns (default 78) |
| `--sampling-interval-ms N` | CPU sampler interval (default 5ms) |
| `--warmup-runs N` | Warm-up iterations before timing (default 2) |
| `--measurement-runs N` | Timed iterations (default 7) |
| `--include PATTERN` / `--exclude PATTERN` | Filter files during scan |
| `--max-depth N` | Limit directory recursion during scan |
| `--seed N` | Seed for sampled correctness comparison |
| `--verbose` | More diagnostic output |

Every subcommand exits nonzero on failure (bad arguments, a profiling
error, a failed correctness check) -- safe to gate a CI step on.

## Static scan vs. runtime profile: what each one can and can't tell you

**`scan`** reads source code with Python's `ast` module. It can tell you
"this function has a nested loop over an array and no existing GPU
decorator" (a static, structural fact). It **cannot** tell you how long
that loop actually takes, whether the array is big enough for a GPU
launch to pay for itself, or what the real speedup would be -- so it
never states one. Every scan result is either a `HIGH`/`MEDIUM`/`LOW
POTENTIAL` label or a list of concrete parse-level blockers, never a
number.

**`profile`** and **`compare`** actually run your code (only the exact
script/test you point them at -- never anything else in your project)
and record what happened. These commands can and do report real
milliseconds, real speedups, and pass/fail correctness results.

## Existing Numba code vs. plain Python: how the scanner treats each

- A function already decorated `@numba.njit` (or the bare `@njit`, or
  `@vectorize`/`@guvectorize`/`@cfunc`/`@stencil`) is flagged HIGH
  confidence immediately -- it's already proven itself compilable by
  Numba, which is most of the work needed before attempting numba-metal.
- A function already decorated `@metal.jit` is also flagged HIGH
  confidence, labeled "Already running on numba-metal" -- the scanner's
  job here is just to surface it, not to suggest converting it again.
- A plain, undecorated Python function gets a LOW/MEDIUM rating from
  structural heuristics alone (nested loops, elementwise array ops,
  reduction patterns) and an explicit `unknowns` list (runtime array
  sizes, actual dtypes, call frequency, transfer cost) -- exactly the
  facts static analysis cannot know, named so you know what `compare`
  would need to measure before a real recommendation is possible.

## How compatibility is determined

`compatibility.py` runs a real, two-stage, **device-free** dry run --
it never touches a live Metal device and works even on a machine with
no Apple Silicon at all:

1. `compile_to_typed_ir(func, arg_types)` -- Numba's own typing pipeline.
   A failure here means Numba itself can't type the function; that's
   reported as `UNABLE_TO_ANALYZE` or `BLOCKED_BY_MISSING_FEATURE`.
2. `MSLKernelLowerer(name, typed).lower()` -- numba-metal's actual MSL
   codegen backend. This is where most real blockers surface (dicts,
   strings, recursion, exceptions, unsupported dtypes, etc.), each with
   the exact message the backend itself raises -- never summarized into
   a vague "not supported."

Five possible results, and the advisor never collapses them into a
single percentage:

- `SUPPORTED` -- both stages succeeded.
- `SUPPORTED_WITH_CHANGES` -- compiles, but with a caveat (e.g. would
  narrow float64 to float32).
- `BLOCKED_BY_MISSING_FEATURE` -- a specific, named construct numba-metal
  doesn't support yet.
- `POOR_GPU_CANDIDATE` -- compiles, but structurally unsuited to a GPU
  (e.g. heavy branching, tiny fixed size).
- `UNABLE_TO_ANALYZE` -- the dry run itself failed unexpectedly; this
  never crashes the surrounding scan.

## How opportunity scoring works

`OpportunityScore` is an explicit sum of six independently-computed,
bounded components -- never a single opaque number:

```
Hotspot (runtime share):      28 / 30   93% of measured total runtime
Parallelism:                  18 / 20   Independent per-element work, no cross-iteration dependency
Arithmetic intensity:         15 / 20   Multiple FP ops per array element
Compatibility:                20 / 20   SUPPORTED with no blockers
Call frequency:                 6 / 10   Called 40 times per run
Transfer/sync penalty:         -3 / 10   One synchronize() per call
-----------------------------------------------
TOTAL                          84 / 100
```

When only static evidence exists (no `compare` has been run), the score
is still computed but every component that would need runtime data
degrades to `STATIC_POTENTIAL` evidence, and the advisor calls the
result "POTENTIAL" rather than an estimated speedup.

## How to read the ASCII flame graphs

Three variants, one shared format (`+ - | = # . [ ]` only, no Unicode):

**CPU baseline** -- from the statistical sampler. Every row is tagged
`[ESTIMATED]` because sampling infers time from snapshot frequency, not
direct measurement, and a compiled Numba/Metal frame it can't see inside
is labeled accordingly rather than guessed at:

```
CPU BASELINE - 22.945 s
Each column represents approximately 310 ms
+----------------------------------------------------------------------------+
| <module>[numba_cpu]() ################################### 100% [ESTIMATED] |
| +-- main[metal_wrapper]() ############################### 100% [ESTIMATED] |
| |   +-- cmd_profile[metal_wrapper]() #################### 100% [ESTIMATED] |
```

**Metal run** -- from real instrumentation events (`profile`'s hooks
into `runtime/context.py`). These are genuinely measured wall-clock
spans, so rows carry no `[ESTIMATED]` tag at all:

```
METAL RUN - 0.179 s
Each column represents approximately 2 ms
+----------------------------------------------------------------------------+
| submission_to_completion ######################  50%                       |
| synchronize ############################  50%                              |
| nbmtl_metal_kernel_0    0%                                                 |
+----------------------------------------------------------------------------+
```

**Differential** -- CPU vs. Metal, side by side, with a fixed legend:

```
DIFFERENTIAL FLAME GRAPH
CPU: 22.945 s    METAL: 0.179 s    SPEEDUP: 128.30x
Legend:
  [-] less time after Metal
  [+] more time after Metal
  [=] materially unchanged
  [N] new Metal overhead
[-] <module>[numba_cpu]   -22.945 s  ######################################
[N] submission_to_completion   +0.089 s  .
[=] nbmtl_metal_kernel_0   +0.000 s  .
```

(All three captured verbatim from a real `numba-metal advisor profile
benchmarks/mandelbrot.py --no-color` run on this repository.)

Every flame graph works identically with `--no-color`: only the ANSI
escape codes around the `[-]`/`[+]`/`[=]`/`[N]` tags disappear, never
the tags themselves -- no meaning is ever carried by color alone.

## Why GPU work needs a separate timeline

**numba-metal exposes no true GPU-side kernel timestamp today.**
Reading `runtime/context.py` and `runtime/dispatcher.py` directly
confirms there is no `addCompletedHandler_` callback and no
`GPUStartTime`/`GPUEndTime` read anywhere in the runtime. So "how long
did the kernel take" can only be answered as a **host-side wall-clock
span**: the time between submitting a command buffer (`commit()`) and
`waitUntilCompleted()` returning. That single measured span is honestly
what this advisor calls "GPU kernel time" -- every such event is tagged
`gpu_timestamp_source: "host_wall_clock"`, never `"device_timestamp"`,
so nothing downstream can accidentally claim a precision the hardware
integration doesn't provide. If a future numba-metal version exposes a
real device timestamp, only that one field needs to change.

Because of this, the timeline treats "GPU queue" (command submitted),
"GPU kernel" (submission-to-completion span), and "CPU sync" (blocked
in `waitUntilCompleted()`) as three separate, correctly-labeled lanes
rather than merging them -- collapsing them would either hide CPU stall
time inside "GPU time" or vice versa:

```
CPU/GPU TIMELINE - 0 ms to 40474 ms
                0         6745      13491     20237     26982     33728
                |---------|---------|---------|---------|---------|----------
GPU queue       [           [                                              [
GPU kernel      [           [                                              []
CPU sync        [           [                                              []
```

Because Apple Silicon uses unified memory, this timeline also never
labels a unified-memory operation "transfer" by default -- `to_device`/
`copy_to_host` calls appear as sync/mapping/materialization events, not
a PCIe-style data-transfer bar, matching what the hardware is actually
doing.

## Cold, warm, and steady state

Every timing result is one of three explicitly separate regimes --
never merged, because they answer different questions:

- **COLD** -- includes MSL source generation, Metal shader compilation,
  and pipeline-state creation. This is what a user pays the very first
  time a kernel runs in a process.
- **WARM** -- the compiled pipeline is reused, but a call may still
  allocate buffers or synchronize.
- **STEADY STATE** -- many repeated calls with reusable buffers; the
  closest approximation to a long-running production workload.

`compare` reports median/min/max/mean/stdev/p90/p95 and sample count
`n` for each regime it measures, and calls out outliers explicitly
rather than silently discarding them. A single-run result is always
labeled `preliminary` and given LOW confidence -- the advisor never
reports a speedup from one run as if it were solid.

## Correctness verification

`compare` always checks correctness before recommending anything, using
exact equality for int/bool outputs and configurable `atol`/`rtol` for
float outputs, with explicit NaN/Inf-aware comparison and deterministic
sampling (a fixed `--seed`) when an output is too large to compare in
full:

```
CORRECTNESS
Result:                 PASS
Elements compared:      4,194,304
Dtype:
  CPU:                  int32
  Metal:                int32
```

**If correctness fails, the tool still reports the measured
performance -- it just refuses to recommend using Metal.** This is a
hard, structurally-enforced rule (`recommendations.py`'s
`_correctness_gate`): a `USE_METAL` recommendation literally cannot be
produced without an attached, passing `CorrectnessResult`. Verified by
an actual test in this repository (`test_correctness_failure_blocks_
use_metal_even_with_huge_speedup`): even a fabricated 10x measured
speedup with a failing correctness check produces only `DO NOT USE
METAL`, never a positive recommendation. The same holds when no
correctness check was run at all -- "we didn't check" is treated the
same as "it failed," not as "it's probably fine."

## Device calibration

`numba-metal advisor calibrate` measures machine-specific baseline
numbers once and caches them (`~/.cache/numba-metal/`, versioned JSON):
dispatch overhead, cold-compile cost, buffer allocation cost, sync
overhead, float32 throughput, memory bandwidth, and an estimated
CPU/GPU crossover size. Real numbers from an Apple M4 Pro:

| Metric | Value |
|---|---|
| Dispatch overhead | ~144 us |
| Cold compile | ~99 ms |
| Buffer allocation | ~62 us |
| Sync overhead | ~333 ns |
| Float32 throughput | ~10 GFLOPS |
| Memory bandwidth | ~60 GB/s |

**These are hints, not promises.** A project-specific measurement from
`compare`, run against your actual workload, always takes precedence
over a generic calibration number -- calibration exists to give a rough
sense of scale (e.g. "is my array big enough that dispatch overhead
won't dominate"), not to replace measuring your own code.

## JSON schema

`profile.json` (schema version 1) contains: a `schema_version` field, a
device/software metadata block (`DeviceInfo`: chip name/family, macOS
version, Python/Numba/numba-metal versions), the raw event list (each
event carries `measured: bool`, explicit units, and, for GPU events, a
`gpu_timestamp_source` tag), candidates, compatibility results,
comparisons (with the full COLD/WARM/STEADY_STATE breakdown),
correctness results, opportunity scores, and recommendations. Source
file paths are relativized to their basename by default
(`source_paths_included: false`) -- pass `--include-source-paths` to
keep absolute paths, and no source code is ever embedded unless you ask
for it. `numba-metal advisor report` reconstructs the exact same objects
from this JSON and renders them deterministically -- rendering the same
JSON twice, or on two different machines, produces byte-identical text
(verified by `tests/advisor/unit/test_export_roundtrip.py`).

## Profiling overhead

Instrumentation hooks (`runtime/context.py`'s `_submission_hooks`/
`_sync_hooks`) are empty lists by default and are only ever populated
while a profiling session is active; every call site checks
`if _submission_hooks:` / `if _sync_hooks and pending:` before doing any
hook-related work, so normal (non-profiled) numba-metal usage pays
nothing. This was verified directly on real Metal hardware: launching
1000 kernels with no profiler installed, with the profiler installed,
and after uninstalling it again produced timing within normal
run-to-run noise in all three cases (`tests/advisor/integration/
test_profiling_overhead.py`).

The default CPU sampling interval is 5ms. Raw events are collected
during the run and rendered only after the workload completes, so
rendering cost never perturbs the measurement itself. Event/sample
counts are bounded (`--sampling-interval-ms` and internal buffer
limits); when the bound is hit, the exact number of dropped events is
reported, never silently discarded.

There is deliberately no per-element instrumentation inside numerical
kernels -- that would dominate the very thing being measured.

## Privacy / local-only operation

The advisor makes no network requests, sends no telemetry, and uploads
nothing. `scan` never executes your code. `profile`/`compare` execute
only the exact script or test path you name, nothing else in your
project. Source file paths are stripped to basenames in saved JSON by
default. No LLM is called anywhere in the recommendation engine --
`recommendations.py` is pure, deterministic rule functions over
measured data (verified by `test_recommendations_never_call_an_llm`,
which asserts identical inputs always produce byte-identical output).

## Known limitations

- **No true GPU-side kernel timestamp.** See "Why GPU work needs a
  separate timeline" above -- every "kernel time" number is a host-side
  submission-to-completion span, not isolated device execution time.
  This is the single biggest fidelity gap in what this tool can measure,
  and it is disclosed everywhere that number appears (`gpu_timestamp_
  source: "host_wall_clock"`), not hidden.
- **The CPU sampler cannot see inside compiled Numba/Metal frames.**
  Time spent inside a JIT-compiled function shows up as the Python-level
  call that invoked it, tagged appropriately (`numba_cpu`/
  `metal_wrapper`), never fabricated as if the sampler could see
  further in.
- **`~/.cache/numba-metal/` as the calibration cache location is an
  assumption**, not a verified existing convention in this repository
  (no `platformdirs` dependency exists here) -- flagged explicitly so a
  future change can correct it without surprise.
- **The interactive TUI's keyboard-driven event loop was verified with
  real `blessed.Terminal()` rendering logic (header/footer/tab bodies),
  but not exercised via an actual interactive TTY session** in automated
  testing -- the non-interactive fallback path (used in CI, redirected
  output, and whenever `blessed` isn't installed) was fully verified.

## Troubleshooting

**"Metal profiling requires macOS on Apple Silicon"** -- `profile`,
`compare`, and `calibrate` all need a real Metal device and print this
exact message (plus what's still available: static scan, saved-report
rendering, and CPU-only profiling if supported) when one isn't found.
`scan` and `report` never print this message, because they never touch
a Metal device.

**A correctness check fails unexpectedly** -- check the reported
`max_abs_error`/`max_rel_error` against your `--atol`/`--rtol`; a
chaotic or ill-conditioned computation (e.g. Mandelbrot near the escape
boundary) can legitimately diverge between float32 GPU and float64 CPU
reference paths well before it's a real bug -- see
`docs/limitations.md`'s notes on this exact effect.

**A candidate function I expect to see isn't in `scan` output** -- the
scanner filters out functions with no signal at all (no decorator, no
structural pattern, no blocker) rather than listing every function in
the project; check `--verbose` or look for it in the JSON output
(`--format json`), which includes every analyzed candidate.

## Examples of good and poor GPU workload candidates

**Good**: `benchmarks/mandelbrot.py`'s kernel -- one independent thread
per pixel, no cross-thread dependency, several arithmetic ops per
element, called once per frame. Measured 15.4x faster on Metal at
2048x2048 on an Apple M4 Pro (see the `compare` example above).

**Poor**: anything in `benchmarks/common.py` -- these are host-side
helper functions (dict/string manipulation, timing utilities) with no
per-element numerical work at all; `scan` correctly rates every one of
them LOW POTENTIAL with a full list of blockers (dict literals, f-strings,
`raise` statements) rather than a HIGH rating.

## Adding a new instrumentation event

Instrumentation lives in `numba_metal/advisor/metal_events.py`, which
wraps two additive, disabled-by-default hook lists added directly to
`numba_metal/runtime/context.py` (`_submission_hooks`, `_sync_hooks`).
To add a new measured event type:

1. If it needs a new hook point in the runtime, add a small, additive
   callback list the same way (never change existing control flow or
   return values -- only add a call to registered hooks after existing
   logic has already run).
2. Add an `EventCollector` method (`_on_<event>`) that builds an `Event`
   from the callback's arguments and calls `self.record(...)`.
3. Register it in `install()`.
4. Give the new `event_type` a lane in `timeline.py`'s
   `_LANE_FOR_EVENT_TYPE` if it deserves its own row, otherwise it falls
   back to its `category`'s lane.

## Adding a new recommendation rule

Recommendation rules live in `numba_metal/advisor/recommendations.py`
as plain functions over `(Candidate, CompatibilityResult, ComparisonResult
| None, CorrectnessResult | None)`, returning a list of `Recommendation`
dataclasses. To add one:

1. Write a pure function -- no I/O, no randomness, no LLM call --
   taking whatever subset of those inputs it needs.
2. If the rule could ever suggest `USE_METAL`, route it through
   `_correctness_gate(correctness)` first; a rule that bypasses this
   gate is a bug, not a feature.
3. Populate every required `Recommendation` field: `supporting_
   measurement`, `file`/`line_start`, `confidence`, `evidence`
   (MEASURED/ESTIMATED/STATIC_POTENTIAL), and `how_to_verify` -- a
   recommendation with no way to verify it is not actionable.
4. Add it to `generate_recommendations()`'s call list.
5. Add a unit test with synthetic, fixed-clock inputs (see
   `tests/advisor/unit/test_recommendations.py` for the pattern) --
   never a test that depends on real machine timing.
