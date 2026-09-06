# Changelog

## Unreleased (reductions, multi-dim device functions, compile caching, and an honest benchmark rewrite)

### Added

- **`metal.reduce_sum`/`reduce_min`/`reduce_max`**
  (`numba_metal/reductions.py`): a host-side helper composed entirely from
  already-tested primitives (`metal.shared_array`, `metal.barrier`,
  `metal.atomic_add`/`atomic_min`/`atomic_max`) -- no new compiler
  intrinsic or MSL codegen was needed. A two-stage algorithm (a
  per-threadgroup partial via a shared-memory tree reduction, then one
  atomic op per threadgroup to combine partials) avoids the O(n) atomic
  contention a naive one-atomic-per-thread reduction would hit, and
  handles non-power-of-two array sizes via bounds-checked identity-element
  padding (not skipping the shared-memory write, which would read back
  stale data from a previous launch -- see
  `test_shared_array_repeated_launches_do_not_leak_stale_data`). Supports
  float32/int32/uint32 1D device arrays; `argmax` is not implemented. See
  `tests/integration/test_reductions.py` (26 tests) and
  `benchmarks/monte_carlo_paths.py`/`benchmarks/asian_option_pricing.py`,
  both of which now offer a GPU-reduced payoff path alongside their
  original CPU-reduced path for direct, honest comparison: the GPU-side
  reduction is measured slower at the smallest sizes tested (dominated by
  a second kernel launch's fixed overhead) and modestly faster (roughly
  3-16%) at the largest, growing with array size as expected from
  avoiding a full-array `copy_to_host()`.
- **Native 2D/3D array arguments for `@metal.device_func`**
  (`compiler/msl_backend.py`'s `_classify_params`,
  `_emit_device_function_signature`, `_call`): a multi-dimensional
  device-function array parameter now gets its own `_dimN` companion
  parameters (plain by-value `uint`, not the kernel-level `constant
  uint&` binding), threaded through from the caller's own
  `arg_{name}_dimN` at every call site -- including transitively through
  nested device-function calls, verified by a dedicated regression test.
  `_flat_index_expr` already read `arg_{name}_dimN` by naming convention
  regardless of parameter kind, so this was purely a parameter-threading
  change, not a new indexing scheme. See
  `tests/integration/test_device_functions.py`'s new 2D/3D/transitive
  -forwarding tests and `benchmarks/heat_diffusion.py`'s
  `_make_metal_kernel_device_func`, which factors the stencil's per-point
  update into a device function taking the 2D grid directly as a
  real-world case study -- measured, not assumed, to cost a real ~1.9-2.4x
  per-iteration slowdown for this specific small, hot-loop update
  (reported honestly rather than presented as a free code-organization
  win).
- **A process-wide `@metal.device_func` compile-time cache**
  (`compiler/msl_backend.py`'s `_device_function_compile_cache`): a
  device function shared by multiple different kernels was previously
  independently re-lowered (Numba frontend + MSL codegen) once per
  calling kernel, even though the generated MSL is identical every time.
  The cache is keyed by `(py_func, arg_types)` and stores every
  transitively-needed source alongside each entry, not just a function's
  own body -- two real bugs (a missing-transitive-dependency bug causing
  an "undeclared identifier" Metal shader-compile error, and a duplicate
  -MSL-function-definition bug in a diamond-dependency case) were found
  and fixed while building it, both covered by dedicated regression
  tests before the cache was trusted. Measured in
  `benchmarks/device_function_compile_cache.py`: cold-compiling 30
  kernels sharing one device function is 1.25-1.89x faster than each
  kernel independently compiling its own copy (varies with system load
  across repeated runs, always a genuine speedup, never a loss -- a
  first, buggy version of this cache measured a 0.92x net loss and was
  not shipped).
- `docs/benchmark-report.html`: a saved, plain-language rendering of the
  full NumPy/Numba/Metal comparison across all nine benchmarks, with a
  "what each benchmark actually computes" section aimed at readers with
  no GPU or compiler background. Linked from the README.
- `docs/research-paper.md` (+ a published HTML rendering): a
  paper-structured writeup of the compiler architecture, the full
  capability surface, a controlled roofline-model demonstration (see
  below), and a full account of a compiler fix that was verified against
  500 randomized cases and the entire test suite before being found to
  silently miscompile a real production kernel and fully reverted --
  presented as the paper's central methodological argument. Includes an
  explicit scope statement: this is a single-session engineering writeup,
  not a peer-reviewed publication. Related-work citations (Apple MLX,
  PyTorch's MPS backend, `numba.cuda`, Triton, Halide, TVM, JAX, and an
  independent arXiv paper on Apple Silicon memory bandwidth) were
  verified against primary sources before writing.

### Changed

- **`benchmarks/heat_diffusion.py` now solves nonlinear (Perona-Malik)
  anisotropic diffusion instead of linear diffusion.** The linear model
  (`next = 0.25 * sum of 4 neighbors`, a constant coefficient everywhere)
  was found to be bandwidth-bound on Apple Silicon's unified memory:
  tested at 10x and 100x the benchmark's largest standard grid's point
  count, Metal got WORSE relative to parallel Numba CPU as the grid grew
  (0.52x and 0.60x, down from ~1.0x parity at 1024²), since bandwidth
  pressure grows with grid size while a constant-coefficient stencil's
  arithmetic does not. The nonlinear replacement keeps the identical
  4-neighbor memory-access pattern but makes the diffusion coefficient at
  each point depend on the local gradient (`exp(-(gradient/kappa)^2)`, 4
  `math.exp` calls per point per iteration instead of one constant
  multiply) -- a real, physically-motivated increase in arithmetic
  intensity, the standard nonlinear diffusion model from the
  image-processing literature (Perona & Malik, IEEE PAMI 1990), not a
  synthetic stand-in. Measured, correctness verified at every size:
  128²→0.89x, 512²→3.32x, 1024²→7.15x (4.6-7.2x observed across repeated
  runs), and at the same 10x/100x scale that broke the linear
  version→**13.05x and 13.66x**, improving rather than degrading with
  scale -- the cleanest available demonstration in this project's own
  suite that arithmetic intensity, not problem size, determines which
  side of the roofline a kernel falls on. The original linear-diffusion
  benchmark and its "roughly parity" finding remain in git history; see
  `docs/performance-guidance.md` for the full before/after writeup.
- **`benchmarks/mandelbrot.py`, `benchmarks/pairwise_distance.py`,
  `benchmarks/monte_carlo_paths.py`, and
  `benchmarks/asian_option_pricing.py`** were converted from manual
  flattened-index arithmetic (`idx // n`, `idx % n`, `i*n+j`) to native
  2D array indexing, applied fairly to both the Metal kernel and the
  Numba-CPU reference simultaneously in every case (never one side
  only -- the exact mistake `heat_diffusion.py`'s own earlier
  investigation found and corrected). Mandelbrot and pairwise_distance
  showed real, measured improvements from the conversion (Mandelbrot:
  16.1x→29.2x at 2048², 17.8x→38.0x at 4096²; pairwise_distance:
  1.14x→1.93x at 2000², 1.75x→2.31x at 5000²) since both have genuinely
  spatial access patterns; monte_carlo_paths and asian_option_pricing
  showed no measurable difference either way, confirming the hypothesis
  that a purely sequential per-thread access pattern gains nothing from
  native indexing over hand-computed offsets, since both compile to the
  same row-major arithmetic.

### Investigated and reverted (not shipped)

- **`break`/`continue` nested inside an `if`/`else` within a `while`
  loop whose body opens with that `if`/`else`** (the `RotatedWhileNode`
  shape). A real, confirmed structural bug was found and fixed in
  `_structure_branch`'s merge-point resolution (a nested if/else's own
  arm, unable to find its own internal merge point, would recurse
  unbounded into a block the enclosing loop region had already committed
  to structuring separately, producing a duplicated copy of the loop's
  remaining body). The fix for `break` passed 500 randomized test cases
  and the full pre-existing test suite with zero mismatches -- and was
  still found, only once tested against `benchmarks/implied_volatility.py`'s
  real Newton-Raphson kernel (whose `abs_diff < tol or vega < 1e-6` exit
  condition lowers as a shared-branch-target pattern no randomized test
  case had exercised), to produce silently incorrect results on all 1,000
  test elements, traced to de-SSA phi-copy placement being sensitive to a
  block's position in the structured tree, not merely its content. Fully
  reverted (confirmed via a clean `git diff`) rather than shipped with a
  caveat. `continue` in the same position remains unsupported for a
  separate, deeper reason (Numba routes it through a guard block that
  looks like a second loop header requiring an incompatible emission
  strategy) and was never attempted this pass. See
  `docs/limitations.md` and `docs/research-paper.md` §6.2 for the full
  account.

## Unreleased (device-function array arguments)

### Added

- **`@metal.device_func` now accepts 1D array arguments**, matching the
  existing constraints on kernel array arguments (1D only, same
  supported-dtype set). Previously scalar-argument-only. A device
  function's array parameter is emitted in the same `device`-address-space
  MSL pointer form a kernel-level array argument already uses, so
  `metal.atomic_*()` intrinsics (which always cast to `device atomic_<T>*`)
  work correctly whether the array being indexed is a kernel argument or
  was forwarded into a device function.
- **`metal.atomic_add`/`atomic_sub`/`atomic_min`/`atomic_max`/
  `atomic_exchange`/`atomic_compare_exchange` are now usable from inside a
  `@metal.device_func` body**, including a `while`-loop compare-and-swap
  retry pattern -- verified under real multi-thread contention against a
  sequential reference, not merely "compiles." This required giving these
  six intrinsics a real (previously `_unimplemented_codegen`) CPU
  `codegen`, compiled via `context.compile_internal`: `@metal.device_func`
  wraps its target in a real `njit` dispatcher so a *caller's* frontend can
  type the call site, and typing a Dispatcher call unavoidably forces
  Numba to fully compile (and lower) that dispatcher, including any atomic
  intrinsic calls in its body -- this CPU-lowered version is never
  actually executed (the MSL backend always re-derives real GPU semantics
  from the original plain function's typed IR), but it must not crash
  during that throwaway compile. A correct single-threaded sequential
  implementation is safe here since no real concurrency exists in that
  throwaway compile. `metal.grid`/`gridsize`/thread-position/
  `local_array`/`shared_array`/`barrier` remain kernel-only and still
  raise if called from a device function (they have no meaningful
  standalone CPU or out-of-dispatch GPU semantics to fall back to).
- New tests: `tests/integration/test_device_functions.py` gains array
  -argument, mixed array/scalar-signature, nested-call array-forwarding,
  atomic-CAS-loop-under-contention, and 2D-array-rejection cases.

## Unreleased (post-MVP hardening pass)

A correctness- and evidence-focused hardening pass over the initial MVP
below. No new user-facing features were added; this pass fixed several
real correctness bugs, closed test-coverage gaps, and corrected several
places where documentation or benchmark methodology overstated what had
actually been verified.

### Fixed

- **Phi-node/SSA-destruction correctness.** Replaced an unsound
  union-find-based phi aliasing scheme with a proper de-SSA pass
  (`compiler/dessa.py`) that associates each phi's incoming values with
  the correct predecessor edge, handles copy cycles and critical edges,
  and preserves loop-carried variables. The prior scheme could silently
  alias distinct variables across divergent branches. See
  `docs/architecture.md`, "De-SSA: correct phi-node elimination."
- **Command-buffer error propagation.** Every submitted Metal command
  buffer is now tracked through completion (`SubmissionRecord`,
  `runtime/context.py`); `metal.synchronize()` inspects the status/error
  of every outstanding buffer and raises `MetalRuntimeError` naming the
  specific failing kernel and submission, instead of relying solely on
  completion-callback exceptions that could be silently lost.
- **Host/device synchronization race.** `copy_to_device()` now
  synchronizes with outstanding GPU work before writing (previously
  could race an in-flight kernel's reads of the same buffer);
  `copy_to_host()`'s existing synchronize-first behavior is now
  documented as part of the same model. See `docs/architecture.md`,
  "Host/device synchronization model."
- **Benchmark cold-timing reuse bug.** Every benchmark's "cold" timing
  previously reused one persistent dispatcher across a problem-size
  sweep, so only the first size measured in a run was ever actually
  uncompiled. Cold measurements now use a brand-new dispatcher/cache
  (or bypass the cache entirely) per measurement, with
  `compiled_before`/`compiled_after` cache-entry counts recorded as
  proof each cold number is genuinely cold.
- **Three real MSL-codegen bugs**, found by the new differential test
  suite: `min()`/`max()` emitting an ambiguous MSL overload for
  mixed-width integer arguments; a for-range loop with no back edge
  (e.g. an unconditional `break`) leaving a `RangeIteratorType`-typed
  temporary undeclared; and a copy-chain-following helper
  (`_find_pair_first_target`) that could walk past the real loop
  variable into unrelated user code.
- **Mislabeled compiler-error test.** A test named as if it exercised a
  real Metal shader compiler failure actually only tested a Numba
  frontend rejection (its own docstring admitted this). Renamed to
  `test_unsupported_construct_rejected_before_metal_compilation`; a new
  test genuinely exercises Apple's Metal compiler via a monkeypatched
  invalid-MSL codegen path.
- **NumPy scalar dtype constructors inside a kernel body** (e.g.
  `np.uint32(x)`) are rejected with `UnsupportedFeatureError`, found and
  now documented while adding feature-matrix test coverage.
- **Zero-block launches** are not silently skipped for zero-length
  arrays: `kernel[0, threads](...)` raises `KernelLaunchError` like any
  other invalid launch geometry, found and now documented while adding
  boundary-value test coverage.

### Added

- **Hypothesis-based differential testing** (`tests/differential/`):
  generates kernels from a closed grammar and compares real Metal
  execution against real Numba CPU execution. Quick profile (25 cases,
  always runs) and exhaustive profile (500 cases, opt-in via
  `NUMBA_METAL_DIFFERENTIAL_EXHAUSTIVE=1`; 500/500 passing on this
  hardware in ~42s).
- **Standardized, separated benchmark timing categories**
  (`benchmarks/common.py`): frontend/lowering time, MSL pipeline-compile
  time, cold-total time, kernel-only warm time, H2D/D2H transfer time,
  end-to-end warm time, and resident-pipeline time are now distinct
  fields, never conflated. Every benchmark reports both parallel
  (`prange`) and single-threaded Numba CPU baselines with thread count
  recorded.
- **Feature-matrix traceability audit** (`docs/feature-traceability.md`):
  every "Supported" claim in `docs/supported-features.md` is now backed
  by a real Metal execution test; 21 new tests
  (`tests/integration/test_feature_matrix_gaps.py`,
  `tests/integration/test_boundary_values.py`) close gaps found during
  the audit (`metal.gridsize`, `min`/`max`, casts, `uint32`/`float16`
  array I/O, `int64`/`bool` as kernel inputs, all comparison/boolean
  operators, `NUMBA_METAL_DUMP_MSL`, dtype boundary values, NaN/inf,
  odd/non-divisible grid sizes, zero-length arrays).
- **Runtime Numba-version compatibility gate**
  (`numba_metal.compat.check_numba_compatible`): raises a specific
  `UnsupportedNumbaVersionError` naming the installed version if it's
  outside the validated `0.67.x` series, rather than trusting the
  packaging dependency bound alone.
- **Upstream documentation**: `docs/numba-rfc.md` (findings for
  numba/numba#5706, informed by actually reading its multi-year
  discussion thread) and `docs/upstream-strategy.md` (assesses staying
  external vs. formal target registration vs. upstreaming; recommends
  staying external).
- `scripts/build_release.sh`: builds sdist+wheel, runs `twine check`,
  verifies no dev/cache/macOS-metadata paths leaked into either
  distribution, installs into a clean throwaway virtualenv, and
  smoke-tests the install (import, capability check, vector-add
  example, full test suite).

### Changed

- Numba dependency narrowed from `>=0.59,<0.68` (9 minor versions, only
  one ever tested) to `>=0.67,<0.68`, matching what has actually been
  validated.
- `requires-python` narrowed from `>=3.10,<3.14` to `>=3.12,<3.14`;
  3.10/3.11 were never available to test in this project's development
  environment. Both 3.12 and 3.13 have the full test suite run and
  passing. CI now matrices across both.
- Corrected benchmark results regenerated
  (`benchmarks/results/corrected_m4pro_<date>.json`) under the new
  timing methodology; the prior results file is kept as legacy
  (different schema, not directly comparable).

## 0.1.0.dev0 (unreleased)

Initial MVP.

### Added

- `@metal.jit` kernel decorator and `kernel[blocks, threads](*args)`
  launch syntax (1D and 2D grids).
- `metal.grid(ndim)` / `metal.gridsize(ndim)` intrinsics.
- Device memory API: `metal.to_device`, `metal.device_array`,
  `metal.device_array_like`, `DeviceNDArray.copy_to_host`/`copy_to_device`.
- `metal.synchronize()`.
- Typed-Numba-IR-to-MSL compiler backend supporting: scalar arithmetic,
  comparison and boolean operators, assignment, 1D array reads/writes,
  flattened multidimensional indexing, `if`/`if-else`, `for x in
  range(...)` with `break`/`continue`, `abs`/`min`/`max`, and
  `math.sqrt`/`exp`/`log`/`sin`/`cos`.
- Support for `float32`, `int32`, `uint32`, `bool` (required) and
  `float16`, `int64` (optional) dtypes.
- In-process compilation cache keyed by kernel source, argument
  signature, and device identity.
- `NUMBA_METAL_DUMP_MSL` / `metal.config.dump_msl` debug MSL dumping.
- Explicit platform/device/toolchain capability checks
  (`UnsupportedPlatformError`, `MetalToolchainError`) that fail fast on
  unsupported hardware.
- Five benchmark programs (vector polynomial, Mandelbrot, heat diffusion,
  Monte Carlo paths, pairwise distance) comparing Python/NumPy/Numba
  CPU/Metal with correctness checks and cold/warm/transfer timing
  breakdown, plus `benchmarks/run_all.py` text + JSON reporting.
- Unit tests (compiler/frontend/codegen, no GPU required) and integration
  tests (`pytest -m metal`, real GPU execution), all passing on an Apple
  M4 Pro during development.
- Full documentation set: installation, quickstart, supported-features
  matrix, architecture (with Mermaid diagram), benchmarking methodology,
  limitations, troubleshooting, and a prioritized roadmap.

### Known limitations

See `docs/limitations.md`. Notably: no `while` loops, no device
functions/recursion, no float64 array support, no zero-copy host<->device
transfer yet, and only one Numba version (0.67.0) has been exercised
end-to-end against real hardware so far.
