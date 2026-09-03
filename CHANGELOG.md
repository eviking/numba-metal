# Changelog

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
