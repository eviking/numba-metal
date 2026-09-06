# Limitations

Blunt and specific, by design. If something isn't listed here as a
limitation, it should be listed as supported in
`docs/supported-features.md`; if it's in neither, that's a documentation
gap, not evidence it works.

## Unsupported Python features

- **`while` loops may contain a nested `if`/`else`, but not `break` or
  `continue` inside one.** Numba's bytecode lowering rotates
  `while cond: body` into a do-while-shaped CFG distinct from
  `for x in range(...)`'s shape, and when the body opens with an
  `if`/`else` (or otherwise doesn't put the condition re-test in the
  same block as the loop-carried phi nodes), the loop's real header and
  its real condition-test block end up as two different blocks --
  detected and handled as a `RotatedWhileNode` (see
  `compiler/structuring.py`'s extensive module comments for the full
  CFG-shape explanation), reusing the same if/else/break/continue
  structuring a `for`-range loop's body already used correctly. This
  covers ordinary if/else logic inside a `while` body (e.g. accumulate
  differently depending on a per-iteration condition, common in
  iterative numerical methods like Newton-Raphson) -- see
  `tests/integration/test_while_loops.py`'s
  `test_while_loop_with_nested_if_else_*` tests, verified correct
  across 1000 real parallel GPU threads with varied data, not just a
  single-thread smoke test.

  `break`/`continue` NESTED INSIDE that if/else remain unsupported and
  explicitly rejected with `UnsupportedFeatureError` at compile time
  (not silently mis-lowered): generalizing the structurer to handle
  that specific combination was found, by direct testing, to require a
  substantially larger rewrite than was in scope when this was fixed --
  two intermediate, silently-wrong-result bugs were found and fixed
  during earlier `while`-loop development, and that specific
  break/continue-inside-if combination remains the one shape not
  proven safe (see `tests/integration/test_while_loops.py`'s
  `test_while_loop_rejects_nested_break`/`_continue` and
  `docs/architecture.md`). A straight-line `while` body, or one with an
  if/else but no break/continue inside it, is fully supported and
  tested. Any other unrecognized loop shape still raises
  `UnsupportedFeatureError`.
- **Calling other Python functions from within a kernel is supported
  only via `@metal.device_func`** (a decorator, not calling an
  arbitrary undecorated function). Scalar arguments and return type,
  and 1D/2D/3D array arguments of a supported dtype (forwarded from the
  caller's own array, including transitively through nested
  device-function calls) -- no `metal.local_array`/`shared_array`
  inside a device function, and no array RETURN type (a device function
  can only read/write a caller-provided array, never allocate or return
  one of its own). Direct recursion is rejected by Numba's own frontend
  at typing time; mutual/transitive recursion between two device
  functions is rejected by numba-metal's own in-progress-compilation
  cycle detection. Each is compiled to a real, separate MSL function
  (never inlined). A known inefficiency (not a correctness issue): the
  same call compiled from two call sites whose Numba-level argument/
  return type tuples differ before narrowing (e.g. one call's literal
  arguments type as float64, another's as already-float32) but are
  identical after numba-metal's float64->float32 narrowing currently
  emits two functionally-identical MSL functions rather than
  deduplicating by the post-narrowing MSL signature.
- No exceptions (`try`/`except`/`raise`) inside kernels.
- No classes, no strings, no dicts/lists/sets, no f-strings, no `print()`.
- No closures over non-constant outer-scope variables -- only
  module-level globals that resolve to `metal`/`math` functions are
  understood.
- No local array allocation inside a kernel (e.g. `np.zeros(...)` in the
  kernel body).

## Unsupported NumPy operations

- No NumPy ufuncs callable from inside a kernel (`np.sqrt(x)` etc. inside
  the kernel body is unsupported -- use `math.sqrt(x)` instead, from the
  small supported subset in `docs/supported-features.md`).
- No broadcasting.
- 2D and 3D arrays ARE supported as kernel arguments (`arr[x, y]`/
  `arr[x, y, z]` indexing with a matching `metal.grid(2)`/`metal.grid(3)`
  launch -- see `tests/integration/test_multidim_arrays.py` and
  `benchmarks/heat_diffusion.py`). 4D and beyond are not; flatten those
  manually. `@metal.device_func` array arguments also support 2D/3D
  (each gets its own `_dimN` companion parameters, threaded through
  from the caller's own binding at every call site, including
  transitively through nested device-function calls -- see
  `msl_backend.py`'s `_emit_device_function_signature`/`_call` and
  `tests/integration/test_device_functions.py`); this was measured on
  `benchmarks/heat_diffusion.py`'s stencil (factoring the 4-neighbor
  sum into a device function taking `cur` directly): correct, but the
  non-inlined MSL function call costs a real, consistent ~1.9-2.4x
  per-iteration slowdown on an Apple M4 Pro -- not free, and not
  recommended for code this small and this hot; see that benchmark's
  module docstring ("Round 3") for the full numbers. There is no
  `.shape` attribute access inside a
  kernel body (only `.size`, the total flattened element count) -- read a
  dimension's size from a separately-passed scalar argument if a kernel
  needs it directly. Manual index-flattening (`arr[x*n+y]`) still works
  and remains necessary for 4D+ data, but was found, directly measured, to
  meaningfully slow down BOTH Numba's own CPU codegen and Metal's
  performance relative to real multi-dimensional indexing at the same
  problem -- see `docs/performance-guidance.md`'s bandwidth-bound section
  for the measured effect size before choosing to flatten by hand when 2D/
  3D support already covers the case.
- **No negative-index wraparound, on any array dimensionality.** Real
  Numba (CPU) implements Python/NumPy's `a[-1]` meaning "last element"
  with real runtime wraparound arithmetic (see
  `numba/np/arrayobj.py`'s `fix_integer_index`) -- numba-metal's MSL
  codegen does not, and never has, for 1D arrays either; this was only
  found by comparing against Numba's own reference implementation while
  building 2D/3D support. A LITERAL negative index (`a[-1]`, `a[-1, 0]`)
  is now rejected at compile time with `UnsupportedFeatureError` rather
  than silently reading/writing an out-of-bounds offset (see
  `tests/test_unsupported.py`'s `test_negative_literal_index_rejected*`
  tests). A runtime-VARIABLE index that happens to go negative
  (`a[x - 1]` where `x` can be 0) cannot be checked this way -- MSL has
  no bounds checking, and this backend has no general bounds-checking
  machinery of its own for any array access -- and remains silent,
  undefined out-of-bounds behavior, same as any other
  out-of-range index in this project.
- **Whole-array reductions (`metal.reduce_sum`/`reduce_min`/
  `reduce_max`) support float32/int32/uint32 1D device arrays only.**
  No 2D/3D array reduction, no other dtype, and no `argmax` (returning
  the winning index alongside the value) -- only the value-only
  reductions are implemented. This is a host-side helper built from
  existing primitives (`metal.shared_array`, `metal.barrier`, atomics),
  not a new compiler intrinsic, so its dtype ceiling is exactly MSL's
  native/CAS-loop atomic dtype set (`_ATOMIC_DTYPES` in
  `compiler/intrinsics.py`) -- there would be no way to combine
  per-threadgroup partials for any other dtype. See
  `numba_metal/reductions.py` and `tests/integration/test_reductions.py`.

## Type restrictions

- **float64 is rejected for array dtypes and kernel scalar arguments.**
  It has not been verified on any Apple GPU family and numba-metal makes
  no claim about it.
- **Local float64 intermediates are silently narrowed to float32.** A
  Python float literal (`x = 0.0`) or a true-division result (`a / b`
  where both are integers) infers as `float64` under Numba's ordinary
  typing rules -- exactly like CPython. Since MSL's `float` is 32-bit,
  numba-metal's backend maps these *local, non-argument* float64 values
  to MSL `float` rather than rejecting every kernel containing a literal
  or a division. This is a real precision reduction, not a bug: a kernel
  computing `x = a[i] / b[i]` where `a[i]`/`b[i]` are `int32` will produce
  a float32-precision result even though the equivalent CPU Python code
  would compute in float64. This was discovered concretely while
  validating the Mandelbrot benchmark (see below) and is why that
  benchmark's CPU reference implementation was rewritten to use explicit
  `np.float32()` casts throughout, for a fair comparison.
- No `int8`/`int16`/`uint8`/`uint16`.
- `uint64` has a type mapping but is not exercised by any kernel in this
  repository's test suite or benchmarks -- treat it as unverified.
- **`int64` device-array kernel ARGUMENTS have been found to trigger a
  real Metal shader-compiler failure for at least one real kernel
  shape** (nested `for` loops with no other unusual features), even
  though the generated MSL text itself looks well-formed and `int64` is
  otherwise documented as supported. Reproduced directly: the identical
  kernel body compiles and runs correctly with an `int32` output array,
  and fails with `AGXMetalG16X ... XPC_ERROR_CONNECTION_INTERRUPTED`
  (a real Metal-driver-level shader-compiler error, not a numba-metal
  exception) with an `int64` one. Not yet root-caused or minimally
  reproduced further -- found incidentally while adding
  `tests/integration/test_while_loops.py` coverage, unrelated to that
  file's own subject (the working test there uses `int32`). Treat
  `int64` OUTPUT arrays specifically as unverified until this is
  investigated; `int64` used only for scalars/loop counters (as
  `metal.grid()` already does internally) has not shown this problem.

## Hardware requirements

- Apple-silicon (arm64) Macs only. Verified and enforced at runtime
  (`UnsupportedPlatformError` on any other architecture).
- macOS 14+, enforced at runtime.
- Requires the Xcode command-line tools *and* the separately-downloadable
  Metal Toolchain component (`xcodebuild -downloadComponent
  MetalToolchain`) -- a fresh Xcode install was observed, while building
  this package, to be missing the Metal Toolchain by default, requiring
  this explicit one-time download before any kernel could compile. See
  `docs/installation.md`.
- All development, testing, and benchmarking for this MVP was done on a
  single machine (Apple M4 Pro / "Metal 4" family GPU). No claims are
  made about behavior on M1/M2/M3 or other GPU family/core-count
  combinations -- they have not been tested.

## Compilation latency

- First-time compilation of a given kernel+signature (Numba typed-IR
  frontend + MSL lowering + Apple's Metal shader/pipeline compilation)
  takes roughly 10-235ms total depending on kernel complexity, measured
  and separated into phases directly in this repository's own benchmark
  suite (`metal_frontend_ns`, `metal_pipeline_compile_ns`,
  `metal_cold_total_ns` in `benchmarks/*.py` / `docs/benchmarking.md`
  output, each confirmed genuinely cold via a before/after compiled-
  kernel-count check). There is no persistent, cross-process compilation
  cache in the MVP -- every new Python process recompiles every kernel
  from scratch on first launch. See `docs/roadmap.md` Phase 1.

## GPU dispatch overhead

- Every kernel launch has fixed overhead (command buffer encoding, buffer
  binding, dispatch, and this project's `synchronize()`-before-returning
  benchmark convention). Measured directly: at small problem sizes (e.g.
  10,000-element vector polynomial, 200x200 pairwise distance), the
  kernel-only warm GPU time is slower than parallel Numba CPU by roughly
  1.4x-5.3x -- see `docs/benchmarking.md` for the full, separated-by-
  category numbers (and note that Numba's own `prange` thread-pool
  dispatch overhead can *also* make its parallel variant slower than
  single-threaded at these same small sizes, an independent effect
  reported there too). This is expected GPU-dispatch behavior, not a
  defect, but it means numba-metal is not a good fit for small,
  latency-sensitive workloads.

## Memory-transfer behavior

- `to_device()`/`copy_to_host()` are **not zero-copy**, even though Apple
  silicon has unified memory. The current implementation performs a
  host-side `memcpy` into/out of a shared-storage-mode `MTLBuffer`'s own
  backing memory, because a NumPy array and an `MTLBuffer` are distinct
  allocations in this implementation. This is documented honestly in
  `docs/architecture.md` rather than described as free; a genuinely
  zero-copy path is on the roadmap.
- No streams, multiple command queues, or asynchronous transfer overlap
  with compute -- a single process-wide serial command queue is used for
  everything.
- **Every `copy_to_host()`/`copy_to_device()` call is a full
  synchronization boundary.** Both wait for *all* currently outstanding
  GPU work (every submitted kernel, not just ones touching the specific
  buffer involved) before touching host/device memory. This is
  deliberately conservative rather than fine-grained per-buffer
  dependency tracking; it is correct but means a host transfer on a
  buffer with no relationship to an unrelated in-flight kernel will still
  wait for that kernel to finish. See `docs/architecture.md`, "Host/device
  synchronization model", for why this is necessary (host reads/writes of
  shared-memory buffers race outstanding GPU work otherwise -- this was
  confirmed by deliberately removing the synchronization and observing an
  in-flight kernel's actual output change) and the roadmap for a possible
  future per-buffer tracker.

## Numerical differences

- **GPU (float32) vs. CPU (float64-by-default) results can diverge
  significantly on chaotic/iterative algorithms**, even when both
  "should" compute the same thing. Concretely measured in this project:
  Mandelbrot escape-iteration counts differed by up to 25 iterations (out
  of 100) for a small number of pixels near the escape boundary when
  comparing a float32 GPU kernel against a float64-inferred Numba CPU
  reference -- purely from floating-point precision, not from any
  codegen defect (verified by making the CPU reference float32 too, which
  produced an exact match). Any kernel with chaotic/sensitive numerics
  should be validated at the precision it will actually run at, not
  assumed to match a higher-precision reference.
- Metal's fast-math shader compilation mode (FMA fusion, reassociation)
  is explicitly disabled by numba-metal to reduce (not eliminate) this
  class of divergence; see `docs/architecture.md`.
- Integer `//` uses MSL's native truncating division, which differs from
  Python's floor division for negative operands. Not an issue for the
  non-negative flattened-index arithmetic used throughout this project's
  benchmarks, but a real difference for kernels that do signed integer
  division.

## Dependency on Apple tooling

- Requires `pyobjc-framework-Metal`/`pyobjc-framework-libdispatch` at
  runtime for all device/buffer/command-queue access.
- Requires `xcrun`/the Metal shader compiler toolchain at runtime for
  every *new* kernel compilation (cached kernels within a process don't
  need it again, but nothing is cached across processes).
- All of the above are Apple-platform-specific and have no equivalent
  fallback; this package does not attempt to run anywhere else.

## Compatibility risk with future Numba versions

- The frontend adapter (`numba_metal/compiler/frontend.py`) depends on
  `numba.core.compiler.CompilerBase`, `numba.core.compiler_machinery`,
  and `numba.core.typed_passes.{NopythonTypeInference, AnnotateTypes}` --
  none of which carry a public API stability guarantee across Numba
  releases. `pyproject.toml` pins a bounded, tested Numba version range;
  a Numba release that changes this internal pipeline shape in a
  breaking way will surface as a `KernelCompilationError` mentioning "did
  not produce a TypedKernelIR" (a defensive check specifically added to
  fail loudly rather than silently misbehave in that scenario -- see
  `docs/architecture.md`).
- Only one Numba version (0.67.0) has actually been exercised end-to-end
  against real GPU execution while building this package. The declared
  compatible range in `pyproject.toml` (`>=0.67,<0.68`) is narrowed to
  match exactly that -- it is not a claim that a wider range has been
  tested.
- A runtime compatibility gate
  (`numba_metal.compat.check_numba_compatible`, invoked by
  `metal.jit`/`metal.get_device_info()` via `check_capable()`) raises a
  specific `UnsupportedNumbaVersionError` if the installed Numba version
  is outside the validated `0.67.x` series, rather than relying solely
  on the packaging dependency bound being enforced (which it might not
  be, e.g. under `pip install --no-deps` or an in-place Numba upgrade).
  This converts "might silently produce wrong MSL from a changed
  internal-IR shape" into "fails immediately with a specific,
  actionable error" -- it does not make an untested version work, only
  makes the failure mode safe. See `docs/numba-rfc.md` for the full
  compatibility table and the internal APIs this depends on.

## What has and hasn't been verified

Everything described as "Supported" in `docs/supported-features.md` has
a corresponding automated test that passed on real Metal-GPU hardware (an
Apple M4 Pro) during development of this package -- see `tests/` and
`pytest -m metal`. Nothing in this repository's documentation describes
behavior that was only reasoned about but never executed.
