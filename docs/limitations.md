# Limitations

Blunt and specific, by design. If something isn't listed here as a
limitation, it should be listed as supported in
`docs/supported-features.md`; if it's in neither, that's a documentation
gap, not evidence it works.

## Unsupported Python features

- No `while` loops -- only `for x in range(...)` loop shapes are
  recognized by the control-flow structurer. A `while` (or any other loop
  the structurer can't match to a natural-loop-with-range-iterator shape)
  raises `UnsupportedFeatureError`.
- No recursion, no calling other Python functions from within a kernel
  (no device-function support yet -- see `docs/roadmap.md`).
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
- No multidimensional (`ndim > 1`) arrays as kernel arguments; flatten
  first and do index arithmetic manually (as `heat_diffusion.py` and
  `pairwise_distance.py` do).

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
