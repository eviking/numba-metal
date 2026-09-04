# Roadmap

Prioritized, not a wish list. Each item states user value, technical
dependency, principal risk, and rough complexity (small/medium/large).

## Phase 1: Harden the MVP

**Broader correctness testing** -- largely done in the post-MVP
hardening pass (see `CHANGELOG.md`): a Hypothesis-based differential
test suite (`tests/differential/`, 500/500 exhaustive cases passing),
integer overflow/wraparound and float32 boundary-value coverage
(`tests/integration/test_boundary_values.py`), and a feature-matrix
traceability audit closing every "Supported"-but-under-tested gap
(`docs/feature-traceability.md`). Remaining: this is a floor, not a
ceiling -- more generated-kernel grammar coverage (device functions once
supported, more dtype combinations) remains valuable as the supported
subset grows.
Dependency: none; extends `tests/`.
Risk: low.
Complexity: small-medium, incrementally, going forward.

**Stable Numba-version adapters**
User value: reduces breakage when users upgrade Numba.
Dependency: testing against multiple Numba versions in the declared
range. Still only 0.67.0 has been exercised against the full test
suite; a runtime compatibility gate now exists
(`numba_metal.compat.check_numba_compatible`) that fails explicitly
rather than silently on any other version, but expanding the actually-
*validated* range still requires running the test suite against each
new candidate version -- that has not happened yet for anything past
0.67.0.
Risk: medium -- `numba.core.typed_passes`/`compiler_machinery` have no
stability guarantee, so this is ongoing maintenance, not a one-time fix.
Complexity: medium.

**Improved diagnostics**
User value: faster debugging when a kernel fails to compile or produces
wrong results.
Dependency: none.
Risk: low.
Complexity: small (e.g. surfacing the offending Python source line number
alongside `UnsupportedFeatureError`, not just the IR var name).

**Persistent compilation caching**
User value: eliminates redundant recompilation across process runs
(currently every new process recompiles every kernel from scratch --
see `docs/limitations.md`).
Dependency: a stable, safe cache-invalidation key (kernel source hash +
signature + numba-metal version + device identity, extending the
existing in-memory `KernelCache` key in
`numba_metal/compiler/pipeline.py`) and a filesystem cache directory
convention.
Risk: low-medium (cache invalidation bugs are the classic failure mode;
must be conservative about what counts as "the same kernel").
Complexity: medium.

**Automated Apple-silicon testing (CI)** -- partially done: a Python
3.12/3.13 matrix now runs lint, format check, and non-GPU unit tests on
every push/PR (`.github/workflows/ci.yml`). The GPU-requiring job
(`gpu-tests`, `pytest -m metal` + a quick benchmark run) is written and
matrixed the same way but still gated `if: false`, since it requires a
self-hosted Apple-silicon runner that has not been provisioned.
User value: confidence that changes don't silently break GPU execution.
Dependency: access to Apple-silicon CI runners (GitHub Actions macOS
arm64 runners, or self-hosted).
Risk: low; mainly a resourcing/cost question, not technical.
Complexity: small (flip `if: false` once a runner is provisioned; the
workflow itself is already written).

**Performance profiling**
User value: understand where time actually goes (compile vs. dispatch vs.
kernel execution vs. transfer) beyond what `benchmarks/common.py`
currently reports.
Dependency: Metal's own GPU capture/profiling tools (`MTLCaptureManager`)
or `os_signpost` integration.
Risk: low.
Complexity: medium.

## Phase 2: Scientific-computing features

**Reductions** (sum/min/max/argmax across a device array) -- the
low-level primitives this depends on are now done (threadgroup memory,
barriers, atomics; see below), but a dedicated multi-stage/tree
reduction *helper* (as opposed to a hand-written single-threadgroup
cooperative reduction, which `tests/integration/
test_local_and_shared_memory.py` already demonstrates) is not yet
built.
User value: avoids the current pattern of downloading results and
reducing on the CPU (as `monte_carlo_paths.py` does today, documented as
a boundary in that benchmark).
Dependency: threadgroup-memory support (done, see below).
Risk: medium -- getting a correct, race-free parallel reduction in MSL
right (threadgroup barriers, non-power-of-two sizes) is a real
correctness hazard.
Complexity: medium.

**Shared/threadgroup memory** -- done. `metal.shared_array(shape,
dtype)` (a `threadgroup`-qualified array, shared by every thread in a
threadgroup) and `metal.barrier()` (`threadgroup_barrier`) are
implemented and tested, including cross-thread visibility after a
barrier and per-threadgroup isolation
(`tests/integration/test_local_and_shared_memory.py`). A real MSL
syntax/runtime-allocation issue was found and fixed along the way: a
`[[threadgroup(n)]]`-attributed kernel parameter must be a pointer, not
a fixed-size array type, and its backing memory must be explicitly
sized per-dispatch via `setThreadgroupMemoryLength:atIndex:` --
omitting either compiles/runs with no error and silently reads back as
zero. `metal.local_array(shape, dtype)` (per-thread-private, not
shared) is also implemented.

**Atomics** -- done. `metal.atomic_add/sub/min/max/exchange` and
`metal.atomic_compare_exchange` are implemented for int32/uint32/float32
device-array elements, verified race-free under real multi-hundred-
thousand-thread contention (`tests/integration/test_atomics.py`).
float32 min/max has no native MSL atomic operation on any Apple GPU
family (a permanent MSL-language limitation, not a device-capability
gap -- verified directly against Apple's Metal compiler) and is lowered
to a compare-and-swap retry loop instead. Building and testing this
also surfaced and fixed two real, silent-wrong-result bugs in `while`
-loop lowering (see the `while` loop entry in Phase 3 below) and one
real MSL-syntax/runtime-allocation gap in threadgroup memory (above).

**More dtypes** (int8/int16/uint8/uint16, verified uint64)
User value: matches more real NumPy array dtypes without a manual cast.
Dependency: extending `numba_metal/types/__init__.py`'s mapping tables;
no architectural change needed.
Risk: low.
Complexity: small.

**More math functions** (the rest of `math.*`: `tan`, `atan2`, `floor`,
`ceil`, `fmod`, etc.)
User value: fewer kernels blocked on an unsupported function.
Dependency: extending `MATH_FUNCS` in `msl_backend.py`; verifying each
against Metal's actual `metal_math` header.
Risk: low.
Complexity: small, incrementally.

**Better multidimensional arrays** (native 2D/3D kernel ARGUMENTS with
real `.shape`, instead of requiring manual flattening -- distinct from
3D launch *grids*, which are done: `metal.grid(3)`/`kernel[(bx,by,bz),
(tx,ty,tz)]` are implemented and tested)
User value: removes the current requirement to flatten and hand-compute
strides (as `heat_diffusion.py`/`pairwise_distance.py` do today).
Dependency: extending the MSL backend's array-parameter handling to carry
shape/stride metadata into generated indexing code.
Risk: medium -- interacts with the 1D-array-only restriction that
currently simplifies buffer binding; needs care to keep argument binding
correct for non-contiguous or multi-dimensional layouts.
Complexity: large.

**Broadcasting**
User value: matches common NumPy usage patterns.
Dependency: multidimensional array support (above).
Risk: medium.
Complexity: large.

**Random-number generation on the GPU**
User value: removes the current CPU-side RNG boundary in
`monte_carlo_paths.py` (documented there and in `docs/limitations.md`),
which is a real bottleneck for large Monte Carlo workloads (RNG-array
generation and host-to-device transfer of the random draws currently
dominates over the actual GPU compute at large path counts).
Dependency: a counter-based GPU-friendly PRNG (e.g. Philox) implemented
directly in generated MSL, with a Numba-side typed intrinsic analogous to
`metal.grid()`.
Risk: medium -- must be reproducible/seedable correctly, and
statistically validated, not just "produces different-looking numbers".
Complexity: large.

## Phase 3: Numba compatibility

**Greater compatibility with Numba CUDA kernel syntax**
User value: lowers the porting cost for existing `numba.cuda` kernels.
Dependency: auditing `numba.cuda`'s public kernel-writing API surface
(`cuda.threadIdx`, `cuda.blockIdx`, `cuda.blockDim`, `cuda.gridDim`, in
addition to `cuda.grid`/`cuda.gridsize` which numba-metal already
mirrors) and adding equivalent intrinsics.
Risk: medium -- some CUDA semantics (warp-level primitives, shared-memory
bank conflicts) have no clean Metal equivalent and would need to be
explicitly marked unsupported rather than silently approximated.
Complexity: medium.

**General `while` loops** (nested `if`/`else`, `break`, or `continue`
inside a `while` body)
User value: removes the current straight-line-only restriction (see
`docs/limitations.md`) -- a `while` loop with any conditional logic in
its body currently raises `UnsupportedFeatureError` rather than risking
the confirmed-possible silent-wrong-result failure mode found while
adding basic `while` support.
Dependency: a genuine rewrite of the control-flow structurer's loop
handling in `numba_metal/compiler/structuring.py`. Numba's bytecode
lowering rotates `while cond: body` into a do-while-shaped CFG (the
"body" block on the natural-exit path is often trivial, with all real
per-iteration work and the next condition test folded into the loop
header itself); the current structurer's `if`/`else` detection was not
designed with this rotated shape in mind and was found, by direct
testing, to mis-lower `break`/`continue` nested inside a conditional
within such a loop (confirmed: swapped/wrong results, not a crash).
Risk: high -- this is exactly the kind of control-flow-shape interaction
bug that is easy to "fix" for one test case while silently breaking
another; would need substantially more differential/property-based test
coverage of generated `while`-loop shapes (extending
`tests/differential/grammar.py`, which currently only generates
`for x in range(...)` loops) before trusting a broader rewrite.
Complexity: large.

**Device functions** -- done for the scalar case. `@metal.device_func`
compiles a scalar-argument, scalar-return helper function to a real,
separate MSL function (not inlined), callable from a kernel or another
device function, with nested calls and full control flow (if/else,
loops) supported. Direct recursion is rejected by Numba's own frontend;
mutual/transitive recursion by numba-metal's own cycle detection. Not
yet done: array-argument device functions (would need buffer-binding
semantics for a non-top-level MSL function -- a real design question,
not yet addressed) and MSL-signature-level deduplication (two call
sites whose Numba-level type tuples differ before numba-metal's
float64->float32 narrowing but are identical after it currently compile
to two separate, functionally-identical MSL functions rather than one
shared one -- an inefficiency, not a correctness issue, see
`docs/limitations.md`).
User value (remaining): array arguments would unlock more code-reuse
patterns (e.g. a helper that reduces over a sub-range of an array).
Dependency: extending the MSL backend's array-parameter/buffer-binding
model to a non-kernel-entry-point function.
Risk: medium.
Complexity: medium.

**Command-buffer batching and scalar-buffer reuse** -- done.
`metal.batch()` encodes multiple launches onto one shared command
buffer (one commit, one tracked `SubmissionRecord`, instead of one per
launch); small scalar-argument/array-size constant buffers are pooled
by byte size and reused across dispatches once `synchronize()` confirms
the previous user's command buffer completed, instead of allocating a
fresh `MTLBuffer` every single launch. Not yet done: automatic/implicit
batching (today `metal.batch()` is an explicit opt-in context manager,
not applied by default), and reuse of the larger per-array device
buffers themselves (only the small scalar/size constant buffers are
pooled; `DeviceNDArray`'s own backing buffer is still allocated once
per `to_device()`/`device_array()` call, which is the correct behavior
since an array's buffer is user-owned for that array's lifetime, not a
per-dispatch temporary).

**Streams and events**
User value: overlapping compute with transfer, concurrent independent
kernel graphs. Distinct from `metal.batch()` (done, above), which
reduces per-launch overhead within the existing single serial queue but
does not provide concurrent/overlapping execution.
Dependency: exposing `MTLCommandQueue`/`MTLEvent` beyond the current
single process-wide serial queue (`numba_metal/runtime/context.py`).
Risk: medium -- concurrency bugs (race conditions in buffer reuse across
overlapping streams) are a real correctness hazard.
Complexity: large.

**Ufunc support / `@vectorize`-equivalent**
User value: matches a common Numba CPU idiom, lets existing
elementwise-function code target the GPU with minimal changes.
Dependency: device-function support (above) plus an auto-generated
launch wrapper.
Risk: low-medium.
Complexity: medium.

**Portable CUDA/Metal source where feasible**
User value: write once, target either CUDA (via `numba-cuda`) or Metal.
Dependency: a shared kernel-writing dialect subset and per-backend
codegen; significant design work to identify what's genuinely portable
(index arithmetic, control flow) versus backend-specific (memory
hierarchy names, warp primitives).
Risk: high -- easy to either over-promise portability or end up with a
dialect so restricted it isn't useful.
Complexity: large.

## Phase 4: Automatic acceleration

**Analysis of `@njit` loops for GPU eligibility**
User value: reduces the manual work of identifying which existing Numba
CPU code could benefit from a GPU port.
Dependency: static analysis over Numba's typed IR (which numba-metal
already knows how to consume) to flag loop shapes matching the supported
subset.
Risk: medium -- false positives (flagging loops that use unsupported
constructs) need to fail gracefully, not silently.
Complexity: medium.

**Automatic parallel-loop extraction**
User value: turns a flagged `@njit(parallel=True)` loop into a numba-metal
kernel without the user hand-writing one.
Dependency: the analysis above, plus a code-generation step from
Numba's `parfor` IR (not the typed-IR shape numba-metal currently
consumes) into a kernel function.
Risk: high -- `parfor` lowering has its own complex IR shape that would
need a dedicated adapter, separate from the existing kernel-focused
frontend.
Complexity: large.

**CPU-versus-GPU cost model**
User value: automatic dispatch decisions instead of the user having to
guess (informed directly by this project's own measured finding that GPU
dispatch overhead makes small workloads *slower* on Metal -- see
`docs/benchmarking.md`).
Dependency: the automatic parallel-loop extraction above, plus empirical
calibration data (problem size vs. measured CPU/GPU crossover point,
which will vary by kernel shape and by GPU core count/family).
Risk: high -- a wrong automatic decision that silently runs something
slower defeats the purpose; must be conservative (opt-in, or clearly
reported) rather than silently substituted, consistent with this
project's no-silent-fallback principle.
Complexity: large.

**Kernel fusion**
User value: avoids materializing intermediate arrays across chained
kernel launches (the motivating example for Benchmark 1 in the MVP, done
manually today by writing one fused kernel by hand).
Dependency: an IR-level fusion pass operating on a sequence of
`@metal.jit` calls; needs data-flow analysis across launches.
Risk: high -- correctness-critical (must preserve exact semantics of
each fused operation) and can interact badly with in-place mutation.
Complexity: large.

**Automatic data-residency management**
User value: automatically keeps arrays GPU-resident across a chain of
operations instead of requiring manual `to_device`/`copy_to_host` calls
(the pattern `heat_diffusion.py` currently does by hand, and explicitly
benchmarks the cost of *not* doing).
Dependency: a tracking layer over `DeviceNDArray` lifetimes and usage
patterns.
Risk: medium -- must not silently keep stale data resident when the host
array it was derived from changes.
Complexity: large.

## Phase 5: Ecosystem and upstreaming

**SciPy / scikit-image integration experiments**
User value: demonstrates numba-metal in a real downstream library
context, not just isolated benchmarks.
Dependency: Phase 2/3 features (multidimensional arrays, device
functions) likely needed first for anything beyond trivial elementwise
ops.
Risk: medium -- these libraries' APIs weren't designed with a
1D-array-only GPU backend in mind.
Complexity: large.

**Research reproducibility packages**
User value: makes it easy for a paper/notebook author to pin an exact
numba-metal + Numba + macOS + hardware combination and reproduce results.
Dependency: the persistent compilation cache (Phase 1) and the JSON
environment-metadata reporting already built into
`benchmarks/common.py`, extended into a general-purpose "environment
manifest" tool.
Risk: low.
Complexity: medium.

**Coordination with Numba maintainers** -- findings written up, not yet
posted: `docs/numba-rfc.md` (paste-ready comment for numba/numba#5706,
informed by actually reading that issue's multi-year discussion
thread) and `docs/upstream-strategy.md` (assesses four paths and
recommends staying external while raising the "frontend-only extension
point" question). Posting the RFC comment itself is the remaining step.
User value: reduces the compatibility-risk items in
`docs/limitations.md` by getting numba-metal's needs (a stable "typed IR
only, no lowering" entry point) considered in Numba's own roadmap.
Dependency: none technical; a relationship/communication effort.
Risk: low technical risk, but outcome-uncertain (maintainer bandwidth
and priorities are outside this project's control; the issue's own
history shows maintainers have consistently said a full Metal target is
unlikely without a contributor driving it).
Complexity: small (from numba-metal's side; the uncertainty is on the
other side of the conversation).

**Small upstream Numba extension-interface PRs**
User value: a stable, public "compile to typed IR only" entry point in
Numba itself would remove the single largest compatibility-risk item in
this project (see `docs/limitations.md`) -- currently numba-metal
depends on internal APIs (`CompilerBase`, `compiler_machinery`,
`typed_passes`) with no stability guarantee.
Dependency: the coordination above; a concrete, minimal PR proposal
(e.g. a documented `compile_to_typed_ir()` public function) informed by
exactly what `numba_metal/compiler/frontend.py` needed to build here.
Risk: medium -- upstream API design is a negotiation, not something this
project controls unilaterally.
Complexity: medium (for the PR itself, once scoped).

**Criteria for official Numba ecosystem recognition**
User value: discoverability and trust signal for users evaluating
GPU-backend options.
Dependency: Numba's own criteria (typically: test coverage, CI, a
maintained release cadence, docs) -- most of which Phase 1 items
directly build toward.
Risk: low technical risk; primarily a matter of sustained maintenance
over time, which is the actual bar such recognition exists to signal.
Complexity: small (procedural), contingent on the substantive work in
earlier phases actually being done and maintained.
