# Architecture

This document describes how numba-metal actually works, based on the
implementation in this repository (not a design proposal). See
`docs/implementation-plan.md` for the investigation record that led to
these decisions.

## Pipeline overview

```mermaid
flowchart TD
    A["Python kernel source\n(@metal.jit function)"] --> B["Numba frontend\nbytecode -> untyped IR\n(numba.core.compiler untyped passes)"]
    B --> C["Numba nopython type inference\n(NopythonTypeInference + AnnotateTypes)"]
    C --> D["Typed Numba IR\n(numba.core.ir.FunctionIR + typemap)"]
    D --> E["numba-metal structuring pass\nCFG -> if/else, for, break/continue"]
    E --> F["numba-metal MSL backend\nstructured tree-walk codegen"]
    F --> G["MSL source text\n(kernel void ...)"]
    G --> H["MTLDevice.newLibraryWithSource\n(Apple Metal shader compiler)"]
    H --> I["MTLComputePipelineState"]
    I --> J["Kernel launch\nkernel[blocks, threads](args)"]
    J --> K["MTLCommandBuffer / ComputeCommandEncoder\nbuffer binding + dispatchThreads"]
    K --> L["GPU execution"]
    L --> M["metal.synchronize() /\nDeviceNDArray.copy_to_host()"]

    subgraph Cache["Compilation cache"]
        N["keyed by: kernel source,\nargument types, device id"]
    end
    E -.-> N
    N -.-> I
```

## Numba frontend integration

numba-metal reuses Numba's real frontend end-to-end: bytecode
disassembly, control-flow reconstruction, and full `nopython`-mode type
inference, exactly as the CPU target does. This happens in
`numba_metal/compiler/frontend.py`, via a custom `CompilerBase` subclass
(`_TypedIROnlyCompiler`) whose pipeline is the standard untyped-IR
pipeline (`DefaultPassBuilder.define_untyped_pipeline`) followed by
`NopythonTypeInference` and `AnnotateTypes` -- and nothing else. No
lowering, no LLVM, no object-mode fallback pass is ever run.

The result is Numba's typed intermediate representation: a
`numba.core.ir.FunctionIR` (a control-flow graph of basic blocks) plus a
`typemap` (`Var name -> Type`) produced entirely by Numba's own type
inference. numba-metal's own code starts only after this point.

### `metal.grid()` / `metal.gridsize()` typing

These are implemented with `numba.extending.intrinsic(prefer_literal=True)`
(`numba_metal/compiler/intrinsics.py`) purely so Numba's type inference
gives them a real, checked signature (an `int32` literal `ndim` argument
mapping to `int64` or `UniTuple(int64, 2)`). Their `codegen` callback --
the half that would normally emit LLVM IR -- is never invoked, because
numba-metal never runs Numba's LLVM lowering. Instead, the MSL backend
pattern-matches `Call` IR nodes whose callee resolves (via a pre-scan of
`Global`/`FreeVar`/`getattr` statements) to these exact function objects,
and emits `thread_position_in_grid`/`threads_per_grid` MSL expressions
directly. This split -- real Numba typing, intercepted lowering -- was
verified empirically against the installed Numba version before being
adopted (see `docs/implementation-plan.md`).

## Typed IR -> MSL lowering

`numba_metal/compiler/msl_backend.py` and
`numba_metal/compiler/structuring.py` implement a structured
tree-walking code generator -- not string substitution on generated text.
It works in two phases:

### 1. Control-flow structuring

Numba's IR is a graph of basic blocks connected by `Branch`/`Jump`
terminators in SSA form (with `phi` nodes at merge points); there is no
"if" or "for" node once the frontend is done. Metal Shading Language, like
C, has no `goto` -- this was verified directly against the Metal compiler
during development, not assumed. So numba-metal reconstructs real
structured control flow from the CFG via a bounded region-based
structurer (`structuring.py`):

- A block ending in `Branch` whose arms reconverge is emitted as
  `if (...) { ... } else { ... }`, with the merge point found via a
  dominance-restricted reachability search.
- A block that is a natural loop header (has a back-edge predecessor
  dominated by the header) is emitted as a loop. The loop's *body region*
  is defined as every block dominated by the loop's body-entry block --
  this dominance-based definition (rather than a naive forward-reachability
  flood-fill) is what correctly distinguishes a `break` target from an
  ordinary post-loop merge block, even when both happen to be reachable
  from inside the loop body (a case the flood-fill approach gets wrong,
  discovered and fixed during this project's own test-driven development).
- If the loop's exit test traces back to a `getiter(range(...))` /
  `iternext` / `pair_first` / `pair_second` chain in the exact shape
  Numba emits for `for x in range(...)`, it is emitted as a native MSL
  `for` loop over the recovered start/stop/step bounds; otherwise it falls
  back to `while (true) { if (exit_cond) break; ... }`.
- Jumps out of the loop's body region are `break`; jumps back to the
  header are `continue`.
- Anything outside these recognized shapes (e.g. a `while` loop with
  side-entry, or genuinely irreducible control flow) raises
  `UnsupportedFeatureError` naming the offending block, rather than
  emitting incorrect MSL.

SSA `phi` nodes are eliminated by a union-find pass
(`_unify_phis`) that assigns one shared MSL variable identifier to a phi
target and all of its incoming values, so ordinary assignment on each
predecessor path already leaves the correct value visible after the
merge -- no MSL is emitted for the phi node itself.

### 2. Statement/expression codegen

Each Numba IR statement (`Assign`, `SetItem`, `StaticSetItem`) and
expression (`binop`, `unary`, `call`, `getitem`, `getattr`, `cast`, ...)
is walked and translated to one MSL statement/expression via a fixed
dispatch table (`numba_metal/compiler/msl_backend.py`). Every operation
not in the supported table raises `UnsupportedFeatureError` naming the
specific IR node, operator, or type -- this is enforced by
`tests/test_unsupported.py`.

Local variables are all declared up front, at function scope (flat, not
block-scoped), from the typemap -- this mirrors how Numba's own SSA form
already treats every `Var` as function-scoped, and sidesteps MSL's
declare-before-use requirement without needing separate scope tracking.

## Deliberate precision decisions

- **float64 narrowing.** Python float literals and `/` true-division
  infer as `float64` under ordinary Numba typing (matching CPython),
  but MSL's `float` is 32-bit and Apple GPU float64 support is not
  verified. numba-metal narrows *local intermediate* float64 values to
  `float32` rather than reject every kernel containing a literal or a
  division; float64 *array dtypes or kernel arguments* remain strictly
  rejected. See `docs/limitations.md` for the numerical consequences,
  discovered concretely while validating the Mandelbrot benchmark (a
  float64 CPU reference and a float32 GPU result diverged near the
  chaotic escape boundary purely from this precision difference, not
  from any codegen bug -- fixed by making the benchmark's CPU reference
  float32 too, for a fair comparison).
- **Fast-math disabled.** `MTLCompileOptions.fastMathEnabled` is
  explicitly set to `False` for all kernel compilation
  (`numba_metal/compiler/pipeline.py`). Metal's default fast-math mode
  permits FMA fusion and reassociation that is invisible for most
  kernels but was observed, during this project's own benchmark
  development, to shift Mandelbrot escape-iteration counts for a
  small fraction of pixels relative to standard (non-fused) float32
  semantics. Disabling it trades a small amount of performance for
  results that match ordinary float32 arithmetic more closely.
- **Integer `//`.** MSL's native integer division truncates toward zero;
  Python's `//` floors toward negative infinity. These agree for
  non-negative operands (the only case exercised by the five benchmark
  kernels, which use `//` for non-negative flattened-index arithmetic)
  and are documented to differ for negative operands.

## Runtime compilation and kernel cache

`numba_metal/compiler/pipeline.py`'s `KernelCache` maps a cache key --
SHA-256 of the kernel's source text (`inspect.getsource`), qualified
name, argument-type tuple, and the Metal device's `registryID` -- to a
`CompiledKernel` (deterministic generated kernel name, MSL source, an
`MTLComputePipelineState`, and metadata needed for argument binding). A
cache hit skips both the Numba typed-IR frontend and Metal shader
compilation entirely. The cache is in-process, in-memory only; there is
no on-disk persistent cache in the MVP (see `docs/roadmap.md`).

Kernel names are deterministic per compilation (`nbmtl_<funcname>_<n>`,
with `n` from a process-wide monotonic counter), so repeated dumps via
`NUMBA_METAL_DUMP_MSL=1` are stable and identifiable.

## Metal device and queue management

`numba_metal/runtime/context.py` holds one process-wide `MTLDevice` and
one `MTLCommandQueue`, created lazily on first use
(`numba_metal.runtime.device.check_capable()` runs the full
platform/device/toolchain validation before anything is created). This is
the one piece of module-level mutable state in the package; it mirrors
how essentially every GPU runtime (CUDA included) treats the device and
queue as process-wide resources, and was chosen over threading a context
object through every public call for no practical benefit in a
single-GPU MVP.

## Buffer ownership and memory model

Device arrays (`numba_metal/runtime/array.py`, `DeviceNDArray`) are
backed by `MTLResourceStorageModeShared` buffers. On Apple silicon's
unified memory architecture, these buffers are directly addressable from
both CPU and GPU -- there is no discrete VRAM to copy across via a
PCIe-style transfer. However, `to_device()`/`copy_to_host()` currently
still perform a host-side `memcpy` into/out of the `MTLBuffer`'s own
backing memory (via `buf.contents().as_buffer(...)`), because a NumPy
array and an `MTLBuffer` are distinct allocations in the current
implementation. This is **not** zero-copy, and is documented honestly as
such rather than described as "unified memory" magic. A true zero-copy
path -- wrapping an existing NumPy allocation's memory directly via
`MTLDevice.newBufferWithBytesNoCopy_length_options_deallocator_` -- is
legal Metal API surface and is listed in `docs/roadmap.md`, deferred from
the MVP to keep buffer lifetime rules simple (Python's own reference
counting owns the `MTLBuffer` for the `DeviceNDArray`'s lifetime; a
no-copy wrapper would need to additionally keep the source NumPy array
alive for as long as the GPU might reference its memory).

## Dispatch and synchronization

`numba_metal/runtime/dispatcher.py`'s `KernelDispatcher` implements
`kernel[blocks, threads](*args)`: it infers each argument's Numba type
(device array dtype -> 1D array type; NumPy/Python scalar -> scalar
type), gets-or-compiles the matching `CompiledKernel` from the cache,
binds buffers (each array argument gets its own buffer slot plus a
hidden trailing `uint` element-count buffer, used to implement `.size`;
each scalar argument gets its own small constant buffer), and encodes a
`dispatchThreads:threadsPerThreadgroup:` call on a fresh command buffer
from the shared queue, then commits it.

`metal.synchronize()` (`context.py`) commits an empty barrier command
buffer and waits on it; because the single command queue executes
command buffers in submission order, waiting on this barrier guarantees
everything submitted before it has completed.

## Error handling

All user-facing failures raise one of the dedicated exception types in
`numba_metal/errors.py` (`UnsupportedPlatformError`,
`MetalToolchainError`, `UnsupportedFeatureError`, `KernelCompilationError`,
`KernelLaunchError`, `MetalRuntimeError`), never a bare `Exception`, and
error messages always name the specific offending construct/dtype/value
rather than a generic failure. `KernelCompilationError` for a Metal
shader-compilation failure includes both the Metal compiler's own
diagnostic text and the full generated MSL source, so a failure is always
debuggable without re-running with `NUMBA_METAL_DUMP_MSL=1`.

## Relevant Numba extension points and private APIs used

| API | Public/private | Why it's needed |
|---|---|---|
| `numba.core.compiler.CompilerBase`, `DefaultPassBuilder` | Semi-public (documented as an extension point, but its exact pass-list shape is compiler-internal) | Build a pipeline that runs the untyped frontend + type inference and stops there |
| `numba.core.compiler_machinery.{PassManager, FunctionPass, register_pass}` | Semi-public | Insert a custom terminal pass (`_CaptureTypedIR`) that copies what's needed out of Numba's internal `state` object |
| `numba.core.typed_passes.{NopythonTypeInference, AnnotateTypes}` | Internal (`numba.core.typed_passes` has no stability guarantee across releases) | The actual type-inference passes reused |
| `numba.core.registry.cpu_target` | Semi-public | Typing/target context source; numba-metal does not register its own Numba target via `numba.core.target_extension` (see "Alternatives considered") |
| `numba.extending.intrinsic(prefer_literal=True)` | Public | Type `metal.grid()`/`gridsize()` |
| `numba.core.ir.*` (`Assign`, `Expr`, `Branch`, etc.) | Public-ish (documented as the IR data model, but full grammar isn't a stability-guaranteed API) | Walking the typed IR |

`numba_metal/compiler/frontend.py` is a small, deliberately isolated
adapter module -- everything above is imported only there and in
`msl_backend.py`/`structuring.py`, not scattered through the codebase, so
a future Numba version's changes are localized to a known surface.

## Alternatives considered

**Register a `"metal"` target via `numba.core.target_extension` and
implement `@lower`-based lowering, like `numba.cuda`'s
`CUDACompiler`/NVVM path.** Rejected for the MVP: that machinery exists
specifically to let multiple backends share Numba's generic
`@overload`/`@lower` registries and ultimately hand *LLVM IR* to
`numba.core.codegen`. Metal Shading Language is text, not LLVM IR, and
Apple's `air64` LLVM dialect is not something llvmlite/Numba know how to
target; reverse-engineering AIR bitcode emission to reuse this path was
judged out of scope for an MVP. Adopting `target_extension` registration
without also using its lowering half would add real version-fragility
(these are internal-ish APIs that move between Numba releases) for no
benefit, since numba-metal's actual backend is a bespoke IR-to-text
walker regardless.

**Emit MSL via string templates / f-string substitution keyed on source
text patterns.** Rejected outright, including as a fallback: this is
explicitly disallowed by the project's requirements (no unconstrained
string-replacement code generation) and would not survive even trivial
kernel variations. The structured tree-walk over typed IR is the only
approach implemented.

**Emit a goto-based MSL translation of the raw CFG.** Investigated and
rejected after directly testing that Metal's shader compiler rejects
`goto`/labeled statements (`"labeled statements are not supported in
Metal"` / `"'goto' is not supported in Metal"`), which is what motivated
building the structured-control-flow reconstruction described above
instead.

**numba/numba#5706.** The task's investigation instructions named this
open feature request as something to review before implementation. It
was not actually reviewed -- no fetch or read of it happened during this
project, despite an earlier draft of this document claiming otherwise;
that claim was false and has been removed. No content from it influenced
anything in this codebase. This should be treated as an open item, not a
completed and negative-result investigation step.

## Why this architecture

The governing constraint is that MSL is a restricted C++14-derived text
language, not an LLVM target -- so any approach that tries to reuse
Numba's LLVM-based lowering infrastructure either doesn't apply (no LLVM
backend for AIR exists) or would require building one (out of scope for
an MVP). Given that, the smallest *honest* pipeline is: reuse everything
in Numba that produces validated, typed IR (frontend + type inference,
which is by far the most complex and highest-value part of "does this
Python code typecheck as a GPU kernel"), and write a small, dedicated,
testable backend for the one thing Numba can't help with -- turning that
IR into MSL text. This kept the amount of Numba-internal-API surface
touched to a minimum (isolated in `frontend.py`) while avoiding any
string-templating code generation.
