# Compiling a Restricted Python Subset to Metal Shading Language: Structured Control-Flow Reconstruction and an Empirical Roofline Study on Apple Silicon

**Jens Schutt**
Independent research and engineering project
September 2026

## Scope statement

This document is written in the structure of a systems/compilers research
paper because that structure is the clearest way to present the technical
contribution, the methodology, and the results honestly. It is **not** a
peer-reviewed publication, and it should not be read as one. It was produced
over a single, long engineering session (see §7 for the exact scope of that
session) by one person working with an AI coding assistant (Claude), not
through a sustained multi-author research program with independent
replication, external code review, or academic peer review. The related-work
citations in §2 were verified to be real, correctly attributed sources at the
time of writing, but the comparison against them is best-effort, not
exhaustive — a proper literature review would likely surface additional prior
art this document does not cover. Every empirical number in §5 and §6 was
measured on exactly one machine (a single Apple M4 Pro) under one software
configuration; no claim in this paper should be read as generalizing to other
Apple Silicon generations, other macOS/Metal versions, or other measurement
methodologies without independent verification. Where results were negative,
wrong, or reverted, that is reported in as much detail as the successes,
because the negative results are, if anything, the more defensible research
contribution of the two: they are falsifiable, they were falsified, and the
falsification is documented.

## Abstract

We present numba-metal, a just-in-time compiler that lowers a restricted
subset of Python — typed via Numba's existing frontend — directly to Metal
Shading Language (MSL) source text, without passing through LLVM at any
stage. This architectural choice is forced, not stylistic: Apple's Metal
compiler toolchain has no publicly documented LLVM IR entry point, and Numba
maintainers have independently and publicly concluded that building an
LLVM-IR-to-Metal backend analogous to `numba.cuda`'s LLVM-to-NVPTX path would
constitute "a research project" in its own right, not an incremental
extension. We instead reconstruct structured control flow (`if`/`else`,
`for`-range loops, a supported subset of `while` loops, `break`/`continue`)
directly from Numba's unstructured static single assignment (SSA) basic-block
control-flow graph (CFG), using a bounded, dominance-based region structurer,
and emit it as MSL text via a tree-walking code generator. We document the
compiler's capability surface precisely (§4), including several boundaries
that are load-bearing correctness decisions rather than oversights — most
notably a nested `if`/`else`-inside-`while` shape where `break` is supported
and proven correct, but `continue` is not, because the two require
irreconcilable code-generation strategies given how Numba's own bytecode
lowering rotates `while` loops (§4.3). We report a full empirical study of
nine benchmark kernels comparing CPU-parallel Numba, single-threaded Numba,
and Metal on identical problem instances, and use one benchmark — a heat
diffusion stencil — as a controlled instrument to directly demonstrate the
roofline model's central prediction on Apple Silicon's unified memory
architecture: an identical memory-access pattern is bandwidth-bound and
*degrades* under 100× problem-size scaling in one arithmetic-intensity
regime (0.60×, i.e. a 40% slowdown relative to parallel CPU), and
compute-bound and *improves* under the same scaling in another (13.66×
speedup), with no change to the kernel's memory-access pattern at all. We
extend this beyond a two-point demonstration with a controlled
arithmetic-intensity sweep (§6.2) that empirically locates the real
crossover point, finding it to be problem-size-dependent and, at the
sizes tested, well below the theoretical ridge point obtained by simply
dividing two independently measured throughput ceilings — ceilings that,
we show, almost certainly originate from the project's own
`numba-metal advisor` tool, a ~4,200-line CLI subpackage for scanning,
profiling, and comparing candidate kernels that we give a full,
evidence-scoped account of for the first time (§6.5). We
also report, in full and without omission, a compiler bug that was found,
partially fixed, verified against 500 randomized test cases and the full
existing test suite, and then found — only once measured against a real,
structurally different production kernel — to produce silently incorrect
results on every one of 1,000 test elements, and was consequently reverted
in its entirety rather than shipped with a caveat (§6.3). We argue this
reversion, not any single benchmark number, is the paper's most important
methodological result.

## 1. Introduction

### 1.1 Motivation

Apple Silicon GPUs are, as of this writing, largely inaccessible to
Python-native numerical and scientific-computing code without either (a)
rewriting the computation into a fixed-op-set array library such as Apple's
own MLX [Hannun et al., 2023] or PyTorch's MPS backend, both of which dispatch
to a closed set of precompiled kernels rather than compiling arbitrary
user-authored kernel *functions*, or (b) writing Metal Shading Language
directly, which requires a different language, a different toolchain, and
forfeits Python's ecosystem entirely. Numba [Lam, Pitrou & Seibert, 2015]
already solves an adjacent problem for NVIDIA hardware via `numba.cuda`, which
type-checks a restricted Python subset with Numba's own frontend and lowers
it through LLVM IR to NVIDIA's NVVM toolchain, ultimately producing PTX. No
equivalent path exists for Metal, and — as we discuss in §2.3 — the reason is
not merely that nobody has built it yet, but that the LLVM-to-Metal path
Numba CUDA exploits for NVIDIA has no public equivalent on Apple's toolchain.

This project asks a narrower, more tractable question: can a *useful*
subset of Python — real control flow, real multi-dimensional array
indexing, atomics, device-callable functions, threadgroup memory — be
compiled directly to MSL text by walking Numba's already-typed intermediate
representation, skipping LLVM entirely? And, separately: on the specific
hardware in question — Apple Silicon's unified-memory architecture, where
CPU and GPU share one memory bandwidth pool rather than each having
dedicated, disjoint bandwidth as on a discrete-GPU system — does moving a
given workload's inner loop to the GPU actually help, and under what
conditions?

### 1.2 Contributions

1. **A working compiler pipeline** (Numba frontend → structured-CFG
   reconstruction → MSL text emission) that handles a real, tested subset of
   Python control flow, typing, and Metal-specific primitives (atomics,
   threadgroup memory, barriers, multi-dimensional arrays, device-callable
   functions), described precisely in §3–4, including a from-scratch
   structured-control-flow reconstruction algorithm since MSL, like C, has no
   `goto` and Numba's IR has no structured control-flow nodes at all.
2. **A precisely bounded capability surface**, where every unsupported
   construct fails at compile time with a specific, named error rather than
   silently miscompiling — a design principle we call *no silent fallback*,
   applied consistently across the type system, the control-flow structurer,
   and the array-indexing subsystem (§4).
3. **A documented, reproduced instance of the roofline model** on real Apple
   Silicon hardware, using one benchmark kernel rewritten in two
   mathematically related but computationally distinct forms (linear vs.
   nonlinear diffusion) to hold the memory-access pattern exactly constant
   while varying arithmetic intensity, and observing the predicted divergence
   directly (§6.1).
4. **A full account of a reverted compiler feature**, including the exact
   mechanism of the bug (de-SSA phi-copy placement sensitivity to a
   structured tree's shape, not merely a block's content), the verification
   protocol that initially passed (500 randomized trials plus the full
   pre-existing test suite), and the specific, more complex real-world
   kernel shape that exposed the failure (§6.3). We present this not as an
   embarrassment to be minimized but as the paper's central methodological
   argument: a fix that passes randomized testing and an existing test suite
   is not thereby proven correct, and shipping it without testing against a
   structurally different, real production kernel would have been a real,
   silent correctness regression.
5. **An empirically measured roofline ridge point**, in place of a purely
   theoretical one: a controlled arithmetic-intensity sweep (§6.2) that
   holds a kernel's memory-access pattern and problem size fixed while
   scaling arithmetic operations per point across a real, multi-point
   range, showing the actual measured crossover is problem-size-dependent
   and, at the sizes tested, sits well below the ~3.09 FLOPs/byte figure
   obtained by simply dividing two independently measured throughput
   ceilings — together with a proper account of the project's own
   `numba-metal advisor` tool (§6.5), the CLI infrastructure those ceiling
   figures and this paper's roofline classifications ultimately depend on.

## 2. Related work

### 2.1 Apple MLX

MLX [ml-explore/mlx, Apple machine-learning research] is an array
programming library, structurally similar in spirit to NumPy, PyTorch, or
JAX, built by Apple's own ML research organization specifically for Apple
Silicon. Its two defining architectural features are lazy evaluation (array
operations build a computation graph and are only materialized on demand,
enabling operator fusion before execution) and a unified-memory-native
design: because Apple Silicon has no separate host and device memory pools,
MLX's array type has no explicit host-to-device transfer API at all — there
is no device boundary to cross.

MLX and numba-metal solve different problems and are not competitors in any
direct sense. MLX provides a fixed, pre-implemented set of operations
(matrix multiplication, convolution, elementwise arithmetic, reductions);
using it means composing those existing operations. numba-metal compiles an
*arbitrary user-authored Python function* — with its own loops, its own
branching, its own array-indexing pattern — into new MSL source code
specific to that function, generated fresh at JIT time. MLX therefore never
needs to solve the problem this paper's compiler is centrally about:
reconstructing structured control flow from an unstructured control-flow
graph, because MLX never compiles a user's control flow in the first place.

### 2.2 PyTorch's MPS backend

PyTorch's Metal Performance Shaders (MPS) backend dispatches ATen operations
to Apple's own precompiled MPS and MPSGraph kernels, encoding the resulting
work through ordinary Metal command buffers and queues, with operator fusion
happening at the ATen-graph level. Structurally, this is the same category
of system as MLX from the perspective of this comparison: it is kernel
*dispatch* to a closed, existing library of GPU kernels, not compilation of
user-authored Python into new kernel code. There is no path in PyTorch's MPS
backend for a user to write an arbitrary Python function with its own
control flow and receive freshly generated Metal code for exactly that
function.

### 2.3 Numba's CUDA backend

`numba.cuda` is the closest prior art to this project in spirit, and the
comparison is worth making precisely because the destination hardware target
differs while the starting point — Numba's own typed intermediate
representation — is identical. Numba's CUDA backend lowers its typed IR
instruction-by-instruction into LLVM IR, which is then compiled through
NVIDIA's NVVM (an LLVM-based PTX-generating toolchain distinct from LLVM's
in-tree NVPTX backend, chosen specifically for better real-device support),
producing PTX that is linked and loaded via the CUDA Driver API at runtime
[Markall, "The Life of a Numba Kernel," RAPIDS AI, and LLVM's own NVPTX
backend documentation].

This project's backend never touches LLVM. There is no LLVM-based, publicly
documented path from an arbitrary compiler's intermediate representation to
Apple's GPU instruction set (AIR, Apple's internal LLVM-derived IR, has no
public backend target). numba-metal instead walks Numba's *typed* IR (post
type-inference, pre-LLVM-lowering) directly, and — because that IR is an
unstructured graph of basic blocks connected by branches and jumps, with no
notion of a structured `if` or `while` at all — reconstructs the structured
control-flow tree itself before ever emitting a line of MSL text (§3). This
is architecturally the more constrained and more fragile path of the two:
LLVM IR is a mature, general-purpose compiler intermediate representation
with decades of correctness tooling behind it, whereas this project's
control-flow structurer is bespoke, project-specific code, and — as §6.3
documents in detail — bespoke control-flow-reconstruction code is exactly
where this project's one serious, reverted correctness bug originated.

### 2.4 Triton

Triton [Tillet, 2021] is a Python-embedded domain-specific language for GPU
kernel authoring, originally targeting CUDA and, since a 2022 rewrite, built
on MLIR with a multi-stage lowering pipeline (Triton IR → Triton GPU IR →
LLVM IR → PTX or AMDGCN). Triton's programming model is block- or
tile-based: a single kernel invocation describes the computation for an
entire block of data, and the compiler is responsible for scheduling
individual hardware threads and memory accesses within that block. This is a
fundamentally different abstraction from both `numba.cuda` and numba-metal,
which are thread-per-element models: one kernel invocation corresponds to
one hardware thread's scalar or array-indexed slice of work, with explicit
per-thread indexing (`cuda.grid()` / `metal.grid()`).

As of this writing, no production Metal backend exists for Triton itself.
Community reports document that Triton can be force-compiled on Apple
Silicon but silently falls back to CPU execution with no real Apple-GPU code
generation — a genuine, documented gap in the existing tooling landscape
that this project's contribution sits inside, though we make no claim that
numba-metal is a substitute for a hypothetical real Triton-Metal backend,
which would bring an entirely different (and, for many workloads, more
scalable) programming model. A separate project, TileLang, has added real
Metal device support using Triton-adjacent tile abstractions, but it is not
Triton itself and was not evaluated as part of this work.

### 2.5 Broader compiler literature

Several established lines of compiler research bear directly on the design
choices in this project, though none of them target Metal specifically:
Halide [Ragan-Kelley et al., PLDI 2013] established that a restricted,
domain-specific language for image-processing pipelines — stencils being a
central example, directly relevant to this project's own heat-diffusion
benchmark — can achieve portable, near-optimal performance across CPU and
GPU targets by separating a pipeline's algorithm from its schedule. TVM
[Chen et al., OSDI 2018] generalizes the same "compile a high-level tensor
program to diverse hardware backends" goal to deep-learning workloads
specifically. JAX [Bradbury et al., 2018; Frostig, Johnson & Leary, SysML
2018] takes a distinct approach relevant by contrast: rather than directly
compiling a Python AST or bytecode-derived IR (as Numba, and by extension
this project, do), JAX traces a Python function's execution to build an
intermediate representation and compiles that trace via XLA. This
trace-then-compile strategy sidesteps control-flow reconstruction almost
entirely for straight-line and structured-Python-control-flow code, at the
cost of requiring the user's Python control flow to be expressible as, or
rewritten into, JAX's own functional control-flow primitives (`lax.cond`,
`lax.while_loop`) for anything data-dependent — a genuinely different
tradeoff from this project's approach of accepting ordinary Python `if`,
`while`, and `for` syntax directly and reconstructing its structure after
the fact.

### 2.6 Apple Silicon unified memory and the roofline model

The empirical core of this paper (§6.1) depends on Apple Silicon's unified
memory architecture, in which the CPU and GPU are not connected to separate,
independently-sized memory pools (as on a typical discrete-GPU PC or a
datacenter accelerator such as an NVIDIA A100, with >1550 GB/s of dedicated
device memory bandwidth) but instead share one physical memory system and
its bandwidth. An independent, peer-reviewable measurement of this — not
performed by this project — is available in Hübner, Hu, Peng & Markidis
(KTH Royal Institute of Technology), "Apple vs. Oranges: Evaluating the
Apple Silicon M-Series SoCs for HPC Performance and Efficiency" (arXiv,
2025), which directly measures M-series CPU and GPU memory bandwidth
(reporting figures in the 100–103 GB/s range for the M4 generation, achieving
roughly 85% of theoretical peak) and states explicitly that Apple Silicon
chips remain bandwidth-bound for many traditional HPC workloads relative to
discrete accelerators. We cite this work as independent corroboration of the
general architectural claim underlying our own roofline framing; we make no
claim that it replicates, validates, or was in any way connected to our own
heat-diffusion measurement, which is a separate, independent empirical
result produced entirely within this project.

## 3. Compiler architecture

### 3.1 Pipeline overview

The compiler consists of five stages, summarized in Table 1, moving a
Python function decorated with `@metal.jit` from source to a compiled Metal
compute pipeline object.

**Table 1: Compiler pipeline stages**

| Stage | Module | Approx. LOC | Responsibility |
|---|---|---:|---|
| Frontend | `compiler/frontend.py` | 173 | Runs Numba's real bytecode-to-IR translation and full `NopythonTypeInference`/`AnnotateTypes` passes through a custom `CompilerBase` subclass, halting before LLVM lowering. Produces a `TypedKernelIR` object (typed `func_ir`, `typemap`, `calltypes`, argument types/names, return type) that is entirely independent of Numba from this point forward. |
| Structuring | `compiler/structuring.py` | 928 | Reconstructs `if`/`else`, `for`-range, supported `while` shapes, and `break`/`continue` from the unstructured SSA basic-block CFG, via a bounded, dominance-based region structurer (§3.2). |
| De-SSA | `compiler/dessa.py` | 349 | An isolated phi-elimination pass, computing edge-correct parallel-copy assignments at CFG merge points. This module replaced an earlier union-find/aliasing scheme after that scheme was recognized as theoretically unsound per the SSA-destruction literature (Briggs et al., 1998; Boissinot et al., 2009, on sequentializing simultaneous parallel copies, e.g. the loop-carried swap pattern `a, b = b, a`), even though no live miscompilation had been demonstrated from the earlier scheme at the time it was replaced. |
| Backend | `compiler/msl_backend.py` | 2,354 | A structured tree-walking code generator — explicitly not string-substitution-based codegen — that emits MSL statement-by-statement by walking the structured tree produced by the structuring stage against the typemap produced by the frontend. |
| Pipeline / dispatch | `compiler/pipeline.py`, `runtime/dispatcher.py` | 197 / 476 | Ties the frontend and backend together behind a compilation cache keyed on kernel source digest, argument types, and device identity; implements CUDA-style launch syntax (`kernel[blocks, threads](*args)`), argument binding, and Metal command encoding. |

The full `src/` tree totals approximately 12,675 lines of Python.

### 3.2 Why structured-CFG reconstruction, and not LLVM

This design was not the first considered. Two alternatives were evaluated
and rejected during the project's design phase:

1. **Registering as a Numba target via `numba.core.target_extension`**,
   reusing Numba's existing `@lower`-based LLVM lowering path exactly as
   `numba.cuda` does for NVIDIA. This was rejected because Apple's AIR
   (Apple Intermediate Representation, an internal LLVM-derived dialect used
   by Metal's shader compiler) has no publicly documented LLVM backend
   target that `llvmlite` or Numba could emit code for. This is not a
   theoretical concern: Numba's own maintainers, in a long-running public
   GitHub issue (numba/numba#5706, active 2020–2025), independently reached
   the same conclusion — that building an LLVM-IR-to-Metal backend would
   constitute "a research project" in its own right, not an incremental
   backend addition. This project's architecture was designed before that
   issue thread was read in full; the issue is cited here as independent
   corroboration of the constraint, not as its origin.
2. **String-template MSL emission and goto-based control-flow translation.**
   Metal Shading Language, like C, has no `goto` and no labeled statements
   (verified directly against Apple's own Metal compiler, which rejects
   labeled statements outright). A goto-preserving translation of Numba's
   unstructured CFG is therefore not an option regardless of implementation
   convenience, and string-template-based code generation was rejected as a
   project requirement independent of this constraint, on the grounds that
   it conflates code generation with string manipulation in a way that
   makes correctness difficult to reason about or test compositionally.

The structuring stage that resulted (`compiler/structuring.py`) implements a
bounded region structurer: a run of blocks with no incoming edges other than
ordinary fallthrough is a linear sequence; a block ending in a two-way branch
whose successors reconverge at a common, dominance-provable merge point is
emitted as `if`/`else`; a block that is the target of a genuine back-edge (a
predecessor it dominates) is a loop header. Only the specific CFG shapes
Numba's own bytecode lowering actually produces for `if`, `for x in
range(...)`, and a bounded subset of `while` are recognized; every other
shape raises a specific, named compile-time error rather than emitting
speculative or partially-correct MSL (§4.1 discusses this "no silent
fallback" policy in more detail, and §6.3 documents a case where a
structuring-stage bug slipped past this policy's testing regime, despite the
policy itself, and was caught only by testing against a real production
kernel shape).

### 3.3 A representative example: multi-dimensional array support

A concrete illustration of how the frontend, structuring, and backend stages
compose: native 2D and 3D array arguments (`arr[x, y]`, `arr[x, y, z]`)
required no change to the structuring stage at all, since array indexing is
an expression-level concern, not a control-flow concern. The backend was
extended to synthesize, for each multi-dimensional array-typed kernel
parameter, one additional scalar buffer parameter per trailing dimension
(`arg_name_dim1`, `arg_name_dim2`, ...), threaded through the dispatcher's
own argument-binding logic at the exact same buffer indices on both the
Python-launch side and the generated-MSL-signature side. A 2D index
`arr[x, y]` then lowers to the row-major flat-offset expression
`x * arg_arr_dim1 + y` at the point of use. This same mechanism was later
extended, without further structural change, to `@metal.device_func`
array arguments, including forwarding the synthesized dimension parameters
correctly through *transitive* device-function-to-device-function calls —
verified by a dedicated regression test constructing exactly that call
shape.

## 4. Capability surface and correctness policy

### 4.1 "No silent fallback"

Every construct this compiler does not support fails at compile time with a
specific `UnsupportedFeatureError` naming the exact unsupported construct,
rather than silently emitting speculative, partially-correct, or
undefined-behavior-inducing MSL. This is applied uniformly across the type
system (e.g., a literal negative array index such as `a[-1]` is rejected at
compile time, since this backend implements no runtime bounds-checking or
wraparound machinery at all — a real, permanent limitation distinct from
Numba's own CPU backend, which does implement Python-compatible negative
indexing), the control-flow structurer (any CFG shape not among the small,
enumerated set the structurer recognizes is rejected by name, never
silently mis-lowered), and the array-indexing subsystem (arrays with more
than three dimensions are rejected by name at compile time rather than
silently truncated or misinterpreted).

### 4.2 Capability matrix

Table 2 summarizes the compiler's tested and documented capability surface
as of this writing.

**Table 2: Capability surface**

| Area | Status | Constraint |
|---|---|---|
| Scalar arithmetic, comparison, boolean, ternary operators | Supported | `//` uses MSL's native truncating (not Python's floor) division for negative operands |
| `if`/`if-else` | Supported | Arbitrarily nested |
| `for x in range(...)` | Supported | 1/2/3-argument form; compiles to a native MSL `for` |
| `break`/`continue` in `for` loops | Supported | |
| `while` loops | Partially supported | See §4.3 |
| Recursion | Unsupported | Rejected at compile time (direct: by Numba's own frontend; mutual/transitive between device functions: by this project's own in-progress-compilation cycle detection), not as a runtime stack overflow |
| Exceptions, classes, strings, containers, f-strings, `print()` | Unsupported | |
| `@metal.device_func` | Supported | Scalar and 1D/2D/3D array arguments (with synthesized dimension parameters forwarded transitively); scalar-only return type; no local/threadgroup-memory arguments or in-body allocation |
| `float32`/`int32`/`uint32`/`bool`/`float16`/`int64` | Supported | |
| `uint64` | Partially supported | Type mapping exists; not exercised by any real kernel in this project's own test suite |
| `float64` (array dtype or kernel argument) | Unsupported | Unverified on any Apple GPU family; rejected explicitly |
| `float64` (local intermediate value) | Silently narrowed to `float32` | A deliberate, documented precision decision, not an unnoticed gap — discovered concretely via a Mandelbrot correctness investigation (chaotic-recurrence float32/float64 divergence) |
| `int8`/`int16`/`uint8`/`uint16` | Unsupported | Absent from the type-mapping tables |
| 1D/2D/3D arrays (kernel and device-function arguments) | Supported | 4D and beyond rejected by name at compile time |
| `.shape` attribute | Unsupported | Only `.size` (total flattened element count) is available |
| `metal.local_array`/`metal.shared_array` | Supported | Fixed, compile-time-constant shape |
| Atomics (add/sub/min/max/exchange/compare-exchange) | Supported | float32 min/max has no native MSL atomic operation on any Apple GPU family (a permanent language-level limitation, confirmed directly against Apple's compiler) and is lowered to a compare-and-swap retry loop, verified race-free under real multi-hundred-thousand-thread contention |
| `metal.reduce_sum`/`reduce_min`/`reduce_max` | Supported | float32/int32/uint32, 1D arrays only; `argmax` not implemented |
| `metal.batch()` (command-buffer fusion) | Supported | Measured directly for this paper (not previously documented elsewhere in the project): fusing 10/50/200 launches of a small kernel onto one command buffer reduced total wall time by 1.29×/1.57×/1.81× respectively relative to one command buffer per launch, growing with launch count as expected, since batching amortizes a fixed per-commit cost |
| Zero-copy host↔device transfer | Unsupported | A real host-side memcpy occurs on every transfer despite the underlying unified-memory architecture; documented explicitly rather than described as free |
| `int64` output device arrays | Unverified / known-broken for at least one kernel shape | A reproducible Metal-driver-level shader-compiler crash for a specific nested-`for`-loop kernel shape with an `int64` output array; not yet root-caused |

### 4.3 The `while`-loop boundary: `break` supported, `continue` is not, and why they are not symmetric

The single most instructive boundary in this compiler's capability surface
concerns `while` loops whose body opens with an `if`/`else` — the shape
Numba's bytecode lowering produces when the loop's condition-test statements
are not the first statements executed in the loop body (common in, for
example, Newton-Raphson-style early-exit iteration). Numba's rotation of
such a loop places the loop-carried state's phi nodes in one basic block
(the loop's real header, the target of the back-edge) while the actual
condition re-test ends up in a different, later block reached via whatever
control flow — here, the `if`/`else` — makes up the rest of the per-iteration
work. This project's structurer detects this shape (a `RotatedWhileNode`,
distinguished from an ordinary loop header purely by whether a genuine
back-edge target's own terminator is a condition test or an arbitrary
branch) and handles it by structuring the region between the header and the
real condition-test block using the same if/else/break/continue machinery
already proven correct for `for`-range loop bodies.

`break` nested inside that `if`/`else` is supported, and the reasoning for
why it required a real, non-trivial fix — not merely a testing gap — is
detailed in §6.3. `continue` nested in the identical syntactic position is
*not* supported, and the reason is architecturally deeper than "not yet
implemented": Numba's lowering routes a `continue` in this position through
a *separate* guard block that re-tests the original loop condition (because
a `continue` that immediately fails the condition must exit the loop rather
than re-enter its body), and that guard block itself becomes, from this
compiler's structurer's perspective, what looks like a second, outer loop
header wrapping the original rotated-while region — even though it is, in
the source program, the same logical loop. The guard block performs real
work of its own (a genuine SSA phi merge of values arriving from both the
loop's initial entry and the `continue` path, plus the condition
recomputation), which is incompatible with this compiler's two existing
loop-emission strategies: the generic (non-for-range) `while` path emits a
guarded `do`/`while` construct that re-executes its header's own statements
once per pass (`LoopNode`'s `pre_test` mechanism), while the rotated-while
path emits a `while (true) { ... }` construct with the condition test
already inlined into the body as an ordinary `if`/`break` (`RotatedWhileNode`).
Reconciling these into a correct third construct for the specific case of
"a rotated-while nested inside its own continue-guard" is, based on direct
investigation, a materially larger undertaking than the `break` fix, and was
scoped out of this project after that investigation rather than attempted
under time pressure. The construct remains rejected at compile time with a
specific error, per §4.1's policy, rather than silently mis-lowered.

## 5. Benchmark methodology

Nine kernels are benchmarked (Table 3), each compared across up to four
execution modes — plain (single-threaded) Python where tractable, NumPy
(vectorized), Numba CPU (`@njit`, both `parallel=True` and single-threaded),
and this project's Metal backend — at multiple problem sizes chosen to span
from GPU-dispatch-overhead-dominated (small) to compute- or
bandwidth-dominated (large). All reported speedups are relative to **parallel
Numba CPU**, not single-threaded Python, since single-threaded Python is not
a serious baseline for any numerical workload a practitioner would actually
consider moving to a GPU. Every benchmark's correctness is verified on every
run: Metal's result is checked against the CPU reference within an explicit,
documented floating-point tolerance before any timing number is reported,
and a run in which correctness fails is reported as a failure regardless of
its timing. All measurements in this paper were taken on a single machine:
an Apple M4 Pro (Apple Silicon), macOS 26.5.1, Python 3.13.5, Numba 0.67.0.
Run-to-run timing variance is real and is disclosed explicitly wherever
observed (§6.1 in particular), rather than reported as a single point
estimate presented as more precise than the measurement process actually
supports.

**Table 3: Benchmark kernels**

| # | Kernel | What it computes |
|---|---|---|
| 1 | Vector polynomial | `out[i] = ((1.75a[i] + 0.25b[i])² − a[i]b[i]) / (|b[i]| + 1)`, applied independently to every array element |
| 2 | Mandelbrot | Escape-time fractal iteration (`z ← z² + c`, up to 100 iterations, divergent per-pixel early exit) |
| 3 | Heat diffusion (nonlinear) | Perona–Malik anisotropic diffusion over a 2D grid, 200 explicit-scheme iterations (§6.1) |
| 4 | Monte Carlo paths | European call option pricing via simulated geometric Brownian motion, 100 timesteps per path |
| 5 | Pairwise distance | Squared Euclidean distance between every pair of points across two point sets |
| 6 | Cyclist aerodynamics (`sweep`) | One instantaneous aerodynamic-drag calculation per rider configuration |
| 7 | Cyclist aerodynamics (`course_energy`) | The same drag model integrated over a 200-segment simulated course (200× the arithmetic of `sweep` per output value, identical memory footprint) |
| 8 | Implied volatility | Newton-Raphson solve for the volatility implied by an option's market price; a real-world example of the `while`+`if`/`else` shape in §4.3 |
| 9 | Asian option pricing | Monte Carlo pricing of an average-price option, 500 timesteps per path with a running-average payoff accumulated every step |

A tenth measurement, the device-function compile-time cache (§6.4), is
compiler infrastructure rather than a numerical workload and is reported
separately.

## 6. Results

### 6.1 The roofline model, demonstrated directly

The heat-diffusion benchmark was used as a deliberately controlled
instrument to test the roofline model's central prediction on Apple
Silicon's unified-memory architecture: since the CPU and GPU share one
memory bandwidth pool rather than each having independent, dedicated
bandwidth, a kernel whose performance is limited by memory bandwidth (not
by computation) should see **no benefit**, and potentially a *disadvantage*,
from running on the GPU, regardless of problem size — because the GPU brings
more compute throughput to a problem that was never bottlenecked on compute
in the first place.

The benchmark's original form computed simple linear diffusion — a
Jacobi-style 4-neighbor stencil with a single constant averaging weight per
neighbor (`next[x,y] = 0.25 · Σ neighbors`) — using an identical, fairly
converted 2D memory-access pattern on both the CPU and GPU sides (both sides
were rewritten together specifically to avoid a documented earlier
methodological error, discussed further below). At the benchmark's standard
sizes, this kernel measured 0.72×, 0.91×, and 1.00× (parity) relative to
parallel Numba CPU at grid sizes of 128², 512², and 1024² respectively —
already a strong hint of bandwidth-bound behavior, since a compute-bound
kernel would be expected to show a clearer GPU advantage at these sizes
(compare Mandelbrot's 8.62×–37.42× at comparable or smaller problem sizes,
Table 4). To test the roofline prediction directly rather than merely
inferring it from a flat trend, the same kernel was measured at 10× and
100× its largest standard grid's point count (3238² and 10240² respectively,
holding the memory-access pattern, iteration count, and methodology
otherwise identical). The result was not merely "no improvement" but active
*degradation*: 0.52× and 0.60× respectively — the GPU implementation became
comparatively *slower*, relative to CPU, as the problem grew, the opposite of
the scaling behavior every compute-bound benchmark in this suite exhibits
(Table 4).

The kernel was then rewritten to solve nonlinear (Perona–Malik) anisotropic
diffusion instead of linear diffusion — the standard nonlinear diffusion
model from the image-processing literature [Perona & Malik, IEEE PAMI,
1990], in which the diffusion coefficient at each point depends on the
local gradient magnitude via `exp(−(gradient/κ)²)`, rather than being a
constant everywhere. This changes the model's real-world meaning (nonlinear
diffusion resists blurring across sharp gradients, the basis of
edge-preserving image smoothing) but was chosen here specifically because it
preserves the *exact same* 4-neighbor memory-access pattern as the linear
version — the same four array reads per grid point, the same output-array
write pattern — while replacing one constant multiply per neighbor with a
transcendental function evaluation (`math.exp`), a real, substantial,
physically motivated increase in arithmetic operations per byte of memory
traffic. Holding the access pattern fixed while deliberately varying
arithmetic intensity is precisely the controlled-variable structure needed
to isolate the roofline model's prediction from confounds such as launch
configuration or array layout, both of which are already known
confounders in this exact benchmark's history (see the discussion of the
original 2D-indexing investigation below).

At the same problem sizes, correctness verified identically at every size
against both a NumPy-vectorized and a scalar Numba-CPU reference, the
nonlinear kernel measured 0.89×, 3.32×, and 7.15× relative to parallel Numba
CPU at 128², 512², and 1024² (with a disclosed run-to-run range of
approximately 4.6×–7.2× observed at 1024² across repeated executions — we
report this range rather than a single favorable point estimate). Extended
to the same 10× and 100× grid-point-count sizes used for the linear
version, the nonlinear kernel measured **13.05× and 13.66×** — improving,
not degrading, with scale, the precise inverse of the linear kernel's
behavior at the identical sizes (Figure 1, described in prose since this
document does not embed rendered figures: linear diffusion's speedup curve
is monotonically decreasing past 1024² down to 0.52×–0.60×; nonlinear
diffusion's speedup curve is monotonically increasing across the same size
range, from 0.89× up to 13.66×). No other variable was changed between the
two kernels' 10×/100× measurements. This is, to our knowledge, the cleanest
within-project demonstration available of the roofline model's core
claim — that arithmetic intensity, not problem size, determines which side
of the memory-bandwidth/compute-throughput boundary a kernel falls on —
because it holds the access pattern exactly fixed and varies only the
arithmetic.

**Table 4: Selected speedups vs. parallel Numba CPU, by kernel and problem size**

| Kernel | Smallest size | Largest standard size | 10× scale | 100× scale |
|---|---:|---:|---:|---:|
| Vector polynomial | 0.65× | 0.53× | — | — |
| Mandelbrot | 8.62× | 37.42× | — | — |
| Heat diffusion, **linear** (historical) | 0.72× | 1.00× | 0.52× | 0.60× |
| Heat diffusion, **nonlinear** (current) | 0.89× | 7.15× | 13.05× | 13.66× |
| Monte Carlo paths | 1.15× | 4.41× | — | — |
| Pairwise distance | 0.53× | 2.32× | — | — |
| Cyclist aero, `sweep` | 0.45×–1.07× (range across sizes) | | — | — |
| Cyclist aero, `course_energy` | 0.76× | 4.86× | — | — |
| Implied volatility | 1.14× | 4.5× | — | — |
| Asian option pricing | 10.01× | 11.46× | — | — |

We note one further, prior methodological finding directly relevant to
Table 4's interpretation: an earlier investigation of this same
heat-diffusion kernel converted *only* the Metal implementation from a
flattened 1D thread-index scheme (manually computing `x = i // n`,
`y = i % n`) to a genuine 2D-indexed launch, while leaving the CPU reference
on its original flattened form. This measured an apparent ~5× improvement
and was, at the time, reported as a clear GPU win. A follow-up investigation
found this to be an unfair, one-sided comparison: the flattened-indexing
scheme independently degraded Numba's own CPU code generation by roughly
1.6×–2× in isolation, at the same problem sizes, for reasons unrelated to
the GPU at all. Once both implementations were converted to 2D indexing
together, the honest comparison at the time (using the linear-diffusion
model, before the nonlinear rewrite) was roughly parity, not a win. This
project's later benchmarks (Mandelbrot, pairwise distance) that also
involved a 1D-to-2D-indexing conversion were, as a direct consequence of
this earlier finding, converted on both the CPU and GPU sides simultaneously
from the outset, specifically to avoid repeating this exact category of
mistake.

### 6.2 Locating the ridge point: a controlled arithmetic-intensity sweep

Section 6.1's linear-vs-nonlinear comparison is a two-point demonstration:
it shows *that* arithmetic intensity determines which side of the roofline
a kernel falls on, using exactly two measured FLOPs/byte values. It does
not, by itself, say *where* the crossover is, nor whether that crossover
is a single fixed number or a function of problem size. Separately,
`docs/performance-guidance.md` states a ridge point of **~3.09 FLOPs/byte**,
derived by dividing two independently-measured single-point ceilings
(~683 GFLOPS float32 compute-bound ceiling, divided by ~221 GB/s
bandwidth-bound ceiling, divided by 4 bytes per float32). That number is a
theoretical construction from two peak-throughput microbenchmarks, not a
measurement of where any real, generated kernel actually crosses over —
and prior to this section, no benchmark or test anywhere in this project
varied arithmetic intensity across more than two discrete points to check
whether the theoretical ridge point matches real kernel behavior.

To close that gap, `benchmarks/arithmetic_intensity_sweep.py` holds the
heat-diffusion stencil's memory-access pattern *and* problem size fixed
within each run (4 float32 reads + 1 float32 write per point, 20
bytes/point, identical to Section 6.1's kernels) and varies only a `REPS`
parameter that repeats a data-dependent `math.exp`-based coefficient
computation `REPS` times per neighbor before combining — a sequential
dependency chain (each repeat consumes the previous repeat's output),
deliberately structured so neither Numba's LLVM backend nor
numba-metal's MSL lowering can hoist or eliminate the repeated work as
loop-invariant. An initial version of this script used a
loop-invariant computation instead and was silently optimized away by
both compilers to identical wall-clock time at every `REPS` value — a
real methodological trap, caught only by checking that the computed
output actually changed with `REPS` before trusting any timing from it.
An additional `REPS = -1` mode reproduces the *original*,
pre-Perona-Malik linear-diffusion kernel exactly (`0.25 × sum of the 4
raw neighbor values`, ~0.2 FLOPs/byte), giving a validated low-end
anchor: this mode's measured speedups (0.79×, 0.94×, 1.05× at 128²,
512², 1024², 200 iterations, GPU-resident buffers) reproduce Table 4's
independently-measured linear-diffusion row (0.72×, 0.91×, 1.00×) to
within normal run-to-run variance, confirming the sweep harness matches
the rest of this paper's own measurement methodology rather than
introducing a second, incompatible one. (One methodological detail
mattered enough to be worth naming: `heat_diffusion.py`'s own
`_metal_resident` helper rebuilds the `@metal.jit` kernel closure on
every call rather than reusing one built outside the timed loop: doing
so costs a real, repeatable ~2× wall-clock penalty at 1024² — 21ms vs.
~11ms per 200-iteration run — evidently because numba-metal's
`KernelCache` does not treat two structurally-identical closures built
from separate Python function objects as the same cache entry. The
sweep script deliberately matches this rebuild-per-call convention so
its numbers stay comparable to Table 4, but this is itself a real,
previously-undocumented cache-key characteristic worth flagging for
anyone else timing numba-metal kernels this way.)

**Table 4a: Empirically measured speedup vs. FLOPs/byte, by grid size**

| FLOPs/byte | 128² | 512² | 1024² |
|---:|---:|---:|---:|
| 0.20 (linear floor) | 0.79× | 0.94× | 1.05× |
| 0.65 | 0.51× | 0.75× | 1.02× |
| 1.65 | 0.74× | 2.74× | 6.18× |
| 2.65 | 0.87× | 3.46× | 6.33× |
| 4.65 | 1.08× | 4.37× | 6.58× |
| 8.65 | 1.64× | 5.24× | 7.20× |
| **Interpolated parity crossover** | **~3.72** | **~0.73** | **between 0.65 and 1.65** |

The headline finding is that the ridge point is not one number: it
depends on problem size, and at the sizes actually exercised by this
benchmark suite, the real crossover sits **well below** the theoretical
3.09 FLOPs/byte figure for anything above a few hundred grid points per
side. At 512² and 1024², parity is reached somewhere under 1 FLOP/byte —
roughly a third to a tenth of the theoretically-derived ridge point — and
by 1.65 FLOPs/byte (four `exp`-based coefficient evaluations per neighbor)
Metal is already winning by 6×–6.3× at those sizes. Only at 128² does the
empirical crossover (~3.72 FLOPs/byte) land close to the theoretical
figure, and at that size dispatch-and-synchronization overhead is large
enough relative to the total work that Metal never clearly wins across
the measured range at all — consistent with this paper's Section 2.6
discussion of per-launch overhead as the other lever besides bandwidth
that a coarse ridge-point number does not capture.

**What this means as practical guidance**, stated with the same
directness as `docs/performance-guidance.md`'s existing rules of thumb:
for a stencil-shaped kernel on this class of Apple Silicon GPU, treat
"a handful of transcendental-function-equivalent operations per neighbor,
at problem sizes of a few hundred points per side or larger" as
comfortably inside the compute-bound regime — the practical floor is
lower than the theoretical ridge point would suggest. Below roughly
one FLOP per byte moved, and especially at small (≲128²) problem sizes,
treat the kernel as bandwidth- or overhead-bound and expect parity or a
Metal loss, matching Section 6.1's own linear-diffusion result. The gray
zone between these — where the answer depends on both problem size and
exact arithmetic intensity, and a single ridge-point number is not
precise enough to predict it — is real and should be measured directly
with `numba-metal advisor compare` (Section 6.5) rather than assumed
from the theoretical figure alone.

This sweep covers one kernel family (a 4-neighbor 2D stencil) at three
problem sizes and one iteration count; it is not a claim that 0.73
FLOPs/byte is *the* crossover for numba-metal in general, only that *a*
real, measured crossover for this kernel family is well below the
theoretically-derived figure, and that the theoretical figure alone
should not be treated as a reliable predictor of real kernel behavior
without checking it this way. Reproducing or extending this sweep to
other kernel shapes (dense matrix-style access patterns, gather/scatter,
reductions) is listed as future work in Section 7.

### 6.3 A reverted compiler fix: when passing tests is not enough

Section 4.3 describes the current, shipped boundary of `while`-loop support:
`break` nested inside a rotated-while's own `if`/`else` is supported;
`continue` in the identical position is not. Reaching that final, shipped
state required first attempting, verifying, and then fully reverting a
different fix, and we report the full arc here because we consider the
reversion itself to be a more important result than either the `break` fix
that was kept or the `continue` limitation that remains.

Investigation of the rotated-while structuring logic (§3.2, §4.3) surfaced a
real, confirmed bug independent of the break/continue question: when a
nested `if`/`else`'s own branch-merge-point search found no internal
reconvergence point for its two arms (a legitimate outcome when one arm
exits the loop via `break`), the structurer would recurse into the region
the *enclosing* loop-structuring call had already committed to structuring
separately as its own continuation — producing a structurally duplicated
copy of the loop's remaining body, nested inside the `if`. A fix was
implemented: the structurer's branch-handling routine was given the
enclosing call's own "stop here" boundary, and, when its own merge search
failed but that boundary was reachable from one of its arms, adopted the
boundary as the merge point rather than recursing past it. The fix was
verified against 500 independently randomized test cases (varied array
contents, varied trip counts, varied break-triggering conditions) with zero
mismatches against a Python reference implementation, and the project's
full pre-existing test suite (332 tests) continued to pass without
modification.

Both verification methods passed. The fix was not shipped on that basis.
Before committing it, it was additionally tested against
`benchmarks/implied_volatility.py`'s actual, real-world Newton-Raphson
kernel — not a synthetic test case, but the specific production kernel this
capability was intended to eventually support — rewritten to use `break`
instead of its original boolean-flag-based early-exit pattern. That kernel's
real exit condition is a boolean `or` of two independent conditions
(`abs_diff < tol or vega < 1e-6`), which Numba's bytecode lowering compiles
as a *shared-branch-target* pattern: two separate two-way branches that both
lead to the same `break` target block, rather than the simple single-branch
diamond every randomized test case up to that point had exercised. Run
against this kernel, at this shape, the fix produced **silently incorrect
results on all 1,000 test elements** in a differential comparison against
the unmodified, known-correct kernel. Root-causing the divergence traced it
to the de-SSA (phi-elimination) pass: the shared `break`-target block was
now reached via two structurally distinct paths in the corrected tree (one
via each of the two branches feeding it), and de-SSA's phi-copy placement is
sensitive to *where in the structured tree* a given jump occurs, not merely
to which raw basic block it targets — a distinction the fix's own
correctness reasoning, and both of its passing verification methods, had
not accounted for.

The fix was reverted in its entirety — confirmed via a clean `git diff`
against the pre-fix state — rather than shipped with the `continue`-adjacent
limitation documented as a caveat on top of it. We consider this the
paper's clearest illustration of a general principle in compiler
correctness engineering that is easy to state and, evidently, easy to
violate in practice even under a deliberately careful verification
protocol: passing a substantial battery of randomized tests and an entire
pre-existing regression suite is evidence of correctness on the *distribution
of cases those tests sample*, and is not, by itself, proof of correctness
against a real, structurally novel input the test distribution did not
happen to cover. The bug here was not caught by writing more random tests
in the same style; it was caught by testing against one specific, real
production kernel whose actual shape had not been anticipated.

### 6.4 A genuine compiler infrastructure improvement: device-function compile-time caching

Not every compiler-level investigation in this project concluded negatively.
Motivated by an earlier finding that factoring a small, hot-loop
computation into a `@metal.device_func` costs a real, measured ~1.9×–2.4×
per-iteration runtime slowdown relative to inlining it directly (a real,
non-negligible cost of a non-inlined function call in generated MSL, not
free code organization), a separate hypothesis was tested: that a device
function's *compile time*, rather than its runtime, could benefit from
being shared across multiple different kernels that call it. Prior to this
investigation, each top-level kernel's compilation independently re-ran the
full Numba-frontend-plus-MSL-lowering pipeline for every device function it
called, even when an identical function (same Python object, same argument
types) had already been compiled once for a different kernel earlier in the
same process.

A process-wide compilation cache, keyed on `(python_function, argument_types)`
and storing each device function's complete transitive MSL source
dependency set (not merely its own body — a device function that itself
calls another device function requires that callee's source to be spliced
into every kernel that calls it, transitively), was added to the MSL
backend. Two real correctness bugs were found and fixed while building this
cache, both confirmed via dedicated regression tests before the cache was
trusted: first, caching only a function's own compiled body (not its
transitive dependencies) produced an "undeclared identifier" Metal
shader-compiler error the first time a second kernel called an
already-cached function that itself, transitively, called a third function
that second kernel had never independently compiled; second, naively
splicing a cache hit's full transitive-dependency list without checking
what the current compilation had already emitted produced a duplicate MSL
function definition in a diamond-dependency case (two device functions in
one kernel sharing one common callee) — a defect Apple's Metal shader
compiler tolerated silently in every case tested, but which we do not
consider safe to rely on. With both bugs fixed and covered by regression
tests, cold-compiling 30 kernels that share one common device function
measured a **1.25×–1.89× reduction in total cold-compile time** compared to
each kernel independently compiling its own copy of the shared function,
reproduced consistently across repeated measurement runs. A first, buggy
version of this same cache — prior to finding and fixing the two defects
above — measured a **0.92× outcome (a net loss)** and was not shipped; only
the corrected, independently reverified version is part of the project.

### 6.5 The `numba-metal advisor` tool

Every roofline claim in this paper and in `docs/performance-guidance.md`
depends on being able to actually measure a candidate kernel's compute-vs-
bandwidth classification and compare its CPU and Metal performance without
hand-writing a bespoke benchmark script each time. `numba-metal` ships a
dedicated tool for exactly this, in `src/numba_metal/advisor/` (roughly
4,200 lines across 15 modules: static scanning, Numba/MSL compatibility
dry-runs, a runtime profiler with instrumentation hooks, CPU-vs-Metal
comparison and correctness verification, opportunity scoring, a
deterministic recommendation engine, ASCII flame-graph/timeline
rendering, an interactive terminal UI, and device calibration), installed
as a real console-script entry point (`numba-metal`, declared in
`pyproject.toml`'s `[project.scripts]`) with its own test suite under
`tests/advisor/`. It is documented in full in `docs/advisor.md`; this
section exists because, despite `docs/performance-guidance.md` citing it
six times as the source of its own roofline classifications, an earlier
draft of this paper mentioned it exactly once, in passing, without
explaining what it is or that it is real, working infrastructure — an
omission this section corrects.

The tool exposes five subcommands: `scan` (static AST analysis of a
codebase to find `@njit`/`@prange` candidates for Metal, without ever
importing or executing the scanned code), `profile` (runtime profiling of
a script or pytest run, producing an ASCII flame graph and CPU/GPU
timeline from real instrumentation events), `compare` (CPU-vs-Metal
timing and correctness comparison for a given script, the subcommand
`docs/performance-guidance.md` cites directly for its `BANDWIDTH_BOUND`/
`COMPUTE_BOUND` classifications), `report` (re-rendering a previously
saved JSON profile), and `calibrate` (measuring this machine's own
dispatch overhead, compile cost, memory bandwidth, and float32 throughput
ceilings, cached to disk). `docs/performance-guidance.md`'s own cited
figures — the ~683 GFLOPS and ~221 GB/s ceilings behind the 3.09
theoretical ridge point discussed in Section 6.2 — are, based on matching
units, order of magnitude, and the absence of any other committed
benchmark script producing these exact numbers, almost certainly the
output of a `numba-metal advisor calibrate` run, though this paper does
not have a recorded calibration-run log confirming that provenance with
certainty.

In the course of preparing this paper, `scan` and `calibrate` were run
directly against this repository and confirmed to produce real,
non-fabricated output: `scan` correctly identified `@njit`/`@prange`
candidates across `benchmarks/` without executing any of them, and
`calibrate` produced a fresh set of device ceiling measurements in the
same units and order of magnitude as those already cited in
`docs/performance-guidance.md`. `compare` has a real implementation and a
passing test suite under `tests/advisor/`, but was not independently
re-run end-to-end against a live kernel in the course of preparing this
specific paper revision; this distinction — direct confirmation for
`scan` and `calibrate`, implementation-plus-passing-tests but not a fresh
live run for `compare` — is stated explicitly here rather than
smoothed into a single blanket claim that "the advisor works."

The honest scope of this section is that the advisor is real, useful,
substantially tested infrastructure that this paper's own quantitative
claims already depend on indirectly, and that it deserved a real
description rather than the single passing citation it received before
this revision — not a claim that every one of its five subcommands has
been independently re-verified against a live kernel in this paper's
preparation.

## 7. Scope, limitations, and threats to validity

This section is deliberately explicit about what this work does not
establish, consistent with the scope statement at the top of this document.

**Single-machine measurement.** Every number in §5–6 was measured on one
physical machine (a single Apple M4 Pro unit) under one software
configuration (macOS 26.5.1, Numba 0.67.0). No claim about other Apple
Silicon generations (M1 through M3, or later M-series chips), other macOS or
Metal Toolchain versions, or other GPU core-count configurations within the
M4 family is made or should be inferred. Run-to-run variance was observed
directly (§6.1's disclosed 4.6×–7.2× range at one problem size) and is
real, not measurement error alone; a more rigorous study would report
confidence intervals computed from substantially more repeated trials per
configuration than this project's own methodology (typically 3–5 repeats
per measurement, following the project's own documented convention) affords.

**No formal proof of the structurer's correctness.** The control-flow
structurer (§3.2) is verified by a combination of unit tests, differential
testing against a Python reference implementation, and — critically, per
§6.3 — testing against specific real-world kernel shapes discovered to
expose bugs the former two methods missed. It is not formally verified, and
§6.3 is direct, first-party evidence that its current test suite, however
substantial (334 tests as of this writing), does not exhaust the space of
CFG shapes Numba's bytecode lowering can produce. The `continue`-in-rotated-
while limitation documented in §4.3 is a conservative response to this same
uncertainty: rather than attempt a second structural fix under the same
verification methodology that already produced one false negative, the
capability was left unimplemented and clearly documented as such.

**One pre-existing, unresolved test failure.** A Hypothesis-based
property test (`tests/differential/test_differential.py::test_quick_
differential`) fails on a randomly generated kernel combining nested
`if`/`else` and a `for`-range loop — a shape containing no `while` loop at
all, and therefore independent of the §4.3/§6.3 discussion — with a
compile-time `UnsupportedFeatureError` ("block reached more than once as a
loop header"). This failure predates the work described in this paper, was
confirmed (via `git stash` bisection) to be unrelated to any change made
during this project's own sessions, and has not been root-caused. It is
disclosed here rather than omitted or silently excluded from the reported
test count.

**Self-reported results.** This document, the code it describes, and the
verification of both were produced by one individual working with a single
AI coding assistant across one project's development history, without
independent code review, external replication, or peer review at any stage.
Readers should weigh every claim in this paper accordingly, and are
encouraged to verify any specific number by running the project's own
benchmark suite (`benchmarks/run_all.py` and the standalone scripts named in
Table 3), which is designed explicitly so that no reported speedup number is
hard-coded anywhere in the repository — every number this paper reports is,
in principle, independently reproducible by re-running the corresponding
script on comparable hardware, modulo the run-to-run variance already
discussed.

**Related-work coverage is best-effort, not exhaustive.** The comparisons in
§2 were verified to be accurate as of the time of writing (real projects,
real authors, real architectural claims, checked against primary sources
where possible), but the search that produced them was not a systematic
literature review, and closely related work — particularly any existing
academic or industrial effort at compiling Python or another
high-level language directly to Metal Shading Language without going
through an intermediate array-operation library — may exist and was not
found.

## 8. Conclusion

We have described a compiler that lowers a restricted, precisely bounded
subset of Python directly to Metal Shading Language by reconstructing
structured control flow from Numba's typed intermediate representation,
motivated by the concrete absence of any LLVM-based path to Apple's GPU
instruction set. We have shown, through a controlled pair of otherwise-
identical benchmark kernels differing only in arithmetic intensity, a clean
empirical demonstration of the roofline model's central prediction on Apple
Silicon's shared-memory-bandwidth architecture: identical memory-access
patterns can be made to either degrade or improve dramatically under
100× problem-size scaling, depending entirely on how much arithmetic
accompanies each memory access. We have also reported, in full, a case where
a compiler fix passed every verification method applied to it except the
one that mattered — testing against a real, structurally novel production
kernel — and was reverted as a direct result. We believe this last result,
more than any single speedup number in this paper, is the honest
methodological contribution: in a project whose central design principle is
"no silent fallback," the discipline of actually reverting a change that
passes its own tests but fails against reality is what that principle
requires in practice, not merely in documentation.

## References

- J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin,
  G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, Q. Zhang. *JAX:
  composable transformations of Python+NumPy programs.* 2018.
- T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Yan, H. Shen, M. Cowan, L. Wang,
  Y. Hu, L. Ceze, C. Guestrin, A. Krishnamurthy. *TVM: An Automated
  End-to-End Optimizing Compiler for Deep Learning.* OSDI 2018.
  arXiv:1802.04799.
- R. Frostig, M. J. Johnson, C. Leary. *Compiling machine learning programs
  via high-level tracing.* SysML 2018.
- A. Hannun, J. Digani, A. Katharopoulos, R. Collobert, et al. *MLX: An
  array framework for Apple silicon.* Apple Machine Learning Research,
  2023. https://github.com/ml-explore/mlx
- L. Hübner, Y. Hu, I. B. Peng, S. Markidis. *Apple vs. Oranges: Evaluating
  the Apple Silicon M-Series SoCs for HPC Performance and Efficiency.*
  arXiv:2502.05317, 2025.
- S. K. Lam, A. Pitrou, S. Seibert. *Numba: A LLVM-based Python JIT
  Compiler.* Proceedings of the Second Workshop on the LLVM Compiler
  Infrastructure in HPC (LLVM-HPC), Supercomputing 2015.
  https://doi.org/10.1145/2833157.2833162
- G. Markall. *The Life of a Numba Kernel: A Compilation Pipeline Taking
  User-Defined Functions in Python to CUDA.* RAPIDS AI, Medium, 2019.
- P. Perona, J. Malik. *Scale-space and edge detection using anisotropic
  diffusion.* IEEE Transactions on Pattern Analysis and Machine
  Intelligence, 1990.
- J. Ragan-Kelley, C. Barnes, A. Adams, S. Paris, F. Durand, S. Amarasinghe.
  *Halide: A Language and Compiler for Optimizing Parallelism, Locality,
  and Recomputation in Image Processing Pipelines.* PLDI 2013.
- P. Tillet, H. T. Kung, D. Cox. *Triton: An Intermediate Language and
  Compiler for Tiled Neural Network Computations.* MAPL 2019 (workshop
  paper preceding the 2021 public release).
- Numba maintainers. GitHub issue numba/numba#5706 (Metal/Apple GPU backend
  discussion), 2020–2025.
- LLVM Project. *User Guide for NVPTX Back-end.* https://llvm.org/docs/NVPTXUsage.html
- PyTorch documentation contributors. *MPS Backend.*
  https://deepwiki.com/pytorch/pytorch/3.3-mps-backend-(metal-performance-shaders)

## Appendix A: Reproducing these results

The full source, test suite, and benchmark scripts referenced throughout
this paper are available in this repository. To reproduce the core suite:

```bash
python benchmarks/run_all.py            # five core benchmarks, text report
python benchmarks/run_all.py --json out.json
python benchmarks/cyclist_aerodynamics.py
python benchmarks/implied_volatility.py
python benchmarks/asian_option_pricing.py
python benchmarks/device_function_compile_cache.py
python benchmarks/arithmetic_intensity_sweep.py --grid 1024 --reps -1 0 1 2 4 8
numba-metal advisor scan .
numba-metal advisor calibrate
```

The full test suite (334 tests as of this writing, one pre-existing and
disclosed failure per §7) is run via:

```bash
pytest tests/ -q
```

See `docs/benchmarking.md` for full methodology and `docs/limitations.md`
and `docs/supported-features.md` for the complete, current capability
surface, which is kept in sync with the compiler's actual tested behavior
as a matter of project policy.
