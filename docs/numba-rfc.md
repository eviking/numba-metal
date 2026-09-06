# numba-metal: findings for numba/numba#5706

Status: draft, paste-ready for posting as a comment on
[numba/numba#5706](https://github.com/numba/numba/issues/5706)
("Please add support for Metal as GPU accelerator"). Link placeholders
below (`<REPO_URL>`) should be filled in once this project has a public
home; nothing in this document should be posted with a placeholder
still in it.

This is not a request for Numba to adopt or merge anything. It's a
report on an independent, out-of-tree experiment addressing the exact
question raised in that issue, written because the issue has been open
since 2020 with no code proposed, and this project may be useful
context for anyone else who finds it.

---

## Project description

**numba-metal** is a small, out-of-tree Python package that lets a
constrained subset of Numba-`@njit`-style kernel functions run on
Apple-silicon GPUs via Metal, without touching Numba's own target
registry or its LLVM-based lowering path. It reuses Numba's frontend
(untyped IR construction + type inference) unmodified, then hands the
resulting typed IR to a from-scratch backend that walks it and emits
Metal Shading Language (MSL) source text directly -- never LLVM IR,
never string-template substitution on Python source.

It is alpha-quality, single-contributor, unaudited by Numba's own
maintainers, and has not been used in production. It is shared here
purely as a data point: **a viable path to Apple GPU support from
Numba-style kernels does exist, and it does not go through LLVM.**

## Architecture, in one paragraph

`@metal.jit` intercepts a Python function, runs it through Numba's
frontend to get validated typed IR (the same intermediate
representation Numba's own CPU/parallel targets consume), reconstructs
structured control flow from that IR's basic-block CFG (dominator-based
loop detection, since Metal's shader compiler rejects `goto`/labeled
statements -- confirmed by directly testing it), performs a from-scratch
SSA/phi-node elimination pass (not reusing anything from
`numba.core.typed_passes`' own lowering, since that path assumes an LLVM
target context), and tree-walks the resulting structured AST into MSL
text. That text is compiled with Apple's own
`MTLDevice.newLibraryWithSource_options_error_` at runtime, producing a
real `MTLComputePipelineState` that is dispatched exactly like any other
Metal compute kernel. Full detail, including the specific typed-IR
internals relied upon, is in `docs/architecture.md` in this repository.

## What's implemented (and what explicitly is not)

A constrained but real subset: scalar `float32`/`int32`/`uint32`/`bool`
(+ `float16`/`int64` as secondary), 1D flattened array arguments,
arithmetic/comparison/boolean operators, `if`/`if`-`else` (arbitrarily
nested), ternary expressions, `for x in range(...)` (1/2/3-argument,
runtime bounds, positive or negative step), `break`/`continue`,
`abs`/`min`/`max`, `float()`/`int()` casts, a handful of `math.*`
functions, 1D and 2D launch grids. Explicitly **not** implemented:
device functions/call graphs beyond a single kernel, local array
allocation, threadgroup (shared) memory, atomics, `while` loops,
recursion, exceptions, and formal registration as a Numba compilation
target (see `docs/upstream-strategy.md` for why the last one is a
deliberate choice, not a gap). The full matrix, including exactly which
test provides evidence for each row, is in
`docs/supported-features.md` and `docs/feature-traceability.md`.

## Test environment

- Hardware: Apple M4 Pro (see `benchmarks/results/*.json` for the exact
  machine-readable environment block from each run, including chip
  identifier and unified-memory confirmation).
- OS: macOS 26.5.1.
- Python: 3.12 and 3.13 (both run against the full test suite; see
  "Supported versions" below).
- Numba: 0.67.0 (the only version validated; see "Supported versions").
- Test suite: 127 unit/integration tests (`pytest -m "not metal"` /
  `pytest -m metal`), plus a Hypothesis-based differential test suite
  generating kernels from a closed grammar and comparing real Metal
  execution against real `@njit` execution -- 500/500 generated cases
  passed in ~42s on this hardware (`artifacts/differential/`).

## Measured performance (summary)

Full methodology, all fifteen size/benchmark combinations, and the
separated timing categories (frontend compile, MSL pipeline compile,
kernel-only warm execution, host<->device transfer, cold vs. warm) are
in `docs/benchmarking.md`; only the headline shape is summarized here,
deliberately without cherry-picking the best numbers:

- Compute-bound, high-parallelism workloads with divergent per-thread
  control flow (Mandelbrot escape-time) saw genuine GPU wins: 2.7x-16.5x
  over 12-thread parallel Numba CPU as image size grows from 512² to
  4096², because per-thread iteration counts vary and the GPU has enough
  independent lanes to hide that divergence.
- Iterative stencil workloads with resident GPU data (heat diffusion)
  saw the largest wins measured (up to ~224x at 512²), because the CPU
  baseline redoes cache-unfriendly nested-loop work every iteration
  while the GPU dispatches once per iteration over already-resident
  buffers.
- Simple elementwise arithmetic and small-working-set workloads
  (vector polynomial, small pairwise-distance sizes) were **slower**
  than parallel Numba CPU at smaller sizes (0.2x-0.7x) -- fixed
  per-launch dispatch overhead dominates until problem size grows large
  enough to amortize it, and Numba's own `prange` is already fast on a
  12-core CPU for this class of problem.
- Every cold-compile measurement (first-time typed-IR frontend + MSL
  lowering + Metal shader/pipeline compilation) took low tens of
  milliseconds; every reported cold number was confirmed to reflect a
  genuinely fresh compilation via a before/after cache-entry-count
  check, not reused warm state.

These are single-machine, single-run numbers, not a claim about other
Apple-silicon chips or a promise of these exact speedups elsewhere.

## Why typed-IR-without-target-registration, not `target_extension`

Numba's `numba.core.target_extension` machinery (used by `numba.cuda`,
`numba.core.registry.cpu_target`, etc.) exists to let multiple backends
share Numba's generic `@overload`/`@lower` registries and ultimately
hand **LLVM IR** to `numba.core.codegen`. That machinery is explicitly
documented as in-development ("all features and APIs described in this
page are in-development and may change at any time without deprecation
notices being issued" -- `developer/target_extension.html`), and more
fundamentally, it assumes the eventual output is LLVM IR. MSL is
C++14-derived text, not LLVM IR, and there is no publicly documented way
to lower LLVM IR to Apple's AIR/`air64` GPU IR (see "What #5706 already
established," below) -- so registering as a formal target would add
real API-surface exposure to Numba's actual internal/unstable
mechanisms for a lowering path (`@lower`-based, LLVM-IR-producing) that
this project cannot use anyway. Given that, the smallest honest design
is: reuse only the frontend (typed IR + type inference -- the highest-
value, most complex part of "does this Python code typecheck as a GPU
kernel"), and write a from-scratch, independently-testable backend for
the one thing Numba's own tooling can't help with. Full alternatives
considered (including a rejected goto-based/CFG-literal translation
attempt and why string-template MSL generation was ruled out entirely)
are in `docs/architecture.md`, "Alternatives considered."

## What #5706 already established (read in full for this writeup)

Reading the issue's five years of discussion (2020-2025) surfaced a
consistent, independently-reached conclusion among maintainers and
contributors that this project's architecture also arrived at
separately:

- No public LLVM-IR-to-Metal-AIR backend exists; Apple's internal
  AIR/`air64` LLVM dialect is undocumented (`@seibert`, `@gmarkall`).
- Apple's Metal Shader Converter (macOS 14+) converts DXIL (a subset of
  LLVM 3.7 IR used for the Game Porting Toolkit), not arbitrary
  Numba-generated LLVM IR; a maintainer was "not optimistic about it
  making a straightforward conversion" (`@gmarkall`).
- A community attempt to route through SPIR-V
  (LLVM-IR -> SPIRV-LLVM-Translator -> SPIRV-Cross -> MSL, all
  Homebrew-installable Khronos tooling) got partway (successfully
  produced `.spv` from Numba-compiled LLVM bitcode) but hit
  Kernel-capability/scope errors converting that SPIR-V to MSL, and the
  thread has no record of it being resolved (`@thipokKub`, 2021).
- Maintainers characterized a real from-LLVM-IR solution as "a research
  project" requiring "significant extra development effort," not
  something likely without a contributor doing or funding a large part
  of it (`@gmarkall`, 2020 and 2024).

This project did not attempt the LLVM-IR route at all -- it treats "no
LLVM path to Metal" as a given (confirmed independently, not just by
reading the issue) and sidesteps it entirely by generating MSL text from
Numba's typed IR instead of from LLVM IR. That is the central technical
finding worth surfacing to the issue: **the LLVM-IR route the issue's
discussion focused on may genuinely be a dead end, but a
typed-IR-to-text-source route is not**, at least for the constrained
kernel subset this project covers.

## Supported versions

| Numba | Status |
|---|---|
| 0.67.0 | Metal tested (full test suite, including real Metal execution and differential tests, run against this exact version) |
| 0.67.x (x != 0) | Unit tested only in the sense that numba-metal's own runtime compatibility gate (`numba_metal.compat.check_numba_compatible`) accepts any 0.67.x patch release without rejection, on the assumption that Numba patch releases do not change the typed-IR/CFG shape this project depends on -- but no patch release other than 0.67.0 has actually been run against this test suite |
| Anything outside 0.67.x | Rejected at runtime with `UnsupportedNumbaVersionError` naming the installed version and explaining why (internal-API dependency, not a generic pin) |

| Python | Status |
|---|---|
| 3.12 | Full test suite run and passing as of an earlier snapshot (127 tests); not re-verified against the current, larger suite on this interpreter version -- see CI (`.github/workflows/ci.yml`) for the current 3.12 matrix result |
| 3.13 | Full test suite run and passing (334 tests as of this writing); this repository's primary development environment |
| 3.10, 3.11 | Not tested on this machine (unavailable in the development environment); `requires-python` was deliberately narrowed to `>=3.12,<3.14` rather than claiming untested support |

`pyproject.toml` pins `numba>=0.67,<0.68` to match exactly what has been
validated; CI (`.github/workflows/ci.yml`) matrices across Python
3.12/3.13.

## Reliance on internal/semi-private Numba APIs

Documented in full in `docs/architecture.md`, "Relevant Numba extension
points and private APIs used" -- summarized here:

| API | Stability | Why needed |
|---|---|---|
| `numba.core.compiler.CompilerBase`, `DefaultPassBuilder` | Semi-public | Build a pipeline that stops after type inference |
| `numba.core.compiler_machinery.{PassManager, FunctionPass, register_pass}` | Semi-public | Insert a terminal pass to capture typed IR |
| `numba.core.typed_passes.{NopythonTypeInference, AnnotateTypes}` | Internal, no stability guarantee | The actual type-inference passes reused |
| `numba.core.registry.cpu_target` | Semi-public | Typing/target context source (not used for lowering) |
| `numba.extending.intrinsic(prefer_literal=True)` | Public | Type `metal.grid()`/`gridsize()` |
| `numba.core.ir.*` | Public-ish, grammar not stability-guaranteed | Walking the typed IR (including `ir.Expr.phi`'s parallel `incoming_values`/`incoming_blocks` arrays for de-SSA) |

This surface is deliberately small and isolated to one adapter module
(`numba_metal/compiler/frontend.py`), and is exactly the reason
`numba_metal.compat.check_numba_compatible()` exists as a runtime gate
rather than trusting a version-range dependency pin alone: a shape
change in any of the above across a future Numba release could produce
silently wrong MSL rather than a clean failure, which this project
treats as unacceptable.

## Questions for maintainers

1. Is there interest in a documented (even if unstable/provisional)
   extension point specifically for "frontend-only" consumers -- i.e.
   third parties that want validated typed IR and nothing else, without
   needing to assemble a `CompilerBase` pipeline via
   `register_pass`/`typed_passes` internals? That is the single largest
   internal-API surface this project depends on, and the pattern seems
   plausibly reusable by other non-LLVM targets.
2. Is `ir.Expr.phi`'s `incoming_values`/`incoming_blocks` shape
   considered stable enough across releases to document explicitly, or
   has it changed in the past in ways a downstream consumer should
   watch for?
3. Independent of this project: is there any interest from the core
   team in revisiting Metal support now that Apple's tooling landscape
   has changed since 2020 (Metal Shader Converter, MPS, MLX), or does
   the 2024 "research project, would need a contributor to drive it"
   assessment still hold? This project does not resolve that question
   (it deliberately avoids the LLVM-IR route entirely), but may be
   useful context either way.

## Proposed extension points (if there is interest)

Not requested or expected to be acted on -- offered only in case the
questions above get a "yes, that would help downstream projects"
response:

- A narrow, documented function (even behind an explicit
  "unstable, may change" banner) that runs exactly Numba's
  untyped-IR-construction + type-inference stages and returns the
  typed `FunctionIR` + `Signature` + `typemap`, without requiring a
  caller to assemble a `CompilerBase` subclass or reach into
  `typed_passes` directly.
- Explicit documentation (even a short paragraph) of `ir.Expr.phi`'s
  field shape and stability expectations, since any non-LLVM backend
  doing its own SSA destruction needs this exact information and
  currently has to read Numba's source to get it.

## Links

- This project: `<REPO_URL>`
- Architecture detail: `<REPO_URL>/blob/main/docs/architecture.md`
- Full benchmark methodology and results: `<REPO_URL>/blob/main/docs/benchmarking.md`
- Feature support matrix and evidence audit: `<REPO_URL>/blob/main/docs/supported-features.md`, `<REPO_URL>/blob/main/docs/feature-traceability.md`
- Upstream strategy assessment (this project's own recommendation on
  whether/how to engage further with Numba upstream):
  `<REPO_URL>/blob/main/docs/upstream-strategy.md`
