"""A closed, controlled grammar for generating small numba-metal kernel
functions, used by the Hypothesis-driven differential test suite
(Workstream 4).

This deliberately does NOT generate unrestricted arbitrary Python and does
NOT `eval()` untrusted content. It builds a small typed expression/
statement tree from a fixed set of node types (see `Expr`/`Stmt` below),
each of which corresponds 1:1 to an operation numba-metal's
`docs/supported-features.md` documents as supported, renders that tree to
Python *source text* through a fixed, reviewable template
(`render_kernel`), and `exec`s the resulting source in a fresh, minimal
namespace containing only `metal`/`math` -- the same trust boundary as
any other Python source file in this repository, not a sandboxing
mechanism for untrusted input. The grammar is closed: every node type
listed here is the complete set Hypothesis can choose from, so a
generated kernel's shape is always traceable back to one of these
constructors.

Every generated kernel's source is retained on the returned `GeneratedKernel`
object so a failing case can be printed verbatim for diagnosis (Hypothesis
example, generated Python source, generated MSL, inputs, expected vs.
actual, max error).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hypothesis import strategies as st

# --- Typed value pool -------------------------------------------------

#: dtypes covered by the grammar, matching the task's required
#: end-to-end-verified subset.
DTYPES = ("float32", "int32", "uint32", "bool_")

# Scalar Python-level types the grammar treats each dtype as, for
# generating literal constants and reference (Numba CPU) computation.
_NUMPY_DTYPE = {
    "float32": "np.float32",
    "int32": "np.int32",
    "uint32": "np.uint32",
    "bool_": "np.bool_",
}


@dataclass
class VarSpec:
    """One kernel-local variable: a name and its dtype (as a grammar
    dtype string, one of DTYPES)."""

    name: str
    dtype: str


@dataclass
class KernelSpec:
    """A fully-specified generated kernel: its array/scalar parameters,
    body statements (as source-text lines, already rendered), and enough
    metadata to build both a `@metal.jit` and a `@njit` version and to
    generate matching random inputs."""

    name: str
    array_params: list[VarSpec]  # 1D arrays, one element read/written per lane
    scalar_params: list[VarSpec]  # runtime scalar arguments
    body_lines: list[str]  # already-indented Python source lines (kernel body)
    output_param: str  # which array_param name is the output
    uses_math: bool = False
    description: str = ""


# --- Source rendering ---------------------------------------------------

_KERNEL_TEMPLATE = """\
def {name}({args}):
    i = metal.grid(1)
    if i < {output}.size:
{body}
"""

_CPU_TEMPLATE = """\
def {name}_cpu({args}):
{scalar_backup}
    for i in range({output}.shape[0]):
{scalar_reset}
{body}
"""


def render_kernel(spec: KernelSpec) -> str:
    """Render a KernelSpec to a `@metal.jit`-decorated kernel's Python
    source text (the decorator itself is applied by the caller after
    `exec`, not baked into this text, so the same body can be reused for
    both the GPU and CPU-reference renderings via `render_cpu_reference`).
    """
    args = ", ".join(v.name for v in (*spec.array_params, *spec.scalar_params))
    body = "\n".join(f"        {line}" for line in spec.body_lines)
    return _KERNEL_TEMPLATE.format(
        name=spec.name, args=args, output=spec.output_param, body=body
    )


def render_cpu_reference(spec: KernelSpec) -> str:
    """Render the same body as a plain Python-loop CPU function (no Numba
    decorator applied by this text -- the caller wraps it in `@njit`),
    used as the correctness oracle. Structurally identical to the GPU
    kernel's body except `i` ranges over a Python `for` loop instead of
    `metal.grid(1)`, since `@njit` CPU code has no grid concept.

    Each GPU thread receives its own independent copy of every scalar
    argument (real Metal semantics: scalar kernel arguments are read from
    a per-dispatch `constant` buffer, and one thread mutating its local
    copy of a scalar parameter has zero effect on any other thread's
    copy). A naive single Python function looping `for i in
    range(n_lanes)` does NOT have this property: if the generated body
    mutates a scalar parameter (a real, grammar-covered case -- e.g.
    `m = 0` inside a per-lane loop, exercising exactly this), that
    mutation would otherwise persist into the next `i` iteration's
    "thread", silently simulating shared mutable state no real dispatch
    has. This was found concretely: a generated kernel mutating its `m`
    scalar argument produced DIFFERENT CPU-reference results depending on
    whether it was run through this multi-lane template or in isolation
    for a single lane, proving the multi-lane template (not numba-metal)
    was the source of the divergence. The reset line below re-establishes
    each scalar argument's original value at the top of every `i`
    iteration, matching real independent-thread semantics.
    """
    args = ", ".join(v.name for v in (*spec.array_params, *spec.scalar_params))
    body = "\n".join(f"        {line}" for line in spec.body_lines)
    if spec.scalar_params:
        backup_lines = "\n".join(
            f"    {s.name}_orig = {s.name}" for s in spec.scalar_params
        )
        reset_lines = "\n".join(
            f"        {s.name} = {s.name}_orig" for s in spec.scalar_params
        )
    else:
        backup_lines = "    pass"
        reset_lines = "        pass"
    return _CPU_TEMPLATE.format(
        name=spec.name,
        args=args,
        output=spec.output_param,
        body=body,
        scalar_backup=backup_lines,
        scalar_reset=reset_lines,
    )


# --- Hypothesis strategies for building KernelSpecs ---------------------

_VALID_IDENTIFIERS = [
    "a",
    "b",
    "c",
    "x",
    "y",
    "z",
    "acc",
    "tmp",
    "val",
    "out",
]


def _literal_for(dtype: str, value: float | int | bool) -> str:
    if dtype == "float32":
        return f"{float(value)!r}"
    if dtype == "bool_":
        return "True" if value else "False"
    return str(int(value))


@dataclass
class _BuildContext:
    """Mutable state threaded through grammar construction.

    `live_vars` contains only variables *definitely initialized on every
    path reaching the current point* -- i.e. it is safe to read any of
    them right now. This is deliberately conservative dataflow tracking:
    a variable assigned inside only one arm of an `if` (no `else`, or an
    `else` that doesn't also assign it) must NOT be added to the parent
    scope's `live_vars` after the `if`, because a path exists (the arm
    that didn't run) where it was never assigned -- exactly the `v3 = v3`
    /`NameError`-shaped bug this tracking exists to prevent generating.
    Callers that open a new scope (`_if_stmt`'s two arms, `_for_stmt`'s
    body) must snapshot `live_vars` with `snapshot()`/`restore()` around
    that scope so speculative variables introduced inside it don't leak
    into a sibling branch or the parent scope's remaining statements
    unless a dataflow-sound merge rule (see `_if_stmt`) explicitly adds
    them back.
    """

    live_vars: list[VarSpec] = field(default_factory=list)
    counter: int = 0

    def fresh_name(self, prefix: str = "v") -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def snapshot(self) -> list[VarSpec]:
        return list(self.live_vars)

    def restore(self, snapshot: list[VarSpec]) -> None:
        self.live_vars = list(snapshot)


# Arithmetic/comparison/boolean operators, restricted to the documented
# supported subset (docs/supported-features.md).
_ARITH_OPS = ["+", "-", "*"]
_COMPARE_OPS = ["<", "<=", ">", ">=", "==", "!="]
_MATH_FUNCS = ["sqrt", "exp", "log", "sin", "cos"]


@st.composite
def _numeric_expr(draw, dtype: str, ctx: _BuildContext, depth: int = 0) -> str:
    """Build a small expression of the given dtype as Python source text.
    `depth` bounds recursion so generated expressions stay small."""
    candidates = [v for v in ctx.live_vars if v.dtype == dtype]
    choices = ["literal"]
    if candidates:
        choices.append("var")
    if depth < 2 and dtype in ("float32", "int32", "uint32"):
        choices.append("binop")
    if depth < 2 and dtype == "float32":
        choices.append("mathfunc")
    if depth < 2:
        choices.append("absmin")

    kind = draw(st.sampled_from(choices))

    if kind == "literal":
        if dtype == "float32":
            v = draw(st.floats(min_value=-100.0, max_value=100.0, width=32))
        elif dtype == "bool_":
            v = draw(st.booleans())
        elif dtype == "uint32":
            v = draw(st.integers(min_value=0, max_value=1000))
        else:
            v = draw(st.integers(min_value=-1000, max_value=1000))
        return _literal_for(dtype, v)

    if kind == "var":
        return draw(st.sampled_from(candidates)).name

    if kind == "binop":
        op = draw(st.sampled_from(_ARITH_OPS))
        lhs = draw(_numeric_expr(dtype, ctx, depth + 1))
        rhs = draw(_numeric_expr(dtype, ctx, depth + 1))
        if op == "-" and dtype == "uint32":
            # avoid unsigned underflow, which is legal but not what this
            # grammar is trying to exercise (wraparound semantics are a
            # different, not-yet-documented feature)
            op = "+"
        return f"({lhs} {op} {rhs})"

    if kind == "mathfunc":
        fn = draw(st.sampled_from(_MATH_FUNCS))
        inner = draw(_numeric_expr(dtype, ctx, depth + 1))
        if fn == "exp":
            # math.exp grows fast enough that even a modest input (e.g.
            # ~90, reachable by chaining `exp` of another already-large
            # value at nesting depth 2) overflows float32's finite range
            # (~3.4e38) while remaining finite in float64. Numba CPU
            # infers float64 for unannotated literals/intermediates
            # (matching CPython), while numba-metal's MSL backend
            # deliberately narrows local float64 intermediates to
            # float32 (see docs/limitations.md) -- so an overflowing
            # `exp` input produces `+inf` on the GPU (and a correct,
            # IEEE-754 `NaN` from the following `cos`/`sin`/etc.) but a
            # large *finite* float64 result on CPU, a genuine, already
            # -documented precision difference (see docs/architecture.md,
            # the Mandelbrot benchmark's float32-vs-float64 investigation)
            # rather than a compiler bug. Clamping exp's input to a small
            # range keeps the grammar exercising real math-function
            # coverage without generating this known, already-understood,
            # separately-documented divergence as a false differential
            # -test failure. `min(..., 10.0)` bounds it after the usual
            # abs()+0.1 shaping.
            return f"math.exp(min(abs({inner}) + 0.1, 10.0))"
        # keep other math function inputs positive-ish and bounded to
        # avoid domain errors (log/sqrt of negative) dominating the
        # search space
        return f"math.{fn}(abs({inner}) + 0.1)"

    if kind == "absmin":
        sub = draw(st.sampled_from(["abs", "min", "max"]))
        if sub == "abs":
            inner = draw(_numeric_expr(dtype, ctx, depth + 1))
            return f"abs({inner})"
        lhs = draw(_numeric_expr(dtype, ctx, depth + 1))
        rhs = draw(_numeric_expr(dtype, ctx, depth + 1))
        return f"{sub}({lhs}, {rhs})"

    raise AssertionError(f"unreachable grammar kind {kind!r}")


@st.composite
def _bool_expr(draw, ctx: _BuildContext, depth: int = 0) -> str:
    numeric_dtypes = [v.dtype for v in ctx.live_vars if v.dtype != "bool_"]
    bool_vars = [v for v in ctx.live_vars if v.dtype == "bool_"]
    choices = []
    if numeric_dtypes:
        choices.append("compare")
    if bool_vars:
        choices.append("boolvar")
    if depth < 2 and bool_vars:
        choices.append("boolop")
    if not choices:
        choices = ["literal"]

    kind = draw(st.sampled_from(choices))
    if kind == "literal":
        return draw(st.sampled_from(["True", "False"]))
    if kind == "boolvar":
        return draw(st.sampled_from(bool_vars)).name
    if kind == "compare":
        dtype = draw(st.sampled_from(numeric_dtypes))
        op = draw(st.sampled_from(_COMPARE_OPS))
        lhs = draw(_numeric_expr(dtype, ctx))
        rhs = draw(_numeric_expr(dtype, ctx))
        return f"({lhs} {op} {rhs})"
    if kind == "boolop":
        op = draw(st.sampled_from(["and", "or"]))
        lhs = draw(_bool_expr(ctx, depth + 1))
        rhs = draw(_bool_expr(ctx, depth + 1))
        return f"({lhs} {op} {rhs})"
    raise AssertionError(f"unreachable grammar kind {kind!r}")
