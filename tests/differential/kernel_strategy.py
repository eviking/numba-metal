"""Top-level Hypothesis strategy assembling grammar.py's expression
builders into complete generated kernels: array reads/writes, nested
if/else, ternary expressions, for loops (runtime bounds, nested,
accumulators, break/continue), covering the combinations Workstream 4
requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st

from .grammar import (
    DTYPES,
    KernelSpec,
    VarSpec,
    _bool_expr,
    _BuildContext,
    _numeric_expr,
)

_NUMPY_DTYPE = {
    "float32": np.float32,
    "int32": np.int32,
    "uint32": np.uint32,
    "bool_": np.bool_,
}


@dataclass
class GeneratedKernel:
    """A fully generated, ready-to-run differential test case."""

    spec: KernelSpec
    gpu_source: str
    cpu_source: str


def _numpy_dtype_for(dtype: str) -> np.dtype:
    return np.dtype(_NUMPY_DTYPE[dtype])


@st.composite
def _statement_block(
    draw, ctx: _BuildContext, array_names: list[str], depth: int, max_stmts: int
) -> list[str]:
    """Build a block of statements: assignments, if/else, and for loops
    over a runtime bound, drawn from the closed grammar. `array_names` is
    the pool of 1D array parameters available for indexed reads (indexed
    by the kernel's own `i` or a loop variable)."""
    lines: list[str] = []
    n_stmts = draw(st.integers(min_value=1, max_value=max_stmts))
    for _ in range(n_stmts):
        kind_choices = ["assign"]
        if depth < 2:
            kind_choices.append("if")
        if depth < 1:
            kind_choices.append("for")
        kind = draw(st.sampled_from(kind_choices))

        if kind == "assign":
            lines.extend(draw(_assign_stmt(ctx)))
        elif kind == "if":
            lines.extend(draw(_if_stmt(ctx, array_names, depth, max_stmts)))
        elif kind == "for":
            lines.extend(draw(_for_stmt(ctx, array_names, depth, max_stmts)))
    return lines


@st.composite
def _assign_stmt(draw, ctx: _BuildContext) -> list[str]:
    dtype = draw(st.sampled_from(DTYPES))
    # Prefer reassigning an existing variable of the same dtype half the
    # time (exercises phi/merge-heavy patterns); otherwise introduce a
    # fresh one. Critically, a freshly-introduced target is NOT added to
    # ctx.live_vars until AFTER its own initializing RHS expression is
    # drawn -- otherwise the RHS expression's variable pool would include
    # the not-yet-initialized target itself as a legal operand (`v3 = v3`
    # on first definition), which is exactly the "variable is not
    # defined" bytecode-analysis failure this ordering exists to prevent.
    existing = [v for v in ctx.live_vars if v.dtype == dtype]
    is_fresh = not existing or not draw(st.booleans())
    if is_fresh:
        target = VarSpec(name=ctx.fresh_name("v"), dtype=dtype)
    else:
        target = draw(st.sampled_from(existing))

    if dtype == "bool_":
        expr = draw(_bool_expr(ctx))
    else:
        expr = draw(_numeric_expr(dtype, ctx))

    if is_fresh:
        ctx.live_vars.append(target)
    return [f"{target.name} = {expr}"]


@st.composite
def _if_stmt(draw, ctx: _BuildContext, array_names, depth, max_stmts) -> list[str]:
    """Build an `if`/`if-else`. Variables introduced inside either arm are
    NOT carried into the parent scope's `live_vars` unless the SAME
    variable (by identity, not just name/dtype) was assigned in *both*
    arms of a two-armed if -- that is the only case where every path
    reaching the statement after the `if` has actually initialized it.
    A one-armed `if` (no `else`) never contributes any newly-introduced
    variable to the parent scope, since the `else`-less path skips it
    entirely. This mirrors real dataflow "definitely assigned" analysis
    and is what prevents generating a read of a variable that Numba's own
    bytecode analysis would (correctly) reject as possibly-undefined --
    the exact `NotDefinedError: Variable 'v3' is not defined` class of
    generator bug found and fixed while building this harness.
    """
    cond = draw(_bool_expr(ctx))
    before = ctx.snapshot()

    then_lines = draw(
        _statement_block(ctx, array_names, depth + 1, max(1, max_stmts - 1))
    )
    then_after = ctx.snapshot()
    ctx.restore(before)

    has_else = draw(st.booleans())
    out = [f"if {cond}:"]
    out.extend(f"    {line}" for line in then_lines)
    if has_else:
        else_lines = draw(
            _statement_block(ctx, array_names, depth + 1, max(1, max_stmts - 1))
        )
        else_after = ctx.snapshot()
        out.append("else:")
        out.extend(f"    {line}" for line in else_lines)
        # Merge: a variable is live after the if/else only if it appears
        # in both arms' resulting live-var sets (by name+dtype identity).
        else_names = {(v.name, v.dtype) for v in else_after}
        merged = [v for v in then_after if (v.name, v.dtype) in else_names]
        ctx.restore(merged if merged else before)
    else:
        # No else: the "skip the if" path never runs then_lines, so
        # nothing it introduced is safe to read afterward.
        ctx.restore(before)
    return out


@st.composite
def _for_stmt(draw, ctx: _BuildContext, array_names, depth, max_stmts) -> list[str]:
    """Build a `for` loop. The loop may execute zero times at runtime (a
    generated bound of 0, or a step that never reaches stop), so nothing
    the body introduces -- including the loop variable itself -- is safe
    to read after the loop; `ctx.live_vars` is restored to its pre-loop
    snapshot once the body is drawn, mirroring `_if_stmt`'s no-else case.
    """
    before = ctx.snapshot()
    loop_var = ctx.fresh_name("j")
    ctx.live_vars.append(VarSpec(name=loop_var, dtype="int32"))
    step = draw(st.sampled_from([1, -1, 2]))
    n_args = draw(st.sampled_from([1, 2, 3]))
    bound_name = draw(st.sampled_from(["n", "m"]))
    if n_args == 1:
        range_src = f"range({bound_name})"
    elif n_args == 2:
        range_src = f"range(0, {bound_name})"
    else:
        range_src = f"range(0, {bound_name}, {step})"

    # `before_body` is the snapshot immediately before the loop body is
    # drawn (but after the loop variable itself is added) -- this is the
    # ONLY set of variables genuinely safe to reference in a guard placed
    # BEFORE the body in the rendered output. Drawing the guard against
    # `ctx` after `body_lines` has already been drawn (which mutates
    # `ctx.live_vars` in place) would let the guard reference a variable
    # the body itself introduces, even though the guard executes first at
    # runtime -- exactly the "variable is not defined" class of bug this
    # snapshot exists to prevent (found by this harness before this fix).
    before_body = ctx.snapshot()
    body_lines = draw(
        _statement_block(ctx, array_names, depth + 1, max(1, max_stmts - 1))
    )
    if draw(st.booleans()) and body_lines:
        # Wrap the last body statement in a break/continue-guarded if,
        # exercising break/continue-with-partial-branch-updates.
        saved = ctx.snapshot()
        ctx.restore(before_body)
        guard_cond = draw(_bool_expr(ctx))
        ctx.restore(saved)
        kind = draw(st.sampled_from(["break", "continue"]))
        body_lines = [f"if {guard_cond}:", f"    {kind}", *body_lines]

    ctx.restore(before)
    out = [f"for {loop_var} in {range_src}:"]
    out.extend(f"    {line}" for line in body_lines)
    if not body_lines:
        out.append("    pass")
    return out


@st.composite
def generated_kernel(draw, n_arrays: int = 2, max_stmts: int = 4) -> GeneratedKernel:
    """Build one complete GeneratedKernel: N input arrays + one output
    array (all float32, for simplicity of the differential numeric
    comparison), two runtime scalar bounds (n, m) for loops, and a
    generated body."""
    ctx = _BuildContext()
    array_names = [f"arr{k}" for k in range(n_arrays)]
    array_params = [VarSpec(name=nm, dtype="float32") for nm in array_names]
    output = VarSpec(name="out", dtype="float32")

    # Seed the variable pool with per-lane array reads (out[i]-style reads
    # are not supported as expressions -- reads happen via a dedicated
    # "seed" assignment `x = arrN[i]`, added as the first statements).
    seed_lines = []
    for arr in array_names:
        local = VarSpec(name=ctx.fresh_name("x"), dtype="float32")
        ctx.live_vars.append(local)
        seed_lines.append(f"{local.name} = {arr}[i]")

    ctx.live_vars.append(VarSpec(name="n", dtype="int32"))
    ctx.live_vars.append(VarSpec(name="m", dtype="int32"))

    body_lines = draw(_statement_block(ctx, array_names, depth=0, max_stmts=max_stmts))

    # Final output write must be float32 (the output array's dtype).
    out_candidates = [v for v in ctx.live_vars if v.dtype == "float32"]
    if out_candidates:
        result_var = draw(st.sampled_from(out_candidates))
        write_line = f"{output.name}[i] = {result_var.name}"
    else:
        write_line = f"{output.name}[i] = {seed_lines[0].split(' = ', 1)[1]}"

    full_body = [*seed_lines, *body_lines, write_line]

    spec = KernelSpec(
        name="gen_kernel",
        array_params=[*array_params, output],
        scalar_params=[
            VarSpec(name="n", dtype="int32"),
            VarSpec(name="m", dtype="int32"),
        ],
        body_lines=full_body,
        output_param=output.name,
        uses_math=True,
    )

    from .grammar import render_cpu_reference, render_kernel

    return GeneratedKernel(
        spec=spec,
        gpu_source=render_kernel(spec),
        cpu_source=render_cpu_reference(spec),
    )
