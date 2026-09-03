"""Typed Numba IR -> Metal Shading Language backend.

This is a structured tree-walking code generator: it never does string
substitution on already-generated source text. It consumes:

- The structured control-flow tree from `numba_metal.compiler.structuring`
  (If/Loop/Break/Continue/Return/Seq/BasicBlockNode nodes reconstructed
  from Numba's basic-block CFG).
- Numba's typed IR statements/expressions within each basic block
  (`numba.core.ir.Assign`, `.SetItem`, `.StaticSetItem`, `.Expr` of kind
  binop/unary/call/getattr/getitem/cast/phi/static_getitem/pair_first/
  pair_second/exhaust_iter).
- The typemap produced by Numba's real type inference.

and emits MSL source text for a single `kernel void` function. Only
operations verified to lower to valid MSL are supported; anything else
raises UnsupportedFeatureError naming the exact IR node/op/type, per
project policy of no silent fallback.

Two SSA-specific patterns get dedicated handling rather than a literal
per-instruction translation, because a literal translation would either be
invalid MSL or drop information the structurer already consumed:

1. **Phi nodes.** MSL (like C) has no SSA phi; a phi target and all of its
   incoming values are unified to one shared MSL variable identifier via a
   union-find pre-pass (`_PhiUnifier`), so ordinary assignment on each
   incoming path already leaves the right value visible after the merge.
2. **`for x in range(...)`.** Numba lowers this to `getiter`/`iternext`/
   `pair_first`/`pair_second` plus a phi-carried loop variable. Rather than
   reconstruct an equivalent iterator protocol in MSL, the backend detects
   this exact shape once per loop and emits a native MSL `for` loop over
   the same bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba.core import ir
from numba.core import types as nb_types

from numba_metal.compiler import intrinsics
from numba_metal.compiler.frontend import TypedKernelIR
from numba_metal.compiler.structuring import (
    BasicBlockNode,
    BreakNode,
    ContinueNode,
    IfNode,
    LoopNode,
    Node,
    ReturnNode,
    Seq,
    structure_function,
)
from numba_metal.errors import UnsupportedFeatureError
from numba_metal.types import numba_scalar_to_msl, numpy_dtype_to_msl

_BINOP_MSL = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "%": "%",
    "==": "==",
    "!=": "!=",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "&": "&",
    "|": "|",
    "^": "^",
    "<<": "<<",
    ">>": ">>",
}

#: Supported `math.<name>` functions (task-mandated subset).
MATH_FUNCS = {"sqrt": "sqrt", "exp": "exp", "log": "log", "sin": "sin", "cos": "cos"}


@dataclass
class ArrayParamInfo:
    """Describes one 1D array kernel parameter for buffer binding."""

    name: str
    dtype: np.dtype
    ndim: int


@dataclass
class KernelSignatureInfo:
    """Classified kernel parameters, used by both MSL signature emission
    and the runtime dispatcher's argument-binding logic."""

    array_params: list[ArrayParamInfo] = field(default_factory=list)
    scalar_params: list[tuple[str, nb_types.Type]] = field(default_factory=list)
    param_order: list[tuple[str, str]] = field(default_factory=list)


def nb_scalar_dtype_to_numpy(ty: nb_types.Type) -> np.dtype:
    """Map a Numba scalar Type to the equivalent NumPy dtype."""
    mapping = {
        nb_types.float32: np.float32,
        nb_types.float16: np.float16,
        nb_types.int32: np.int32,
        nb_types.uint32: np.uint32,
        nb_types.int64: np.int64,
        nb_types.boolean: np.bool_,
    }
    for numba_ty, np_ty in mapping.items():
        if ty == numba_ty:
            return np.dtype(np_ty)
    raise UnsupportedFeatureError(f"Unsupported array element type {ty!r}.")


def _numba_fn_to_opstr(fn) -> str:
    for opstr, opfn in ir.BINOPS_TO_OPERATORS.items():
        if opfn is fn:
            return opstr
    for opstr, opfn in ir.UNARY_BUITINS_TO_OPERATORS.items():
        if opfn is fn:
            return opstr
    return getattr(fn, "__name__", str(fn))


class _UnionFind:
    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _unify_phis(blocks: dict[int, ir.Block]) -> dict[str, str]:
    """Union every phi target with each of its incoming vars; return a map
    from every SSA var name that participates in a phi to one canonical
    representative name (used as the MSL identifier for all of them)."""
    uf = _UnionFind()
    for block in blocks.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                if stmt.value.op == "phi":
                    for incoming in stmt.value.incoming_values:
                        uf.union(stmt.target.name, incoming.name)
    return uf


@dataclass
class _RangeCallInfo:
    start: str
    stop: str
    step: str


class _ForRangeLoopInfo:
    """Detected shape of a `for x in range(...)` loop, if the loop's exit
    test traces back to a `range()`-derived iterator in the expected shape.
    """

    def __init__(self, loop_var_ident: str, loop_var_name: str, bounds: _RangeCallInfo):
        self.loop_var_ident = loop_var_ident
        self.loop_var_name = loop_var_name
        self.bounds = bounds


class MSLFunctionBuilder:
    """Accumulates indented MSL source lines during structured code
    generation."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0

    def write(self, text: str) -> None:
        """Append one line at the current indentation level."""
        self._lines.append(("    " * self._indent) + text)

    def block(self):
        """Context manager: indent one level for the duration of the `with`."""
        return _IndentGuard(self)

    def source(self) -> str:
        """Return the accumulated MSL source text."""
        return "\n".join(self._lines) + "\n"


class _IndentGuard:
    def __init__(self, builder: MSLFunctionBuilder):
        self._b = builder

    def __enter__(self):
        self._b._indent += 1

    def __exit__(self, *exc):
        self._b._indent -= 1


class MSLKernelLowerer:
    """Lowers one TypedKernelIR to a complete MSL `kernel void` function."""

    def __init__(self, kernel_name: str, typed: TypedKernelIR):
        self.kernel_name = kernel_name
        self.typed = typed
        self.typemap = typed.typemap
        self.func_ir = typed.func_ir
        self.builder = MSLFunctionBuilder()
        self.sig = KernelSignatureInfo()
        self._globals: dict[str, object] = {}
        self._range_calls: dict[str, list[str]] = {}
        self._phi_uf: _UnionFind | None = None
        self._array_names: set[str] = set()
        self._scalar_names: set[str] = set()
        # Var names that exist only to implement Python's iterator protocol
        # (getiter/iternext/pair_first/pair_second results and the plain
        # copy-assigns chaining them) for loops recognized as `for x in
        # range(...)`. These never get an MSL declaration (their Numba
        # types -- iterator/Pair -- have no MSL representation) and their
        # bookkeeping is fully replaced by the native `for` loop header
        # numba-metal emits instead, so any leftover Assign statement
        # targeting one of them must be skipped rather than emitted.
        self._suppressed_names: set[str] = set()
        # Statement identities (by id()) to skip even though their target
        # variable is NOT suppressed -- currently only the per-iteration
        # `it = <pair_first-derived temp>` copy in a detected for-range
        # loop; see _suppress_loop_protocol_names.
        self._suppressed_copy_stmts: set[int] = set()

    def lower(self) -> str:
        """Run the full typed-IR-to-MSL lowering and return the generated
        `kernel void` function source (excluding the `#include`/`using`
        prelude, added by the caller)."""
        self._classify_params()
        self._phi_uf = _unify_phis(self.func_ir.blocks)
        self._collect_globals()
        self._collect_range_calls()
        entry = min(self.func_ir.blocks.keys())
        structured = structure_function(self.func_ir.blocks, entry)
        self._compute_suppressed_names(structured)
        self._emit_signature()
        with self.builder.block():
            self._declare_locals()
            self._emit_node(structured)
        self.builder.write("}")
        return self.builder.source()

    # -- parameters -----------------------------------------------------

    def _classify_params(self) -> None:
        for name, ty in zip(self.typed.arg_names, self.typed.arg_types, strict=True):
            if isinstance(ty, nb_types.Array):
                if ty.ndim != 1:
                    raise UnsupportedFeatureError(
                        f"Kernel argument {name!r} has {ty.ndim} dimensions; "
                        "numba-metal only supports 1D arrays as kernel "
                        "arguments in this MVP (use flattened indexing for "
                        "multi-dimensional data -- see docs/limitations.md)."
                    )
                np_dtype = nb_scalar_dtype_to_numpy(ty.dtype)
                self.sig.array_params.append(ArrayParamInfo(name, np_dtype, ty.ndim))
                self.sig.param_order.append(("array", name))
                self._array_names.add(name)
            elif isinstance(ty, (nb_types.Integer, nb_types.Float, nb_types.Boolean)):
                self.sig.scalar_params.append((name, ty))
                self.sig.param_order.append(("scalar", name))
                self._scalar_names.add(name)
            else:
                raise UnsupportedFeatureError(
                    f"Kernel argument {name!r} has unsupported type {ty!r}. "
                    "Supported argument types: 1D arrays of a supported "
                    "dtype, and scalar int32/uint32/int64/float32/bool."
                )

    def _emit_signature(self) -> None:
        params: list[str] = []
        buffer_index = 0
        for kind, name in self.sig.param_order:
            ident = f"arg_{name}"
            if kind == "array":
                info = next(a for a in self.sig.array_params if a.name == name)
                msl_ty = numpy_dtype_to_msl(info.dtype)
                params.append(f"device {msl_ty}* {ident} [[buffer({buffer_index})]]")
                buffer_index += 1
                params.append(f"constant uint& {ident}_size [[buffer({buffer_index})]]")
                buffer_index += 1
            else:
                _, ty = next(s for s in self.sig.scalar_params if s[0] == name)
                msl_ty = numba_scalar_to_msl(ty)
                params.append(f"constant {msl_ty}& {ident} [[buffer({buffer_index})]]")
                buffer_index += 1
        params.append("uint3 numba_metal_tid [[thread_position_in_grid]]")
        params.append("uint3 numba_metal_grid_size [[threads_per_grid]]")
        joined = ",\n    ".join(params)
        self.builder.write(f"kernel void {self.kernel_name}(")
        self.builder.write(f"    {joined})")
        self.builder.write("{")

    # -- local declarations -----------------------------------------------

    def _canonical(self, var_name: str) -> str:
        if self._phi_uf is not None:
            return self._phi_uf.find(var_name)
        return var_name

    @staticmethod
    def _ident(canonical_name: str) -> str:
        safe = "".join(
            ch if (ch.isalnum() or ch == "_") else "_" for ch in canonical_name
        )
        return "v_" + safe

    def _declare_locals(self) -> None:
        seen_canonical: dict[str, str] = {}
        for name, ty in self.typemap.items():
            if name in self._array_names or name in self._scalar_names:
                continue
            if name.startswith("arg.") and name[4:] in (
                self._array_names | self._scalar_names
            ):
                # Numba's frontend keeps a pre-SSA-renamed `arg.<name>`
                # typemap entry alongside the SSA-renamed `<name>` for
                # every parameter; it is never referenced by any emitted
                # statement (all real uses go through the SSA name), so it
                # must not be declared or type-checked.
                continue
            if name in self._range_call_target_names():
                continue
            if name in self._suppressed_names:
                continue
            if isinstance(ty, (nb_types.Omitted, nb_types.NoneType)):
                continue
            if isinstance(
                ty, (nb_types.FunctionType, nb_types.Function, nb_types.RangeType)
            ):
                continue
            if isinstance(ty, nb_types.RangeIteratorType):
                continue
            canonical = self._canonical(name)
            if canonical in seen_canonical:
                continue
            try:
                msl_ty = self._msl_type_for(ty)
            except UnsupportedFeatureError as exc:
                raise UnsupportedFeatureError(f"{exc} (variable {name!r})") from exc
            if msl_ty is None:
                continue
            ident = self._ident(canonical)
            self.builder.write(f"{msl_ty} {ident};")
            seen_canonical[canonical] = ident

    def _range_call_target_names(self) -> set[str]:
        names: set[str] = set()
        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                    if stmt.value.op == "call":
                        callee = self._resolve_global(stmt.value.func.name)
                        if callee is range:
                            names.add(stmt.target.name)
        return names

    def _msl_type_for(self, ty: nb_types.Type) -> str | None:
        if isinstance(ty, nb_types.Literal):
            ty = ty.literal_type
        if isinstance(ty, nb_types.UniTuple) and ty.dtype == nb_types.int64:
            if ty.count == 2:
                return "long2"
            raise UnsupportedFeatureError(
                f"Unsupported tuple type {ty!r}: only 2-tuples of int64 "
                "(from metal.grid(2)) are supported."
            )
        if isinstance(ty, nb_types.Pair):
            return None  # iternext pair results never materialize in MSL
        if isinstance(ty, nb_types.Module):
            return None  # module refs (e.g. `metal`, `math`) never materialize
        if ty == nb_types.float64:
            # Python float literals and float-producing operations (e.g.
            # `/` true division) infer as float64 under ordinary Numba
            # typing, same as CPython. MSL's `float` is 32-bit and Apple
            # GPU float64 support is not verified (see
            # docs/limitations.md), so numba-metal narrows float64
            # *local intermediate values* to float32 rather than reject
            # every kernel that contains a literal or a division -- this
            # is a documented, deliberate precision narrowing, distinct
            # from silently accepting a float64 *array* dtype (rejected
            # in numba_scalar_to_msl/numpy_dtype_to_msl, which this local
            # -only narrowing does not affect).
            return "float"
        if isinstance(ty, (nb_types.Integer, nb_types.Float, nb_types.Boolean)):
            return numba_scalar_to_msl(ty)
        if isinstance(ty, nb_types.Array):
            raise UnsupportedFeatureError(
                "Local array variables are not supported inside kernels."
            )
        raise UnsupportedFeatureError(
            f"Unsupported local variable type {ty!r}; no MSL representation."
        )

    # -- globals pre-scan --------------------------------------------------

    def _collect_globals(self) -> None:
        # Two shapes resolve to a callable "global": a direct
        # Global/FreeVar load (`from numba_metal import metal; metal.grid`
        # imported as just `grid`), and a module-level Global/FreeVar
        # followed by a `getattr` (the `metal.grid(...)` attribute-access
        # form used in real kernels). Both are pre-scanned here so `_call`
        # can resolve the callee regardless of which form the user wrote.
        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                if not isinstance(stmt, ir.Assign):
                    continue
                if isinstance(stmt.value, (ir.Global, ir.FreeVar)):
                    self._globals[stmt.target.name] = stmt.value.value
                elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "getattr":
                    base = self._globals.get(stmt.value.value.name)
                    if base is not None and hasattr(base, stmt.value.attr):
                        self._globals[stmt.target.name] = getattr(base, stmt.value.attr)

    def _resolve_global(self, var_name: str):
        return self._globals.get(var_name)

    def _collect_range_calls(self) -> None:
        """Pre-scan (before any MSL emission) every `range(...)` call site
        so `_detect_for_range` and `_compute_suppressed_names` can use
        `self._range_calls` regardless of block visitation order."""
        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                if not (
                    isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)
                ):
                    continue
                if stmt.value.op != "call":
                    continue
                callee = self._resolve_global(stmt.value.func.name)
                if callee is range:
                    self._range_calls[stmt.target.name] = [
                        self._read(a) for a in stmt.value.args
                    ]

    def _compute_suppressed_names(self, node: Node) -> None:
        """Walk the structured tree once, and for every LoopNode that
        successfully matches the for-range pattern, mark its loop-protocol
        bookkeeping vars (getiter/iternext/pair_first/pair_second results
        and the header's phi copy of the iterator) as suppressed so
        `_emit_stmt` skips assignments to them."""
        if isinstance(node, Seq):
            for item in node.items:
                self._compute_suppressed_names(item)
        elif isinstance(node, IfNode):
            self._compute_suppressed_names(node.then_branch)
            if node.else_branch is not None:
                self._compute_suppressed_names(node.else_branch)
        elif isinstance(node, LoopNode):
            for_info = self._detect_for_range(node)
            if for_info is not None:
                self._suppress_loop_protocol_names(node, for_info.loop_var_name)
            self._compute_suppressed_names(node.body)

    def _suppress_loop_protocol_names(self, node: LoopNode, loop_var_name: str) -> None:
        header_block = self.func_ir.blocks[node.header_label]
        for stmt in header_block.body:
            if not (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)):
                continue
            if stmt.value.op in ("iternext", "pair_first", "pair_second", "getiter"):
                self._suppressed_names.add(stmt.target.name)
        # The `getiter()` call itself lives in the block that jumps into
        # the loop header (e.g. block 66 for header 156), not in the
        # header block, so it is not caught by the scan above; find it
        # directly from the range() call site instead.
        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                if (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "getiter"
                    and stmt.value.value.name in self._range_calls
                ):
                    self._suppressed_names.add(stmt.target.name)
        # Copy assignments chain loop-protocol values through plain
        # `ir.Var` assigns (not `op=="phi"` exprs -- see
        # _trace_to_range_call), both to seed the iterator into the
        # header's phi slot (`$phiHDR.0 = $NNget_iter.M`) and to bind the
        # loop variable each iteration (`it = $phiNNN.1`, i.e. a copy of
        # the pair_first result). Propagate suppression transitively
        # across such copies (fixed point) so both are caught regardless
        # of how many hops separate them from the original getiter/
        # pair_first/pair_second target.
        changed = True
        while changed:
            changed = False
            for block in self.func_ir.blocks.values():
                for stmt in block.body:
                    if not (
                        isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var)
                    ):
                        continue
                    if stmt.target.name == loop_var_name:
                        # This is the copy that binds the user-visible loop
                        # variable each iteration (e.g. `it = $phiNNN.1`).
                        # The native `for` loop header already declares
                        # and updates `it` directly, so this specific copy
                        # statement is redundant and must not be emitted
                        # -- but `it` itself must stay declared/assignable,
                        # so it is excluded from _suppressed_names (which
                        # also blocks declaration) and instead recorded as
                        # a suppressed *statement* by object identity.
                        self._suppressed_copy_stmts.add(id(stmt))
                        continue
                    if (
                        stmt.value.name in self._suppressed_names
                        and stmt.target.name not in self._suppressed_names
                    ):
                        self._suppressed_names.add(stmt.target.name)
                        changed = True

    # -- structured-node emission ------------------------------------------

    def _emit_node(self, node: Node) -> None:
        if isinstance(node, Seq):
            for item in node.items:
                self._emit_node(item)
        elif isinstance(node, BasicBlockNode):
            self._emit_basic_block(node)
        elif isinstance(node, IfNode):
            self._emit_if(node)
        elif isinstance(node, LoopNode):
            self._emit_loop(node)
        elif isinstance(node, ReturnNode):
            self.builder.write("return;")
        elif isinstance(node, BreakNode):
            self.builder.write("break;")
        elif isinstance(node, ContinueNode):
            self.builder.write("continue;")
        else:  # pragma: no cover - defensive
            raise UnsupportedFeatureError(
                f"Internal error: unknown structured node {type(node)!r}."
            )

    def _emit_if(self, node: IfNode) -> None:
        cond_expr = self._read(node.cond)
        self.builder.write(f"if ({cond_expr}) {{")
        with self.builder.block():
            self._emit_node(node.then_branch)
        if node.else_branch is not None:
            self.builder.write("} else {")
            with self.builder.block():
                self._emit_node(node.else_branch)
        self.builder.write("}")

    def _emit_loop(self, node: LoopNode) -> None:
        for_info = self._detect_for_range(node)
        if for_info is not None:
            start, stop, step = (
                for_info.bounds.start,
                for_info.bounds.stop,
                for_info.bounds.step,
            )
            var = for_info.loop_var_ident
            self.builder.write(
                f"for ({var} = {start}; "
                f"({step} > 0) ? ({var} < {stop}) : ({var} > {stop}); "
                f"{var} += {step}) {{"
            )
            with self.builder.block():
                self._emit_node(node.body)
            self.builder.write("}")
            return

        self.builder.write("while (true) {")
        with self.builder.block():
            cond_expr = self._read(node.exit_cond)
            if node.exit_cond_negated:
                self.builder.write(f"if (!({cond_expr})) {{ break; }}")
            else:
                self.builder.write(f"if ({cond_expr}) {{ break; }}")
            self._emit_node(node.body)
        self.builder.write("}")

    def _detect_for_range(self, node: LoopNode):
        """Recognize the getiter/iternext/pair_first/pair_second pattern
        that Numba emits for `for x in range(...)` and, if matched, return
        a `_ForRangeLoopInfo` describing the equivalent MSL `for` bounds.
        Returns None if the loop is not in this exact shape (e.g. a
        different iterable), which is otherwise unsupported and falls back
        to the generic while(true) form only for genuinely while-shaped
        loops; range-shaped loops that fail to match raise an error instead
        of silently emitting a wrong translation.
        """
        header_block = self.func_ir.blocks[node.header_label]
        iternext_val_name = None
        pair_second_name = None
        for stmt in header_block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                if stmt.value.op == "iternext":
                    iternext_val_name = stmt.value.value.name
                if stmt.value.op == "pair_second":
                    pair_second_name = stmt.target.name
        if iternext_val_name is None:
            return None
        if pair_second_name is None or self._canonical(
            pair_second_name
        ) != self._canonical(self._exit_cond_source_name(node)):
            return None

        range_target = self._trace_to_range_call(iternext_val_name)
        if range_target is None:
            return None
        args = self._range_calls[range_target]
        if len(args) == 1:
            bounds = _RangeCallInfo(start="0", stop=args[0], step="1")
        elif len(args) == 2:
            bounds = _RangeCallInfo(start=args[0], stop=args[1], step="1")
        elif len(args) == 3:
            bounds = _RangeCallInfo(start=args[0], stop=args[1], step=args[2])
        else:
            return None

        pair_first_name = self._find_pair_first_target(header_block, iternext_val_name)
        if pair_first_name is None:
            return None
        loop_var_canonical = self._canonical(pair_first_name)
        return _ForRangeLoopInfo(
            self._ident(loop_var_canonical), pair_first_name, bounds
        )

    def _exit_cond_source_name(self, node: LoopNode) -> str:
        return node.exit_cond.name

    def _find_pair_first_target(self, header_block, iternext_val_name: str):
        # First find the `iternext()` call's own result var (the Pair),
        # which `pair_first`/`pair_second` both read from -- NOT the same
        # as `iternext_val_name` (that is iternext's *input*, the
        # iterator itself).
        iternext_result_name = None
        for stmt in header_block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                if (
                    stmt.value.op == "iternext"
                    and stmt.value.value.name == iternext_val_name
                ):
                    iternext_result_name = stmt.target.name
        if iternext_result_name is None:
            return None

        first_target = None
        for stmt in header_block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                if (
                    stmt.value.op == "pair_first"
                    and stmt.value.value.name == iternext_result_name
                ):
                    first_target = stmt.target.name
        if first_target is None:
            return None
        # The user-visible loop variable (`it`) is whatever the pair_first
        # result eventually gets copied into, via a chain of plain
        # var-to-var copy assigns (commonly two hops: pair_first's target
        # -> the header's per-iteration phi slot -> `it` in the loop body
        # entry block). These are ir.Var assigns, not phi exprs, so the
        # union-find doesn't unify them (see _trace_to_range_call); follow
        # the copy chain to its end instead of assuming a single hop.
        current = first_target
        for _ in range(len(self.func_ir.blocks) + 1):
            next_hop = None
            for block in self.func_ir.blocks.values():
                for stmt in block.body:
                    if (
                        isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Var)
                        and stmt.value.name == current
                    ):
                        next_hop = stmt.target.name
                        break
                if next_hop is not None:
                    break
            if next_hop is None:
                return current
            current = next_hop
        return current

    def _trace_to_range_call(self, var_name: str) -> str | None:
        # var_name is what `iternext` reads. Numba emits the iterator
        # variable feeding a loop header's `iternext` as a plain copy
        # assignment (`$phi92.0 = $90get_iter.5`, NOT an `op=="phi"` expr,
        # despite the "$phi" name prefix Numba's IR pretty-printer uses for
        # it) from the block that jumps into the header. Follow one hop of
        # such plain Var-to-Var copy assignment to find the real
        # `getiter()` result, then check if that traces to a tracked
        # range() call.
        seen: set[str] = set()
        current = var_name
        for _ in range(len(self.func_ir.blocks) + 1):
            if current in seen:
                return None
            seen.add(current)
            found_next = None
            for block in self.func_ir.blocks.values():
                for stmt in block.body:
                    if not isinstance(stmt, ir.Assign) or stmt.target.name != current:
                        continue
                    if isinstance(stmt.value, ir.Expr) and stmt.value.op == "getiter":
                        src = stmt.value.value.name
                        return src if src in self._range_calls else None
                    if isinstance(stmt.value, ir.Var):
                        found_next = stmt.value.name
            if found_next is None:
                return None
            current = found_next
        return None

    def _emit_basic_block(self, node: BasicBlockNode) -> None:
        for stmt in node.body:
            self._emit_stmt(stmt)

    # -- statements ----------------------------------------------------

    def _emit_stmt(self, stmt) -> None:
        if id(stmt) in self._suppressed_copy_stmts:
            return
        if isinstance(stmt, ir.Assign):
            self._emit_assign(stmt)
        elif isinstance(stmt, ir.SetItem):
            target = self._read(stmt.target)
            index = self._read(stmt.index)
            value = self._read(stmt.value)
            self.builder.write(f"{target}[{index}] = {value};")
        elif isinstance(stmt, ir.StaticSetItem):
            target = self._read(stmt.target)
            value = self._read(stmt.value)
            self.builder.write(f"{target}[{stmt.index}] = {value};")
        elif isinstance(stmt, ir.Del):
            pass
        else:
            raise UnsupportedFeatureError(
                f"Unsupported IR statement {type(stmt).__name__}: {stmt}"
            )

    def _emit_assign(self, stmt: ir.Assign) -> None:
        target_name = stmt.target.name
        value = stmt.value

        if target_name in self._suppressed_names:
            return  # pure for-range loop-protocol bookkeeping; see
            # _compute_suppressed_names / _suppress_loop_protocol_names.
        if isinstance(value, ir.Arg):
            return  # bound to the MSL parameter identifier already
        if isinstance(value, (ir.Global, ir.FreeVar)):
            return  # resolved at call sites via self._globals
        if isinstance(value, ir.Var):
            self._assign(target_name, self._read(value))
            return
        if isinstance(value, ir.Const):
            self._assign(target_name, self._const_text(value.value, target_name))
            return
        if isinstance(value, ir.Expr):
            self._emit_expr_assign(target_name, value)
            return
        raise UnsupportedFeatureError(
            f"Unsupported assignment value {type(value).__name__} for "
            f"{target_name}: {value}"
        )

    def _assign(self, target_name: str, expr_text: str) -> None:
        if target_name not in self.typemap:
            return
        ty = self.typemap[target_name]
        if isinstance(ty, (nb_types.Omitted, nb_types.NoneType)):
            return
        if isinstance(
            ty,
            (
                nb_types.FunctionType,
                nb_types.Function,
                nb_types.RangeType,
                nb_types.Pair,
            ),
        ):
            return
        canonical = self._canonical(target_name)
        ident = self._ident(canonical)
        self.builder.write(f"{ident} = {expr_text};")

    # -- expressions -----------------------------------------------------

    def _emit_expr_assign(self, target_name: str, expr: ir.Expr) -> None:
        op = expr.op
        if op == "binop" or op == "inplace_binop":
            self._assign(target_name, self._binop(expr))
        elif op == "unary":
            self._assign(target_name, self._unary(expr))
        elif op == "cast":
            self._assign(target_name, self._read(expr.value))
        elif op == "getitem":
            self._assign(
                target_name, f"{self._read(expr.value)}[{self._read(expr.index)}]"
            )
        elif op == "static_getitem":
            self._assign(target_name, f"{self._read(expr.value)}[{expr.index}]")
        elif op == "getattr":
            if target_name in self._globals:
                return  # resolved as a callable (e.g. metal.grid); no MSL emitted
            self._assign(target_name, self._getattr(expr))
        elif op == "call":
            self._call(target_name, expr)
        elif op == "phi":
            pass  # handled by union-find; no MSL emitted for the phi itself
        elif op == "exhaust_iter":
            # Numba inserts `exhaust_iter` when unpacking a fixed-size
            # tuple (`x, y = metal.grid(2)`); it is an identity copy of
            # the tuple value that the following static_getitem(s) index
            # into. It is NOT part of the getiter/iternext for-loop
            # protocol despite the similar-sounding name.
            self._assign(target_name, self._read(expr.value))
        elif op in ("pair_first", "pair_second", "iternext", "getiter"):
            pass  # consumed structurally by for-range loop detection
        elif op == "build_tuple":
            raise UnsupportedFeatureError(
                "Tuple construction is only supported as the direct result "
                "of metal.grid(2); general tuple literals are unsupported."
            )
        else:
            raise UnsupportedFeatureError(f"Unsupported IR expression op {op!r}.")

    def _binop(self, expr: ir.Expr) -> str:
        opname = _numba_fn_to_opstr(expr.fn)
        lhs = self._read(expr.lhs)
        rhs = self._read(expr.rhs)
        if opname == "**":
            return f"pow(float({lhs}), float({rhs}))"
        if opname == "//":
            lhs_ty = self.typemap.get(expr.lhs.name)
            rhs_ty = self.typemap.get(expr.rhs.name)
            both_int = isinstance(lhs_ty, nb_types.Integer) and isinstance(
                rhs_ty, nb_types.Integer
            )
            if both_int:
                # Native MSL integer division truncates toward zero; Python
                # `//` floors toward negative infinity. These agree for
                # non-negative operands (the only case exercised by the
                # supported kernels/benchmarks, which use `//` for
                # non-negative flattened-index arithmetic) and are
                # documented to differ for negative operands -- see
                # docs/limitations.md.
                return f"({lhs} / {rhs})"
            return f"floor({lhs} / {rhs})"
        if opname == "/":
            # Python true division always produces a float, even for two
            # integer operands (`4 / 3 == 1.333...`), but MSL's `/` on two
            # integer-typed operands truncates like C. Force a float
            # context to match Python/Numba's typing of this expression
            # (which is why the *result* var was declared `float` by the
            # float64-narrowing rule in _msl_type_for).
            lhs_ty = self.typemap.get(expr.lhs.name)
            rhs_ty = self.typemap.get(expr.rhs.name)
            if isinstance(lhs_ty, nb_types.Integer) or isinstance(
                rhs_ty, nb_types.Integer
            ):
                return f"(float({lhs}) / float({rhs}))"
            return f"({lhs} / {rhs})"
        if opname not in _BINOP_MSL:
            raise UnsupportedFeatureError(f"Unsupported binary operator {opname!r}.")
        return f"({lhs} {_BINOP_MSL[opname]} {rhs})"

    def _unary(self, expr: ir.Expr) -> str:
        opname = _numba_fn_to_opstr(expr.fn)
        val = self._read(expr.value)
        if opname in ("-", "neg"):
            return f"(-{val})"
        if opname in ("not", "not_"):
            return f"(!{val})"
        if opname in ("+", "pos"):
            return f"(+{val})"
        raise UnsupportedFeatureError(f"Unsupported unary operator {opname!r}.")

    def _getattr(self, expr: ir.Expr) -> str:
        attr = expr.attr
        val_name = expr.value.name
        val_ty = self.typemap.get(val_name)
        if isinstance(val_ty, nb_types.Array) and attr == "size":
            if val_name in self._array_names:
                return f"arg_{val_name}_size"
        raise UnsupportedFeatureError(
            f"Unsupported attribute access `.{attr}` on {val_ty!r}. Only "
            "`.size` on a 1D kernel-argument array is supported."
        )

    def _call(self, target_name: str, expr: ir.Expr) -> None:
        callee = self._resolve_global(expr.func.name)
        args = [self._read(a) for a in expr.args]

        if callee is bool:
            self._assign(target_name, args[0])
            return
        if callee is intrinsics.grid or callee is intrinsics.gridsize:
            self._assign(target_name, self._grid_expr(callee, expr))
            return
        if callee is range:
            self._range_calls[target_name] = args
            return
        if callee is abs:
            self._assign(target_name, f"abs({args[0]})")
            return
        if callee is min:
            self._assign(target_name, f"min({args[0]}, {args[1]})")
            return
        if callee is max:
            self._assign(target_name, f"max({args[0]}, {args[1]})")
            return
        if callee is float:
            self._assign(target_name, f"float({args[0]})")
            return
        if callee is int:
            self._assign(target_name, f"int({args[0]})")
            return

        if callee is not None and getattr(callee, "__module__", None) == "math":
            fname = callee.__name__
            if fname in MATH_FUNCS:
                self._assign(target_name, f"{MATH_FUNCS[fname]}({args[0]})")
                return
            raise UnsupportedFeatureError(
                f"Unsupported math function math.{fname}(); supported: "
                f"{', '.join('math.' + n for n in MATH_FUNCS)}."
            )

        raise UnsupportedFeatureError(
            f"Unsupported function call to {callee!r} in kernel."
        )

    def _grid_expr(self, callee, expr: ir.Expr) -> str:
        ndim_var = expr.args[0]
        ndim_ty = self.typemap.get(ndim_var.name)
        if not isinstance(ndim_ty, nb_types.IntegerLiteral):
            raise UnsupportedFeatureError(
                "metal.grid()/gridsize() require a literal integer ndim "
                "argument known at compile time."
            )
        ndim = ndim_ty.literal_value
        source = (
            "numba_metal_grid_size"
            if callee is intrinsics.gridsize
            else "numba_metal_tid"
        )
        if ndim == 1:
            return f"long({source}.x)"
        if ndim == 2:
            return f"long2(long({source}.x), long({source}.y))"
        raise UnsupportedFeatureError(
            f"metal.grid(ndim)/gridsize(ndim) only support ndim in (1, 2); got {ndim}."
        )

    def _read(self, var: ir.Var) -> str:
        name = var.name
        if name in self._array_names or name in self._scalar_names:
            return f"arg_{name}"
        if name in self.typemap:
            canonical = self._canonical(name)
            return self._ident(canonical)
        raise UnsupportedFeatureError(f"Reference to undeclared variable {name!r}.")

    def _const_text(self, py_val, target_name: str) -> str:
        if isinstance(py_val, bool):
            return "true" if py_val else "false"
        if isinstance(py_val, int):
            return str(py_val)
        if isinstance(py_val, float):
            if py_val != py_val:
                return "NAN"
            if py_val == float("inf"):
                return "INFINITY"
            if py_val == float("-inf"):
                return "-INFINITY"
            return repr(py_val) + "f"
        if py_val is None:
            return "0"
        raise UnsupportedFeatureError(f"Unsupported constant value {py_val!r}.")
