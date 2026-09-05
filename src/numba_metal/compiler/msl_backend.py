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

1. **Phi nodes.** MSL (like C) has no SSA phi. Phi elimination is handled
   by the isolated `numba_metal.compiler.dessa` pass, which computes
   edge-correct parallel-copy assignments (never aliases two SSA variables
   to one shared identifier -- see `dessa.py`'s module docstring for why
   that would be unsound). Every SSA variable, including every phi target,
   gets its own independent MSL declaration; this backend only asks
   `DeSSAResult` for (a) which names are phi targets that must be declared
   but never assigned via their `phi` statement itself, and (b) which
   `Copy` assignments to emit immediately after each basic block's own
   statements.
2. **`for x in range(...)`.** Numba lowers this to `getiter`/`iternext`/
   `pair_first`/`pair_second` plus a phi-carried loop variable. Rather than
   reconstruct an equivalent iterator protocol in MSL, the backend detects
   this exact shape once per loop and emits a native MSL `for` loop over
   the same bounds.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

import numpy as np
from numba.core import ir
from numba.core import types as nb_types

from numba_metal.compiler import intrinsics
from numba_metal.compiler.dessa import DeSSAPass, DeSSAResult
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

#: Process-wide monotonic counter for @metal.device_func MSL function
#: names, mirroring pipeline.py's own `_next_kernel_name` counter but
#: kept independent to avoid a circular import (pipeline.py already
#: imports this module).
_device_function_name_counter = itertools.count()
_device_function_name_lock = threading.Lock()


def _next_device_function_name(func_name: str) -> str:
    with _device_function_name_lock:
        n = next(_device_function_name_counter)
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in func_name)
    return f"nbmtl_devfn_{safe}_{n}"


#: metal.atomic_min()/atomic_max(): native MSL atomic_fetch_min/max exist
#: for int32/uint32 but NOT for float32 (verified directly against
#: Apple's Metal compiler -- "no matching function," a permanent
#: MSL-language limitation for every Apple GPU family, not a
#: device-capability gap); float32 is lowered to a compare-and-swap
#: retry loop instead. See docs/architecture.md.
_ATOMIC_MINMAX_OPS: dict[object, str] = {
    intrinsics.atomic_min: "min",
    intrinsics.atomic_max: "max",
}

#: All metal.atomic_<op>() intrinsics handled by _emit_atomic_fetch_op
#: (fetch-and-modify ops returning the previous value; excludes
#: atomic_compare_exchange, which returns a (old_value, success) tuple
#: and is handled separately by _emit_atomic_compare_exchange).
_ATOMIC_FETCH_OP_MSL: dict[object, str] = {
    intrinsics.atomic_add: "fetch_add",
    intrinsics.atomic_sub: "fetch_sub",
    intrinsics.atomic_exchange: "exchange",
    intrinsics.atomic_min: "fetch_min",
    intrinsics.atomic_max: "fetch_max",
}


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
    # `metal.shared_array()` declarations, in the exact order their
    # `[[threadgroup(n)]]` attribute index was assigned in
    # `_emit_signature` -- the dispatcher must call
    # `setThreadgroupMemoryLength:atIndex:` for each one (with `n` equal
    # to its position in this list) before every dispatch, or the
    # threadgroup memory the kernel reads/writes is simply unallocated
    # (silently reads/writes as zero, with no error from Metal -- found
    # by direct testing while implementing this feature).
    threadgroup_arrays: list[_ThreadgroupArrayInfo] = field(default_factory=list)


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


def _reject_unsupported_while_body(node: Node) -> None:
    """Raise UnsupportedFeatureError if `node` (a rotated-`while` loop's
    body region, per `_emit_loop`'s docstring) contains anything beyond
    the one verified-correct shape: a straight-line sequence of basic
    blocks ending in the trivial back-edge block (empty statements, one
    unconditional ContinueNode). Any IfNode, or any BreakNode/ContinueNode
    other than that single trivial one, means this loop's real per-
    iteration control flow is more complex than what this backend's
    rotated-while lowering has been verified correct for -- see
    `_emit_loop`'s comment for the confirmed-wrong-output cases this
    guards against (break/continue nested inside an if within a while
    body)."""
    if isinstance(node, Seq):
        for item in node.items:
            _reject_unsupported_while_body(item)
        return
    if isinstance(node, BasicBlockNode):
        return
    if isinstance(node, ContinueNode):
        return  # the one allowed occurrence: the trivial back-edge block
    raise UnsupportedFeatureError(
        "Unsupported `while` loop shape: a `while` loop's body may only "
        "contain a straight-line sequence of statements (no nested "
        "`if`/`else`, `break`, or `continue`) in this MVP -- see "
        "docs/limitations.md. Restructure the loop, or use `for x in "
        "range(...)` if applicable."
    )


def _numba_fn_to_opstr(fn) -> str:
    for opstr, opfn in ir.BINOPS_TO_OPERATORS.items():
        if opfn is fn:
            return opstr
    for opstr, opfn in ir.UNARY_BUITINS_TO_OPERATORS.items():
        if opfn is fn:
            return opstr
    return getattr(fn, "__name__", str(fn))


@dataclass
class _RangeCallInfo:
    start: str
    stop: str
    step: str


@dataclass
class _LocalArrayInfo:
    """A `metal.local_array(shape, dtype)` call site: declared as an
    ordinary MSL function-body array, private to the calling thread."""

    target_name: str
    ident: str
    msl_elem_ty: str
    count: int


@dataclass
class _ThreadgroupArrayInfo:
    """A `metal.shared_array(shape, dtype)` call site: declared as a
    `threadgroup`-qualified kernel-function parameter (MSL does not
    permit `threadgroup`-qualified locals inside a function body), shared
    by every thread in the same threadgroup."""

    target_name: str
    ident: str
    msl_elem_ty: str
    count: int
    elem_itemsize: int

    @property
    def byte_size(self) -> int:
        return self.count * self.elem_itemsize


class _ForRangeLoopInfo:
    """Detected shape of a `for x in range(...)` loop, if the loop's exit
    test traces back to a `range()`-derived iterator in the expected shape.
    """

    def __init__(
        self,
        loop_var_ident: str,
        loop_var_name: str,
        bounds: _RangeCallInfo,
        range_target: str,
    ):
        self.loop_var_ident = loop_var_ident
        self.loop_var_name = loop_var_name
        self.bounds = bounds
        # The SSA name of the `range(...)` call feeding this specific
        # loop's iterator, used to scope iterator-protocol suppression to
        # this loop only (see _suppress_loop_protocol_names).
        self.range_target = range_target


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

    def __init__(
        self,
        kernel_name: str,
        typed: TypedKernelIR,
        *,
        device_function: bool = False,
        return_type: nb_types.Type | None = None,
        device_function_cache: dict | None = None,
    ):
        # `device_function=True` selects the small set of differences
        # needed to emit a real, standalone MSL function (`<msl_ty> name
        # (params) { ...; return expr; }`) instead of a `kernel void`:
        # `_classify_params` accepts only scalar arguments (no
        # device-buffer binding, no thread-position parameters -- a
        # device function has no independent notion of "which thread",
        # it only ever runs as part of whatever kernel called it),
        # `_emit_signature` emits an ordinary C-style function signature,
        # and `ReturnNode` emits `return <value>;` instead of the
        # kernel-only bare `return;` (kernels never return a value; see
        # `_emit_node`). Every other codegen path (statement/expression
        # walking, control-flow structuring, de-SSA) is identical and
        # fully shared -- this is deliberately NOT a second, duplicated
        # lowerer class, to avoid the two ever silently drifting apart.
        self.device_function = device_function
        self.device_function_return_type = return_type
        # Shared across one kernel's full compilation (including every
        # device function it calls, transitively): maps
        # (original_py_func, arg_types) -> already-emitted MSL function
        # name, so the same device-function+signature pair compiled from
        # multiple call sites (or called by more than one kernel in the
        # same process) is only ever lowered to MSL once. Also used to
        # detect direct or transitive recursion (see
        # `_compile_device_function`).
        self._device_function_cache: dict = (
            device_function_cache if device_function_cache is not None else {}
        )
        # MSL source for every device function this lowering transitively
        # needed, in the order they were first compiled (dependencies
        # before dependents is not required in C/MSL as long as every
        # function is declared before its own body is emitted at the top
        # level, which this ordering naturally satisfies since a callee
        # is always fully compiled, by recursion, before its caller's own
        # call-site code is emitted).
        self.device_function_sources: list[str] = []
        self.kernel_name = kernel_name
        self.typed = typed
        self.typemap = typed.typemap
        self.func_ir = typed.func_ir
        self.builder = MSLFunctionBuilder()
        self.sig = KernelSignatureInfo()
        self._globals: dict[str, object] = {}
        self._range_calls: dict[str, list[str]] = {}
        self._dessa: DeSSAResult | None = None
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
        # Monotonic counter for the range-bound snapshot temporaries
        # `_emit_loop` introduces (v__range_start_N/stop_N/step_N), so
        # that a kernel with multiple/nested for-range loops gets a
        # distinct, non-colliding set of MSL identifiers for each one.
        self._range_snapshot_counter = 0
        # Threadgroup ("shared") memory declarations collected from
        # metal.shared_array() calls found while pre-scanning the kernel
        # body (see _collect_special_arrays); emitted as
        # `threadgroup <type> <ident>[<count>]` kernel-function parameters
        # (MSL's convention for threadgroup-address-space arrays -- they
        # cannot be declared as ordinary function-body locals).
        self._threadgroup_arrays: list[_ThreadgroupArrayInfo] = []
        # metal.local_array() call sites, keyed by SSA target name; each
        # gets its own ordinary MSL function-body array declaration (see
        # _declare_locals) instead of going through the generic
        # scalar/tuple _msl_type_for path.
        self._local_arrays: dict[str, _LocalArrayInfo] = {}
        # SSA target names of metal.shared_array() calls, keyed the same
        # way as _local_arrays but resolved via self._threadgroup_arrays
        # (declared once, as a kernel parameter -- see above -- not
        # per-occurrence).
        self._threadgroup_array_targets: dict[str, _ThreadgroupArrayInfo] = {}
        self._special_array_counter = 0
        # SSA name -> (old_value_ident, success_ident) for every
        # metal.atomic_compare_exchange(...) call result (and every
        # identity-copy of one produced by tuple-unpacking's
        # exhaust_iter) -- see _emit_atomic_compare_exchange and the
        # static_getitem/exhaust_iter branches in _emit_expr_assign.
        self._atomic_cas_results: dict[str, tuple[str, str]] = {}
        # SSA names structurally typed as a 2-tuple of (atomic-eligible
        # scalar, bool) -- populated by _collect_atomic_cas_targets and
        # consulted by _declare_locals to skip the generic per-name MSL
        # declaration path for them (see that pre-scan method's
        # docstring for why this structural type check, rather than
        # call-graph tracing, is sufficient and exact).
        self._atomic_cas_target_names: set[str] = set()

    def lower(self) -> str:
        """Run the full typed-IR-to-MSL lowering and return the generated
        `kernel void` function source (excluding the `#include`/`using`
        prelude, added by the caller)."""
        self._classify_params()
        self._collect_globals()
        self._collect_special_arrays()
        self._collect_atomic_cas_targets()
        self._collect_range_calls()
        entry = min(self.func_ir.blocks.keys())
        structured = structure_function(self.func_ir.blocks, entry)
        self._dessa = DeSSAPass(self.func_ir.blocks).run(structured)
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
                    kind = (
                        "@metal.device_func argument"
                        if self.device_function
                        else ("Kernel argument")
                    )
                    raise UnsupportedFeatureError(
                        f"{kind} {name!r} has {ty.ndim} dimensions; "
                        "numba-metal only supports 1D arrays as kernel or "
                        "device-function arguments in this MVP (use "
                        "flattened indexing for multi-dimensional data -- "
                        "see docs/limitations.md)."
                    )
                np_dtype = nb_scalar_dtype_to_numpy(ty.dtype)
                self.sig.array_params.append(ArrayParamInfo(name, np_dtype, ty.ndim))
                self.sig.param_order.append(("array", name))
                self._array_names.add(name)
                continue
            if isinstance(ty, (nb_types.Integer, nb_types.Float, nb_types.Boolean)):
                self.sig.scalar_params.append((name, ty))
                self.sig.param_order.append(("scalar", name))
                self._scalar_names.add(name)
                continue
            if self.device_function:
                raise UnsupportedFeatureError(
                    f"@metal.device_func argument {name!r} has "
                    f"unsupported type {ty!r}. Device functions only "
                    "support scalar (int32/uint32/int64/float32/float16/"
                    "bool) arguments and 1D arrays of a supported dtype -- "
                    "no metal.local_array()/shared_array()."
                )
            raise UnsupportedFeatureError(
                f"Kernel argument {name!r} has unsupported type {ty!r}. "
                "Supported argument types: 1D arrays of a supported "
                "dtype, and scalar int32/uint32/int64/float32/bool."
            )

    def _emit_signature(self) -> None:
        if self.device_function:
            self._emit_device_function_signature()
            return
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
        params.append("uint3 numba_metal_tgid [[threadgroup_position_in_grid]]")
        params.append("uint3 numba_metal_tid_in_tg [[thread_position_in_threadgroup]]")
        params.append("uint3 numba_metal_tg_size [[threads_per_threadgroup]]")
        if self._threadgroup_arrays:
            for tg_index, tg in enumerate(self._threadgroup_arrays):
                # MSL requires a `[[threadgroup(n)]]`-attributed
                # parameter to be a pointer, not a fixed-size array type
                # (confirmed directly against Apple's Metal compiler:
                # `threadgroup float name[8] [[threadgroup(0)]]` is
                # rejected with "'threadgroup' attribute cannot be
                # applied to types"; `threadgroup float* name
                # [[threadgroup(0)]]` compiles). The actual backing size
                # is supplied at dispatch time via
                # `setThreadgroupMemoryLength:atIndex:` (see
                # runtime/dispatcher.py), matching this parameter's index
                # -- `tg.count` is enforced only by `[]` indexing being
                # the sole way generated code ever touches this pointer,
                # never by the MSL type itself.
                params.append(
                    f"threadgroup {tg.msl_elem_ty}* {tg.ident} "
                    f"[[threadgroup({tg_index})]]"
                )
                self.sig.threadgroup_arrays.append(tg)
        joined = ",\n    ".join(params)
        self.builder.write(f"kernel void {self.kernel_name}(")
        self.builder.write(f"    {joined})")
        self.builder.write("{")

    def _emit_device_function_signature(self) -> None:
        """Emit an ordinary C-style MSL function signature for a
        `@metal.device_func` (no `[[buffer(n)]]`/thread-position
        parameters -- those only make sense for a top-level `kernel
        void` entry point; a device function runs as part of whatever
        kernel called it, taking plain by-value scalar arguments and/or
        `device T*`-typed array arguments -- the same `device` address
        space every kernel-level array argument already lives in, since a
        device function is only ever called (directly or transitively)
        from within a kernel body operating on that same buffer, never
        from `threadgroup`/`constant` address space. This is also why
        `metal.atomic_*()` intrinsics work identically inside a device
        function's body: `_atomic_address_expr` always casts to `(device
        atomic_<T>*)`, matching the address space an array parameter is
        declared with here exactly, regardless of whether the array
        currently being indexed is a kernel argument or was passed down
        into a device function."""
        params = []
        for kind, name in self.sig.param_order:
            if kind == "array":
                info = next(a for a in self.sig.array_params if a.name == name)
                msl_ty = numpy_dtype_to_msl(info.dtype)
                params.append(f"device {msl_ty}* arg_{name}")
                params.append(f"uint arg_{name}_size")
                continue
            _, ty = next(s for s in self.sig.scalar_params if s[0] == name)
            # Same float64->float32 narrowing as the return type below:
            # a caller passing a bare Python float literal (e.g.
            # `clamp(x, 0.0, 10.0)`) makes Numba infer that PARAMETER as
            # float64 (ordinary CPython-matching literal typing), not
            # because the caller actually wants float64 precision.
            msl_ty = self._msl_type_for(ty)
            params.append(f"{msl_ty} arg_{name}")
        joined = ", ".join(params)
        # Uses the same float64->float32 local-intermediate narrowing as
        # ordinary local variables (_msl_type_for), not the stricter
        # numba_scalar_to_msl (which rejects float64 outright, correct
        # for kernel ARGUMENTS/array dtypes but not for a device
        # function's return type -- Python's `return x * 2.0 + 1.0`
        # infers float64 under ordinary Numba typing rules exactly like
        # any other local expression, and should be narrowed the same
        # documented way, not rejected).
        return_ty = self._msl_type_for(self.device_function_return_type)
        self.builder.write(f"{return_ty} {self.kernel_name}({joined})")
        self.builder.write("{")

    # -- local declarations -----------------------------------------------

    def _ident(self, var_name: str) -> str:
        """MSL identifier for a given SSA variable name. Every distinct
        SSA name maps to its own distinct identifier -- no aliasing.

        `metal.local_array()`/`metal.shared_array()` call targets are
        special-cased to their pre-assigned array identifier (see
        `_collect_special_arrays`) rather than the default `v_<name>`
        scheme, since a threadgroup array's identifier must match the
        one already emitted as a kernel-function parameter in
        `_emit_signature`.
        """
        if var_name in self._local_arrays:
            return self._local_arrays[var_name].ident
        if var_name in self._threadgroup_array_targets:
            return self._threadgroup_array_targets[var_name].ident
        safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in var_name)
        return "v_" + safe

    def _declare_locals(self) -> None:
        # Every distinct SSA variable name -- including every phi target --
        # gets its own independent MSL declaration. Nothing is aliased or
        # collapsed: this is the core correctness property the de-SSA
        # rewrite establishes (see dessa.py's module docstring). A phi
        # target (name in self._dessa.phi_targets) is declared exactly
        # like any other local; its value comes from the Copy assignments
        # DeSSAPass computed for each incoming edge, emitted by
        # _emit_basic_block, not from the (never-emitted) `phi` statement.
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
            if name in self._threadgroup_array_targets:
                # Already declared as a `threadgroup`-qualified
                # kernel-function parameter in _emit_signature; MSL does
                # not permit a second, ordinary function-body declaration
                # for the same identifier.
                continue
            if name in self._local_arrays:
                info = self._local_arrays[name]
                self.builder.write(f"{info.msl_elem_ty} {info.ident}[{info.count}];")
                continue
            if name in self._atomic_cas_target_names:
                # A metal.atomic_compare_exchange(...) call result (or an
                # exhaust_iter identity-copy of one) -- its two
                # underlying MSL locals are declared inline at the real
                # call site by _emit_atomic_compare_exchange, not here;
                # this SSA name itself never gets its own declaration
                # (MSL has no tuple type to declare it as).
                continue
            if isinstance(ty, (nb_types.Omitted, nb_types.NoneType)):
                continue
            if isinstance(
                ty,
                (
                    nb_types.FunctionType,
                    nb_types.Function,
                    nb_types.RangeType,
                    nb_types.NumberClass,
                    nb_types.Dispatcher,
                ),
            ):
                # NumberClass is the type of a dtype *reference* itself
                # (e.g. the `np.float32` argument passed to
                # metal.local_array()/shared_array()) -- a compile-time
                # value with no MSL representation, never read as data.
                # Dispatcher is the type of an @njit/@metal.device_func
                # global reference itself (the callee var at a call
                # site, e.g. `helper` in `helper(x)`) -- also a
                # compile-time-only value, resolved by `_call`/
                # `_resolve_global`, never declared or read as data.
                continue
            if isinstance(ty, nb_types.RangeIteratorType):
                continue
            try:
                msl_ty = self._msl_type_for(ty)
            except UnsupportedFeatureError as exc:
                raise UnsupportedFeatureError(f"{exc} (variable {name!r})") from exc
            if msl_ty is None:
                continue
            ident = self._ident(name)
            self.builder.write(f"{msl_ty} {ident};")
        self._declare_dessa_temporaries()

    def _declare_dessa_temporaries(self) -> None:
        """Declare the synthetic temporaries DeSSAPass introduces to break
        copy cycles (e.g. a loop-carried swap `a, b = b, a`). Each such
        temp is a plain copy of some phi target's value, so it shares that
        target's already-validated MSL type -- see
        `dessa._sequentialize_parallel_copies`, which names every such
        temp `__dessa_tmp<n>__<target>` specifically so its origin target
        (and therefore its type) can be recovered here."""
        if self._dessa is None:
            return
        declared: set[str] = set()
        for copies in self._dessa.edge_copies.values():
            for copy in copies:
                if not copy.target.startswith("__dessa_tmp"):
                    continue
                if copy.target in declared:
                    continue
                # copy.source at declaration time is always the original
                # phi-target variable this temporary preserves the value
                # of (see _sequentialize_parallel_copies: the temp is
                # created as `Copy(target=temp, source=victim.target)`).
                origin_ty = self.typemap.get(copy.source)
                if origin_ty is None:
                    raise UnsupportedFeatureError(
                        f"Internal error: de-SSA temporary {copy.target!r} "
                        f"has no type information for its origin "
                        f"{copy.source!r}."
                    )
                msl_ty = self._msl_type_for(origin_ty)
                if msl_ty is None:
                    continue
                ident = self._ident(copy.target)
                self.builder.write(f"{msl_ty} {ident};")
                declared.add(copy.target)

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
            if ty.count == 3:
                return "long3"
            raise UnsupportedFeatureError(
                f"Unsupported tuple type {ty!r}: only 2-tuples or 3-tuples "
                "of int64 (from metal.grid(2)/metal.grid(3) and the other "
                "thread/threadgroup-position intrinsics) are supported."
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
                "Array-typed local variables are only supported when "
                "produced by metal.local_array(shape, dtype) or "
                "metal.shared_array(shape, dtype); general array-typed "
                "expressions (e.g. slicing) are not supported inside "
                "kernels."
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

    def _collect_special_arrays(self) -> None:
        """Pre-scan every `metal.local_array(...)`/`metal.shared_array(...)`
        call site (requires `self._globals` already populated by
        `_collect_globals`) and record shape/dtype/identifier info for
        each. Threadgroup arrays are additionally appended to
        `self._threadgroup_arrays` here, since `_emit_signature` (which
        declares them as kernel-function parameters) runs before any
        statement is walked."""
        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                if not (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "call"
                ):
                    continue
                callee = self._resolve_global(stmt.value.func.name)
                if callee is not intrinsics.local_array and (
                    callee is not intrinsics.shared_array
                ):
                    continue
                target_name = stmt.target.name
                shape_var, dtype_var = stmt.value.args[0], stmt.value.args[1]
                shape_ty = self.typemap.get(shape_var.name)
                if not isinstance(shape_ty, nb_types.IntegerLiteral):
                    raise UnsupportedFeatureError(
                        "metal.local_array()/shared_array() require a "
                        "literal integer shape known at compile time."
                    )
                count = shape_ty.literal_value
                dtype_ty = self.typemap.get(dtype_var.name)
                elem_ty = getattr(dtype_ty, "instance_type", None)
                if elem_ty is None:
                    raise UnsupportedFeatureError(
                        "metal.local_array()/shared_array() require a "
                        "NumPy scalar dtype (e.g. np.float32) as the "
                        "second argument."
                    )
                msl_elem_ty = numba_scalar_to_msl(elem_ty)
                ident = f"v_special_arr_{self._special_array_counter}"
                self._special_array_counter += 1
                if callee is intrinsics.local_array:
                    self._local_arrays[target_name] = _LocalArrayInfo(
                        target_name=target_name,
                        ident=ident,
                        msl_elem_ty=msl_elem_ty,
                        count=count,
                    )
                else:
                    elem_itemsize = nb_scalar_dtype_to_numpy(elem_ty).itemsize
                    info = _ThreadgroupArrayInfo(
                        target_name=target_name,
                        ident=ident,
                        msl_elem_ty=msl_elem_ty,
                        count=count,
                        elem_itemsize=elem_itemsize,
                    )
                    self._threadgroup_arrays.append(info)
                    self._threadgroup_array_targets[target_name] = info

    def _collect_atomic_cas_targets(self) -> None:
        """Pre-scan every SSA name typed as a 2-tuple of
        (int32/uint32/float32, bool) -- the exact and only shape
        `metal.atomic_compare_exchange(...)` (or an `exhaust_iter`
        identity-copy of its result) can produce, since general tuple
        construction is unsupported (see the `build_tuple` rejection in
        `_emit_expr_assign`) -- and records them in
        `self._atomic_cas_target_names` so `_declare_locals` skips the
        generic (tuple-incapable) `_msl_type_for` declaration path for
        them; their actual MSL declarations are emitted inline by
        `_emit_atomic_compare_exchange` at the real call site instead.
        """
        atomic_elem_types = (nb_types.int32, nb_types.uint32, nb_types.float32)
        for name, ty in self.typemap.items():
            if (
                isinstance(ty, nb_types.Tuple)
                and len(ty.types) == 2
                and ty.types[0] in atomic_elem_types
                and ty.types[1] == nb_types.boolean
            ):
                self._atomic_cas_target_names.add(name)

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
                self._suppress_loop_protocol_names(
                    node, for_info.loop_var_name, for_info.range_target
                )
            self._compute_suppressed_names(node.body)

    def _suppress_loop_protocol_names(
        self, node: LoopNode, loop_var_name: str, range_target: str
    ) -> None:
        """Mark this *specific* loop's iterator-protocol bookkeeping
        variables as suppressed. Scoped strictly to `range_target` (the
        exact `range(...)` call feeding this loop's `getiter`) so that,
        with nested for-range loops, resolving the inner loop's protocol
        names can never walk into and re-suppress the outer loop's -- each
        loop's transitive copy-chain closure is seeded fresh from only its
        own getiter/iternext/pair_first/pair_second names, never from the
        shared, ever-growing `self._suppressed_names` accumulated by
        previously processed loops (that was the root cause of a real bug
        found by tests/integration/test_phi_dessa.py's nested-loop case:
        the inner loop's closure scan matched on "source name is already
        suppressed" against the *global* set, which by then already
        contained the outer loop's getiter result, and followed its copy
        chain all the way to the outer loop's own `j`).
        """
        header_block = self.func_ir.blocks[node.header_label]
        local: set[str] = set()
        for stmt in header_block.body:
            if not (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)):
                continue
            if stmt.value.op in ("iternext", "pair_first", "pair_second", "getiter"):
                local.add(stmt.target.name)
        # The `getiter()` call itself lives in the block that jumps into
        # the loop header (e.g. block 66 for header 156), not in the
        # header block, so it is not caught by the scan above; find it
        # directly from this loop's own range() call site instead.
        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                if (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "getiter"
                    and stmt.value.value.name == range_target
                ):
                    local.add(stmt.target.name)
        # Copy assignments chain loop-protocol values through plain
        # `ir.Var` assigns (not `op=="phi"` exprs -- see
        # _trace_to_range_call), both to seed the iterator into the
        # header's phi slot (`$phiHDR.0 = $NNget_iter.M`) and to bind the
        # loop variable each iteration (`it = $phiNNN.1`, i.e. a copy of
        # the pair_first result). Propagate suppression transitively
        # across such copies (fixed point) so both are caught regardless
        # of how many hops separate them from the original getiter/
        # pair_first/pair_second target -- but seeded from and bounded by
        # `local` (this loop only), not the global accumulated set.
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
                    if stmt.value.name in local and stmt.target.name not in local:
                        local.add(stmt.target.name)
                        changed = True
        self._suppressed_names |= local

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
            if self.device_function and node.value is not None:
                ty = self.typemap.get(node.value.name)
                if not isinstance(ty, nb_types.NoneType):
                    self.builder.write(f"return {self._read(node.value)};")
                    return
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
            var = for_info.loop_var_ident
            # Python's `range(start, stop, step)` evaluates start/stop/step
            # exactly ONCE, when the loop begins -- they are never re-read
            # even if the loop body later reassigns the variable(s) that
            # fed them (e.g. `for j in range(m): ... m = 0`, a real case
            # found by the differential/property-based test suite,
            # Workstream 4). A native MSL `for (init; cond; incr)` header
            # re-evaluates its `cond` expression every iteration, so
            # substituting the raw stop/step expressions directly into
            # `cond` (as an earlier version of this function did) would
            # silently pick up any later mutation of the variables they
            # reference -- changing the trip count mid-loop and diverging
            # from Python/Numba CPU semantics. Capturing each bound into
            # its own MSL-declared, uniquely-named temporary evaluated
            # once, immediately before the `for` header, reproduces the
            # correct once-only evaluation regardless of what the loop
            # body does to the source variables afterward.
            snapshot_id = self._range_snapshot_counter
            self._range_snapshot_counter += 1
            start_var = f"v__range_start_{snapshot_id}"
            stop_var = f"v__range_stop_{snapshot_id}"
            step_var = f"v__range_step_{snapshot_id}"
            self.builder.write(f"long {start_var} = {for_info.bounds.start};")
            self.builder.write(f"long {stop_var} = {for_info.bounds.stop};")
            self.builder.write(f"long {step_var} = {for_info.bounds.step};")
            self.builder.write(
                f"for ({var} = {start_var}; "
                f"({step_var} > 0) ? ({var} < {stop_var}) : ({var} > {stop_var}); "
                f"{var} += {step_var}) {{"
            )
            with self.builder.block():
                self._emit_node(node.body)
            self.builder.write("}")
            return

        # Only a straight-line loop body (no nested if/else, break, or
        # continue anywhere in the rotated body/pre_test regions) is
        # supported by this generic (non-for-range) path. Numba's
        # loop-rotation transform for `while` interacts with this
        # backend's if/else structurer in a way that was found, by
        # direct testing, to silently mis-lower `break`/`continue` nested
        # inside a conditional within a `while` body (confirmed: a
        # continue-inside-if test and a break-inside-if test produced
        # each other's expected results, i.e. genuinely swapped/wrong
        # control flow, not just a crash) -- fully generalizing this is
        # a substantially larger control-flow-structurer project than a
        # bounded bug fix, so it is deliberately out of scope here.
        # Rejecting this shape explicitly, rather than emitting the
        # confirmed-wrong MSL, is required by this project's no-silent-
        # incorrect-results policy. Straight-line while bodies (the
        # common case: recompute some values, test a condition, repeat --
        # e.g. a compare-and-swap retry loop) ARE genuinely fixed and
        # tested by this method; see docs/limitations.md.
        _reject_unsupported_while_body(node.body)

        # numba-metal only emits this generic form for a loop header
        # whose OWN statements are the per-iteration condition-feeding
        # work (Numba's bytecode lowering rotates `while cond: body`
        # into an if-guarded do-while: the "body" region reached via
        # `body_target`, structured into `node.body`, is a trivial
        # back-edge block -- an empty BasicBlockNode plus an unconditional
        # ContinueNode -- while the header block (`node.header_label`)
        # contains one full iteration's real work followed by the NEXT
        # condition test, captured in `node.pre_test`; see structuring
        # .py's LoopNode.pre_test docstring for the exact CFG shape this
        # was verified against).
        #
        # That back-edge block's raw IR statement list is empty, but it
        # is NOT a no-op: DeSSAPass may have attached phi-resolution edge
        # -copies to ITS outgoing edge specifically (e.g. `total_for_next
        # _read = total_just_computed;`), separately from the copies
        # attached to the header's own edge (already re-run every pass
        # via `_emit_edge_copies(node.header_label)` below). Skipping
        # them (an earlier version of this method did, on the mistaken
        # assumption that an empty statement list meant nothing to emit)
        # left every read inside the loop body referencing a stale,
        # never-updated value -- confirmed by direct testing: a summation
        # while-loop only ever added its first element, every subsequent
        # pass re-reading index 0. The literal `continue;` itself is
        # still elided (MSL's `continue` would jump past `pre_test`,
        # skipping every real statement there every single pass), but
        # the back-edge block's own edge-copies must still run --
        # `_emit_basic_block` normally does exactly this (statements then
        # edge-copies) for every ordinary block, so this handles the one
        # case that needs the statements suppressed but the edge-copies
        # kept.
        #
        # Emitted as a real MSL `do { ... } while (cond);`, not a
        # `while (true) { ... }` with a break: a `do-while`'s `continue`
        # jumps to the trailing `while (cond)` test, which is exactly
        # where control needs to end up after any GENUINE `continue` a
        # nested if/else inside `node.body` might contain (this matters
        # once `node.body` carries real conditional break/continue logic,
        # not just the trivial back-edge case elided above).
        #
        # The whole `do-while` is further guarded by `if (cond)`: a
        # `do-while` always executes its body at least once, but the
        # pre-loop `Seq` wrapper in structuring.py has ALREADY executed
        # one full pass (the header, i.e. this loop's real per-iteration
        # work) before this method runs at all, and already computed
        # THAT pass's own exit condition. If that already-computed
        # condition says to stop, the do-while must not run a second,
        # unwanted pass -- omitting this guard was a real, confirmed bug
        # (found via a hand-rolled atomic-compare-exchange retry loop):
        # a `do-while` unconditionally repeats its body once regardless
        # of what the condition already evaluated to, producing exactly
        # one extra, spurious CAS attempt (which -- being a real,
        # correctly-race-free CAS -- sometimes even "succeeded" again,
        # corrupting the count) beyond the correct one-attempt-that-
        # matters-per-thread flow.
        cond_expr = self._read(node.exit_cond)
        guard_expr = cond_expr if node.exit_cond_negated else f"!({cond_expr})"
        self.builder.write(f"if ({guard_expr}) {{")
        with self.builder.block():
            trivial_back_edge_label = self._trivial_back_edge_label(node.body)
            self.builder.write("do {")
            with self.builder.block():
                if trivial_back_edge_label is None:
                    self._emit_node(node.body)
                else:
                    self._emit_edge_copies(trivial_back_edge_label)
                for stmt in node.pre_test:
                    self._emit_stmt(stmt)
                self._emit_edge_copies(node.header_label)
            # `node.exit_cond_negated=True` means "break when `cond` is
            # false" (the loop continues while `cond` is true), so the
            # do-while's own trailing continuation test is `cond`
            # unchanged; `exit_cond_negated=False` means "break when
            # `cond` is true" (continues while `cond` is false), so the
            # trailing test must be negated to match -- the mirror image
            # of the break-style condition, not the same expression (and
            # the same expression as `guard_expr` above, reused here).
            cond_for_continue = (
                cond_expr if node.exit_cond_negated else f"!({cond_expr})"
            )
            self.builder.write(f"}} while ({cond_for_continue});")
        self.builder.write("}")

    def _trivial_back_edge_label(self, node: Node) -> int | None:
        """If `node` is exactly an empty-statement-list BasicBlockNode
        followed by an unconditional ContinueNode (Numba's rotated-
        `while` lowering's back-edge block -- see `_emit_loop`'s
        docstring comment), return that BasicBlockNode's label (so the
        caller can still run ITS edge-copies, just not its -- empty --
        statement list or the ContinueNode's literal `continue;`).
        Returns None for any other shape. Recurses through Seq wrappers,
        which is how the structurer represents this shape."""
        if isinstance(node, Seq):
            items = node.items
            if len(items) == 1:
                return self._trivial_back_edge_label(items[0])
            if len(items) == 2:
                first, second = items
                if (
                    isinstance(first, BasicBlockNode)
                    and not first.body
                    and isinstance(second, ContinueNode)
                ):
                    return first.label
            return None
        return None

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
        if pair_second_name is None or pair_second_name != self._exit_cond_source_name(
            node
        ):
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
        return _ForRangeLoopInfo(
            self._ident(pair_first_name), pair_first_name, bounds, range_target
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
        # entry block). These are ir.Var assigns, not phi exprs (see
        # _trace_to_range_call), so DeSSAPass never touches them -- follow
        # the copy chain to its end instead of assuming a single hop.
        # Stop as soon as the chain reaches a real, user-visible Python
        # variable name (Numba's compiler-generated SSA temporaries are
        # always prefixed with `$`; a plain identifier like `j3` never
        # is). This is required, not just an optimization: continuing to
        # follow "any assignment that happens to copy this value" beyond
        # that point is unsound, because a real user statement --
        # entirely unrelated to the loop-iterator-protocol chain, e.g.
        # `n = j3` inside the loop body -- is itself exactly such a copy
        # assignment, and the previous unconditional "follow every copy
        # forward" logic would walk straight into it and misidentify an
        # unrelated variable as the loop's induction variable. Found by
        # the exhaustive differential test suite (Workstream 4): a
        # generated kernel containing `for j3 in range(...): ... n = j3`
        # (reassigning a scalar kernel *argument* from the loop variable)
        # produced MSL where the `for` loop's own induction variable was
        # wrongly identified as `n` instead of `j3`, leaving `j3` itself
        # referenced-but-undeclared elsewhere in the body.
        if not first_target.startswith("$"):
            return first_target
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
            if not next_hop.startswith("$"):
                return next_hop
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
        self._emit_edge_copies(node.label)

    def _emit_edge_copies(self, block_label: int) -> None:
        """Emit the parallel-copy assignments DeSSAPass computed for the
        CFG edge leaving `block_label` (i.e. this block is a phi's
        `incoming_block`). This is the single place phi resolution
        produces MSL text; see dessa.py for why per-edge copies -- not
        aliasing -- are required for correctness, and structuring.py for
        why `block_label`'s position in the structured tree is always the
        unique, correct point for "control just took this edge"."""
        if self._dessa is None:
            return
        for copy in self._dessa.edge_copies.get(block_label, []):
            target_ident = self._ident(copy.target)
            source_ident = self._ident_for_copy_source(copy.source)
            self.builder.write(f"{target_ident} = {source_ident};")

    def _ident_for_copy_source(self, name: str) -> str:
        """Resolve a Copy's source name to an MSL expression: either a
        kernel argument, a declared local/phi-temp, or (for a temporary
        DeSSAPass introduced to break a copy cycle) its own identifier."""
        if name in self._array_names or name in self._scalar_names:
            return f"arg_{name}"
        return self._ident(name)

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
                nb_types.RangeIteratorType,
                nb_types.NumberClass,
                nb_types.Dispatcher,
            ),
        ):
            # No MSL declaration exists for these types (see
            # _declare_locals/_msl_type_for, which skip them the same
            # way) -- they are Python-iterator-protocol bookkeeping
            # (getiter()'s own RangeIteratorType result, iternext()'s
            # Pair result) with no runtime representation in generated
            # MSL. This check must stay in sync with _declare_locals's
            # skip list: previously RangeIteratorType was declared-skip
            # only, not assign-skip, which let a plain `ir.Var` copy
            # assignment whose target has this type (e.g. seeding a
            # loop's iterator into its per-iteration phi slot) reach this
            # method and emit an assignment to an identifier that was
            # never declared. This only manifested when the surrounding
            # loop had no back-edge left in the CFG for the structurer to
            # recognize as a LoopNode (e.g. a `for` loop whose body
            # provably breaks unconditionally on iteration 0, which
            # Numba's own optimizer reduces to a back-edge-free CFG) --
            # in that shape, _detect_for_range/_suppress_loop_protocol_names
            # never runs at all (they are only invoked for a recognized
            # LoopNode), so nothing else suppressed this assignment.
            # Found by the differential/property-based test suite
            # (Workstream 4): `for j in range(n): if True: break` (i.e.
            # a loop that unconditionally exits on its first iteration)
            # produced MSL referencing undeclared identifiers before this
            # fix. Skipping by type here, unconditionally, is strictly
            # more robust than the structural (LoopNode-gated) suppression
            # this bug was found in: no assignment to a value of a type
            # with no MSL representation should ever be emitted, whether
            # or not the surrounding control flow was recognized as a
            # loop.
            return
        ident = self._ident(target_name)
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
            cas_pair = self._atomic_cas_results.get(expr.value.name)
            if cas_pair is not None:
                # Indexing into a metal.atomic_compare_exchange(...)
                # result tuple: [0] is the pre-swap value, [1] is the
                # success flag, each already its own independent MSL
                # local (see _emit_atomic_compare_exchange) rather than
                # an indexable array/struct -- MSL has no tuple type to
                # index into here.
                self._assign(target_name, cas_pair[expr.index])
                return
            self._assign(target_name, f"{self._read(expr.value)}[{expr.index}]")
        elif op == "getattr":
            if target_name in self._globals:
                return  # resolved as a callable (e.g. metal.grid); no MSL emitted
            self._assign(target_name, self._getattr(expr))
        elif op == "call":
            self._call(target_name, expr)
        elif op == "phi":
            # The phi statement itself is never emitted -- its target is
            # declared as an ordinary local (see _declare_locals) and
            # assigned only via the per-edge Copy statements DeSSAPass
            # computed, emitted by _emit_edge_copies wherever each
            # incoming block's own statements finish. See dessa.py.
            pass
        elif op == "exhaust_iter":
            # Numba inserts `exhaust_iter` when unpacking a fixed-size
            # tuple (`x, y = metal.grid(2)`); it is an identity copy of
            # the tuple value that the following static_getitem(s) index
            # into. It is NOT part of the getiter/iternext for-loop
            # protocol despite the similar-sounding name.
            cas_pair = self._atomic_cas_results.get(expr.value.name)
            if cas_pair is not None:
                # Propagate the (old_value_ident, success_ident) mapping
                # to this identity-copy's own target, so the
                # static_getitem(s) that follow (which index into THIS
                # target, not the original call's) still resolve
                # correctly -- see the static_getitem branch above.
                self._atomic_cas_results[target_name] = cas_pair
                return
            self._assign(target_name, self._read(expr.value))
        elif op in ("pair_first", "pair_second", "iternext", "getiter"):
            pass  # consumed structurally by for-range loop detection
        elif op == "build_tuple":
            raise UnsupportedFeatureError(
                "Tuple construction is only supported as the direct result "
                "of a thread/threadgroup-position intrinsic called with "
                "ndim=2 or ndim=3 (e.g. metal.grid(2)); general tuple "
                "literals are unsupported."
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

    def _compile_device_function(
        self, py_func, arg_types: tuple[nb_types.Type, ...]
    ) -> str:
        """Recursively compile a `@metal.device_func`-wrapped `py_func`
        at `arg_types` to a real, standalone MSL function (memoized by
        `(py_func, arg_types)` in `self._device_function_cache`, shared
        across this entire kernel compilation including transitively
        -called device functions), returning the MSL function's name for
        the caller to emit a normal call expression against. Detects
        direct or transitive recursion and raises rather than recursing
        into Python's own call stack (which would eventually hit
        RecursionError, or worse, silently produce an infinite MSL
        function-emission loop for a mutually-recursive pair) --
        recursion is explicitly out of scope, matching numba-metal's
        existing "no recursion" restriction for kernels themselves.
        """
        from numba_metal.compiler.frontend import compile_to_typed_ir

        cache_key = (py_func, arg_types)
        if cache_key in self._device_function_cache:
            cached = self._device_function_cache[cache_key]
            if cached is None:
                raise UnsupportedFeatureError(
                    f"Recursive call to @metal.device_func "
                    f"{getattr(py_func, '__name__', py_func)!r} detected "
                    "(directly or transitively) -- recursion is not "
                    "supported."
                )
            return cached
        # Mark as "in progress" (None) before recursing into the
        # callee's own body, so a call back to this exact
        # (py_func, arg_types) pair from within that body -- direct
        # self-recursion, or a transitive cycle through another device
        # function -- is detected above instead of recursing forever.
        self._device_function_cache[cache_key] = None

        typed = compile_to_typed_ir(py_func, arg_types, return_type=None)
        name = _next_device_function_name(getattr(py_func, "__name__", "device_func"))
        lowerer = MSLKernelLowerer(
            name,
            typed,
            device_function=True,
            return_type=typed.return_type,
            device_function_cache=self._device_function_cache,
        )
        body_src = lowerer.lower()
        # Any device functions THIS device function itself called are
        # already fully compiled (recursively) by the nested lowerer;
        # splice their sources in before this function's own, so every
        # callee is textually declared before its caller (not required
        # by MSL/C at file scope for functions -- MSL does not require
        # forward declaration order for top-level function definitions
        # within one compiled source -- but keeping this ordering makes
        # the generated source readable top-to-bottom regardless).
        self.device_function_sources.extend(lowerer.device_function_sources)
        self.device_function_sources.append(body_src)
        self._device_function_cache[cache_key] = name
        return name

    def _call(self, target_name: str, expr: ir.Expr) -> None:
        callee = self._resolve_global(expr.func.name)
        args = [self._read(a) for a in expr.args]

        from numba_metal.runtime.dispatcher import (
            device_function_py_func,
            is_device_function,
        )

        if is_device_function(callee):
            arg_types = tuple(self.typemap[a.name] for a in expr.args)
            device_func_name = self._compile_device_function(
                device_function_py_func(callee), arg_types
            )
            # Every array-typed argument expands to two MSL call
            # arguments (the `device T*` pointer plus its `_size`
            # companion), matching _emit_device_function_signature's
            # two-parameter-per-array emission -- an array argument is
            # always a plain SSA variable reference here (arrays cannot
            # be produced by an inline expression), so `a.name` is always
            # a real typemap/array-name entry, never a temporary.
            call_args = []
            for a, arg_expr in zip(expr.args, args, strict=True):
                call_args.append(arg_expr)
                if a.name in self._array_names:
                    call_args.append(f"arg_{a.name}_size")
            self._assign(target_name, f"{device_func_name}({', '.join(call_args)})")
            return

        if callee is bool:
            self._assign(target_name, args[0])
            return
        if callee in (
            intrinsics.grid,
            intrinsics.gridsize,
            intrinsics.threadgroup_position,
            intrinsics.thread_in_threadgroup,
            intrinsics.threads_per_threadgroup,
        ):
            self._assign(target_name, self._grid_expr(callee, expr))
            return
        if callee is range:
            self._range_calls[target_name] = args
            return
        if callee is intrinsics.local_array or callee is intrinsics.shared_array:
            # The target's array declaration was already emitted in
            # _declare_locals (an ordinary function-body array for
            # local_array, a threadgroup-qualified kernel parameter for
            # shared_array) -- MSL has no single-expression array
            # initializer syntax to assign here, so this call site emits
            # nothing further.
            return
        if callee is intrinsics.barrier:
            # Not routed through _assign: the call's SSA target is
            # None-typed (a bare `metal.barrier()` statement's result is
            # discarded), and _assign deliberately drops None-typed
            # targets (see its docstring) -- this is a statement with a
            # real side effect, not a value-producing expression, so it
            # must be written unconditionally.
            self.builder.write("threadgroup_barrier(mem_flags::mem_threadgroup);")
            return
        if callee in _ATOMIC_FETCH_OP_MSL:
            self._emit_atomic_fetch_op(target_name, callee, expr)
            return
        if callee is intrinsics.atomic_compare_exchange:
            self._emit_atomic_compare_exchange(target_name, expr)
            return
        if callee is abs:
            self._assign(target_name, f"abs({args[0]})")
            return
        if callee is min or callee is max:
            # MSL's min/max are C++-style overloaded functions requiring
            # both arguments to share exactly one concrete type; Metal's
            # shader compiler reports "call to 'min'/'max' is ambiguous"
            # rather than silently promoting mixed-width arguments (e.g.
            # `min(int32_val, int64_val)` -- Numba's own type inference
            # already promotes such a call's *result* type to the wider
            # operand's type, matching ordinary Python/Numba numeric
            # promotion, so casting both arguments to that already-known
            # result type here reproduces Numba's promotion semantics
            # exactly rather than reimplementing promotion rules
            # independently. Found by the differential/property-based
            # test suite (Workstream 4): `min(int32_arg, int64_literal)`
            # inside a generated kernel failed real Metal compilation
            # with exactly this "ambiguous call" error before this fix.
            result_ty = self.typemap.get(target_name)
            msl_result_ty = (
                self._msl_type_for(result_ty) if result_ty is not None else None
            )
            fname = "min" if callee is min else "max"
            if msl_result_ty is not None:
                casted = [f"{msl_result_ty}({a})" for a in args]
                self._assign(target_name, f"{fname}({casted[0]}, {casted[1]})")
            else:
                self._assign(target_name, f"{fname}({args[0]}, {args[1]})")
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

    #: Maps each thread/threadgroup-position intrinsic to the MSL
    #: parameter identifier carrying the corresponding
    #: `[[attribute]]`-qualified uint3 value (declared in
    #: `_emit_signature`). All four share one lowering shape: read the
    #: literal ndim, then build a scalar/2-tuple/3-tuple long expression
    #: from that uint3's `.x`/`.y`/`.z` fields.
    _POSITION_INTRINSIC_SOURCE = {
        "grid": "numba_metal_tid",
        "gridsize": "numba_metal_grid_size",
        "threadgroup_position": "numba_metal_tgid",
        "thread_in_threadgroup": "numba_metal_tid_in_tg",
        "threads_per_threadgroup": "numba_metal_tg_size",
    }

    def _grid_expr(self, callee, expr: ir.Expr) -> str:
        ndim_var = expr.args[0]
        ndim_ty = self.typemap.get(ndim_var.name)
        if not isinstance(ndim_ty, nb_types.IntegerLiteral):
            raise UnsupportedFeatureError(
                "metal thread/threadgroup-position intrinsics require a "
                "literal integer ndim argument known at compile time."
            )
        ndim = ndim_ty.literal_value
        intrinsic_name = getattr(callee, "__name__", None)
        source = self._POSITION_INTRINSIC_SOURCE.get(intrinsic_name)
        if source is None:  # pragma: no cover - defensive, unreachable via _call
            raise UnsupportedFeatureError(
                f"Unknown thread-position intrinsic {callee!r}."
            )
        if ndim == 1:
            return f"long({source}.x)"
        if ndim == 2:
            return f"long2(long({source}.x), long({source}.y))"
        if ndim == 3:
            return f"long3(long({source}.x), long({source}.y), long({source}.z))"
        raise UnsupportedFeatureError(
            f"metal thread/threadgroup-position intrinsics only support "
            f"ndim in (1, 2, 3); got {ndim}."
        )

    def _read(self, var: ir.Var) -> str:
        name = var.name
        if name in self._array_names or name in self._scalar_names:
            return f"arg_{name}"
        if name in self._globals:
            # A direct reference to a module-level global/freevar (e.g. a
            # module-level `N = 32` used as a range() bound, an array
            # size, or any other operand). `_emit_assign` deliberately
            # never emits a declaration or assignment for an
            # ir.Global/ir.FreeVar-valued SSA name (see its comment
            # "resolved at call sites via self._globals") -- that scheme
            # only actually worked for *callable* globals (metal.grid,
            # range, math.sqrt, ...), which are consumed exclusively via
            # `_resolve_global`/`_call` and never flow through `_read` at
            # all. A non-callable global (a plain int/float/bool
            # constant) DOES flow through `_read` whenever it's used as a
            # value -- e.g. `range(N)`, `x + N`, `if i < N` -- and
            # without this branch, `_read` fell through to `_ident(name)`
            # and returned an identifier for a SSA variable that was
            # never declared or assigned anything, since `_declare_locals`
            # has no exclusion for it either: MSL then default-initializes
            # that `long`/`int`/etc. to (observed) zero, silently making
            # every such reference read as 0 instead of the real constant
            # (concretely: `for j in range(N)` with a zero stop bound
            # runs its body zero times). Resolve it to the same literal
            # MSL text a `Const` of the same value would produce, exactly
            # as `_emit_assign` already does for `ir.Const` via
            # `_const_text`, so a global constant behaves identically to
            # writing the literal inline.
            #
            # This check also has to accept numpy scalar types
            # (np.float32/np.int32/np.bool_/...), not just Python's own
            # bool/int/float -- numpy scalars are NOT subclasses of the
            # builtins (`isinstance(np.float32(1.0), float)` is False),
            # so a module-level constant declared with an explicit numpy
            # dtype (the idiomatic way to pin a physical constant to
            # float32 for a kernel, e.g. `SIGMA = np.float32(5.67e-8)`)
            # used to fall straight through this branch to the
            # `_ident(name)` case below, referencing an SSA name that was
            # never declared -- reading uninitialized MSL stack memory at
            # runtime (observed directly: silently 0.0 in one kernel,
            # NaN in another, depending on what garbage happened to be on
            # the stack). `_const_text` itself normalizes numpy scalars
            # to plain Python values, so it's the single source of truth
            # for "is this a constant-like value" here too.
            value = self._globals[name]
            if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                return self._const_text(value, name)
        if name in self.typemap:
            return self._ident(name)
        raise UnsupportedFeatureError(f"Reference to undeclared variable {name!r}.")

    #: MSL scalar type name -> MSL atomic type name, for the three dtypes
    #: metal.atomic_*() support (see intrinsics._ATOMIC_DTYPES).
    _MSL_ATOMIC_TYPE = {
        "int": "atomic_int",
        "uint": "atomic_uint",
        "float": "atomic_float",
    }

    def _atomic_address_expr(self, expr: ir.Expr) -> tuple[str, str]:
        """Given an `ir.Expr` call to one of the `metal.atomic_*()`
        intrinsics, return `(address_expr, msl_elem_ty)`: `address_expr`
        is a `device atomic_<T>*`-typed MSL expression pointing at
        `array[index]` (an in-place C-style pointer-address-and-cast --
        verified directly against Apple's Metal compiler to produce
        correct, race-free atomics, including under real multi-thread
        contention -- see docs/architecture.md), and `msl_elem_ty` is the
        array's plain (non-atomic) MSL element type name (needed by
        callers to cast literal operands to the matching type).
        """
        arr_var, idx_var = expr.args[0], expr.args[1]
        arr_ty = self.typemap.get(arr_var.name)
        elem_ty = arr_ty.dtype
        msl_elem_ty = numba_scalar_to_msl(elem_ty)
        atomic_ty = self._MSL_ATOMIC_TYPE.get(msl_elem_ty)
        if atomic_ty is None:  # pragma: no cover - unreachable; intrinsics.py
            # already restricts to _ATOMIC_DTYPES at the typing step.
            raise UnsupportedFeatureError(
                f"metal.atomic_*() does not support element type " f"{msl_elem_ty!r}."
            )
        arr_expr = self._read(arr_var)
        idx_expr = self._read(idx_var)
        address = f"(device {atomic_ty}*)(&{arr_expr}[{idx_expr}])"
        return address, msl_elem_ty

    def _emit_atomic_fetch_op(self, target_name: str, callee, expr: ir.Expr) -> None:
        address, msl_elem_ty = self._atomic_address_expr(expr)
        value_expr = self._read(expr.args[2])
        if callee in _ATOMIC_MINMAX_OPS and msl_elem_ty == "float":
            # No native MSL atomic_fetch_min/max exists for float on any
            # Apple GPU family (see _ATOMIC_MINMAX_OPS's module-level
            # comment) -- lowered to a compare-and-swap retry loop that
            # is still race-free: every thread's CAS either installs its
            # value (if it is still the extremum relative to whatever the
            # memory currently holds) or discovers a newer value and
            # retries against that.  This exact shape was verified to
            # compile and produce correct results under real contention
            # -- see docs/architecture.md and the atomics test suite.
            op = _ATOMIC_MINMAX_OPS[callee]
            cmp = ">" if op == "max" else "<"
            tmp = f"__nbmtl_atomic_{self._special_array_counter}"
            self._special_array_counter += 1
            self.builder.write(f"float {tmp}_expected;")
            self.builder.write(f"float {tmp}_val = {value_expr};")
            self.builder.write(
                f"{tmp}_expected = atomic_load_explicit({address}, "
                "memory_order_relaxed);"
            )
            self.builder.write(f"while ({tmp}_val {cmp} {tmp}_expected) {{")
            with self.builder.block():
                self.builder.write(
                    f"if (atomic_compare_exchange_weak_explicit({address}, "
                    f"&{tmp}_expected, {tmp}_val, memory_order_relaxed, "
                    "memory_order_relaxed)) { break; }"
                )
            self.builder.write("}")
            self._assign(target_name, f"{tmp}_expected")
            return
        msl_op = _ATOMIC_FETCH_OP_MSL[callee]
        casted_value = f"{msl_elem_ty}({value_expr})"
        self._assign(
            target_name,
            f"atomic_{msl_op}_explicit({address}, {casted_value}, "
            "memory_order_relaxed)",
        )

    def _emit_atomic_compare_exchange(self, target_name: str, expr: ir.Expr) -> None:
        address, msl_elem_ty = self._atomic_address_expr(expr)
        expected_expr = self._read(expr.args[2])
        desired_expr = self._read(expr.args[3])
        # atomic_compare_exchange_weak_explicit takes `expected` by
        # pointer and overwrites it with the CURRENT value on failure
        # (and leaves it unchanged -- equal to what was passed in -- on
        # success), so a local variable is required regardless of
        # whether the caller wants to observe the failure-case value;
        # `_assign` below always declares `target_name` (a Tuple result)
        # normally, so this temp is purely an implementation detail of
        # calling the by-pointer MSL API, not a second declaration of the
        # visible tuple result.
        tmp = f"__nbmtl_cas_{self._special_array_counter}"
        self._special_array_counter += 1
        self.builder.write(f"{msl_elem_ty} {tmp} = {msl_elem_ty}({expected_expr});")
        success_expr = (
            f"atomic_compare_exchange_weak_explicit({address}, &{tmp}, "
            f"{msl_elem_ty}({desired_expr}), memory_order_relaxed, "
            "memory_order_relaxed)"
        )
        # The typed IR target of an `atomic_compare_exchange(...)` call
        # is a 2-tuple (old_value, success); MSL has no tuple type, so
        # this is represented as two independent MSL locals, named by
        # convention `<ident>_0`/`<ident>_1`, matching how
        # `static_getitem(<call target>, 0/1)` will read them back after
        # tuple-unpacking (`old, ok = metal.atomic_compare_exchange(...)`)
        # -- see _emit_expr_assign's static_getitem handling, which reads
        # `self._ident(value.name)` + `[index]` for an ordinary tuple,
        # but a 2-tuple call result here is never itself declared as an
        # MSL array/struct, so static_getitem must be special-cased for
        # this exact call shape (see _read/_getitem interception below).
        ident = self._ident(target_name)
        self._atomic_cas_results[target_name] = (f"{ident}_0", f"{ident}_1")
        self.builder.write(f"{msl_elem_ty} {ident}_0;")
        self.builder.write("bool " + f"{ident}_1;")
        self.builder.write(f"{ident}_1 = {success_expr};")
        self.builder.write(f"{ident}_0 = {tmp};")

    def _const_text(self, py_val, target_name: str) -> str:
        # A module-level global declared with an explicit numpy scalar
        # type (e.g. `STEFAN_BOLTZMANN = np.float32(5.67e-8)`, the
        # idiomatic way to pin a physical constant's dtype) arrives here
        # as a `numpy.float32`/`numpy.int32`/`numpy.bool_` instance, none
        # of which are subclasses of Python's own `bool`/`int`/`float` --
        # every isinstance check below would silently miss it and fall
        # through to the UnsupportedFeatureError at the bottom, EXCEPT
        # that (found the hard way, via a NaN in a kernel using exactly
        # this pattern) this function is also reached from `_read()`'s
        # global-constant path, which does its own narrower isinstance
        # check before ever calling here -- so a numpy-scalar global was
        # actually falling through *there* to an unassigned/undeclared
        # variable reference, reading uninitialized MSL stack memory
        # (observed as 0.0 or NaN depending on luck), never even reaching
        # this function or its error path. Normalizing numpy scalars to
        # plain Python values up front fixes both call sites at once.
        if isinstance(py_val, np.bool_):
            py_val = bool(py_val)
        elif isinstance(py_val, np.integer):
            py_val = int(py_val)
        elif isinstance(py_val, np.floating):
            py_val = float(py_val)
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
