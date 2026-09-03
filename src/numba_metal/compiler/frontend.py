"""Numba frontend integration: bytecode -> typed IR, stopping before lowering.

This module contains the only code in numba-metal that touches Numba's
compiler-internal APIs (`numba.core.compiler.CompilerBase`,
`numba.core.compiler_machinery`, `numba.core.typed_passes`). It is kept
small and isolated (per the task's "adapter layer" instruction) because
these are semi-private/compiler-internal APIs whose shape can change
between Numba releases; see docs/architecture.md for the full rationale
and the specific APIs used.

The output of `compile_to_typed_ir` is Numba's typed intermediate
representation: a `numba.core.ir.FunctionIR` with a `typemap` (Var name ->
Type) and `calltypes` (call-site -> Signature). numba-metal's own backend
(numba_metal.compiler.msl_backend) walks this structure directly to emit
MSL; Numba's own LLVM-based lowering passes are never run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numba.core import types as nb_types
from numba.core.compiler import CompilerBase, DefaultPassBuilder, Flags
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.core.typed_passes import AnnotateTypes, NopythonTypeInference

from numba_metal.errors import KernelCompilationError


@dataclass
class TypedKernelIR:
    """Result of running Numba's frontend + type inference on a kernel.

    Attributes mirror the subset of `numba.core.compiler.StateDict` fields
    the MSL backend needs; kept as a plain dataclass so the backend never
    has to reach back into Numba's internal `state` object.
    """

    func_ir: Any  # numba.core.ir.FunctionIR
    typemap: dict[str, nb_types.Type]
    calltypes: dict[Any, Any]
    return_type: nb_types.Type
    arg_types: tuple[nb_types.Type, ...]
    arg_names: tuple[str, ...]


@register_pass(mutates_CFG=False, analysis_only=True)
class _CaptureTypedIR(FunctionPass):
    """Final pass of the typed-IR-only pipeline: copies what the MSL backend
    needs out of the compiler state into a plain, Numba-independent object.

    Required because `CompilerBase.compile_extra` demands `state.cr` be set
    to something by the end of the pipeline; there is no public "just give
    me the typed IR" entry point in this Numba version, so a one-pass
    pipeline that captures state right after typing is the smallest
    adapter that achieves it (see docs/architecture.md).
    """

    _name = "numba_metal_capture_typed_ir"

    def __init__(self) -> None:
        FunctionPass.__init__(self)

    def run_pass(self, state) -> bool:
        """Numba FunctionPass entry point: copy the typed IR out of
        Numba's internal compiler state into a plain TypedKernelIR."""
        state.cr = TypedKernelIR(
            func_ir=state.func_ir,
            typemap=dict(state.typemap),
            calltypes=dict(state.calltypes),
            return_type=state.return_type,
            arg_types=tuple(state.args),
            arg_names=tuple(state.func_ir.arg_names),
        )
        return True


class _TypedIROnlyCompiler(CompilerBase):
    """A Numba compiler pipeline that runs the standard untyped frontend
    passes plus type inference, then stops -- no lowering, no LLVM, no
    object mode fallback. If type inference fails, Numba raises a
    `TypingError` as usual, which `compile_to_typed_ir` converts to a
    `KernelCompilationError`.
    """

    def define_pipelines(self):
        """Numba CompilerBase entry point: build the untyped-frontend +
        type-inference-only pass pipeline."""
        pm = PassManager("numba_metal_typed_ir_only")
        untyped_pipeline = DefaultPassBuilder.define_untyped_pipeline(self.state)
        for pass_cls, pass_name in untyped_pipeline.passes:
            pm.add_pass(pass_cls, pass_name)
        pm.add_pass(NopythonTypeInference, "nopython frontend type inference")
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(_CaptureTypedIR, "capture typed ir for numba-metal backend")
        pm.finalize()
        return [pm]


def make_flags() -> Flags:
    """Build the Numba compiler Flags used for typed-IR-only compilation
    (no NRT, no CPython/cfunc wrappers -- this pipeline never lowers)."""
    flags = Flags()
    flags.nrt = False
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True
    flags.error_model = "numpy"
    return flags


def compile_to_typed_ir(func, arg_types: tuple[nb_types.Type, ...]) -> TypedKernelIR:
    """Run Numba's frontend and nopython type inference on `func`.

    Reuses Numba's bytecode-to-IR translation, control-flow reconstruction,
    and full nopython type inference exactly as the CPU target does;
    numba-metal only takes over after this point. Object-mode fallback is
    disabled (`nopython` semantics only, matching CUDA-style kernel
    languages) so an untypeable kernel fails loudly here rather than
    silently compiling to something else.
    """
    from numba.core.compiler import compile_extra

    flags = make_flags()
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    typingctx.refresh()
    targetctx.refresh()

    try:
        result = compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=func,
            args=arg_types,
            return_type=nb_types.void,
            flags=flags,
            locals={},
            pipeline_class=_TypedIROnlyCompiler,
        )
    except TypingError as exc:
        raise KernelCompilationError(
            f"Numba type inference failed for kernel "
            f"{getattr(func, '__name__', func)!r}: {exc}"
        ) from exc

    if not isinstance(result, TypedKernelIR):
        # Defensive: should be unreachable because _CaptureTypedIR always
        # sets state.cr to a TypedKernelIR, but a future Numba version could
        # change control flow (e.g. an early object-mode fallback) in a way
        # that bypasses our pass. Fail loudly rather than silently
        # returning something the backend can't walk.
        raise KernelCompilationError(
            "Internal error: numba-metal's typed-IR-only compiler pipeline "
            f"did not produce a TypedKernelIR (got {type(result)!r}). This "
            "usually means the installed Numba version changed compiler "
            "pipeline behavior in a way numba-metal's frontend adapter "
            "does not yet handle."
        )
    return result
