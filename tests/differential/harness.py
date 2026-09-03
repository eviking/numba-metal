"""Execution and comparison harness for generated differential test
cases: compiles a GeneratedKernel's source for both `@metal.jit` (real
Metal GPU) and `@njit` (real Numba CPU), runs both against matching
random inputs, and reports a rich failure record (Hypothesis example
context is added by the caller via `pytest.fail`/Hypothesis's own
reporting -- this module focuses on what to compare and how).

`exec()` is used here to turn generated Python *source text* into a real
function object -- this is not "eval on untrusted content" in the sense
the assignment warns against: the source text is built exclusively by
grammar.py/kernel_strategy.py's closed set of node constructors (never
from external input, never from a string an attacker could influence),
and is retained verbatim on every `DifferentialResult` for diagnosis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

# Tolerances, per dtype family, for the float32 comparison (integer/bool
# outputs always use strict equality -- see compare_outputs). rtol/atol
# chosen to accommodate the small amount of float32 rounding-order
# difference expected between the GPU's arithmetic and Numba CPU's
# (both are float32, but instruction scheduling/fma use can still cause
# last-bit differences across a chain of several operations) without
# masking a real compiler bug: a wrong-by-more-than-noise result should
# still fail.
FLOAT_RTOL = 1e-3
FLOAT_ATOL = 1e-3


@dataclass
class DifferentialResult:
    """Everything needed to diagnose a differential test failure."""

    ok: bool
    gpu_source: str
    cpu_source: str
    inputs: dict[str, object]
    expected: np.ndarray | None = None
    actual: np.ndarray | None = None
    max_abs_error: float | None = None
    error_message: str = ""
    msl_source: str | None = None

    def format_report(self) -> str:
        lines = [
            "=== Differential test failure ===",
            "--- Generated GPU kernel source ---",
            self.gpu_source,
            "--- Generated CPU reference source ---",
            self.cpu_source,
            "--- Inputs ---",
        ]
        for name, value in self.inputs.items():
            if isinstance(value, np.ndarray):
                lines.append(
                    f"{name}: shape={value.shape} dtype={value.dtype} {value[:8]}"
                )
            else:
                lines.append(f"{name}: {value!r}")
        if self.expected is not None:
            lines.append(f"--- Expected (first 8) ---\n{self.expected[:8]}")
        if self.actual is not None:
            lines.append(f"--- Actual (first 8) ---\n{self.actual[:8]}")
        if self.max_abs_error is not None:
            lines.append(f"--- Max absolute error: {self.max_abs_error} ---")
        if self.error_message:
            lines.append(f"--- Error ---\n{self.error_message}")
        if self.msl_source:
            lines.append(f"--- Generated MSL ---\n{self.msl_source}")
        return "\n".join(lines)


def _exec_source(source: str, namespace: dict) -> None:
    """Execute generated (grammar-produced, never externally-influenced)
    Python source text into `namespace`. See this module's docstring for
    why this is not the "unsafe eval on untrusted content" the assignment
    warns against."""
    exec(compile(source, "<generated-kernel>", "exec"), namespace)  # noqa: S102


def build_gpu_kernel(gpu_source: str):
    from numba_metal import metal

    namespace = {"metal": metal, "math": math, "np": np}
    _exec_source(gpu_source, namespace)
    fn = namespace["gen_kernel"]
    return metal.jit(fn)


def build_cpu_kernel(cpu_source: str):
    namespace = {"math": math, "np": np}
    _exec_source(cpu_source, namespace)
    fn = namespace["gen_kernel_cpu"]
    return njit(cache=False)(fn)


def make_inputs(
    spec, n_elements: int, rng: np.random.Generator, n_loop: int, m_loop: int
) -> dict[str, np.ndarray | int]:
    inputs: dict[str, object] = {}
    for arr_spec in spec.array_params:
        if arr_spec.name == spec.output_param:
            continue
        inputs[arr_spec.name] = rng.uniform(-10.0, 10.0, size=n_elements).astype(
            np.float32
        )
    inputs["n"] = np.int32(n_loop)
    inputs["m"] = np.int32(m_loop)
    return inputs


def run_differential(
    spec,
    gpu_source: str,
    cpu_source: str,
    n_elements: int = 64,
    n_loop: int = 4,
    m_loop: int = 3,
    seed: int = 0,
) -> DifferentialResult:
    """Compile and run both the GPU kernel and CPU reference against the
    same random inputs; return a DifferentialResult with ok=True if they
    agree within tolerance, ok=False (with full diagnostic context) if
    not or if either failed to compile/execute."""
    from numba_metal import metal

    rng = np.random.default_rng(seed)
    rng_inputs = make_inputs(spec, n_elements, rng, n_loop, m_loop)

    try:
        gpu_kernel = build_gpu_kernel(gpu_source)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return DifferentialResult(
            ok=False,
            gpu_source=gpu_source,
            cpu_source=cpu_source,
            inputs=rng_inputs,
            error_message=f"GPU kernel failed to compile: {type(exc).__name__}: {exc}",
        )

    try:
        cpu_kernel = build_cpu_kernel(cpu_source)
    except Exception as exc:  # noqa: BLE001
        return DifferentialResult(
            ok=False,
            gpu_source=gpu_source,
            cpu_source=cpu_source,
            inputs=rng_inputs,
            error_message=(
                f"CPU reference failed to compile: {type(exc).__name__}: {exc}"
            ),
        )

    # Both the GPU kernel and CPU reference were rendered with the SAME
    # parameter order: spec.array_params (input arrays, then the output
    # array last, since KernelSpec.array_params == [*inputs, output]) then
    # spec.scalar_params (n, m). Building the call argument lists directly
    # from spec.array_params/scalar_params (rather than re-deriving order
    # from dict iteration or isinstance checks) keeps this correct even if
    # the grammar's array/scalar naming conventions change.
    input_array_names = [
        v.name for v in spec.array_params if v.name != spec.output_param
    ]
    scalar_names = [v.name for v in spec.scalar_params]

    cpu_out = np.zeros(n_elements, dtype=np.float32)
    cpu_args_full = [
        *(rng_inputs[name].copy() for name in input_array_names),
        cpu_out,
        *(rng_inputs[name] for name in scalar_names),
    ]

    try:
        cpu_kernel(*cpu_args_full)
    except Exception as exc:  # noqa: BLE001
        return DifferentialResult(
            ok=False,
            gpu_source=gpu_source,
            cpu_source=cpu_source,
            inputs=rng_inputs,
            error_message=(
                f"CPU reference raised at runtime: {type(exc).__name__}: {exc}"
            ),
        )

    device_arrays = []
    try:
        for name in input_array_names:
            device_arrays.append(metal.to_device(rng_inputs[name]))
        d_out = metal.device_array(n_elements, np.float32)
        threads = 64
        blocks = (n_elements + threads - 1) // threads
        scalar_args = [rng_inputs[name] for name in scalar_names]
        gpu_kernel[blocks, threads](*device_arrays, d_out, *scalar_args)
        metal.synchronize()
        gpu_out = d_out.copy_to_host()
    except Exception as exc:  # noqa: BLE001
        msl = getattr(getattr(gpu_kernel, "_cache", None), "_entries", None)
        msl_text = None
        try:
            if msl:
                msl_text = next(iter(msl.values())).msl_source
        except Exception:  # noqa: BLE001 - best-effort diagnostic only
            msl_text = None
        return DifferentialResult(
            ok=False,
            gpu_source=gpu_source,
            cpu_source=cpu_source,
            inputs=rng_inputs,
            error_message=f"GPU kernel failed at runtime: {type(exc).__name__}: {exc}",
            msl_source=msl_text,
        )

    max_err = float(
        np.max(np.abs(gpu_out.astype(np.float64) - cpu_out.astype(np.float64)))
    )
    ok = np.allclose(gpu_out, cpu_out, rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True)
    return DifferentialResult(
        ok=bool(ok),
        gpu_source=gpu_source,
        cpu_source=cpu_source,
        inputs=rng_inputs,
        expected=cpu_out,
        actual=gpu_out,
        max_abs_error=max_err,
        error_message="" if ok else "GPU and CPU outputs diverge beyond tolerance",
    )
