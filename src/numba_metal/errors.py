"""Dedicated exception types for numba-metal.

All errors that originate from user-facing failure modes (unsupported
syntax, unsupported types, launch-configuration mistakes, Metal compiler
or runtime failures) raise one of these instead of a generic ``Exception``,
so that callers can distinguish "your kernel is unsupported" from "the
Metal driver rejected this" from "your Mac cannot run this at all".
"""

from __future__ import annotations


class NumbaMetalError(Exception):
    """Base class for all numba-metal errors."""


class UnsupportedPlatformError(NumbaMetalError):
    """Raised when the current machine/OS cannot run numba-metal at all.

    This covers non-Apple-silicon hardware, unsupported macOS versions, and
    a missing Metal device -- conditions that are fatal and not specific to
    any one kernel.
    """


class MetalToolchainError(NumbaMetalError):
    """Raised when the Metal compiler toolchain is missing or unusable."""


class UnsupportedFeatureError(NumbaMetalError):
    """Raised at compile time when a kernel uses an unsupported Python
    construct, operator, function, or type.

    The message always names the specific unsupported node/operation/type;
    numba-metal never silently falls back to another execution path.
    """


class KernelCompilationError(NumbaMetalError):
    """Raised when a kernel fails to compile.

    This wraps both numba-metal's own typing/lowering failures and errors
    reported by Apple's Metal shader compiler (``MTLDevice`` library
    compilation). The original Metal compiler diagnostic text, if any, is
    preserved in the exception message.
    """


class KernelLaunchError(NumbaMetalError):
    """Raised for invalid launch configurations (grid/block sizes, argument
    count/type mismatches) detected before or during dispatch.
    """


class MetalRuntimeError(NumbaMetalError):
    """Raised when the Metal runtime reports an error during buffer
    allocation, command submission, or execution (e.g. a GPU command
    buffer completing with an error status).
    """
