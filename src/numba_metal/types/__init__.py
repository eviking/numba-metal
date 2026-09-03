"""Type mapping between Numba scalar/array types and Metal Shading Language.

This module is the single source of truth for which dtypes numba-metal
supports and what MSL type/name each maps to. Nothing outside this module
should hard-code an MSL type name for a NumPy/Numba dtype.
"""

from __future__ import annotations

import numpy as np
from numba.core import types as nb_types

from numba_metal.errors import UnsupportedFeatureError

#: Required scalar dtypes (task-mandated minimum).
REQUIRED_DTYPES: frozenset[np.dtype] = frozenset(
    {
        np.dtype(np.float32),
        np.dtype(np.int32),
        np.dtype(np.uint32),
        np.dtype(np.bool_),
    }
)

#: Optional scalar dtypes, supported but with documented caveats.
OPTIONAL_DTYPES: frozenset[np.dtype] = frozenset(
    {
        np.dtype(np.float16),
        np.dtype(np.int64),
    }
)

#: All dtypes numba-metal will accept for device arrays / kernel arguments.
SUPPORTED_DTYPES: frozenset[np.dtype] = REQUIRED_DTYPES | OPTIONAL_DTYPES

# NumPy dtype -> MSL scalar type name.
_NUMPY_TO_MSL: dict[np.dtype, str] = {
    np.dtype(np.float32): "float",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.bool_): "bool",
    np.dtype(np.float16): "half",
    np.dtype(np.int64): "long",
}

# Numba scalar Type -> MSL scalar type name.
_NUMBA_SCALAR_TO_MSL: dict[type, str] = {
    nb_types.float32: "float",
    nb_types.int32: "int",
    nb_types.uint32: "uint",
    nb_types.boolean: "bool",
    nb_types.float16: "half",
    nb_types.int64: "long",
    # int64 also arises as the type of metal.grid()/loop counters; treat
    # Python-int-derived intp/int64 the same as explicit int64.
    nb_types.intp: "long",
    nb_types.uint64: "ulong",
    # float64 intermediate results (e.g. from literal float folding) are
    # mapped to MSL's "float" (32-bit) because double precision is not
    # verified/supported -- see docs/limitations.md. This is a deliberate,
    # documented precision narrowing, not silent data corruption: values
    # simply are not eligible to be *inputs* typed as float64 (that raises
    # UnsupportedFeatureError below), only intermediate literals do this.
}


def numpy_dtype_to_msl(dtype: np.dtype) -> str:
    """Return the MSL scalar type name for a supported NumPy dtype.

    Raises UnsupportedFeatureError if the dtype is not one numba-metal
    supports (see SUPPORTED_DTYPES).
    """
    dtype = np.dtype(dtype)
    if dtype not in SUPPORTED_DTYPES:
        supported = ", ".join(sorted(d.name for d in SUPPORTED_DTYPES))
        raise UnsupportedFeatureError(
            f"Unsupported array dtype {dtype!r}. Supported dtypes: {supported}."
        )
    return _NUMPY_TO_MSL[dtype]


def numba_scalar_to_msl(ty: nb_types.Type) -> str:
    """Return the MSL scalar type name for a supported Numba scalar Type."""
    if ty == nb_types.float64:
        raise UnsupportedFeatureError(
            "float64 values are not supported as kernel arguments or array "
            "elements (unverified precision/hardware support on Apple GPU "
            "families -- see docs/limitations.md). Use float32."
        )
    for numba_ty, msl_name in _NUMBA_SCALAR_TO_MSL.items():
        if ty == numba_ty:
            return msl_name
    raise UnsupportedFeatureError(
        f"Unsupported scalar type {ty!r} in kernel; no MSL mapping exists."
    )


def msl_c_size(dtype: np.dtype) -> int:
    """Return itemsize in bytes for a supported dtype (host-side buffer sizing)."""
    return np.dtype(dtype).itemsize
