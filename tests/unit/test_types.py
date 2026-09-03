"""Tests for the Numba<->MSL type mapping module."""

from __future__ import annotations

import numpy as np
import pytest
from numba.core import types as nb_types

from numba_metal.errors import UnsupportedFeatureError
from numba_metal.types import numba_scalar_to_msl, numpy_dtype_to_msl


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float32, "float"),
        (np.int32, "int"),
        (np.uint32, "uint"),
        (np.bool_, "bool"),
        (np.float16, "half"),
        (np.int64, "long"),
    ],
)
def test_numpy_dtype_to_msl_supported(dtype, expected) -> None:
    assert numpy_dtype_to_msl(np.dtype(dtype)) == expected


def test_numpy_dtype_to_msl_rejects_float64() -> None:
    with pytest.raises(UnsupportedFeatureError):
        numpy_dtype_to_msl(np.dtype(np.float64))


def test_numpy_dtype_to_msl_rejects_object_dtype() -> None:
    with pytest.raises(UnsupportedFeatureError):
        numpy_dtype_to_msl(np.dtype(object))


def test_numba_scalar_to_msl_rejects_float64() -> None:
    with pytest.raises(UnsupportedFeatureError, match="float64"):
        numba_scalar_to_msl(nb_types.float64)


def test_numba_scalar_to_msl_accepts_float32() -> None:
    assert numba_scalar_to_msl(nb_types.float32) == "float"
