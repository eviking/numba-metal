"""Tests for launch-configuration parsing/validation. Pure Python logic,
no GPU required.
"""

from __future__ import annotations

import pytest

from numba_metal.errors import KernelLaunchError
from numba_metal.runtime.dispatcher import _parse_launch_config


def test_1d_launch_config_parses() -> None:
    blocks, threads = _parse_launch_config((4, 256))
    assert blocks == 4
    assert threads == 256


def test_2d_launch_config_parses() -> None:
    blocks, threads = _parse_launch_config(((1, 1), (16, 16)))
    assert blocks == (1, 1)
    assert threads == (16, 16)


def test_rejects_non_tuple_launch_config() -> None:
    with pytest.raises(KernelLaunchError):
        _parse_launch_config(256)


def test_rejects_wrong_length_tuple() -> None:
    with pytest.raises(KernelLaunchError):
        _parse_launch_config((1, 2, 3))


def test_rejects_float_dims() -> None:
    with pytest.raises(KernelLaunchError):
        _parse_launch_config((1.5, 256))


def test_rejects_bool_dims() -> None:
    with pytest.raises(KernelLaunchError):
        _parse_launch_config((True, 256))


def test_rejects_mixed_1d_2d() -> None:
    with pytest.raises(KernelLaunchError):
        _parse_launch_config((1, (16, 16)))
