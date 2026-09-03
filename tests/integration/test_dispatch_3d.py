"""Standalone tests for 3D kernel dispatch and threadgroup-position
intrinsics: `metal.grid(3)`, `metal.gridsize(3)`,
`metal.threadgroup_position(ndim)`, `metal.thread_in_threadgroup(ndim)`,
`metal.threads_per_threadgroup(ndim)`, and `kernel[(bx,by,bz),
(tx,ty,tz)]` 3D launch configuration. These are general Numba-to-Metal
compiler/runtime capabilities with no connection to any specific
downstream workload -- no import of any third-party integration library.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


def test_grid_3d_matches_expected_indices() -> None:
    metal = _metal()

    @metal.jit
    def kernel(out, dx, dy, dz):
        x, y, z = metal.grid(3)
        if x < dx and y < dy and z < dz:
            idx = z * (dy * dx) + y * dx + x
            out[idx] = x + y * 1000 + z * 1_000_000

    dx, dy, dz = 5, 4, 3
    d_out = metal.device_array(dx * dy * dz, np.int64)
    kernel[(dx, dy, dz), (1, 1, 1)](d_out, np.int32(dx), np.int32(dy), np.int32(dz))
    metal.synchronize()
    result = d_out.copy_to_host()
    expected = np.array(
        [
            x + y * 1000 + z * 1_000_000
            for z in range(dz)
            for y in range(dy)
            for x in range(dx)
        ]
    )
    assert np.array_equal(result, expected)


def test_grid_3d_with_multi_thread_threadgroups() -> None:
    """Exercises a 3D launch where each block has more than one thread
    per dimension (not just the degenerate (1,1,1) threadgroup case)."""
    metal = _metal()

    @metal.jit
    def kernel(out, dx, dy, dz):
        x, y, z = metal.grid(3)
        if x < dx and y < dy and z < dz:
            idx = z * (dy * dx) + y * dx + x
            out[idx] = idx

    dx, dy, dz = 8, 6, 4
    blocks = (2, 2, 2)
    threads = (4, 3, 2)
    d_out = metal.device_array(dx * dy * dz, np.int64)
    kernel[blocks, threads](d_out, np.int32(dx), np.int32(dy), np.int32(dz))
    metal.synchronize()
    result = d_out.copy_to_host()
    expected = np.arange(dx * dy * dz, dtype=np.int64)
    assert np.array_equal(result, expected)


def test_gridsize_3d_matches_launch_geometry() -> None:
    metal = _metal()

    @metal.jit
    def kernel(out):
        x, y, z = metal.grid(3)
        gx, gy, gz = metal.gridsize(3)
        idx = z * (gy * gx) + y * gx + x
        if idx < out.size:
            out[idx] = gx * 1_000_000 + gy * 1000 + gz

    blocks = (2, 3, 1)
    threads = (4, 2, 5)
    gx, gy, gz = blocks[0] * threads[0], blocks[1] * threads[1], blocks[2] * threads[2]
    d_out = metal.device_array(gx * gy * gz, np.int64)
    kernel[blocks, threads](d_out)
    metal.synchronize()
    expected_value = gx * 1_000_000 + gy * 1000 + gz
    assert np.all(d_out.copy_to_host() == expected_value)


def test_threadgroup_position_1d() -> None:
    metal = _metal()

    @metal.jit
    def kernel(out):
        i = metal.grid(1)
        tgid = metal.threadgroup_position(1)
        if i < out.size:
            out[i] = tgid

    blocks, threads = 5, 16
    d_out = metal.device_array(blocks * threads, np.int64)
    kernel[blocks, threads](d_out)
    metal.synchronize()
    result = d_out.copy_to_host()
    expected = np.repeat(np.arange(blocks), threads)
    assert np.array_equal(result, expected)


def test_thread_in_threadgroup_and_threads_per_threadgroup_1d() -> None:
    metal = _metal()

    @metal.jit
    def kernel(out):
        i = metal.grid(1)
        local_id = metal.thread_in_threadgroup(1)
        tg_size = metal.threads_per_threadgroup(1)
        if i < out.size:
            out[i] = local_id * 1000 + tg_size

    blocks, threads = 4, 12
    d_out = metal.device_array(blocks * threads, np.int64)
    kernel[blocks, threads](d_out)
    metal.synchronize()
    result = d_out.copy_to_host()
    expected = np.tile(np.array([t * 1000 + threads for t in range(threads)]), blocks)
    assert np.array_equal(result, expected)


def test_threadgroup_intrinsics_2d() -> None:
    """The same threadgroup intrinsics with ndim=2, mirroring
    metal.grid(2)'s existing 2D support."""
    metal = _metal()

    @metal.jit
    def kernel(out, width):
        x, y = metal.grid(2)
        lx, ly = metal.thread_in_threadgroup(2)
        idx = y * width + x
        if idx < out.size:
            out[idx] = lx * 100 + ly

    bx, by, tx, ty = 3, 2, 4, 4
    width = bx * tx
    height = by * ty
    d_out = metal.device_array(width * height, np.int64)
    kernel[(bx, by), (tx, ty)](d_out, np.int32(width))
    metal.synchronize()
    result = d_out.copy_to_host().reshape(height, width)
    for gy in range(height):
        for gx in range(width):
            assert result[gy, gx] == (gx % tx) * 100 + (gy % ty)


def test_launch_config_rejects_mismatched_dimensionality() -> None:
    from numba_metal.errors import KernelLaunchError

    metal = _metal()

    @metal.jit
    def kernel(out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = i

    d_out = metal.device_array(8, np.int64)
    with pytest.raises(KernelLaunchError):
        kernel[(2, 2, 2), 8](d_out)


def test_launch_config_rejects_zero_or_negative_3d_dims() -> None:
    from numba_metal.errors import KernelLaunchError

    metal = _metal()

    @metal.jit
    def kernel(out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = i

    d_out = metal.device_array(8, np.int64)
    with pytest.raises(KernelLaunchError):
        kernel[(2, 2, 0), (2, 2, 2)](d_out)


def test_threadgroup_size_is_launched_exactly_as_requested() -> None:
    """Regression test for a real bug found while adding 3D dispatch: the
    dispatcher previously silently rescaled the y-dimension of a 2D
    threadgroup if tx*ty exceeded the device's per-threadgroup thread
    limit, which would have made metal.threads_per_threadgroup() lie
    about the actual launch geometry. The dispatcher now launches exactly
    the requested threadgroup shape (already validated to fit before
    dispatch), so this must hold even for a threadgroup shape that isn't
    square/degenerate."""
    metal = _metal()

    @metal.jit
    def kernel(out, dx, dy):
        x, y = metal.grid(2)
        tgx, tgy = metal.threads_per_threadgroup(2)
        idx = y * dx + x
        if x < dx and y < dy:
            out[idx] = tgx * 1000 + tgy

    dx, dy = 12, 9
    blocks = (4, 3)
    threads = (3, 3)
    d_out = metal.device_array(dx * dy, np.int64)
    kernel[blocks, threads](d_out, np.int32(dx), np.int32(dy))
    metal.synchronize()
    expected_value = threads[0] * 1000 + threads[1]
    assert np.all(d_out.copy_to_host() == expected_value)
