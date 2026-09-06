"""Standalone tests for real 2D/3D array kernel arguments
(`arr[x, y]`/`arr[x, y, z]` indexing), instead of the manual flattened
`arr[x*n+y]` scheme every prior benchmark/kernel used.

Motivated directly by a real, measured finding: heat_diffusion.py's
Metal kernel originally used a flattened 1D thread index recovering 2D
coordinates by hand (`i = metal.grid(1)`, `x = i // n`, `y = i % n`)
even though the underlying grid is genuinely 2D. Switching that single
kernel to a real `metal.grid(2)` launch -- with the SAME arithmetic and
memory addresses, nothing else changed -- measured a ~5x improvement at
1024x1024 on an Apple M4 Pro (see docs/performance-guidance.md). But
writing `x, y = metal.grid(2)` and then still manually flattening every
array access (`arr[x*n+y]`) leaves exactly the class of off-by-one/
wrong-stride bugs that hand-flattened indexing is prone to. This
feature closes that gap: numba-metal now accepts and correctly indexes
2D/3D device arrays directly.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


def test_2d_array_read_and_write_matches_numpy() -> None:
    metal = _metal()

    @metal.jit
    def add2d(a, b, out, n, m):
        x, y = metal.grid(2)
        if x < n and y < m:
            out[x, y] = a[x, y] + b[x, y]

    n, m = 7, 11
    rng = np.random.default_rng(0)
    a = rng.random((n, m)).astype(np.float32)
    b = rng.random((n, m)).astype(np.float32)
    d_a = metal.to_device(a)
    d_b = metal.to_device(b)
    d_out = metal.device_array((n, m), np.float32)
    add2d[(1, 1), (n, m)](d_a, d_b, d_out, np.int32(n), np.int32(m))
    metal.synchronize()
    np.testing.assert_array_equal(d_out.copy_to_host(), a + b)


def test_3d_array_read_and_write_matches_numpy() -> None:
    metal = _metal()

    @metal.jit
    def add3d(a, b, out, n0, n1, n2):
        x, y, z = metal.grid(3)
        if x < n0 and y < n1 and z < n2:
            out[x, y, z] = a[x, y, z] + b[x, y, z]

    n0, n1, n2 = 3, 4, 5
    rng = np.random.default_rng(1)
    a = rng.random((n0, n1, n2)).astype(np.float32)
    b = rng.random((n0, n1, n2)).astype(np.float32)
    d_a = metal.to_device(a)
    d_b = metal.to_device(b)
    d_out = metal.device_array((n0, n1, n2), np.float32)
    add3d[(1, 1, 1), (n0, n1, n2)](
        d_a, d_b, d_out, np.int32(n0), np.int32(n1), np.int32(n2)
    )
    metal.synchronize()
    np.testing.assert_array_equal(d_out.copy_to_host(), a + b)


def test_2d_stencil_matches_numpy_reference() -> None:
    """A real 5-point stencil, indexed with arr[x, y] directly -- the
    actual motivating case (heat_diffusion.py's access pattern), not a
    synthetic toy."""
    metal = _metal()

    @metal.jit
    def stencil(cur, nxt, n, m):
        x, y = metal.grid(2)
        if x < n and y < m:
            if 1 <= x < n - 1 and 1 <= y < m - 1:
                nxt[x, y] = 0.25 * (
                    cur[x - 1, y] + cur[x + 1, y] + cur[x, y - 1] + cur[x, y + 1]
                )
            else:
                nxt[x, y] = cur[x, y]

    n, m = 16, 20
    rng = np.random.default_rng(2)
    grid = rng.random((n, m)).astype(np.float32)

    def numpy_step(g):
        nxt = g.copy()
        nxt[1:-1, 1:-1] = 0.25 * (
            g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
        )
        return nxt

    expected = numpy_step(grid)

    d_cur = metal.to_device(grid)
    d_nxt = metal.device_array((n, m), np.float32)
    threads = (8, 8)
    blocks = ((n + 7) // 8, (m + 7) // 8)
    stencil[blocks, threads](d_cur, d_nxt, np.int32(n), np.int32(m))
    metal.synchronize()
    np.testing.assert_allclose(d_nxt.copy_to_host(), expected, rtol=1e-5, atol=1e-5)


def test_static_constant_2d_index_matches_numpy() -> None:
    """A literal-constant tuple index (`a[0, 1]`), which Numba
    constant-folds to a `static_getitem` with a plain-tuple `.index`
    rather than a `getitem` with a Var `.index` -- a structurally
    different IR shape from the variable-index case, exercised
    separately here."""
    metal = _metal()

    @metal.jit
    def read_fixed_cell(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[1, 2]

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    d_a = metal.to_device(a)
    d_out = metal.device_array(5, np.float32)
    read_fixed_cell[1, 5](d_a, d_out)
    metal.synchronize()
    expected = np.full(5, a[1, 2], dtype=np.float32)
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_static_constant_2d_setitem_matches_numpy() -> None:
    metal = _metal()

    @metal.jit
    def write_fixed_cell(out, value):
        i = metal.grid(1)
        if i == 0:
            out[2, 1] = value

    d_out = metal.to_device(np.zeros((4, 4), dtype=np.float32))
    write_fixed_cell[1, 1](d_out, np.float32(99.0))
    metal.synchronize()
    result = d_out.copy_to_host()
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[2, 1] = 99.0
    np.testing.assert_array_equal(result, expected)


def test_different_shaped_2d_arrays_get_independent_dim_parameters() -> None:
    """Two 2D array arguments with DIFFERENT shapes must each use their
    own `_dim1` companion value -- not silently share one, which would
    read/write the wrong flat offset for whichever array has a
    different row width."""
    metal = _metal()

    @metal.jit
    def copy_row0(src, dst, src_cols, dst_cols):
        col = metal.grid(1)
        if col < dst_cols:
            dst[0, col] = src[0, col]

    src = np.arange(3 * 7, dtype=np.float32).reshape(3, 7)
    dst = np.zeros((5, 7), dtype=np.float32)
    d_src = metal.to_device(src)
    d_dst = metal.to_device(dst)
    copy_row0[1, 7](d_src, d_dst, np.int32(7), np.int32(7))
    metal.synchronize()
    result = d_dst.copy_to_host()
    expected = dst.copy()
    expected[0, :] = src[0, :]
    np.testing.assert_array_equal(result, expected)


def test_2d_array_interacts_correctly_with_device_function() -> None:
    """A @metal.device_func call inside a kernel that ALSO uses 2D
    array indexing -- device functions themselves remain 1D-only (see
    msl_backend.py's _classify_params), but a kernel mixing both
    features must still compile and run correctly."""
    metal = _metal()

    @metal.device_func
    def double_it(v):
        return v * 2.0

    @metal.jit
    def kernel(a, out, n, m):
        x, y = metal.grid(2)
        if x < n and y < m:
            out[x, y] = double_it(a[x, y])

    n, m = 5, 6
    a = np.arange(n * m, dtype=np.float32).reshape(n, m)
    d_a = metal.to_device(a)
    d_out = metal.device_array((n, m), np.float32)
    kernel[(1, 1), (n, m)](d_a, d_out, np.int32(n), np.int32(m))
    metal.synchronize()
    np.testing.assert_array_equal(d_out.copy_to_host(), a * 2.0)


def test_2d_array_repeated_launches_are_correct() -> None:
    """Ping-ponging two 2D-indexed buffers across several launches (the
    exact heat_diffusion.py usage pattern) must remain correct across
    repeated dispatches, not just a single one."""
    metal = _metal()

    @metal.jit
    def increment(cur, nxt, n, m):
        x, y = metal.grid(2)
        if x < n and y < m:
            nxt[x, y] = cur[x, y] + 1.0

    n, m = 6, 6
    grid = np.zeros((n, m), dtype=np.float32)
    d_cur = metal.to_device(grid)
    d_nxt = metal.device_array((n, m), np.float32)
    iterations = 5
    for _ in range(iterations):
        increment[(1, 1), (n, m)](d_cur, d_nxt, np.int32(n), np.int32(m))
        d_cur, d_nxt = d_nxt, d_cur
    metal.synchronize()
    result = d_cur.copy_to_host()
    np.testing.assert_array_equal(result, np.full((n, m), float(iterations)))
