"""Standalone tests for `@metal.device_func`: calling a separately
-defined helper function from inside a `@metal.jit` kernel body (or from
another device function). General Numba-to-Metal compiler capability,
no connection to any specific downstream workload.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


def test_device_function_basic_call() -> None:
    metal = _metal()

    @metal.device_func
    def double_plus_one(x):
        return x * 2.0 + 1.0

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = double_plus_one(a[i])

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, 4](d_a, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), a * 2.0 + 1.0)


def test_device_function_multiple_arguments() -> None:
    metal = _metal()

    @metal.device_func
    def weighted_sum(a, b, wa, wb):
        return a * wa + b * wb

    @metal.jit
    def kernel(a, b, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = weighted_sum(a[i], b[i], 0.25, 0.75)

    a = np.array([4.0, 8.0], dtype=np.float32)
    b = np.array([2.0, 6.0], dtype=np.float32)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_out = metal.device_array_like(a)
    kernel[1, 2](d_a, d_b, d_out)
    metal.synchronize()
    expected = a * 0.25 + b * 0.75
    assert np.allclose(d_out.copy_to_host(), expected)


def test_device_function_multiple_call_sites_same_kernel() -> None:
    metal = _metal()

    @metal.device_func
    def square(x):
        return x * x

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = square(a[i]) + square(a[i] + 1.0)

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, 3](d_a, d_out)
    metal.synchronize()
    expected = a**2 + (a + 1.0) ** 2
    assert np.allclose(d_out.copy_to_host(), expected)


def test_device_function_called_from_two_different_kernels() -> None:
    metal = _metal()

    @metal.device_func
    def cube(x):
        return x * x * x

    @metal.jit
    def kernel_a(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = cube(a[i])

    @metal.jit
    def kernel_b(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = cube(a[i]) + 1.0

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out_a = metal.device_array_like(a)
    d_out_b = metal.device_array_like(a)
    kernel_a[1, 3](d_a, d_out_a)
    kernel_b[1, 3](d_a, d_out_b)
    metal.synchronize()
    assert np.allclose(d_out_a.copy_to_host(), a**3)
    assert np.allclose(d_out_b.copy_to_host(), a**3 + 1.0)


def test_device_function_transitive_dependency_shared_across_kernels() -> None:
    """A device function (`outer`) that itself calls another device
    function (`inner`), called from two DIFFERENT kernels -- the second
    kernel's compile hits the process-wide device-function compile cache
    (see msl_backend.py's `_device_function_compile_cache`) for `outer`,
    which must still splice `inner`'s MSL source into that second
    kernel's own compiled output (a real bug found and fixed while
    building this cache: caching only a device function's own body,
    not its transitive dependencies, produced an 'undeclared identifier'
    Metal shader-compile error for the second kernel)."""
    metal = _metal()

    @metal.device_func
    def inner(x):
        return x * 2.0

    @metal.device_func
    def outer(x):
        return inner(x) + 1.0

    @metal.jit
    def kernel1(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = outer(a[i])

    @metal.jit
    def kernel2(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = outer(a[i]) + 100.0

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out1 = metal.device_array_like(a)
    d_out2 = metal.device_array_like(a)
    kernel1[1, 3](d_a, d_out1)
    kernel2[1, 3](d_a, d_out2)  # hits the process-wide cache for `outer`
    metal.synchronize()
    assert np.allclose(d_out1.copy_to_host(), a * 2.0 + 1.0)
    assert np.allclose(d_out2.copy_to_host(), a * 2.0 + 1.0 + 100.0)


def test_device_function_diamond_dependency_across_kernels() -> None:
    """Two device functions (`a_fn`, `b_fn`) sharing one common
    dependency (`shared_c`), called together from two different kernels
    -- the second kernel's compile hits the process-wide cache for BOTH
    `a_fn` and `b_fn`, each of which independently lists `shared_c` as a
    transitive dependency. This must not emit `shared_c`'s MSL function
    definition twice in the second kernel's own compiled source (a real
    bug found and fixed while building this cache: naively splicing
    every cache hit's full transitive-dependency list, without checking
    what this specific compile had already appended, produced a
    duplicate MSL function definition -- tolerated silently by Metal's
    compiler in testing, but not something to rely on)."""
    metal = _metal()

    @metal.device_func
    def shared_c(x):
        return x * 2.0

    @metal.device_func
    def a_fn(x):
        return shared_c(x) + 1.0

    @metal.device_func
    def b_fn(x):
        return shared_c(x) + 2.0

    @metal.jit
    def kernel1(x, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a_fn(x[i]) + b_fn(x[i])

    @metal.jit
    def kernel2(x, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a_fn(x[i]) - b_fn(x[i])

    xs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_x = metal.to_device(xs)
    d_out1 = metal.device_array_like(xs)
    d_out2 = metal.device_array_like(xs)
    kernel1[1, 3](d_x, d_out1)
    kernel2[1, 3](d_x, d_out2)  # hits the process-wide cache for both
    metal.synchronize()
    assert np.allclose(d_out1.copy_to_host(), (xs * 2.0 + 1.0) + (xs * 2.0 + 2.0))
    assert np.allclose(d_out2.copy_to_host(), (xs * 2.0 + 1.0) - (xs * 2.0 + 2.0))


def test_device_function_calling_another_device_function() -> None:
    """Nested device functions: one device function calling another."""
    metal = _metal()

    @metal.device_func
    def square(x):
        return x * x

    @metal.device_func
    def hypot2(a, b):
        return square(a) + square(b)

    @metal.jit
    def kernel(a, b, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = hypot2(a[i], b[i])

    a = np.array([3.0, 5.0, 8.0], dtype=np.float32)
    b = np.array([4.0, 12.0, 15.0], dtype=np.float32)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_out = metal.device_array_like(a)
    kernel[1, 3](d_a, d_b, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), a**2 + b**2)


def test_device_function_with_control_flow() -> None:
    """A device function with nested if/return -- exercises that
    structured control-flow reconstruction and multi-exit-point
    lowering work identically inside a device function as inside a
    kernel."""
    metal = _metal()

    @metal.device_func
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            v1 = clamp(a[i], 0.0, 10.0)
            v2 = clamp(v1 * 2.0, 0.0, 10.0)
            out[i] = v2

    a = np.array([-5.0, 3.0, 8.0, 20.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[1, 4](d_a, d_out)
    metal.synchronize()
    expected = np.clip(np.clip(a, 0, 10) * 2, 0, 10)
    assert np.allclose(d_out.copy_to_host(), expected)


def test_device_function_with_loop() -> None:
    metal = _metal()

    @metal.device_func
    def sum_to_n(n):
        total = 0.0
        for k in range(n):
            total = total + float(k)
        return total

    @metal.jit
    def kernel(out, n):
        i = metal.grid(1)
        if i < out.size:
            out[i] = sum_to_n(n)

    n = 5
    d_out = metal.device_array(1, np.float32)
    kernel[1, 1](d_out, np.int32(n))
    metal.synchronize()
    assert d_out.copy_to_host()[0] == pytest.approx(sum(range(n)))


def test_device_function_int_and_bool_types() -> None:
    metal = _metal()

    @metal.device_func
    def is_positive(x):
        return x > 0

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = is_positive(a[i])

    a = np.array([-3, 0, 5, -1, 2], dtype=np.int32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(len(a), np.bool_)
    kernel[1, len(a)](d_a, d_out)
    metal.synchronize()
    assert np.array_equal(d_out.copy_to_host(), a > 0)


def test_device_function_rejects_direct_recursion() -> None:
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    @metal.device_func
    def recurse(x):
        return recurse(x) + 1.0

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = recurse(a[i])

    d_a = metal.to_device(np.array([1.0], dtype=np.float32))
    d_out = metal.device_array(1, np.float32)
    with pytest.raises(NumbaMetalError):
        kernel[1, 1](d_a, d_out)


def test_device_function_rejects_mutual_recursion() -> None:
    """A cycle between two device functions: each function's own typing
    succeeds in isolation (Numba's frontend cannot see across separate
    compile_to_typed_ir calls), so this specifically exercises
    numba-metal's own in-progress-compilation cycle detection, not
    Numba's built-in direct-recursion check."""
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    @metal.device_func
    def is_even(n):
        if n == 0:
            return True
        return is_odd(n - 1)

    @metal.device_func
    def is_odd(n):
        if n == 0:
            return False
        return is_even(n - 1)

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = is_even(a[i])

    d_a = metal.to_device(np.array([4], dtype=np.int32))
    d_out = metal.device_array(1, np.bool_)
    with pytest.raises(NumbaMetalError):
        kernel[1, 1](d_a, d_out)


def test_device_function_accepts_1d_array_argument() -> None:
    """1D array arguments to @metal.device_func: general capability
    (device functions used to be scalar-argument-only), needed for any
    device function that reads/writes a caller-provided buffer directly
    (e.g. a reusable atomic read-modify-write helper -- see the CAS
    tests below)."""
    metal = _metal()

    @metal.device_func
    def read_and_add(arr, idx, val):
        return arr[idx] + val

    @metal.jit
    def kernel(arr, vals, out, n):
        i = metal.grid(1)
        if i < n:
            out[i] = read_and_add(arr, i, vals[i])

    arr = np.arange(10, dtype=np.float32)
    vals = np.arange(10, dtype=np.float32) * 10.0
    d_arr, d_vals = metal.to_device(arr), metal.to_device(vals)
    d_out = metal.device_array_like(arr)
    kernel[1, 10](d_arr, d_vals, d_out, np.int32(10))
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), arr + vals)


def test_device_function_array_argument_mixed_with_scalars() -> None:
    """Array and scalar parameters mixed in one device-function signature
    -- exercises that array/scalar param-order bookkeeping isn't broken
    by mixing kinds (unlike a kernel, which always lists array params via
    the same param_order mechanism, but this path is exercised
    independently for device functions since their signature emission is
    a separate code path -- see msl_backend.py's
    `_emit_device_function_signature`)."""
    metal = _metal()

    @metal.device_func
    def scaled_lookup(arr, idx, scale, offset):
        return arr[idx] * scale + offset

    @metal.jit
    def kernel(arr, out, scale, offset, n):
        i = metal.grid(1)
        if i < n:
            out[i] = scaled_lookup(arr, i, scale, offset)

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_arr = metal.to_device(arr)
    d_out = metal.device_array_like(arr)
    kernel[1, 3](d_arr, d_out, np.float32(2.0), np.float32(1.0), np.int32(3))
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), arr * 2.0 + 1.0)


def test_device_function_array_argument_forwarded_to_nested_call() -> None:
    """An array argument threaded through one device function into
    another (not just used directly) -- exercises the call-site
    array-forwarding path (each array argument expands to a
    pointer+size pair at every call site, including a device-function
    -to-device-function call) rather than only a kernel-to-device-function
    call site."""
    metal = _metal()

    @metal.device_func
    def inner_read(arr, idx):
        return arr[idx]

    @metal.device_func
    def outer_read_plus_one(arr, idx):
        return inner_read(arr, idx) + 1.0

    @metal.jit
    def kernel(arr, out, n):
        i = metal.grid(1)
        if i < n:
            out[i] = outer_read_plus_one(arr, i)

    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    d_arr = metal.to_device(arr)
    d_out = metal.device_array_like(arr)
    kernel[1, 3](d_arr, d_out, np.int32(3))
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), arr + 1.0)


def test_device_function_atomic_compare_exchange_cas_loop_under_contention() -> None:
    """The motivating use case: a device function wrapping a NaN-agnostic
    atomic-min update as a compare-and-swap retry loop, called by many
    threads targeting a small number of shared array slots (heavy
    contention). Verified against a manual sequential reference, not
    just "doesn't crash" -- this proves both array arguments AND
    metal.atomic_compare_exchange() work correctly when called from
    inside a @metal.device_func body, under real multi-thread races."""
    metal = _metal()

    @metal.device_func
    def atomic_min_update(agg, idx, field):
        done = False
        while not done:
            cur = agg[idx]
            new = min(field, cur)
            _prev, ok = metal.atomic_compare_exchange(agg, idx, cur, new)
            done = ok
        return new

    @metal.jit
    def kernel(agg, vals, n_slots, n):
        i = metal.grid(1)
        if i < n:
            idx = i % n_slots
            atomic_min_update(agg, idx, vals[i])

    rng = np.random.default_rng(0)
    n = 200_000
    n_slots = 8
    vals = rng.uniform(-1000, 1000, n).astype(np.float32)
    agg = np.full(n_slots, np.inf, dtype=np.float32)

    d_agg = metal.to_device(agg)
    d_vals = metal.to_device(vals)
    blocks = max(1, (n + 255) // 256)
    kernel[blocks, 256](d_agg, d_vals, np.int32(n_slots), np.int32(n))
    metal.synchronize()
    result = d_agg.copy_to_host()

    expected = np.full(n_slots, np.inf, dtype=np.float32)
    for i in range(n):
        s = i % n_slots
        expected[s] = min(expected[s], vals[i])

    np.testing.assert_array_equal(result, expected)


def test_device_function_accepts_2d_array_argument() -> None:
    """2D array arguments to @metal.device_func: a multi-dim
    device-function array parameter gets its own `_dimN` companion
    parameters (see msl_backend.py's `_emit_device_function_signature`),
    threaded through from the caller's own `arg_{name}_dimN` at the
    call site -- exercises a real spatial (stencil-style) access
    pattern, the exact shape heat_diffusion.py/mandelbrot.py's own 2D
    conversion targeted, factored into a reusable helper."""
    metal = _metal()

    @metal.device_func
    def neighbor_sum(grid, x, y):
        return grid[x - 1, y] + grid[x + 1, y] + grid[x, y - 1] + grid[x, y + 1]

    @metal.jit
    def kernel(cur, out, n, m):
        x, y = metal.grid(2)
        if x < n and y < m:
            if 1 <= x < n - 1 and 1 <= y < m - 1:
                out[x, y] = 0.25 * neighbor_sum(cur, x, y)
            else:
                out[x, y] = cur[x, y]

    n = 8
    grid = np.zeros((n, n), dtype=np.float32)
    grid[3:5, 3:5] = 100.0
    d_cur = metal.to_device(grid)
    d_out = metal.device_array_like(grid)
    kernel[(1, 1), (n, n)](d_cur, d_out, np.int32(n), np.int32(n))
    metal.synchronize()
    result = d_out.copy_to_host()

    expected = grid.copy()
    expected[1:-1, 1:-1] = 0.25 * (
        grid[:-2, 1:-1] + grid[2:, 1:-1] + grid[1:-1, :-2] + grid[1:-1, 2:]
    )
    assert np.allclose(result, expected)


def test_device_function_accepts_3d_array_argument() -> None:
    metal = _metal()

    @metal.device_func
    def get3d(a, x, y, z):
        return a[x, y, z]

    @metal.jit
    def kernel(a, out, nx, ny, nz):
        x, y, z = metal.grid(3)
        if x < nx and y < ny and z < nz:
            out[x, y, z] = get3d(a, x, y, z) * 2.0

    nx, ny, nz = 4, 5, 6
    a = np.arange(nx * ny * nz, dtype=np.float32).reshape(nx, ny, nz)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    kernel[(1, 1, 1), (nx, ny, nz)](
        d_a, d_out, np.int32(nx), np.int32(ny), np.int32(nz)
    )
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), a * 2.0)


def test_device_function_multidim_array_forwarded_to_nested_call() -> None:
    """A 2D array argument threaded through one device function into
    another -- mirrors
    test_device_function_array_argument_forwarded_to_nested_call but for
    a multi-dim array, exercising that the `_dimN` companion arguments
    (not just the pointer+size pair) are forwarded correctly at a
    device-function-to-device-function call site, not only from a
    kernel."""
    metal = _metal()

    @metal.device_func
    def inner_read(arr, x, y):
        return arr[x, y]

    @metal.device_func
    def outer_read_plus_one(arr, x, y):
        return inner_read(arr, x, y) + 1.0

    @metal.jit
    def kernel(arr, out, n, m):
        x, y = metal.grid(2)
        if x < n and y < m:
            out[x, y] = outer_read_plus_one(arr, x, y)

    n, m = 3, 4
    arr = np.arange(n * m, dtype=np.float32).reshape(n, m)
    d_arr = metal.to_device(arr)
    d_out = metal.device_array_like(arr)
    kernel[(1, 1), (n, m)](d_arr, d_out, np.int32(n), np.int32(m))
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), arr + 1.0)


def test_device_function_rejects_4d_array_argument() -> None:
    """4D+ remains unsupported, matching the exact restriction kernel
    arguments already have (numba-metal only supports 1D/2D/3D
    anywhere). A 4D device-function argument can never actually reach
    `_classify_params`'s own check via a real kernel launch (the
    top-level kernel-argument boundary already rejects a 4D array
    earlier, in dispatcher.py's `_infer_arg_type`) -- this exercises
    `_classify_params`'s device-function branch directly, the same way
    test_atomics.py's `test_atomic_rejects_2d_array` isolates an
    otherwise-unreachable-in-practice typing-level check."""
    from numba.core import types

    from numba_metal.compiler.frontend import compile_to_typed_ir
    from numba_metal.compiler.msl_backend import MSLKernelLowerer
    from numba_metal.errors import UnsupportedFeatureError

    def bad(arr):
        return arr[0, 0, 0, 0]

    typed = compile_to_typed_ir(bad, (types.float32[:, :, :, ::1],), return_type=None)
    lowerer = MSLKernelLowerer(
        "bad", typed, device_function=True, return_type=typed.return_type
    )
    with pytest.raises(UnsupportedFeatureError):
        lowerer.lower()


def test_device_function_rejects_non_array_non_scalar_argument() -> None:
    """A genuinely unsupported argument type (not an array, not a
    scalar) must still be rejected -- confirms the classification
    branch's fallback error still fires now that arrays are no longer
    universally rejected for device functions."""
    from numba_metal.errors import NumbaMetalError

    metal = _metal()

    @metal.device_func
    def bad(pair):
        return pair[0]

    @metal.jit
    def kernel(a, out):
        i = metal.grid(1)
        if i < a.size:
            out[i] = bad((a[i], a[i]))

    d_a = metal.to_device(np.array([1.0, 2.0], dtype=np.float32))
    d_out = metal.device_array_like(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(NumbaMetalError):
        kernel[1, 2](d_a, d_out)


def test_device_function_cannot_be_called_from_ordinary_python() -> None:
    metal = _metal()

    @metal.device_func
    def helper(x):
        return x + 1.0

    # A @metal.device_func-decorated function is a real @njit dispatcher
    # (see docs/architecture.md), so it IS callable from ordinary Python
    # via Numba's own CPU compilation -- numba-metal does not prohibit
    # this (unlike an earlier design considered during development that
    # would have made it raise TypeError outside a kernel). This test
    # documents that actual, intentional behavior rather than asserting
    # a restriction that was not ultimately implemented.
    assert helper(2.0) == 3.0
