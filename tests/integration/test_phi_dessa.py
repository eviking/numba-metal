"""Differential tests for correct SSA phi-node elimination (de-SSA).

Each test targets one of the twelve required adversarial phi patterns
identified during Workstream 1 (see docs/architecture.md, "De-SSA
algorithm", for the full design rationale and why the previous
union-find-based aliasing scheme was theoretically unsound).

Every test compares REAL Metal GPU execution against a REAL Numba
`@njit` CPU compilation of the equivalent algorithm -- never against a
hand-computed Python constant and never by inspecting generated MSL text
alone. A test that only searches MSL source for a substring is
insufficient to demonstrate correctness (per the assignment's explicit
requirement) and is not used here.

Requires a working Metal device; run with `pytest -m metal`.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import njit

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


# 1. A phi result where one incoming variable remains live after the merge.
def test_incoming_variable_remains_live_after_merge() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(a):
        out = np.empty_like(a)
        for i in range(a.shape[0]):
            x = a[i]
            z = 0.0
            if x > 0.0:
                z = x * 2.0
            # x must still read as a[i] here, not be clobbered by z's phi.
            out[i] = x + z
        return out

    @metal.jit
    def gpu_kernel(a, out):
        i = metal.grid(1)
        if i < out.size:
            x = a[i]
            z = 0.0
            if x > 0.0:
                z = x * 2.0
            out[i] = x + z

    rng = np.random.default_rng(0)
    a = rng.standard_normal(2000).astype(np.float32)
    expected = cpu(a)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    gpu_kernel[8, 256](d_a, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), expected, rtol=1e-5, atol=1e-5)


# 2. Two phi nodes on the same merge edge.
def test_two_phi_nodes_on_same_merge_edge() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(a):
        out_x = np.empty_like(a)
        out_y = np.empty_like(a)
        for i in range(a.shape[0]):
            x = a[i]
            y = -a[i]
            if x > 0.0:
                x = x * 2.0
                y = y * 3.0
            out_x[i] = x
            out_y[i] = y
        return out_x, out_y

    @metal.jit
    def gpu_kernel(a, out_x, out_y):
        i = metal.grid(1)
        if i < out_x.size:
            x = a[i]
            y = -a[i]
            if x > 0.0:
                x = x * 2.0
                y = y * 3.0
            out_x[i] = x
            out_y[i] = y

    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000).astype(np.float32)
    ex, ey = cpu(a)
    d_a = metal.to_device(a)
    d_ox, d_oy = metal.device_array_like(a), metal.device_array_like(a)
    gpu_kernel[8, 256](d_a, d_ox, d_oy)
    metal.synchronize()
    assert np.allclose(d_ox.copy_to_host(), ex, rtol=1e-5, atol=1e-5)
    assert np.allclose(d_oy.copy_to_host(), ey, rtol=1e-5, atol=1e-5)


# 3. Parallel-copy behavior (simultaneous cross-assignment via tuple swap).
def test_parallel_copy_tuple_swap() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(a, b):
        out_x = np.empty_like(a)
        out_y = np.empty_like(a)
        for i in range(a.shape[0]):
            x = a[i]
            y = b[i]
            if x > y:
                x, y = y, x
            out_x[i] = x
            out_y[i] = y
        return out_x, out_y

    @metal.jit
    def gpu_kernel(a, b, out_x, out_y):
        i = metal.grid(1)
        if i < out_x.size:
            x = a[i]
            y = b[i]
            if x > y:
                x, y = y, x
            out_x[i] = x
            out_y[i] = y

    rng = np.random.default_rng(2)
    a = rng.standard_normal(2000).astype(np.float32)
    b = rng.standard_normal(2000).astype(np.float32)
    ex, ey = cpu(a, b)
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_ox, d_oy = metal.device_array_like(a), metal.device_array_like(a)
    gpu_kernel[8, 256](d_a, d_b, d_ox, d_oy)
    metal.synchronize()
    assert np.allclose(d_ox.copy_to_host(), ex, rtol=1e-5, atol=1e-5)
    assert np.allclose(d_oy.copy_to_host(), ey, rtol=1e-5, atol=1e-5)


# 4. A copy cycle (loop-carried swap with no temporary in source, the
# closest valid pattern to a genuine cyclic parallel-copy Numba generates).
def test_loop_carried_swap_cycle() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(n_steps):
        a = 1
        b = 2
        for _ in range(n_steps):
            a, b = b, a
        return a

    @metal.jit
    def gpu_kernel(out, n):
        i = metal.grid(1)
        if i < out.size:
            a = 1
            b = 2
            for _ in range(n):
                a, b = b, a
            out[i] = a

    for n_steps in (0, 1, 2, 3, 7, 10):
        expected = cpu(n_steps)
        d_out = metal.device_array(16, np.int32)
        gpu_kernel[1, 16](d_out, np.int32(n_steps))
        metal.synchronize()
        result = d_out.copy_to_host()
        assert np.all(result == expected), f"n_steps={n_steps}: {result} != {expected}"


# 5. Nested if/else merges.
def test_nested_if_else_merges() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(a):
        out = np.empty_like(a)
        for i in range(a.shape[0]):
            x = a[i]
            z = 0.0
            if x > 0.0:
                if x > 10.0:
                    z = 1.0
                else:
                    z = 2.0
            else:
                if x < -10.0:
                    z = 3.0
                else:
                    z = 4.0
            out[i] = z
        return out

    @metal.jit
    def gpu_kernel(a, out):
        i = metal.grid(1)
        if i < out.size:
            x = a[i]
            z = 0.0
            if x > 0.0:
                if x > 10.0:
                    z = 1.0
                else:
                    z = 2.0
            else:
                if x < -10.0:
                    z = 3.0
                else:
                    z = 4.0
            out[i] = z

    rng = np.random.default_rng(3)
    a = (rng.standard_normal(2000) * 20).astype(np.float32)
    expected = cpu(a)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    gpu_kernel[8, 256](d_a, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), expected)


# 6. A ternary expression.
def test_ternary_expression() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(a):
        out = np.empty_like(a)
        for i in range(a.shape[0]):
            out[i] = a[i] * 2.0 if a[i] > 0.0 else a[i] * 3.0
        return out

    @metal.jit
    def gpu_kernel(a, out):
        i = metal.grid(1)
        if i < out.size:
            out[i] = a[i] * 2.0 if a[i] > 0.0 else a[i] * 3.0

    rng = np.random.default_rng(4)
    a = rng.standard_normal(2000).astype(np.float32)
    expected = cpu(a)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    gpu_kernel[8, 256](d_a, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), expected, rtol=1e-5, atol=1e-5)


# 7. Loop-carried accumulation.
def test_loop_carried_accumulation() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(a, n):
        out = np.empty(a.shape[0], dtype=np.float32)
        for i in range(a.shape[0]):
            acc = np.float32(0.0)
            for j in range(n):
                acc = acc + a[i] * np.float32(j)
            out[i] = acc
        return out

    @metal.jit
    def gpu_kernel(a, out, n):
        i = metal.grid(1)
        if i < out.size:
            acc = 0.0
            for j in range(n):
                acc = acc + a[i] * j
            out[i] = acc

    rng = np.random.default_rng(5)
    a = rng.standard_normal(500).astype(np.float32)
    n = 20
    expected = cpu(a, n)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    gpu_kernel[2, 256](d_a, d_out, np.int32(n))
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), expected, rtol=1e-4, atol=1e-4)


# 8. Nested loops with loop-carried values.
def test_nested_loops_with_loop_carried_values() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(n, m):
        out = np.empty(4, dtype=np.int32)
        for i in range(4):
            total = 0
            for j in range(n):
                inner = 0
                for k in range(m):
                    inner = inner + k
                total = total + inner + j
            out[i] = total
        return out

    @metal.jit
    def gpu_kernel(out, n, m):
        i = metal.grid(1)
        if i < out.size:
            total = 0
            for j in range(n):
                inner = 0
                for k in range(m):
                    inner = inner + k
                total = total + inner + j
            out[i] = total

    n, m = 5, 4
    expected = cpu(n, m)
    d_out = metal.device_array(4, np.int32)
    gpu_kernel[1, 4](d_out, np.int32(n), np.int32(m))
    metal.synchronize()
    assert np.array_equal(d_out.copy_to_host(), expected)


# 9. `break` from a loop followed by use of a pre-loop value.
def test_break_followed_by_use_of_pre_loop_value() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(a, n):
        out = np.empty(a.shape[0], dtype=np.float32)
        for i in range(a.shape[0]):
            pre = a[i] * 10.0
            acc = np.float32(0.0)
            for j in range(n):
                if a[i] > 100.0:
                    break
                acc = acc + np.float32(j)
            out[i] = pre + acc
        return out

    @metal.jit
    def gpu_kernel(a, out, n):
        i = metal.grid(1)
        if i < out.size:
            pre = a[i] * 10.0
            acc = 0.0
            for j in range(n):
                if a[i] > 100.0:
                    break
                acc = acc + j
            out[i] = pre + acc

    a = np.array([1.0, 200.0, -5.0, 50.0], dtype=np.float32)
    n = 10
    expected = cpu(a, n)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    gpu_kernel[1, 4](d_a, d_out, np.int32(n))
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), expected, rtol=1e-5, atol=1e-5)


# 10. `continue` with values modified in only one branch.
def test_continue_with_values_modified_in_only_one_branch() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(a, n):
        out = np.empty(a.shape[0], dtype=np.float32)
        for i in range(a.shape[0]):
            count = np.float32(0.0)
            for j in range(n):
                if a[j] < 0.0:
                    continue
                count = count + a[j]
            out[i] = count
        return out

    @metal.jit
    def gpu_kernel(a, out, n):
        i = metal.grid(1)
        if i < out.size:
            count = 0.0
            for j in range(n):
                if a[j] < 0.0:
                    continue
                count = count + a[j]
            out[i] = count

    rng = np.random.default_rng(6)
    a = rng.standard_normal(20).astype(np.float32)
    n = 20
    expected = cpu(a, n)
    d_a = metal.to_device(a)
    d_out = metal.device_array(4, np.float32)
    gpu_kernel[1, 4](d_a, d_out, np.int32(n))
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), expected[:4], rtol=1e-4, atol=1e-4)


# 11. Multiple exits converging on a common block.
def test_multiple_exits_converging_on_common_block() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(a):
        out = np.empty_like(a)
        for i in range(a.shape[0]):
            x = a[i]
            if x > 50.0:
                z = 1.0
            elif x > 0.0:
                if x > 25.0:
                    z = 2.0
                else:
                    z = 3.0
            else:
                z = 4.0
            out[i] = z
        return out

    @metal.jit
    def gpu_kernel(a, out):
        i = metal.grid(1)
        if i < out.size:
            x = a[i]
            if x > 50.0:
                z = 1.0
            elif x > 0.0:
                if x > 25.0:
                    z = 2.0
                else:
                    z = 3.0
            else:
                z = 4.0
            out[i] = z

    a = np.array([100.0, 30.0, 10.0, -5.0], dtype=np.float32)
    expected = cpu(a)
    d_a = metal.to_device(a)
    d_out = metal.device_array_like(a)
    gpu_kernel[1, 4](d_a, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), expected)


# 12. Negative and positive range steps.
def test_negative_and_positive_range_steps() -> None:
    metal = _metal()

    @njit(cache=True)
    def cpu(start, stop, step):
        acc = 0
        for j in range(start, stop, step):
            acc = acc + j
        return acc

    @metal.jit
    def gpu_kernel(out, start, stop, step):
        i = metal.grid(1)
        if i < out.size:
            acc = 0
            for j in range(start, stop, step):
                acc = acc + j
            out[i] = acc

    cases = [(0, 10, 1), (10, 0, -1), (0, 20, 3), (20, 0, -3), (5, 5, 1)]
    for start, stop, step in cases:
        expected = cpu(start, stop, step)
        d_out = metal.device_array(4, np.int32)
        gpu_kernel[1, 4](d_out, np.int32(start), np.int32(stop), np.int32(step))
        metal.synchronize()
        result = d_out.copy_to_host()
        assert np.all(
            result == expected
        ), f"range({start},{stop},{step}): {result} != {expected}"
