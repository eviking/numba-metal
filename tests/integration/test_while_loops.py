"""Standalone tests for numba-metal's `while` loop support: a real
compiler-correctness capability (not merely a documented gap), added and
fixed while adding compare-and-swap-based atomics.

Only straight-line `while` bodies are supported (no nested `if`/`else`,
`break`, or `continue` inside the loop) -- see docs/limitations.md for
exactly why: Numba's bytecode lowering rotates `while cond: body` into a
CFG shape distinct from `for x in range(...)`, and generalizing this
backend's control-flow structurer to handle break/continue nested inside
a conditional within a rotated while body was found, by direct testing,
to require a substantially larger rewrite than this project's scope for
this pass -- so that combination is explicitly rejected at compile time
rather than risking the confirmed-possible silent-wrong-result failure
mode.

Requires a working Apple-silicon Metal device; see tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metal


def _metal():
    from numba_metal import metal

    return metal


def test_while_loop_accumulation_matches_expected_sum() -> None:
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = 0
        total = 0.0
        while i < a.size:
            total = total + a[i]
            i = i + 1
        out[0] = total

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(1, np.float32)
    kernel[1, 1](d_a, d_out)
    metal.synchronize()
    assert d_out.copy_to_host()[0] == pytest.approx(float(a.sum()))


def test_while_loop_iteration_count_is_exact() -> None:
    """Regression test for the exact bug this feature was fixed for: an
    earlier broken version executed the loop body exactly once
    regardless of the true trip count."""
    metal = _metal()

    @metal.jit
    def kernel(a, count):
        i = 0
        n = 0
        while i < a.size:
            i = i + 1
            n = n + 1
        count[0] = n

    for size in (1, 2, 5, 17, 100):
        a = np.zeros(size, dtype=np.float32)
        d_a = metal.to_device(a)
        d_count = metal.device_array(1, np.int64)
        kernel[1, 1](d_a, d_count)
        metal.synchronize()
        assert d_count.copy_to_host()[0] == size


def test_while_loop_false_initial_condition_never_enters_body() -> None:
    """A `while` whose condition is false on entry must run zero
    iterations, not one -- regression test for the do-while-style lowering
    unconditionally running its body at least once before this was fixed
    with an entry guard."""
    metal = _metal()

    @metal.jit
    def kernel(out):
        i = 0
        n = 0
        while i < 0:  # never true
            n = n + 1
            i = i + 1
        out[0] = n

    d_out = metal.device_array(1, np.int64)
    kernel[1, 1](d_out)
    metal.synchronize()
    assert d_out.copy_to_host()[0] == 0


def test_while_loop_single_iteration() -> None:
    metal = _metal()

    @metal.jit
    def kernel(out):
        i = 0
        n = 0
        while i < 1:
            n = n + 1
            i = i + 1
        out[0] = n

    d_out = metal.device_array(1, np.int64)
    kernel[1, 1](d_out)
    metal.synchronize()
    assert d_out.copy_to_host()[0] == 1


def test_while_loop_no_double_execution_on_transition() -> None:
    """Regression test for the exact off-by-one this feature was fixed
    for a second time: an intermediate broken version ran one extra,
    spurious pass immediately after the loop's already-computed exit
    condition should have stopped it -- caught via a compare-and-swap
    retry loop where the spurious pass performed a second, real
    (race-free, and therefore silently "successful") atomic operation
    beyond the one that should have happened."""
    metal = _metal()

    @metal.jit
    def kernel(counter, n):
        i = metal.grid(1)
        if i < n:
            done = False
            while not done:
                old = counter[0]
                _prev, ok = metal.atomic_compare_exchange(counter, 0, old, old + 1)
                done = ok

    for n in (1, 2, 4, 10, 1000):
        d_counter = metal.to_device(np.zeros(1, dtype=np.int32))
        kernel[1, n](d_counter, np.int32(n))
        metal.synchronize()
        assert d_counter.copy_to_host()[0] == n


def test_while_loop_multiple_variables_updated_per_iteration() -> None:
    metal = _metal()

    @metal.jit
    def kernel(out_sum, out_product, out_count):
        i = 1
        total = 0.0
        product = 1.0
        n = 0
        while i <= 5:
            total = total + float(i)
            product = product * float(i)
            n = n + 1
            i = i + 1
        out_sum[0] = total
        out_product[0] = product
        out_count[0] = n

    d_sum = metal.device_array(1, np.float32)
    d_product = metal.device_array(1, np.float32)
    d_count = metal.device_array(1, np.int64)
    kernel[1, 1](d_sum, d_product, d_count)
    metal.synchronize()
    assert d_sum.copy_to_host()[0] == pytest.approx(15.0)  # 1+2+3+4+5
    assert d_product.copy_to_host()[0] == pytest.approx(120.0)  # 5!
    assert d_count.copy_to_host()[0] == 5


def test_while_loop_rejects_nested_continue() -> None:
    from numba_metal.errors import UnsupportedFeatureError

    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = 0
        total = 0.0
        while i < a.size:
            if a[i] < 0.0:
                i = i + 1
                continue
            total = total + a[i]
            i = i + 1
        out[0] = total

    d_a = metal.to_device(np.array([1.0, -2.0], dtype=np.float32))
    d_out = metal.device_array(1, np.float32)
    with pytest.raises(UnsupportedFeatureError):
        kernel[1, 1](d_a, d_out)


def test_while_loop_rejects_nested_break() -> None:
    from numba_metal.errors import UnsupportedFeatureError

    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = 0
        total = 0.0
        while i < a.size:
            if a[i] < 0.0:
                break
            total = total + a[i]
            i = i + 1
        out[0] = total

    d_a = metal.to_device(np.array([1.0, -2.0], dtype=np.float32))
    d_out = metal.device_array(1, np.float32)
    with pytest.raises(UnsupportedFeatureError):
        kernel[1, 1](d_a, d_out)


def test_while_loop_rejects_nested_if_without_break_or_continue() -> None:
    """Even a nested if/else with no break/continue at all is rejected:
    this backend's rotated-while lowering was only verified correct for
    a straight-line body, and a conditional inside the loop changes the
    CFG shape regardless of whether it contains a break/continue."""
    from numba_metal.errors import UnsupportedFeatureError

    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = 0
        total = 0.0
        while i < a.size:
            if a[i] < 0.0:
                total = total - a[i]
            else:
                total = total + a[i]
            i = i + 1
        out[0] = total

    d_a = metal.to_device(np.array([1.0, -2.0], dtype=np.float32))
    d_out = metal.device_array(1, np.float32)
    with pytest.raises(UnsupportedFeatureError):
        kernel[1, 1](d_a, d_out)
