"""Standalone tests for numba-metal's `while` loop support: a real
compiler-correctness capability (not merely a documented gap), added and
fixed while adding compare-and-swap-based atomics, and later extended
to support a nested `if`/`else` inside the loop body.

A `while` body may be straight-line, OR contain a nested `if`/`else`
with no `break`/`continue` inside it -- see docs/limitations.md for the
full CFG-shape explanation: Numba's bytecode lowering rotates
`while cond: body` into a shape distinct from `for x in range(...)`,
and when the body opens with an if/else, the loop's real header (the
back-edge target, holding the loop-carried phi nodes) and its real
condition-test block end up as two DIFFERENT blocks -- detected and
handled as a `RotatedWhileNode` (see `compiler/structuring.py`).
`break`/`continue` NESTED INSIDE that if/else remain unsupported and
explicitly rejected at compile time: generalizing the structurer to
handle that specific combination was found, by direct testing, to
require a substantially larger rewrite than was in scope when this was
fixed, and two intermediate, silently-wrong-result bugs were found and
fixed along the way -- so that combination stays rejected rather than
risking a third.

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


def test_while_loop_with_nested_if_else_and_no_break_or_continue() -> None:
    """A nested if/else with no `break`/`continue` anywhere -- unlike
    the two rejected shapes above -- is genuinely supported: Numba's
    rotated-`while` lowering puts the loop's condition test in a
    DIFFERENT block from the loop-carried phi nodes when the body opens
    with an if/else (a native `for x in range(...)` loop never has this
    split, since it always gets a dedicated header block regardless of
    body content), which the structurer now detects and handles as a
    `RotatedWhileNode` -- see structuring.py's docstring for the full
    CFG-shape explanation and compiler/structuring.py's own extensive
    module comments. `break`/`continue` NESTED INSIDE the if/else remain
    unsupported (see the two tests above) -- that combination was
    confirmed, by direct testing, to produce genuinely swapped/wrong
    control flow in this backend's earlier lowering, and fixing it is
    out of scope for this fix, which only covers the narrower,
    dominance-provably-safe case exercised here (every arm of the
    if/else falls through to the loop's own next-iteration test, never
    jumping anywhere else)."""
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

    a = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(1, np.float32)
    kernel[1, 1](d_a, d_out)
    metal.synchronize()
    expected = float(np.sum(np.abs(a)))
    assert d_out.copy_to_host()[0] == pytest.approx(expected)


def test_while_loop_with_nested_if_else_matches_numpy_across_many_threads() -> None:
    """Same shape as above, but launched across many real GPU threads
    at once with varied per-thread data, to verify correctness under
    real parallelism -- not just a single-thread smoke test."""
    metal = _metal()

    @metal.jit
    def kernel(a, out, n):
        idx = metal.grid(1)
        if idx < out.size:
            i = 0
            total = 0.0
            while i < n:
                if a[idx * n + i] > 0.0:
                    total = total + a[idx * n + i]
                else:
                    total = total - a[idx * n + i]
                i = i + 1
            out[idx] = total

    n_threads = 1000
    n_per_thread = 37
    rng = np.random.default_rng(123)
    a = rng.uniform(-5, 5, n_threads * n_per_thread).astype(np.float32)
    expected = np.abs(a.reshape(n_threads, n_per_thread)).sum(axis=1)

    d_a = metal.to_device(a)
    d_out = metal.device_array(n_threads, np.float32)
    threads = 256
    blocks = (n_threads + threads - 1) // threads
    kernel[blocks, threads](d_a, d_out, np.int32(n_per_thread))
    metal.synchronize()
    np.testing.assert_allclose(d_out.copy_to_host(), expected, rtol=1e-5, atol=1e-4)


def test_while_loop_with_one_sided_if_no_else() -> None:
    """A ONE-SIDED `if` (no `else`) inside a `while` body -- a real,
    separate regression found while testing the two-sided if/else fix
    above: `_is_loop_header`'s dominance check alone produced a false
    positive here, misidentifying the if's own merge block (which
    happens to also contain the loop's real condition test, and so
    trivially dominates the eventual back-edge) as a `for`-range-style
    loop body split. Fixed by additionally requiring that the
    candidate exit branch cannot reach the candidate body branch by
    ordinary forward flow (see `_is_genuine_loop_split`/`_can_reach` in
    compiler/structuring.py) before trusting the dominance check."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = 0
        total = 0.0
        while i < a.size:
            if a[i] > 0.0:
                total = total + a[i]
            i = i + 1
        out[0] = total

    a = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float32)
    d_a = metal.to_device(a)
    d_out = metal.device_array(1, np.float32)
    kernel[1, 1](d_a, d_out)
    metal.synchronize()
    expected = float(np.sum(a[a > 0.0]))
    assert d_out.copy_to_host()[0] == pytest.approx(expected)


def test_while_loop_with_if_else_never_entering_body() -> None:
    """The loop condition is false from the very first check -- the
    if/else inside it must never execute at all, and the loop-carried
    variables' pre-loop values must pass through unchanged."""
    metal = _metal()

    @metal.jit
    def kernel(a, out):
        i = 100
        total = 0.0
        while i < 10:
            if a[i - 100] > 0.0:
                total = total + 1.0
            else:
                total = total - 1.0
            i = i + 1
        out[0] = total

    d_a = metal.to_device(np.array([1.0], dtype=np.float32))
    d_out = metal.device_array(1, np.float32)
    kernel[1, 1](d_a, d_out)
    metal.synchronize()
    assert d_out.copy_to_host()[0] == pytest.approx(0.0)


def test_nested_for_loops_still_correct_after_while_if_else_fix() -> None:
    """Regression coverage for a real intermediate bug found while
    fixing the while+if/else shapes above: an early, over-broad version
    of the loop-header-vs-merge-block distinguishing check (see
    `_can_reach` in compiler/structuring.py) treated an OUTER loop's
    exit target as able to reach an INNER loop's body target simply
    because it technically can, by looping all the way back around the
    outer loop's own back-edge first -- wrongly rejecting genuinely
    nested `for`-range loops that have no `if`/`while` of their own at
    all. Fixed by excluding back-edge crossings from that reachability
    search. This test has no `if`/`else` or `while` in it whatsoever --
    it exists purely to prove the while-loop fix didn't regress
    ordinary nested for-loops."""
    metal = _metal()

    @metal.jit
    def kernel(out, n, m):
        i = metal.grid(1)
        if i < out.size:
            total = 0
            for j in range(n):
                inner = 0
                for k in range(m):
                    inner = inner + k
                total = total + inner + j
            out[i] = total

    n, m = 5, 6
    d_out = metal.device_array(4, np.int32)
    kernel[1, 4](d_out, np.int32(n), np.int32(m))
    metal.synchronize()
    expected = sum(sum(range(m)) + j for j in range(n))
    np.testing.assert_array_equal(d_out.copy_to_host(), np.full(4, expected))
