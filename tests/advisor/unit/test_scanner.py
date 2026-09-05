"""Test 1 (existing Numba candidate), Test 2 (standard Python candidate),
Test 3 (unsupported Python doesn't crash the scan) from the governing spec.

No @pytest.mark.metal here -- static scanning requires no Metal device
and must run everywhere (spec section 17 / Test 16 regression scope).
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

from numba_metal.advisor.models import Confidence
from numba_metal.advisor.scanner import scan, scan_file


def _write_and_scan(source: str):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(textwrap.dedent(source))
        path = Path(f.name)
    try:
        return scan_file(path)
    finally:
        path.unlink()


def test_existing_njit_candidate_is_detected_high_confidence():
    """Test 1: A function decorated with @njit containing an independent
    numerical loop is detected and ranked as a candidate."""
    candidates, error = _write_and_scan("""
        import numba

        @numba.njit
        def simulate_paths(prices, out):
            for i in range(prices.size):
                out[i] = prices[i] * 1.01
        """)
    assert error is None
    assert len(candidates) == 1
    c = candidates[0]
    assert c.qualified_name == "simulate_paths"
    assert c.is_numba_decorated is True
    assert c.confidence == Confidence.HIGH
    assert any("njit" in r for r in c.reasons)


def test_bare_njit_import_is_detected():
    """This project's OWN benchmarks (benchmarks/mandelbrot.py) use
    `from numba import njit` then bare `@njit(...)` -- must be detected
    the same as the fully-qualified `@numba.njit` form."""
    candidates, error = _write_and_scan("""
        from numba import njit

        @njit(parallel=True)
        def add_one(x, out):
            for i in range(x.size):
                out[i] = x[i] + 1.0
        """)
    assert error is None
    assert len(candidates) == 1
    assert candidates[0].is_numba_decorated is True
    assert candidates[0].confidence == Confidence.HIGH


def test_nested_closure_kernel_is_detected():
    """This project's OWN benchmarks define the actual decorated kernel
    as a nested closure returned by a plain outer function (e.g.
    benchmarks/mandelbrot.py's _make_numba_cpu_impl/_make_metal_kernel).
    A scanner that only looks at top-level defs finds nothing real."""
    candidates, error = _write_and_scan("""
        from numba import njit

        def make_impl():
            @njit
            def inner(x, out):
                for i in range(x.size):
                    out[i] = x[i] * 2.0
            return inner
        """)
    assert error is None
    names = [c.qualified_name for c in candidates]
    assert "make_impl.<locals>.inner" in names
    inner = next(
        c for c in candidates if c.qualified_name == "make_impl.<locals>.inner"
    )
    assert inner.is_numba_decorated is True
    assert inner.confidence == Confidence.HIGH


def test_existing_metal_jit_is_detected_and_labeled_already_running():
    from numba_metal.advisor.scanner import scan_file as _sf  # noqa: F401

    candidates, error = _write_and_scan("""
        from numba_metal import metal

        @metal.jit
        def kernel(x, out):
            i = metal.grid(1)
            if i < out.size:
                out[i] = x[i] + 1.0
        """)
    assert error is None
    assert len(candidates) == 1
    c = candidates[0]
    assert c.is_metal_decorated is True
    assert c.confidence == Confidence.HIGH
    assert any("numba-metal" in r for r in c.reasons)


def test_undecorated_python_loop_is_potential_not_a_speedup_claim():
    """Test 2: An undecorated numerical loop is detected as potentially
    suitable, but no speedup is claimed without runtime evidence."""
    candidates, error = _write_and_scan("""
        def elementwise_scale(a, b, out):
            for i in range(len(a)):
                out[i] = a[i] * b[i]
        """)
    assert error is None
    assert len(candidates) == 1
    c = candidates[0]
    assert c.is_numba_decorated is False
    assert c.is_metal_decorated is False
    # No numeric speedup anywhere on the Candidate model at all -- this
    # is a structural guarantee (models.Candidate has no such field), not
    # just a value check.
    assert not hasattr(c, "speedup")
    assert not hasattr(c, "estimated_speedup")
    # Unknowns must be populated -- the model's honesty mechanism for
    # "static-only, no runtime data" candidates.
    assert len(c.unknowns) > 0


def test_unsupported_python_reports_precise_blockers_without_crashing():
    """Test 3: A function using dictionaries, strings, exceptions, or
    object allocation is reported with precise blockers without
    crashing the scan."""
    candidates, error = _write_and_scan("""
        def messy(data):
            cache = {}
            try:
                for k, v in data.items():
                    cache[k] = str(v)
            except KeyError:
                raise ValueError("bad")
            return cache
        """)
    assert error is None  # the FILE parsed fine; blockers are per-function
    assert len(candidates) == 1
    c = candidates[0]
    assert len(c.blockers) >= 4
    joined = " ".join(c.blockers)
    assert "dict" in joined
    assert "try/except" in joined
    assert "raise" in joined or "exceptions" in joined
    assert "string" in joined


def test_scan_never_executes_scanned_code():
    """scan() must never import/exec the scanned file -- verified by
    scanning a file that would raise/crash/have a side effect if
    executed, and confirming scan_file completes normally."""
    candidates, error = _write_and_scan("""
        raise RuntimeError("this file must never be executed by the scanner")

        def f(x, out):
            for i in range(x.size):
                out[i] = x[i]
        """)
    # If the scanner had executed this file, this call would have raised
    # RuntimeError and pytest would report an error, not a clean assert.
    assert error is None
    assert len(candidates) == 1


def test_syntax_error_is_recorded_not_raised():
    """A file scan() cannot even parse must be recorded as a ScanError,
    never raised out of scan_file -- one bad file cannot crash a batch
    scan of many files."""
    candidates, error = _write_and_scan("def f(:\n    pass\n")
    assert candidates == []
    assert error is not None
    assert "syntax" in error.message.lower()


def test_scan_directory_aggregates_across_files_and_survives_one_bad_file():
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "good.py"
        good.write_text(
            "import numba\n\n@numba.njit\ndef f(x, out):\n"
            "    for i in range(x.size):\n        out[i] = x[i]\n"
        )
        bad = Path(d) / "bad.py"
        bad.write_text("def g(:\n  pass\n")
        result = scan(d)
        assert result.files_scanned == 2
        assert len(result.errors) == 1
        assert any(c.qualified_name == "f" for c in result.candidates)
