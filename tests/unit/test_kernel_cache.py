"""Unit tests for KernelCache's source-digest memoization, using a fake
`get_context`/`_compile_kernel` -- no GPU or Metal framework required.

Verifies the fix for a measured, real overhead: `_cache_key` used to
call `inspect.getsource(func)` and re-hash it on every single dispatch,
even on a warm cache hit -- ~22us per call, about 15% of an entire
warm kernel launch's ~150us total (measured directly on real Metal
hardware during this investigation). `KernelCache` now memoizes the
source-derived digest per `id(func)` so it is computed at most once per
distinct function object.
"""

from __future__ import annotations

from unittest.mock import patch

from numba_metal.compiler.pipeline import CompiledKernel, KernelCache, _source_digest


def _fake_compiled(name: str) -> CompiledKernel:
    return CompiledKernel(
        name=name,
        msl_source="",
        signature=None,
        arg_types=(),
        pipeline_state=None,
        max_threads_per_threadgroup=1,
    )


def _some_func():
    return 1


def _other_func():
    return 2


class _FakeDeviceInfo:
    registry_id = 1


class _FakeContext:
    info = _FakeDeviceInfo()


def test_second_call_reuses_memoized_source_digest():
    cache = KernelCache()
    with (
        patch("numba_metal.compiler.pipeline.get_context", return_value=_FakeContext()),
        patch(
            "numba_metal.compiler.pipeline._compile_kernel",
            return_value=_fake_compiled("k1"),
        ) as mock_compile,
        patch(
            "numba_metal.compiler.pipeline._source_digest", wraps=_source_digest
        ) as mock_digest,
    ):
        cache.get_or_compile(_some_func, ())
        cache.get_or_compile(_some_func, ())
        assert mock_digest.call_count == 1
        assert mock_compile.call_count == 1


def test_different_functions_get_independent_digests():
    cache = KernelCache()
    with (
        patch("numba_metal.compiler.pipeline.get_context", return_value=_FakeContext()),
        patch(
            "numba_metal.compiler.pipeline._compile_kernel",
            side_effect=lambda func, arg_types, **kw: _fake_compiled(func.__name__),
        ),
    ):
        compiled_a = cache.get_or_compile(_some_func, ())
        compiled_b = cache.get_or_compile(_other_func, ())
        assert compiled_a.name == "_some_func"
        assert compiled_b.name == "_other_func"
        assert len(cache) == 2


def test_cache_hit_returns_same_compiled_kernel_object():
    cache = KernelCache()
    with (
        patch("numba_metal.compiler.pipeline.get_context", return_value=_FakeContext()),
        patch(
            "numba_metal.compiler.pipeline._compile_kernel",
            return_value=_fake_compiled("k1"),
        ) as mock_compile,
    ):
        first = cache.get_or_compile(_some_func, ())
        second = cache.get_or_compile(_some_func, ())
        assert first is second
        mock_compile.assert_called_once()


def test_clear_forgets_memoized_digests_too():
    cache = KernelCache()
    with (
        patch("numba_metal.compiler.pipeline.get_context", return_value=_FakeContext()),
        patch(
            "numba_metal.compiler.pipeline._compile_kernel",
            return_value=_fake_compiled("k1"),
        ) as mock_compile,
    ):
        cache.get_or_compile(_some_func, ())
        cache.clear()
        assert len(cache) == 0
        assert cache._source_digests == {}
        cache.get_or_compile(_some_func, ())
        assert mock_compile.call_count == 2
