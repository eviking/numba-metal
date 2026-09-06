"""Instrumentation glue between numba-metal's runtime and the advisor's
normalized `Event` model.

Two small, additive hook points now exist directly in
`numba_metal.runtime.context` (added for this feature, four new lines
plus two small helper functions -- see that module's own comments):

1. `context.add_submission_hook(hook)` -- `hook(record: SubmissionRecord)`
   is called once per kernel launch (or once per `metal.batch()` block),
   right after `register_submission` records it. Empty/no-op by default.
2. `context.add_sync_hook(hook)` -- `hook(start_ns, duration_ns,
   pending_count)` is called once per `synchronize()` call that actually
   waited on at least one outstanding command buffer, with wall-clock
   timing around the existing `waitUntilCompleted()` loop. Empty/no-op by
   default.

Neither hook point changes `register_submission`'s or `synchronize`'s
control flow, return value, or exception behavior in any way -- both are
called AFTER the existing logic has already decided what to do next; see
`context.py`'s own inline comments at each call site.

Compilation cold/warm timing (`time_compile` below) needs no NEW hook
in `compiler/pipeline.py`: `KernelCache.get_or_compile`'s own hit/miss
branch can be observed from outside by checking `KernelCache._entries`
under its existing `_lock`, using the exact same cache key
`_cache_key()` computes from `_source_digest()` -- a documented,
minimal, private-API read (no mutation), consistent with this repo's
own `benchmarks/common.py` precedent of reaching into
`compiler.pipeline._MSL_PRELUDE`/`_next_kernel_name` directly. (`
_cache_key`/`_source_digest`'s split into two functions, and
`KernelCache._source_digests`'s memoization, were added later to avoid
re-deriving a kernel's source digest -- an `inspect.getsource()` call
plus a hash -- on every dispatch; `time_compile` mirrors that same
memoization here so checking hit/miss never forces the expensive path
unnecessarily.)

No true GPU-side kernel timestamp exists anywhere in numba-metal (see
models.GpuTimestampSource's docstring) -- every event this module
produces is `measured=True` for a real, observed wall-clock span, but is
tagged `HOST_WALL_CLOCK` (submission-to-completion) rather than
`DEVICE_TIMESTAMP`, and this module never invents a device-side duration.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from numba_metal.advisor.models import Event, GpuTimestampSource, now_ns


@dataclass(frozen=True, slots=True)
class SubmissionSnapshot:
    """Plain, numba-metal-independent copy of the fields an Event needs
    from a `SubmissionRecord` -- never hands out the real record (which
    holds live Metal objects) to advisor code outside this module."""

    kernel_name: str
    sequence: int
    resource_count: int


class EventCollector:
    """Thread-safe sink turning raw hook callbacks into `Event` objects.
    One instance per profiling session (created and owned by
    `profiler.py`); never a process-wide singleton, so multiple
    profiling runs in the same process (e.g. sequential `compare` calls)
    never leak events between each other."""

    def __init__(self, *, max_events: int | None = None) -> None:
        self._lock = threading.Lock()
        self._events: list[Event] = []
        self._max_events = max_events
        self._dropped = 0

    def record(self, event: Event) -> None:
        with self._lock:
            if self._max_events is not None and len(self._events) >= self._max_events:
                self._dropped += 1
                return
            self._events.append(event)

    def events(self) -> tuple[Event, ...]:
        with self._lock:
            return tuple(self._events)

    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._dropped = 0

    # -- hook callbacks, bound and passed to context.add_*_hook ---------

    def _on_submission(self, record) -> None:
        snap = SubmissionSnapshot(
            kernel_name=record.kernel_name,
            sequence=record.sequence,
            resource_count=len(record.resources),
        )
        self.record(
            Event(
                event_type="command_submit",
                name=snap.kernel_name,
                category="gpu",
                start_ns=now_ns(),
                duration_ns=0,
                measured=True,
                gpu_timestamp_source=GpuTimestampSource.NOT_APPLICABLE,
                extra={
                    "sequence": snap.sequence,
                    "resource_count": snap.resource_count,
                },
            )
        )

    def _on_synchronize(
        self, start_ns: int, duration_ns: int, pending_count: int
    ) -> None:
        self.record(
            Event(
                event_type="synchronize",
                name="synchronize",
                category="sync",
                start_ns=start_ns,
                duration_ns=duration_ns,
                measured=True,
                thread_id=threading.get_ident(),
                gpu_timestamp_source=GpuTimestampSource.HOST_WALL_CLOCK,
                extra={"pending_submissions": pending_count},
            )
        )
        # The wait span just measured is also the closest thing this
        # runtime can report as "GPU kernel" time today (submission to
        # completion confirmation), so it is ALSO recorded under a
        # separate "metal_kernel" event type with the same honest
        # HOST_WALL_CLOCK tag -- never DEVICE_TIMESTAMP, since no
        # per-kernel device-side duration is available (see
        # docs/advisor.md's "Why GPU work requires a separate timeline").
        self.record(
            Event(
                event_type="metal_kernel",
                name="submission_to_completion",
                category="gpu",
                start_ns=start_ns,
                duration_ns=duration_ns,
                measured=True,
                thread_id=threading.get_ident(),
                gpu_timestamp_source=GpuTimestampSource.HOST_WALL_CLOCK,
                extra={"pending_submissions": pending_count},
            )
        )

    def on_compile(
        self, *, kernel_name: str, cache_hit: bool, start_ns: int, duration_ns: int
    ) -> None:
        self.record(
            Event(
                event_type="metal_cache_hit" if cache_hit else "metal_cache_miss",
                name=kernel_name,
                category="compile",
                start_ns=start_ns,
                duration_ns=duration_ns,
                measured=True,
                cold_start=not cache_hit,
                gpu_timestamp_source=GpuTimestampSource.NOT_APPLICABLE,
            )
        )

    def on_python_call(
        self, *, name: str, category: str, start_ns: int, duration_ns: int
    ) -> None:
        self.record(
            Event(
                event_type=category,
                name=name,
                category="cpu",
                start_ns=start_ns,
                duration_ns=duration_ns,
                measured=True,
            )
        )


def install(collector: EventCollector) -> None:
    """Register `collector`'s hook methods with `runtime.context`. Call
    `uninstall(collector)` with the SAME collector to remove them again
    -- `profiler.activate()`/`deactivate()` do exactly this."""
    from numba_metal.runtime import context as _context_module

    _context_module.add_submission_hook(collector._on_submission)
    _context_module.add_sync_hook(collector._on_synchronize)


def uninstall_all() -> None:
    """Remove every hook currently registered on runtime.context,
    regardless of which collector installed them. Used by
    `profiler.deactivate()` -- there is at most one active profiling
    session at a time in this MVP (see profiler.py), so this is
    equivalent to removing exactly `collector`'s own hooks in practice,
    but does not require the caller to keep a reference around."""
    from numba_metal.runtime import context as _context_module

    _context_module.clear_hooks()


def time_compile(func, arg_types, *, kernel_cache, collector: EventCollector):
    """Call `kernel_cache.get_or_compile(func, arg_types)`, timing it and
    reporting cache hit/miss to `collector`, without modifying
    `KernelCache` itself. See module docstring for why the private
    `_entries`/`_lock`/`_cache_key` read is safe and sufficient."""
    from numba_metal.compiler.pipeline import _cache_key, _source_digest
    from numba_metal.runtime.context import get_context

    ctx = get_context()
    device_id = ctx.info.registry_id
    # Mirrors get_or_compile's own digest-memoization (see pipeline.py's
    # KernelCache._source_digests) so a hit here doesn't force an
    # otherwise-unnecessary inspect.getsource() just to check hit/miss.
    cached = kernel_cache._source_digests.get(id(func))
    digest = (
        cached[1] if cached is not None and cached[0] is func else _source_digest(func)
    )
    key = _cache_key(digest, arg_types, device_id)
    with kernel_cache._lock:
        cache_hit = key in kernel_cache._entries

    start = now_ns()
    compiled = kernel_cache.get_or_compile(func, arg_types)
    duration = now_ns() - start

    collector.on_compile(
        kernel_name=compiled.name,
        cache_hit=cache_hit,
        start_ns=start,
        duration_ns=duration,
    )
    return compiled


def timed_call(
    name: str, category: str, collector: EventCollector, fn, *args, **kwargs
):
    """Generic timing wrapper for a plain Python/Numba-CPU callable,
    producing one Event. There is nothing runtime-internal to hook for
    ordinary CPU code, so this is just `time.monotonic_ns()` around the
    call -- consistent with every other measured Event in this module."""
    start = now_ns()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    duration_ns = int((time.perf_counter() - t0) * 1e9)
    collector.on_python_call(
        name=name, category=category, start_ns=start, duration_ns=duration_ns
    )
    return result
