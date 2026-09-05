"""Statistical CPU call-stack sampler.

Runs on a background thread, periodically snapshotting every other
thread's Python call stack via `sys._current_frames()` and folding
identical stacks together into the conventional folded-stack format
(`main;caller;callee count`).

This module makes no claim to see inside compiled Numba-CPU or Metal
kernel code -- a frame executing inside a `@njit`/`@metal.jit` function
is invisible to `sys._current_frames()` once Numba's own dispatcher has
handed off to compiled machine code (there is no Python frame for it to
sample). Frames that look like they belong to compiled code are
classified as such (see `_classify_frame`) but their time is NOT invented
from the sampling gap -- see `advisor/metal_events.py` for how compiled
work is actually measured, via explicit instrumentation events, never
inferred from where the sampler stopped seeing Python frames.

Uses only the stdlib (`sys`, `threading`, `time`) -- no `py-spy`/`viztracer`
dependency, matching the spec's "avoid a mandatory dependency on an
external GUI profiler" and this repo's general dependency discipline.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter

from numba_metal.advisor.models import FoldedStack, SamplingResult

DEFAULT_INTERVAL_MS = 5.0


def _classify_frame_file(filename: str) -> str:
    """Best-effort tag for one stack frame's origin, for callers that want
    to color/filter by category. This is a naming heuristic over
    `co_filename`, not a guarantee -- see module docstring."""
    if "numba_metal" in filename:
        return "metal_wrapper"
    if "numba" in filename or filename.startswith("<numba"):
        return "numba_cpu"
    if filename.startswith("<") or not filename:
        return "unknown"
    return "python"


def _stack_to_frame_names(frame) -> list[str]:
    names: list[str] = []
    f = frame
    while f is not None:
        code = f.f_code
        tag = _classify_frame_file(code.co_filename)
        label = f"{code.co_name}"
        if tag != "python":
            label = f"{label}[{tag}]"
        names.append(label)
        f = f.f_back
    names.reverse()
    return names


class StackSampler:
    """Background-thread statistical sampler. Not started automatically;
    call `start()`/`stop()` explicitly (mirrors `profiler.activate()` /
    `deactivate()`'s lifecycle, and keeps this class independently
    testable without importing profiler.py at all)."""

    def __init__(
        self,
        *,
        interval_ms: float = DEFAULT_INTERVAL_MS,
        target_thread_id: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self._interval_s = interval_ms / 1000.0
        self._interval_ms = interval_ms
        self._target_thread_id = target_thread_id
        self._max_samples = max_samples
        self._stacks: Counter[tuple[str, ...]] = Counter()
        self._total_samples = 0
        self._dropped_samples = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Set from INSIDE _run() once the sampler thread actually starts,
        # not here in __init__ (which runs on the caller's thread) --
        # otherwise, when target_thread_id is that same caller thread
        # (the common case: sampling the thread that constructed this
        # sampler), _sample_once's "skip my own thread" check would
        # incorrectly skip the very thread being sampled. Found via a
        # real repro: total_samples stayed 0 even while the target
        # thread was busy the whole interval.
        self._own_thread_id: int | None = None
        self._overhead_ns = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("StackSampler already started")
        self._stop_event.clear()
        started = threading.Event()

        def _run_with_id() -> None:
            self._own_thread_id = threading.get_ident()
            started.set()
            self._run()

        self._thread = threading.Thread(
            target=_run_with_id, name="numba-metal-advisor-sampler", daemon=True
        )
        self._thread.start()
        started.wait(timeout=5.0)

    def stop(self) -> SamplingResult:
        if self._thread is None:
            raise RuntimeError("StackSampler was never started")
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self._interval_s * 10))
        self._thread = None
        stacks = tuple(
            FoldedStack(frames=frames, count=count)
            for frames, count in sorted(
                self._stacks.items(), key=lambda kv: (-kv[1], kv[0])
            )
        )
        return SamplingResult(
            stacks=stacks,
            total_samples=self._total_samples,
            interval_ms=self._interval_ms,
            dropped_samples=self._dropped_samples,
            profiler_overhead_ns=self._overhead_ns,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            self._sample_once()
            elapsed = time.perf_counter() - t0
            self._overhead_ns += int(elapsed * 1e9)
            remaining = self._interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _sample_once(self) -> None:
        frames = sys._current_frames()
        target = self._target_thread_id
        for tid, frame in frames.items():
            if tid == self._own_thread_id:
                continue
            if target is not None and tid != target:
                continue
            names = tuple(_stack_to_frame_names(frame))
            if not names:
                continue
            if (
                self._max_samples is not None
                and self._total_samples >= self._max_samples
            ):
                self._dropped_samples += 1
                continue
            self._stacks[names] += 1
            self._total_samples += 1


def sample_callable(
    fn,
    *args,
    interval_ms: float = DEFAULT_INTERVAL_MS,
    max_samples: int | None = None,
    **kwargs,
):
    """Run `fn(*args, **kwargs)` on the calling thread while a background
    sampler observes it. Returns (result, SamplingResult)."""
    sampler = StackSampler(
        interval_ms=interval_ms,
        target_thread_id=threading.get_ident(),
        max_samples=max_samples,
    )
    sampler.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        sampling_result = sampler.stop()
    return result, sampling_result
