"""Top-level profiling orchestration: runs a target script/pytest
invocation with instrumentation hooks and the statistical sampler both
active, producing a `Report`.

Hook activation/deactivation lifecycle lives here (not in
metal_events.py, which only defines the hooks themselves) -- `activate()`
installs the runtime.context hooks; `deactivate()` removes them. Test 15
("Profiling disabled") requires proving these are inert when NOT
activated; see tests/advisor/unit/test_profiler_overhead.py.
"""

from __future__ import annotations

import runpy
import sys
from dataclasses import dataclass

from numba_metal.advisor.metal_events import EventCollector
from numba_metal.advisor.metal_events import install as install_metal_hooks
from numba_metal.advisor.models import DeviceInfo, Report, now_ns
from numba_metal.advisor.sampling import DEFAULT_INTERVAL_MS, StackSampler


@dataclass
class ProfilingSession:
    """Active profiling state, returned by `activate()` and passed to
    `deactivate()` -- avoids any process-wide singleton so two
    sequential profiling runs in the same process never see each other's
    events (each ProfilingSession owns its own EventCollector)."""

    collector: EventCollector
    sampler: StackSampler | None


_active_session: ProfilingSession | None = None


def activate(*, sampling_interval_ms: float | None = None) -> ProfilingSession:
    """Install runtime.context hooks and (if `sampling_interval_ms` is
    given) start the statistical sampler. Only one session may be active
    at a time in this process (raises RuntimeError otherwise) -- matches
    `metal_events.uninstall_all`'s own "at most one active session"
    assumption."""
    global _active_session
    if _active_session is not None:
        raise RuntimeError(
            "A profiling session is already active in this process; call "
            "deactivate() before activate()ing another one."
        )
    collector = EventCollector()
    install_metal_hooks(collector)
    sampler = None
    if sampling_interval_ms is not None:
        sampler = StackSampler(interval_ms=sampling_interval_ms)
        sampler.start()
    _active_session = ProfilingSession(collector=collector, sampler=sampler)
    return _active_session


def deactivate(session: ProfilingSession) -> tuple:
    """Remove hooks/stop the sampler, returning (events, sampling_result)
    -- sampling_result is None if no sampler was started."""
    global _active_session
    from numba_metal.advisor.metal_events import uninstall_all

    uninstall_all()
    sampling_result = None
    if session.sampler is not None:
        sampling_result = session.sampler.stop()
    events = session.collector.events()
    dropped = session.collector.dropped_count()
    _active_session = None
    return events, sampling_result, dropped


def build_device_info() -> DeviceInfo:
    import platform
    import sys as _sys

    device_name = None
    family = None
    has_unified_memory = None
    try:
        from numba_metal.runtime.device import get_device_info

        info = get_device_info()
        device_name = info.name
        # numba-metal's own DeviceInfo has no dedicated "chip family"
        # field -- `.machine` is platform.machine() ("arm64" on every
        # Apple Silicon Mac, not useful here) and `.name` is the actual
        # MTLDevice name (e.g. "Apple M4 Pro"), which already contains
        # the real chip family. Derived by stripping the "Apple " prefix
        # rather than duplicating a second, separately-maintained lookup.
        family = (
            device_name[len("Apple ") :]
            if device_name and device_name.startswith("Apple ")
            else device_name
        )
        has_unified_memory = info.has_unified_memory
    except Exception:  # noqa: BLE001 -- device info is best-effort context
        # for the report header; its absence must never prevent a report
        # from being produced (e.g. running `scan`/`report` on a non-
        # Apple-Silicon machine that still wants a DeviceInfo stub).
        pass

    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = "unknown"
    try:
        import numba_metal

        numba_metal_version = numba_metal.__version__
    except (ImportError, AttributeError):
        numba_metal_version = "unknown"

    return DeviceInfo(
        device_name=device_name,
        apple_silicon_family=family,
        macos_version=platform.mac_ver()[0] or None,
        python_version=_sys.version.split()[0],
        numba_version=numba_version,
        numba_metal_version=numba_metal_version,
        has_unified_memory=has_unified_memory,
    )


def profile_script(
    script_path: str,
    script_args: list[str],
    *,
    sampling_interval_ms: float = DEFAULT_INTERVAL_MS,
    max_events: int | None = None,
) -> Report:
    """Execute `script_path` as `__main__` (via `runpy`, matching how
    `python script.py args...` itself runs it) with instrumentation and
    sampling active, then build a Report from what was observed.

    This is the ONLY function in the advisor package that executes
    arbitrary user code -- and only the exact script the user named on
    the command line, with the exact arguments they supplied (spec:
    "profile only executes the explicitly supplied script or tests.").
    """
    session = activate(sampling_interval_ms=sampling_interval_ms)
    if max_events is not None:
        session.collector._max_events = max_events

    old_argv = sys.argv
    sys.argv = [script_path, *script_args]
    try:
        runpy.run_path(script_path, run_name="__main__")
    finally:
        sys.argv = old_argv
        events, sampling_result, dropped = deactivate(session)

    return Report(
        schema_version=1,
        generated_at_ns=now_ns(),
        device=build_device_info(),
        events=events,
        candidates=(),
        compatibility=(),
        comparisons=(),
        correctness=(),
        scores=(),
        recommendations=(),
        sampling=sampling_result,
        source_paths_included=True,
        dropped_event_count=dropped,
    )


def profile_pytest(
    test_path: str,
    pytest_args: list[str],
    *,
    sampling_interval_ms: float = DEFAULT_INTERVAL_MS,
    max_events: int | None = None,
) -> Report:
    """Run pytest against `test_path` in-process (via `pytest.main`) with
    instrumentation and sampling active. Only the explicitly supplied
    test path/args are used -- no test discovery beyond what pytest
    itself would do for that path."""
    import pytest

    session = activate(sampling_interval_ms=sampling_interval_ms)
    if max_events is not None:
        session.collector._max_events = max_events

    try:
        pytest.main([test_path, *pytest_args])
    finally:
        events, sampling_result, dropped = deactivate(session)

    return Report(
        schema_version=1,
        generated_at_ns=now_ns(),
        device=build_device_info(),
        events=events,
        candidates=(),
        compatibility=(),
        comparisons=(),
        correctness=(),
        scores=(),
        recommendations=(),
        sampling=sampling_result,
        source_paths_included=True,
        dropped_event_count=dropped,
    )
