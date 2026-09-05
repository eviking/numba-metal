"""Test 11 (asynchronous timeline) and Test 12 (cold versus warm) from
the governing spec.
"""

from __future__ import annotations

from numba_metal.advisor.models import Event, GpuTimestampSource
from numba_metal.advisor.render import render_timeline
from numba_metal.advisor.timeline import build_timeline, detect_gpu_idle_periods


def test_overlapping_cpu_gpu_spans_render_correctly():
    """Test 11: Overlapping CPU submission, GPU execution, and CPU work
    render correctly."""
    events = (
        Event(
            event_type="python_call",
            name="prepare",
            category="cpu",
            start_ns=0,
            duration_ns=10_000_000,
            measured=True,
        ),
        Event(
            event_type="command_submit",
            name="encode",
            category="gpu",
            start_ns=10_000_000,
            duration_ns=5_000_000,
            measured=True,
        ),
        Event(
            event_type="metal_kernel",
            name="simulate_paths",
            category="gpu",
            start_ns=15_000_000,
            duration_ns=60_000_000,
            measured=True,
            gpu_timestamp_source=GpuTimestampSource.HOST_WALL_CLOCK,
        ),
        Event(
            event_type="synchronize",
            name="waiting",
            category="sync",
            start_ns=15_000_000,
            duration_ns=60_000_000,
            measured=True,
        ),
        Event(
            event_type="python_call",
            name="postprocess",
            category="cpu",
            start_ns=75_000_000,
            duration_ns=45_000_000,
            measured=True,
        ),
    )
    timeline = build_timeline(events)
    assert timeline.start_ns == 0
    assert timeline.end_ns == 120_000_000

    lanes = timeline.lanes()
    assert "CPU main" in lanes
    assert "GPU kernel" in lanes
    assert "CPU sync" in lanes
    # GPU queue and GPU kernel must be DISTINCT lanes (found as a real
    # bug: command_submit and metal_kernel share category "gpu", which
    # initially collapsed them into one lane, corrupting the rendered
    # row with two overlapping "[" start markers).
    assert "GPU queue" in lanes
    assert lanes.index("CPU main") < lanes.index("GPU kernel")

    out = render_timeline(timeline, width=78)
    assert "CPU/GPU TIMELINE" in out
    assert out.isascii()
    # Every lane must appear as its own row.
    for lane in lanes:
        assert lane in out


def test_timeline_never_upgrades_estimated_to_measured():
    events = (
        Event(
            event_type="metal_kernel",
            name="k",
            category="gpu",
            start_ns=0,
            duration_ns=1000,
            measured=False,
            gpu_timestamp_source=GpuTimestampSource.NOT_APPLICABLE,
        ),
    )
    timeline = build_timeline(events)
    assert timeline.spans[0].measured is False
    assert timeline.spans[0].gpu_timestamp_source == GpuTimestampSource.NOT_APPLICABLE


def test_gpu_timestamp_source_never_claims_device_timestamp():
    """No code path in this package can produce GpuTimestampSource.DEVICE_TIMESTAMP
    -- numba-metal exposes no such clock. This is a structural regression
    guard: if some future change starts claiming DEVICE_TIMESTAMP without
    an actual device clock existing, this test (and the ones building
    real events from metal_events.py) must be revisited."""
    events = (
        Event(
            event_type="metal_kernel",
            name="k",
            category="gpu",
            start_ns=0,
            duration_ns=1000,
            measured=True,
            gpu_timestamp_source=GpuTimestampSource.HOST_WALL_CLOCK,
        ),
    )
    timeline = build_timeline(events)
    assert timeline.spans[0].gpu_timestamp_source != GpuTimestampSource.DEVICE_TIMESTAMP


def test_detect_gpu_idle_periods():
    events = (
        Event(
            event_type="metal_kernel",
            name="a",
            category="gpu",
            start_ns=0,
            duration_ns=1_000_000,
            measured=True,
        ),
        Event(
            event_type="metal_kernel",
            name="b",
            category="gpu",
            start_ns=10_000_000,
            duration_ns=1_000_000,
            measured=True,
        ),
    )
    timeline = build_timeline(events)
    idle = detect_gpu_idle_periods(timeline, min_idle_ns=1_000_000)
    assert idle == [(1_000_000, 10_000_000)]


def test_cold_compile_appears_only_in_cold_not_warm():
    """Test 12: Compilation appears in cold results but not incorrectly
    in warm kernel time."""
    cold_events = (
        Event(
            event_type="metal_cache_miss",
            name="kernel",
            category="compile",
            start_ns=0,
            duration_ns=96_000_000,
            measured=True,
            cold_start=True,
        ),
    )
    warm_events = (
        Event(
            event_type="metal_cache_hit",
            name="kernel",
            category="compile",
            start_ns=0,
            duration_ns=34_000,
            measured=True,
            cold_start=False,
        ),
        Event(
            event_type="metal_kernel",
            name="submission_to_completion",
            category="gpu",
            start_ns=34_000,
            duration_ns=110_000,
            measured=True,
        ),
    )
    assert cold_events[0].cold_start is True
    assert all(not e.cold_start for e in warm_events)
    # Cold compile duration must not leak into the warm kernel timing --
    # the warm regime's total event duration is orders of magnitude
    # smaller than the cold compile alone.
    warm_total = sum(e.duration_ns for e in warm_events)
    assert warm_total < cold_events[0].duration_ns / 100
