"""CPU/GPU timeline data model: lanes of time-bounded spans on a shared
time axis, built from `models.Event`s.

Pure data-shaping, no ASCII rendering (render.py's job) and no new
measurement (every span here comes directly from an Event's
start_ns/duration_ns). The lane a span is assigned to is a presentation
grouping decision, but it is a DETERMINISTIC one (by event category),
not a guess -- see `_LANE_FOR_CATEGORY`.
"""

from __future__ import annotations

from dataclasses import dataclass

from numba_metal.advisor.models import Event, GpuTimestampSource

_LANE_FOR_CATEGORY: dict[str, str] = {
    "cpu": "CPU main",
    "python": "CPU main",
    "compile": "GPU compile",
    "gpu": "GPU kernel",
    "sync": "CPU sync",
    "transfer": "GPU transfer",
}
#: Finer-grained override, keyed on event_type, checked before the
#: category-based mapping above -- several distinct event_types share
#: category "gpu" (submission vs. actual kernel execution) but belong on
#: visually separate timeline lanes, matching the spec's example layout
#: ("GPU queue" / "GPU transfer" / "GPU kernel" as distinct rows).
_LANE_FOR_EVENT_TYPE: dict[str, str] = {
    "command_submit": "GPU queue",
    "command_encode": "GPU queue",
    "metal_cache_hit": "GPU compile",
    "metal_cache_miss": "GPU compile",
    "buffer_alloc": "GPU transfer",
    "buffer_reuse": "GPU transfer",
    "result_materialize": "GPU transfer",
}
DEFAULT_LANE = "Other"

#: Fixed lane display order (top to bottom), matching the spec's example
#: timeline layout -- CPU work first, then queue/transfer/kernel/sync.
LANE_ORDER: tuple[str, ...] = (
    "CPU main",
    "GPU queue",
    "GPU transfer",
    "GPU compile",
    "GPU kernel",
    "CPU sync",
    DEFAULT_LANE,
)


@dataclass(frozen=True, slots=True)
class TimelineSpan:
    label: str
    start_ns: int
    end_ns: int
    lane: str
    measured: bool
    gpu_timestamp_source: GpuTimestampSource
    category: str

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True, slots=True)
class Timeline:
    spans: tuple[TimelineSpan, ...]
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def lanes(self) -> list[str]:
        """Lanes actually present in this timeline, in the fixed
        LANE_ORDER (never insertion order) -- keeps repeated renders of
        the same data stable."""
        present = {s.lane for s in self.spans}
        ordered = [lane for lane in LANE_ORDER if lane in present]
        # Any lane not in the fixed list (should not normally happen)
        # still appears, sorted, after the known ones -- never silently
        # dropped.
        extra = sorted(present - set(ordered))
        return ordered + extra

    def spans_in_lane(self, lane: str) -> list[TimelineSpan]:
        return sorted(
            (s for s in self.spans if s.lane == lane), key=lambda s: s.start_ns
        )


def build_timeline(events: tuple[Event, ...]) -> Timeline:
    """Convert a flat Event stream into a Timeline. Every span's
    `measured`/`gpu_timestamp_source` are carried through UNCHANGED from
    the source Event -- this module never upgrades an estimated span to
    "measured" or a HOST_WALL_CLOCK span to "device timestamp"."""
    if not events:
        return Timeline(spans=(), start_ns=0, end_ns=0)
    spans = []
    for e in events:
        lane = _LANE_FOR_EVENT_TYPE.get(
            e.event_type, _LANE_FOR_CATEGORY.get(e.category, DEFAULT_LANE)
        )
        spans.append(
            TimelineSpan(
                label=e.name,
                start_ns=e.start_ns,
                end_ns=e.start_ns + max(e.duration_ns, 0),
                lane=lane,
                measured=e.measured,
                gpu_timestamp_source=e.gpu_timestamp_source,
                category=e.category,
            )
        )
    start = min(s.start_ns for s in spans)
    end = max(s.end_ns for s in spans)
    return Timeline(spans=tuple(spans), start_ns=start, end_ns=end)


def detect_gpu_idle_periods(
    timeline: Timeline, *, min_idle_ns: int = 1_000_000
) -> list[tuple[int, int]]:
    """Return (start_ns, end_ns) gaps of at least `min_idle_ns` between
    consecutive GPU-lane spans -- a real, derivable fact from the span
    data (not a new measurement), useful for a renderer/recommendation
    rule to point out "GPU idle for Xms here"."""
    gpu_spans = sorted(
        (
            s
            for s in timeline.spans
            if s.lane in ("GPU kernel", "GPU compile", "GPU transfer", "GPU queue")
        ),
        key=lambda s: s.start_ns,
    )
    idle_periods = []
    for prev, curr in zip(gpu_spans, gpu_spans[1:], strict=False):
        gap = curr.start_ns - prev.end_ns
        if gap >= min_idle_ns:
            idle_periods.append((prev.end_ns, curr.start_ns))
    return idle_periods
