"""Flame graph data model: pure data-shaping (widths, percentages, stable
ordering) with zero measurement logic and zero ASCII rendering -- see
render.py's module docstring for the collection/analysis/presentation
boundary this package maintains throughout.

Consumes `models.FoldedStack` (sampling-based) and/or `models.Event`
(instrumentation-based) and produces a `FlameNode` tree that `render.py`
walks to draw the ASCII flame graph. This module never decides how wide a
terminal column is or what characters to draw -- only relative
proportions and stable node ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numba_metal.advisor.models import Event, FoldedStack


@dataclass
class FlameNode:
    """One frame in the flame graph tree. `self_ns`/`total_ns` (or
    `self_count`/`total_count` for sample-based graphs) are set by
    `build_tree`; `children` is keyed by frame label for stable,
    deterministic ordering across repeated builds of the same input
    (spec: "Stable ordering across repeated reports")."""

    label: str
    total_ns: int = 0
    self_ns: int = 0
    measured: bool = True
    category: str | None = None
    children: dict[str, FlameNode] = field(default_factory=dict)

    def fraction_of(self, root_total_ns: int) -> float:
        if root_total_ns <= 0:
            return 0.0
        return self.total_ns / root_total_ns

    def sorted_children(self) -> list[FlameNode]:
        """Stable order: by descending total_ns, tie-broken by label --
        never insertion order, which would vary run to run for
        dict-backed aggregation."""
        return sorted(self.children.values(), key=lambda n: (-n.total_ns, n.label))


def build_tree_from_folded_stacks(
    stacks: tuple[FoldedStack, ...], *, unit_ns_per_sample: float
) -> FlameNode:
    """Build a FlameNode tree from folded-stack sample counts, converting
    sample counts to an approximate duration via `unit_ns_per_sample`
    (the sampler's own interval -- an ESTIMATE of time spent, since a
    statistical sampler counts occurrences, not exact durations; callers
    must not present this as `measured=True` wall-clock time -- see
    render.py's labeling)."""
    root = FlameNode(label="<root>", measured=False)
    for stack in stacks:
        node = root
        node.total_ns += int(stack.count * unit_ns_per_sample)
        for frame_label in stack.frames:
            child = node.children.get(frame_label)
            if child is None:
                child = FlameNode(label=frame_label, measured=False)
                node.children[frame_label] = child
            child.total_ns += int(stack.count * unit_ns_per_sample)
            node = child
        node.self_ns += int(stack.count * unit_ns_per_sample)
    return root


def build_tree_from_events(events: tuple[Event, ...]) -> FlameNode:
    """Build a FlameNode tree from a flat list of instrumentation Events
    using each event's `name` as a single-level child of the root
    (instrumentation events do not carry a call-stack path the way
    sampled stacks do -- they are individually-named spans). Events with
    the same name are summed. `measured` is True only if EVERY
    contributing event was itself measured=True."""
    root = FlameNode(label="<root>", measured=True)
    for event in events:
        child = root.children.get(event.name)
        if child is None:
            child = FlameNode(
                label=event.name, measured=event.measured, category=event.category
            )
            root.children[event.name] = child
        child.total_ns += event.duration_ns
        child.self_ns += event.duration_ns
        child.measured = child.measured and event.measured
        root.total_ns += event.duration_ns
    return root


def merge_trees(
    cpu_tree: FlameNode, metal_tree: FlameNode
) -> dict[str, tuple[int, int]]:
    """Return {label: (cpu_ns_or_0, metal_ns_or_0)} for every label
    appearing in either tree's direct children -- the raw data
    `render.py`'s differential flame graph classifies into
    faster/slower/unchanged/new-overhead buckets. Classification itself
    (what counts as "materially unchanged") is a rendering/presentation
    decision made in render.py, not here."""
    labels = set(cpu_tree.children) | set(metal_tree.children)
    result: dict[str, tuple[int, int]] = {}
    for label in labels:
        cpu_ns = cpu_tree.children[label].total_ns if label in cpu_tree.children else 0
        metal_ns = (
            metal_tree.children[label].total_ns if label in metal_tree.children else 0
        )
        result[label] = (cpu_ns, metal_ns)
    return result
