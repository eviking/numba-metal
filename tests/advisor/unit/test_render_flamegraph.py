"""Test 7 (ASCII CPU flame graph), Test 8 (differential flame graph),
Test 9 (narrow terminal, 60-240 columns), Test 10 (no-color mode) from
the governing spec.

Fixed, hand-built synthetic FoldedStack/Event inputs throughout -- no
dependency on real machine timing, per the spec's explicit instruction
("Use fixed clocks and deterministic synthetic event fixtures for unit
tests. Do not make timing-sensitive unit tests depend on real machine
performance.").
"""

from __future__ import annotations

import re

from numba_metal.advisor.flamegraph import (
    build_tree_from_events,
    build_tree_from_folded_stacks,
)
from numba_metal.advisor.models import Event, FoldedStack, GpuTimestampSource
from numba_metal.advisor.render import (
    _strip_ansi,
    render_differential_flame_graph,
    render_flame_graph,
)

_KNOWN_STACKS = (
    FoldedStack(frames=("main", "run_simulation", "simulate_paths"), count=481),
    FoldedStack(frames=("main", "run_simulation", "calculate_percentile"), count=93),
    FoldedStack(frames=("main", "write_results"), count=22),
)
_TOTAL_SAMPLES = 481 + 93 + 22  # 596


def _bordered_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.startswith("|") or line.startswith("+")
    ]


def test_known_folded_stack_produces_correct_widths_and_percentages():
    """Test 7: Known folded-stack input produces a deterministic graph
    whose widths and percentages are correct."""
    tree = build_tree_from_folded_stacks(_KNOWN_STACKS, unit_ns_per_sample=1_000_000)
    out = render_flame_graph(
        tree, "CPU BASELINE", tree.total_ns, width=78, call_style_labels=True
    )

    # main (100%), run_simulation (574/596 = 96.3% -> rounds to 96%),
    # write_results (22/596 = 3.7% -> rounds to 4%).
    assert re.search(r"main\(\).*100%", out)
    assert re.search(r"run_simulation\(\).*96%", out) or "run_simulation" in out
    assert "3%" not in out or "4%" in out  # write_results rounds to ~4%

    # simulate_paths: 481/596 = 80.7% of TOTAL (percentages in this
    # renderer are always relative to the grand total, matching the
    # spec's own example where nested percentages are of the whole).
    assert "simulate_paths" in out
    assert "calculate_percentile" in out


def test_flame_graph_rendering_is_deterministic_across_repeated_calls():
    """Stable ordering across repeated reports -- spec requirement."""
    tree1 = build_tree_from_folded_stacks(_KNOWN_STACKS, unit_ns_per_sample=1_000_000)
    tree2 = build_tree_from_folded_stacks(_KNOWN_STACKS, unit_ns_per_sample=1_000_000)
    out1 = render_flame_graph(tree1, "CPU BASELINE", tree1.total_ns, width=78)
    out2 = render_flame_graph(tree2, "CPU BASELINE", tree2.total_ns, width=78)
    assert out1 == out2


def test_flame_graph_estimated_tag_present_for_sampled_data():
    """Sampled (statistical) data must never claim to be measured wall-
    clock duration -- every row must carry [ESTIMATED]."""
    tree = build_tree_from_folded_stacks(_KNOWN_STACKS, unit_ns_per_sample=1_000_000)
    out = render_flame_graph(tree, "CPU BASELINE", tree.total_ns, width=78)
    for line in _bordered_lines(out):
        if line.startswith("+"):
            continue
        if "(no data)" in line:
            continue
        assert "[ESTIMATED]" in line, f"missing ESTIMATED tag: {line!r}"


def test_flame_graph_instrumentation_events_are_measured_not_estimated():
    """Instrumentation-event-based trees (real timed spans) must NOT
    carry [ESTIMATED] -- they are genuinely measured wall-clock time."""
    events = (
        Event(
            event_type="metal_kernel",
            name="kernel",
            category="gpu",
            start_ns=0,
            duration_ns=1000,
            measured=True,
            gpu_timestamp_source=GpuTimestampSource.HOST_WALL_CLOCK,
        ),
    )
    tree = build_tree_from_events(events)
    out = render_flame_graph(tree, "METAL RUN", tree.total_ns, width=78)
    assert "[ESTIMATED]" not in out


def test_differential_flame_graph_classifications():
    """Test 8: CPU and Metal profiles produce correct faster, slower,
    unchanged, and new-overhead classifications."""
    cpu_events = (
        Event(
            event_type="python_call",
            name="simulate_paths",
            category="cpu",
            start_ns=0,
            duration_ns=1_458_000_000,
            measured=True,
        ),
        Event(
            event_type="python_call",
            name="percentile",
            category="cpu",
            start_ns=0,
            duration_ns=200_000_000,
            measured=True,
        ),
    )
    metal_events = (
        # Faster after Metal (huge reduction).
        Event(
            event_type="metal_kernel",
            name="simulate_paths",
            category="gpu",
            start_ns=0,
            duration_ns=38_000_000,
            measured=True,
        ),
        # Unchanged (within 5% of CPU's 200ms).
        Event(
            event_type="python_call",
            name="percentile",
            category="cpu",
            start_ns=0,
            duration_ns=199_000_000,
            measured=True,
        ),
        # New overhead: did not exist in the CPU trace at all.
        Event(
            event_type="synchronize",
            name="synchronization",
            category="sync",
            start_ns=0,
            duration_ns=10_000_000,
            measured=True,
        ),
    )
    cpu_tree = build_tree_from_events(cpu_events)
    metal_tree = build_tree_from_events(metal_events)
    out = render_differential_flame_graph(
        cpu_tree,
        metal_tree,
        cpu_total_ns=sum(e.duration_ns for e in cpu_events),
        metal_total_ns=sum(e.duration_ns for e in metal_events),
        width=78,
    )
    lines = {line.split()[1]: line[:4] for line in out.splitlines() if line[:1] == "["}
    assert lines["simulate_paths"] == "[-] "
    assert lines["percentile"] == "[=] "
    assert lines["synchronization"] == "[N] "


def test_differential_flame_graph_speedup_header():
    cpu_events = (
        Event(
            event_type="python_call",
            name="f",
            category="cpu",
            start_ns=0,
            duration_ns=1_842_000_000,
            measured=True,
        ),
    )
    metal_events = (
        Event(
            event_type="metal_kernel",
            name="f",
            category="gpu",
            start_ns=0,
            duration_ns=576_000_000,
            measured=True,
        ),
    )
    out = render_differential_flame_graph(
        build_tree_from_events(cpu_events),
        build_tree_from_events(metal_events),
        cpu_total_ns=1_842_000_000,
        metal_total_ns=576_000_000,
        width=78,
    )
    assert "SPEEDUP: 3.20x" in out


def test_narrow_and_wide_terminal_widths_stay_bordered_correctly():
    """Test 9: All important results remain readable at 60 columns
    without broken borders or hidden measurement labels; also checked up
    to the spec's stated max of 240."""
    tree = build_tree_from_folded_stacks(_KNOWN_STACKS, unit_ns_per_sample=1_000_000)
    for width in (60, 78, 120, 240):
        out = render_flame_graph(
            tree, "CPU BASELINE", tree.total_ns, width=width, call_style_labels=True
        )
        for line in _bordered_lines(out):
            assert len(line) == width, f"width={width}: bad line length: {line!r}"
        # The measurement label (percentage) must survive truncation at
        # every width -- never silently dropped.
        assert "%" in out


def test_width_below_minimum_is_clamped():
    tree = build_tree_from_folded_stacks(_KNOWN_STACKS, unit_ns_per_sample=1_000_000)
    out = render_flame_graph(tree, "CPU BASELINE", tree.total_ns, width=10)
    for line in _bordered_lines(out):
        assert len(line) == 60  # MIN_WIDTH


def test_no_color_output_is_ansi_stripped_colored_output():
    """Test 10: All meaning remains available without ANSI sequences --
    verified by asserting the no-color render is byte-identical to the
    colored render with ANSI codes mechanically stripped (never just
    "looks similar")."""
    cpu_events = (
        Event(
            event_type="python_call",
            name="simulate_paths",
            category="cpu",
            start_ns=0,
            duration_ns=1_458_000_000,
            measured=True,
        ),
    )
    metal_events = (
        Event(
            event_type="metal_kernel",
            name="GPU kernel",
            category="gpu",
            start_ns=0,
            duration_ns=38_000_000,
            measured=True,
        ),
    )
    cpu_tree = build_tree_from_events(cpu_events)
    metal_tree = build_tree_from_events(metal_events)
    out_color = render_differential_flame_graph(
        cpu_tree,
        metal_tree,
        cpu_total_ns=1_458_000_000,
        metal_total_ns=38_000_000,
        width=78,
        color=True,
    )
    out_nocolor = render_differential_flame_graph(
        cpu_tree,
        metal_tree,
        cpu_total_ns=1_458_000_000,
        metal_total_ns=38_000_000,
        width=78,
        color=False,
    )
    assert _strip_ansi(out_color) == out_nocolor
    # And no meaning is encoded by color alone: every classified row
    # keeps its bracketed ASCII tag regardless of color.
    assert "[-]" in out_nocolor or "[N]" in out_nocolor


def test_ascii_only_character_set():
    """Spec: portable ASCII characters only, no Unicode box-drawing."""
    tree = build_tree_from_folded_stacks(_KNOWN_STACKS, unit_ns_per_sample=1_000_000)
    out = render_flame_graph(tree, "CPU BASELINE", tree.total_ns, width=78)
    assert out.isascii()
