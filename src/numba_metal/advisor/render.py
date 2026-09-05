"""ASCII-only rendering. This module contains NO measurement logic --
every number it prints comes from an already-built model
(models.py/flamegraph.py/timeline.py); it only formats.

Character set restricted to the spec's portable ASCII list throughout:
`+ - | = # . < > [ ]` plus ordinary letters/digits/punctuation. No
Unicode box-drawing characters anywhere. ANSI color codes are optional
and additive -- every colored run of text also carries an ASCII label
(`[-]`/`[+]`/`[=]`/`[N]`, "MEASURED"/"ESTIMATED", etc.) so stripping color
never removes information (spec: "Never encode meaning using color
alone.").
"""

from __future__ import annotations

from numba_metal.advisor.flamegraph import FlameNode, merge_trees
from numba_metal.advisor.models import (
    Candidate,
    ComparisonResult,
    CompatibilityResult,
    CorrectnessResult,
    EvidenceKind,
    OpportunityScore,
    Recommendation,
)
from numba_metal.advisor.timeline import Timeline

DEFAULT_WIDTH = 78
MIN_WIDTH = 60
MAX_WIDTH = 240

# ANSI codes -- only ever used when color=True; every call site that uses
# one also emits the plain-ASCII label the code merely decorates.
_RESET = "\033[0m"
_GREEN = "\033[32m"
_BLUE = "\033[34m"
_RED = "\033[31m"
_GRAY = "\033[90m"
_MAGENTA = "\033[35m"
_YELLOW = "\033[33m"


def _clamp_width(width: int) -> int:
    return max(MIN_WIDTH, min(MAX_WIDTH, width))


def _color(text: str, code: str, *, color: bool) -> str:
    if not color:
        return text
    return f"{code}{text}{_RESET}"


def _truncate(label: str, max_len: int) -> str:
    if len(label) <= max_len:
        return label
    if max_len <= 3:
        return label[:max_len]
    return label[: max_len - 3] + "..."


def _bar(fraction: float, width: int, *, fill: str = "#") -> str:
    fraction = max(0.0, min(1.0, fraction))
    n = int(round(fraction * width))
    return fill * n


def render_header(title: str, *, width: int) -> str:
    width = _clamp_width(width)
    return title[: width - 1]


# ---------------------------------------------------------------------------
# Flame graph rendering
# ---------------------------------------------------------------------------


def _flame_line(
    label: str,
    depth: int,
    fraction: float,
    *,
    width: int,
    measured: bool,
    color: bool,
    bar_color_code: str | None,
) -> str:
    """One row of an ASCII flame graph, matching the spec's example:

        | main() ###########...####  100% |
        | +-- run_simulation() ###...  82% |
        | |   +-- simulate_paths() #.  67% |

    `depth` 0 is the root ("main()"); each deeper level prefixes with
    "|   " for every ancestor above it, then "+-- " for itself -- never
    plain spaces, so the tree structure survives even with color
    disabled and even after whitespace-collapsing terminals/pagers.
    """
    if depth == 0:
        indent = ""
    else:
        indent = "|   " * (depth - 1) + "+-- "
    pct_text = f"{fraction * 100:3.0f}%"
    tag = "" if measured else " [ESTIMATED]"
    left_border = "| "
    right_border = " |"
    # Layout: left_border + indent + label + " " + bar + " " + pct + tag + right_border
    fixed_overhead = (
        len(left_border)
        + len(indent)
        + 1  # space between label and bar
        + 1  # space between bar and pct_text
        + len(pct_text)
        + len(tag)
        + len(right_border)
    )
    label_field_width = max(8, width - fixed_overhead - 10)
    label_text = _truncate(label, label_field_width)
    bar_width = max(1, width - fixed_overhead - len(label_text))
    bar = _bar(fraction, bar_width)
    if bar_color_code is not None:
        bar = _color(bar, bar_color_code, color=color)
    core = f"{left_border}{indent}{label_text} {bar} {pct_text}{tag}"
    inner_target = width - len(right_border)
    pad = inner_target - len(_strip_ansi(core))
    if pad > 0:
        core = core + " " * pad
    return f"{core}{right_border}"


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)


def render_flame_graph(
    root: FlameNode,
    title: str,
    total_label_ns: int,
    *,
    width: int = DEFAULT_WIDTH,
    color: bool = False,
    max_depth: int | None = None,
    call_style_labels: bool = False,
) -> str:
    """Render one flame graph (CPU baseline OR Metal run), matching the
    spec's example layout: a bordered box, header line with total time,
    a scale-per-column note, then one line per frame with a bar,
    percentage, and MEASURED/ESTIMATED tag.

    `call_style_labels`: append "()" to each label for display, matching
    the spec's `main()`/`run_simulation()` example style -- appropriate
    for a tree built from sampled call-stack frames
    (`build_tree_from_folded_stacks`), where every node genuinely is a
    function call. Left False for instrumentation-event-based trees
    (`build_tree_from_events`), whose node labels (e.g. "synchronize",
    "GPU:simulate_paths") are event names, not necessarily bare function
    calls -- this is a purely cosmetic rendering choice, not a data fact,
    which is why it lives here rather than in flamegraph.py."""
    width = _clamp_width(width)
    lines: list[str] = []
    total_s = total_label_ns / 1e9
    lines.append(f"{title} - {total_s:.3f} s")
    if root.total_ns > 0:
        approx_col_ns = root.total_ns / max(1, width - 4)
        lines.append(f"Each column represents approximately {approx_col_ns/1e6:.0f} ms")
    border = "+" + "-" * (width - 2) + "+"
    lines.append(border)

    def _display_label(label: str) -> str:
        if call_style_labels and not label.endswith(")"):
            return f"{label}()"
        return label

    def _walk(node: FlameNode, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        fraction = node.fraction_of(root.total_ns) if root.total_ns > 0 else 0.0
        lines.append(
            _flame_line(
                _display_label(node.label),
                depth,
                fraction,
                width=width,
                measured=node.measured,
                color=color,
                bar_color_code=None,
            )
        )
        for child in node.sorted_children():
            _walk(child, depth + 1)

    if root.total_ns > 0:
        # `root` itself (labeled "<root>" by flamegraph.py) is never
        # rendered as a row -- its children are the program's real
        # top-level frames (e.g. the folded stack's own "main", or every
        # distinct instrumentation Event name). No synthetic "main()"
        # row is invented here: if the underlying data already has its
        # own single top-level call, that IS row 0, at 100%; if there
        # are several unrelated top-level frames (e.g. multiple
        # instrumentation events with no shared caller), each starts its
        # own tree at depth 0 rather than being falsely nested under a
        # fabricated common root.
        for child in root.sorted_children():
            _walk(child, 0)
    else:
        pad = width - 2 - len("(no data)")
        lines.append(f"| (no data){' ' * max(0, pad)}|")
    lines.append(border)
    return "\n".join(lines)


_DIFF_LEGEND = """Legend:
  [-] less time after Metal
  [+] more time after Metal
  [=] materially unchanged
  [N] new Metal overhead"""

_MATERIALLY_UNCHANGED_FRACTION = 0.05


def classify_diff(cpu_ns: int, metal_ns: int) -> str:
    """Returns one of "-", "+", "=", "N" -- classification logic lives
    here (not in flamegraph.py, which only computes the raw pair) since
    it is a presentation decision about what counts as "materially
    unchanged", tunable independent of the data model."""
    if cpu_ns == 0 and metal_ns > 0:
        return "N"
    if cpu_ns == 0 and metal_ns == 0:
        return "="
    delta_fraction = abs(metal_ns - cpu_ns) / cpu_ns
    if delta_fraction <= _MATERIALLY_UNCHANGED_FRACTION:
        return "="
    return "-" if metal_ns < cpu_ns else "+"


def render_differential_flame_graph(
    cpu_root: FlameNode,
    metal_root: FlameNode,
    *,
    cpu_total_ns: int,
    metal_total_ns: int,
    width: int = DEFAULT_WIDTH,
    color: bool = False,
    sort_by: str = "diff",
) -> str:
    width = _clamp_width(width)
    lines = ["DIFFERENTIAL FLAME GRAPH"]
    cpu_s = cpu_total_ns / 1e9
    metal_s = metal_total_ns / 1e9
    speedup = (cpu_total_ns / metal_total_ns) if metal_total_ns > 0 else None
    speedup_text = f"{speedup:.2f}x" if speedup is not None else "N/A"
    lines.append(
        f"CPU: {cpu_s:.3f} s    METAL: {metal_s:.3f} s    SPEEDUP: {speedup_text}"
    )
    lines.append(_DIFF_LEGEND)

    pairs = merge_trees(cpu_root, metal_root)
    rows = []
    for label, (cpu_ns, metal_ns) in pairs.items():
        tag = classify_diff(cpu_ns, metal_ns)
        delta_ns = metal_ns - cpu_ns
        rows.append((label, tag, delta_ns, cpu_ns, metal_ns))

    if sort_by == "total":
        rows.sort(key=lambda r: (-(r[3] + r[4]), r[0]))
    elif sort_by == "self":
        rows.sort(key=lambda r: (-r[4], r[0]))
    else:  # "diff" default
        rows.sort(key=lambda r: (-abs(r[2]), r[0]))

    max_abs_delta = max((abs(r[2]) for r in rows), default=1) or 1
    color_map = {"-": _GREEN, "+": _RED, "=": _GRAY, "N": _MAGENTA}
    for label, tag, delta_ns, _cpu_ns, _metal_ns in rows:
        delta_s = delta_ns / 1e9
        signed_text = f"{delta_s:+8.3f} s"
        label_text = _truncate(label, max(8, width - 40))
        bar_width = max(1, width - 40)
        if tag == "=":
            bar = "."
        else:
            bar = _bar(abs(delta_ns) / max_abs_delta, bar_width)
            if not bar:
                bar = "."
        bar_colored = _color(bar, color_map.get(tag, ""), color=color)
        lines.append(f"[{tag}] {label_text:<20s} {signed_text}  {bar_colored}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timeline rendering
# ---------------------------------------------------------------------------


def render_timeline(timeline: Timeline, *, width: int = DEFAULT_WIDTH) -> str:
    width = _clamp_width(width)
    if not timeline.spans:
        return "CPU/GPU TIMELINE - (no events recorded)"
    total_ms = timeline.duration_ns / 1e6
    lines = [f"CPU/GPU TIMELINE - 0 ms to {total_ms:.0f} ms"]

    label_width = 16
    axis_width = width - label_width - 1
    n_ticks = 6
    tick_positions = [int(i * axis_width / n_ticks) for i in range(n_ticks + 1)]
    tick_labels = [str(int(i * total_ms / n_ticks)) for i in range(n_ticks + 1)]
    header_chars = [" "] * axis_width
    for pos, label in zip(tick_positions, tick_labels, strict=True):
        for j, ch in enumerate(label):
            if pos + j < axis_width:
                header_chars[pos + j] = ch
    lines.append(" " * label_width + "".join(header_chars))

    ruler_chars = ["-"] * axis_width
    for pos in tick_positions:
        if 0 <= pos < axis_width:
            ruler_chars[pos] = "|"
    lines.append(" " * label_width + "".join(ruler_chars))

    def _ns_to_col(ns: int) -> int:
        if timeline.duration_ns <= 0:
            return 0
        return int((ns - timeline.start_ns) / timeline.duration_ns * (axis_width - 1))

    for lane in timeline.lanes():
        row = [" "] * axis_width
        for span in timeline.spans_in_lane(lane):
            start_col = max(0, min(axis_width - 1, _ns_to_col(span.start_ns)))
            end_col = max(start_col, min(axis_width - 1, _ns_to_col(span.end_ns)))
            fill_char = "#" if span.category == "gpu" else "."
            if end_col == start_col:
                row[start_col] = "["
            else:
                row[start_col] = "["
                row[end_col] = "]"
                for i in range(start_col + 1, end_col):
                    row[i] = fill_char
        label_text = _truncate(lane, label_width - 1).ljust(label_width)
        lines.append(f"{label_text}{''.join(row)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary / compatibility / correctness / score / recommendation text
# ---------------------------------------------------------------------------


def render_candidate(candidate: Candidate, *, width: int = DEFAULT_WIDTH) -> str:
    width = _clamp_width(width)
    header = (
        f"[{candidate.confidence.value} POTENTIAL] {candidate.file}:"
        f"{candidate.line_start} {candidate.qualified_name}"
    )
    lines = [_truncate(header, width)]
    if candidate.reasons:
        lines.append("Why:")
        for r in candidate.reasons:
            lines.append(f"  - {r}")
    if candidate.blockers:
        lines.append("Blockers:")
        for b in candidate.blockers:
            lines.append(f"  - {b}")
    if candidate.unknowns:
        lines.append("Unknown:")
        for u in candidate.unknowns:
            lines.append(f"  - {u}")
    return "\n".join(lines)


def render_compatibility(
    result: CompatibilityResult, *, width: int = DEFAULT_WIDTH
) -> str:
    lines = [f"Compatibility: {result.status.value}"]
    lines.append("Function:")
    lines.append(f"  {result.file}:{result.line_start} {result.qualified_name}")
    if result.supported_features:
        lines.append("Supported:")
        for f in result.supported_features:
            lines.append(f"  - {f}")
    if result.blockers:
        lines.append("Blocked:")
        for b in result.blockers:
            lines.append(f"  - {b}")
    if result.recommendation:
        lines.append("Recommendation:")
        lines.append(f"  {result.recommendation}")
    return "\n".join(lines)


def render_correctness(result: CorrectnessResult) -> str:
    lines = ["CORRECTNESS"]
    lines.append(f"Result:                 {'PASS' if result.passed else 'FAIL'}")
    lines.append(
        f"Elements compared:      {result.elements_compared:,}"
        + (f" (sampled from {result.elements_total:,})" if result.sampled else "")
    )
    if result.max_abs_error is not None:
        lines.append(f"Maximum absolute error: {result.max_abs_error:.3g}")
    if result.max_rel_error is not None:
        lines.append(f"Maximum relative error: {result.max_rel_error:.3g}")
    if result.atol is not None:
        lines.append(f"Allowed atol:           {result.atol:.3g}")
    if result.rtol is not None:
        lines.append(f"Allowed rtol:           {result.rtol:.3g}")
    lines.append("Dtype:")
    lines.append(f"  CPU:                  {result.cpu_dtype}")
    lines.append(f"  Metal:                {result.metal_dtype}")
    if not result.passed and result.failure_reason:
        lines.append(f"Failure reason:         {result.failure_reason}")
    return "\n".join(lines)


def render_opportunity_score(score: OpportunityScore) -> str:
    title = (
        "OPPORTUNITY SCORE"
        if score.evidence == EvidenceKind.MEASURED
        else "POTENTIAL SCORE (static evidence only, not a speedup estimate)"
    )
    lines = [f"{title}: {score.total:.0f}/{score.max_total:.0f}"]
    for c in score.components:
        sign = "+" if c.value >= 0 else ""
        lines.append(
            f"  {c.name}: {sign}{c.value:.0f}/{c.max_value:.0f} -- {c.explanation}"
        )
    lines.append(f"Confidence: {score.confidence.value}")
    return "\n".join(lines)


def render_recommendation(rec: Recommendation) -> str:
    lines = [f"Recommendation ({rec.direction.value}): {rec.text}"]
    lines.append(f"  Supporting measurement: {rec.supporting_measurement}")
    lines.append(f"  Location: {rec.file}:{rec.line_start}")
    lines.append(f"  Confidence: {rec.confidence.value}")
    lines.append(f"  Evidence: {rec.evidence.value}")
    lines.append(f"  How to verify: {rec.how_to_verify}")
    return "\n".join(lines)


def render_comparison_summary(result: ComparisonResult) -> str:
    lines = [f"COMPARISON: {result.qualified_name} ({result.file}:{result.line_start})"]
    for regime in result.regimes:
        lines.append(f"-- {regime.mode.value} --")
        if regime.cpu is not None:
            c = regime.cpu
            lines.append(
                f"  CPU:   median={c.median_ns/1e6:.3f}ms "
                f"min={c.min_ns/1e6:.3f}ms max={c.max_ns/1e6:.3f}ms "
                f"mean={c.mean_ns/1e6:.3f}ms stdev={c.stdev_ns/1e6:.3f}ms n={c.n}"
            )
            if c.outliers_ns:
                lines.append(f"    outliers (not removed): {len(c.outliers_ns)}")
        if regime.metal is not None:
            m = regime.metal
            lines.append(
                f"  Metal: median={m.median_ns/1e6:.3f}ms "
                f"min={m.min_ns/1e6:.3f}ms max={m.max_ns/1e6:.3f}ms "
                f"mean={m.mean_ns/1e6:.3f}ms stdev={m.stdev_ns/1e6:.3f}ms n={m.n}"
            )
            if m.outliers_ns:
                lines.append(f"    outliers (not removed): {len(m.outliers_ns)}")
        if regime.speedup is not None:
            prelim = " (PRELIMINARY -- too few runs)" if regime.preliminary else ""
            lines.append(f"  Speedup: {regime.speedup:.2f}x{prelim}")
    if result.whole_program_speedup is not None:
        prelim = " (PRELIMINARY)" if result.whole_program_preliminary else ""
        lines.append(
            f"Whole-program speedup: {result.whole_program_speedup:.2f}x{prelim}"
        )
    if result.profiler_overhead_ns is not None:
        lines.append(
            f"Profiler overhead (estimated): {result.profiler_overhead_ns/1e6:.3f}ms"
        )
    return "\n".join(lines)
