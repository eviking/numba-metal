"""Interactive terminal UI, built on `blessed` (the one new, justified
dependency for this feature -- optional `[tui]` extra, imported lazily
here only, never required by scan/report/non-interactive profile/compare).

Non-interactive contexts (redirected output, no TTY, CI) fall back to
the same plain ASCII text `render.py` produces for file export -- this
module NEVER contains its own rendering logic distinct from render.py;
it only adds keyboard navigation and screen layout around calls to the
same render_* functions everything else uses. `report PROFILE.json`
never imports this module at all (spec: "Do not require a terminal UI
to read saved profiles.").
"""

from __future__ import annotations

import os
import sys

from numba_metal.advisor.models import Report


def is_interactive() -> bool:
    """True iff stdout is a real terminal with a known TERM and `blessed`
    is importable. Any False here means the caller should use plain
    text output instead of attempting the TUI."""
    if not sys.stdout.isatty():
        return False
    if not os.environ.get("TERM"):
        return False
    try:
        import blessed  # noqa: F401
    except ImportError:
        return False
    return True


_TABS = ["Summary", "Candidates", "Flame Graph", "Timeline", "Details"]


class AdvisorTUI:
    """Keyboard-driven, single-screen terminal UI over one `Report`.

    Uses `blessed.Terminal`'s `fullscreen()`/`cbreak()`/`hidden_cursor()`
    context managers for state restoration on exit -- including on an
    uncaught exception or Ctrl-C, matching `curses.wrapper()`'s own
    guarantee (blessed documents these context managers as exception-safe
    via their own __exit__, so a `with` block here is sufficient; no
    separate try/finally is needed beyond what `run()` already has for
    its own event loop's KeyboardInterrupt handling).
    """

    def __init__(self, report: Report, *, color: bool = True, width: int | None = None):
        self.report = report
        self.color = color
        self._forced_width = width
        self.active_tab = 0
        self.selected_index = 0
        self.sort_by = "total"
        self.filter_text = ""
        self.mode = "warm"

    def _render_body(self, width: int) -> str:
        from numba_metal.advisor.export import (
            render_all_text_reports,
            render_summary_text,
        )

        tab = _TABS[self.active_tab]
        if tab == "Summary":
            return render_summary_text(self.report)
        if tab == "Candidates":
            from numba_metal.advisor.render import render_candidate

            lines = []
            for c in self.report.candidates:
                if (
                    self.filter_text
                    and self.filter_text.lower() not in c.qualified_name.lower()
                ):
                    continue
                lines.append(render_candidate(c, width=width))
                lines.append("")
            return "\n".join(lines) or "(no candidates match the current filter)"
        if tab == "Flame Graph":
            outputs = render_all_text_reports(
                self.report, width=width, color=self.color
            )
            parts = []
            for key in (
                "flamegraph-cpu.txt",
                "flamegraph-metal.txt",
                "flamegraph-diff.txt",
            ):
                if key in outputs:
                    parts.append(outputs[key])
            return "\n\n".join(parts) or "(no event data to build a flame graph from)"
        if tab == "Timeline":
            outputs = render_all_text_reports(
                self.report, width=width, color=self.color
            )
            return outputs.get("timeline.txt", "(no events recorded)")
        if tab == "Details":
            from numba_metal.advisor.render import (
                render_comparison_summary,
                render_correctness,
                render_opportunity_score,
                render_recommendation,
            )

            lines = []
            for cmp_result in self.report.comparisons:
                lines.append(render_comparison_summary(cmp_result))
                lines.append("")
            for corr in self.report.correctness:
                lines.append(render_correctness(corr))
                lines.append("")
            for score in self.report.scores:
                lines.append(render_opportunity_score(score))
                lines.append("")
            for rec in self.report.recommendations:
                lines.append(render_recommendation(rec))
                lines.append("")
            return "\n".join(lines) or "(no comparison/correctness/score data)"
        return ""

    def _header(self, term, width: int) -> list[str]:
        lines = []
        title = " NUMBA-METAL ADVISOR "
        pad = width - len(title) - 2
        lines.append("+" + title + "-" * max(0, pad) + "+")
        device = self.report.device.device_name or "unknown device"
        lines.append(f"| Device: {device[: width - 12]}".ljust(width - 1) + "|")
        tab_line = "| "
        for i, name in enumerate(_TABS):
            marker = f"[{i + 1}] {name}  "
            tab_line += marker
        tab_line = tab_line[: width - 1].ljust(width - 1) + "|"
        lines.append(tab_line)
        lines.append("+" + "-" * (width - 2) + "+")
        return lines

    def _footer(self, width: int) -> str:
        text = "q Quit | Tab Next | f Filter | s Sort | c Cold/Warm"
        return text[: width - 1].ljust(width - 1)

    def run(self) -> int:
        import blessed

        term = blessed.Terminal()
        width = self._forced_width or term.width or 78
        with term.fullscreen(), term.cbreak(), term.hidden_cursor():
            while True:
                width = self._forced_width or term.width or 78
                print(term.home + term.clear, end="")
                for line in self._header(term, width):
                    print(line)
                body = self._render_body(width)
                body_height = max(1, term.height - 6)
                body_lines = body.splitlines()[:body_height]
                for line in body_lines:
                    print(line[:width])
                print(self._footer(width))

                key = term.inkey(timeout=None)
                if key.lower() == "q":
                    return 0
                if key.code == term.KEY_TAB or key == "\t":
                    self.active_tab = (self.active_tab + 1) % len(_TABS)
                elif key.isdigit() and 1 <= int(key) <= len(_TABS):
                    self.active_tab = int(key) - 1
                elif key.lower() == "c":
                    self.mode = "cold" if self.mode == "warm" else "warm"
                elif key.lower() == "f":
                    print(term.move_y(term.height - 1) + "Filter: ", end="", flush=True)
                    self.filter_text = input()
                elif key.lower() == "s":
                    self.sort_by = {
                        "total": "self",
                        "self": "diff",
                        "diff": "total",
                    }[self.sort_by]


def run_tui_or_fallback(
    report: Report, *, color: bool = True, width: int | None = None
) -> int:
    """Run the interactive TUI if the environment supports it; otherwise
    print the same content as plain text and return 0. Never raises out
    of this function due to environment limitations -- a non-interactive
    context is a normal, expected case, not an error."""
    if not is_interactive():
        from numba_metal.advisor.export import render_summary_text

        print(render_summary_text(report))
        return 0
    tui = AdvisorTUI(report, color=color, width=width)
    try:
        return tui.run()
    except KeyboardInterrupt:
        return 130
