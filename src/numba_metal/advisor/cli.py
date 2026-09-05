"""`numba-metal` command-line entry point.

No console_scripts entry point existed in numba-metal before this
feature (verified: no [project.scripts] in pyproject.toml, no argparse/
click/typer usage at the package level anywhere in src/) -- this module
IS that entry point (`numba-metal = "numba_metal.advisor.cli:main"`,
added to pyproject.toml alongside this file).

Uses argparse (stdlib), matching the only existing CLI precedent in this
repo (benchmarks/run_all.py's own standalone argparse usage) rather than
adding a new CLI-framework dependency.

Every subcommand returns a process exit code; `main()` calls
`sys.exit(...)` with it. Invalid invocations and profiling failures both
return nonzero (spec section 3's explicit requirement) -- this module
never silently swallows an error into a zero exit code.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _no_color(args: argparse.Namespace) -> bool:
    return (
        bool(getattr(args, "no_color", False)) or os.environ.get("NO_COLOR") is not None
    )


def _require_metal(command_name: str) -> tuple[bool, str | None]:
    """Returns (available, error_message). Never raises -- every command
    that needs a live Metal device calls this first and prints the
    spec-mandated fallback message on failure, rather than letting an
    UnsupportedPlatformError traceback reach the user."""
    try:
        from numba_metal.runtime.device import check_capable

        check_capable()
        return True, None
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: ANY
        # reason Metal profiling can't run here (wrong OS, wrong
        # silicon, no toolchain, missing pyobjc) must produce the same
        # clear, spec-mandated message, not a raw traceback specific to
        # numba-metal's internal exception hierarchy.
        message = (
            "Metal profiling requires macOS on Apple Silicon.\n"
            "Available here:\n"
            "  - Static project scan (numba-metal advisor scan)\n"
            "  - Existing profile rendering (numba-metal advisor report)\n"
            "  - CPU-only profiling, if supported\n"
            f"\n({command_name} needs a real Metal device: {exc})"
        )
        return False, message


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--format",
        choices=["terminal", "text", "json"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument(
        "--width", type=int, default=78, help="Terminal width for ASCII output"
    )
    parser.add_argument(
        "--sampling-interval-ms", type=float, default=5.0, help="Stack sampler interval"
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=2, help="Warmup runs before timing"
    )
    parser.add_argument(
        "--measurement-runs", type=int, default=7, help="Timed repetitions"
    )
    parser.add_argument(
        "--include", type=str, default=None, help="Glob pattern to include"
    )
    parser.add_argument(
        "--exclude", type=str, default=None, help="Glob pattern to exclude"
    )
    parser.add_argument(
        "--max-depth", type=int, default=None, help="Max directory depth to scan"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for sampling"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numba-metal",
        description="numba-metal: an out-of-tree Metal GPU backend for a "
        "constrained subset of Numba-style Python kernels.",
    )
    subparsers = parser.add_subparsers(dest="command")

    advisor = subparsers.add_parser(
        "advisor", help="Numba-Metal Advisor: analyze and profile Python/Numba code"
    )
    advisor_sub = advisor.add_subparsers(dest="advisor_command")

    p_scan = advisor_sub.add_parser(
        "scan",
        help="Statically scan a file or directory for GPU-candidate functions "
        "(never executes any scanned code)",
    )
    p_scan.add_argument("path", type=str, help="File or directory to scan")
    _add_common_args(p_scan)

    p_profile = advisor_sub.add_parser(
        "profile",
        help="Run a script or pytest target with profiling instrumentation active",
    )
    p_profile.add_argument("script", type=str, nargs="?", help="Script to profile")
    p_profile.add_argument(
        "--pytest", type=str, default=None, help="Pytest target to profile"
    )
    _add_common_args(p_profile)
    # NOTE: script_args is deliberately NOT an argparse argument (no
    # nargs=REMAINDER) -- REMAINDER greedily swallows every token after
    # the script positional, INCLUDING flags meant for numba-metal
    # itself (--output, --warmup-runs, etc.), silently discarding them.
    # Found directly: `numba-metal advisor compare script.py
    # --measurement-runs 3 --output DIR` produced --output=None and
    # script_args=['--measurement-runs', '3', '--output', 'DIR']. Fixed
    # by splitting sys.argv on a literal "--" BEFORE calling
    # parser.parse_args() at all (see main() below) -- only the tokens
    # after "--" are ever treated as the target script's own arguments,
    # matching the standard git/cargo-style passthrough convention.

    p_compare = advisor_sub.add_parser(
        "compare", help="Run a script under both CPU and Metal, comparing results"
    )
    p_compare.add_argument("script", type=str, help="Script to compare")
    _add_common_args(p_compare)

    p_report = advisor_sub.add_parser(
        "report", help="Render a saved profile.json without re-running anything"
    )
    p_report.add_argument("profile_json", type=str, help="Path to a saved profile.json")
    _add_common_args(p_report)

    p_calibrate = advisor_sub.add_parser(
        "calibrate", help="Measure and store local Metal device calibration data"
    )
    p_calibrate.add_argument(
        "--delete", action="store_true", help="Delete stored calibration data"
    )

    return parser


def cmd_scan(args: argparse.Namespace) -> int:
    import dataclasses

    from numba_metal.advisor.render import render_candidate
    from numba_metal.advisor.scanner import scan

    result = scan(
        args.path, include=args.include, exclude=args.exclude, max_depth=args.max_depth
    )

    if args.format == "json":
        import json

        print(
            json.dumps(
                {
                    "root": result.root,
                    "files_scanned": result.files_scanned,
                    "candidates": [c.to_json_dict() for c in result.candidates],
                    "errors": [
                        {"file": e.file, "message": e.message} for e in result.errors
                    ],
                },
                indent=2,
            )
        )
    else:
        n_cand = len(result.candidates)
        n_err = len(result.errors)
        print(f"Scanned {result.files_scanned} file(s) under {result.root}")
        print(f"Found {n_cand} candidate(s), {n_err} error(s)")
        print()
        root_abs = Path(result.root).resolve()
        if root_abs.is_file():
            root_abs = root_abs.parent
        _NO_SIGNAL = "No strong GPU-candidate signal detected"
        for c in result.candidates:
            if not c.blockers and c.reasons == (_NO_SIGNAL,):
                continue
            # Display path relative to the scanned root when possible --
            # scanner.scan_file always stores absolute paths (needed so
            # `compare`'s single-file candidate lookup and JSON output
            # stay unambiguous), but an absolute path can crowd out the
            # qualified name entirely once _truncate() cuts a long
            # header down to `--width` columns. This is presentation
            # only: c itself (and the JSON branch above) still carries
            # the real absolute path.
            try:
                display_path = str(Path(c.file).resolve().relative_to(root_abs))
            except ValueError:
                display_path = c.file
            display_candidate = dataclasses.replace(c, file=display_path)
            print(render_candidate(display_candidate, width=args.width))
            print()
        for e in result.errors:
            print(f"[SCAN ERROR] {e.file}: {e.message}")

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    ok, message = _require_metal("advisor profile")
    if not ok:
        print(message, file=sys.stderr)
        return 1

    from numba_metal.advisor.export import (
        render_all_text_reports,
        render_summary_text,
        save_report,
    )
    from numba_metal.advisor.profiler import profile_pytest, profile_script

    if args.pytest:
        report = profile_pytest(
            args.pytest,
            args.script_args,
            sampling_interval_ms=args.sampling_interval_ms,
        )
    elif args.script:
        report = profile_script(
            args.script,
            args.script_args,
            sampling_interval_ms=args.sampling_interval_ms,
        )
    else:
        print(
            "numba-metal advisor profile: provide a SCRIPT.py or --pytest TEST_PATH",
            file=sys.stderr,
        )
        return 2

    color = not _no_color(args)
    if args.format == "json":
        import json

        print(json.dumps(report.to_json_dict(), indent=2))
    else:
        print(render_summary_text(report))

    if args.output:
        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        save_report(report, outdir / "profile.json")
        for name, content in render_all_text_reports(
            report, width=args.width, color=color
        ).items():
            (outdir / name).write_text(content)
        print(f"Wrote report to {outdir}/")
    return 0


_COMPARE_CONTRACT_HELP = """numba-metal advisor compare expects the target
script to define an `advisor_workload()` function returning an
AdvisorWorkload (see numba_metal.advisor.workload.AdvisorWorkload) --
this is the documented contract in docs/advisor.md's "Static scan
versus runtime profile" section. Example:

    from numba_metal.advisor.workload import AdvisorWorkload

    def advisor_workload():
        ...set up arrays, compile dispatchers...

        def run_metal_warm():
            kernel[blocks, threads](d_x, d_out)
            metal.synchronize()

        return AdvisorWorkload(
            qualified_name="my_module.my_kernel",
            cpu_fn=lambda: cpu_reference(x),
            metal_warm_fn=run_metal_warm,
            get_cpu_result=lambda: cpu_reference(x),
            get_metal_result=lambda: d_out.copy_to_host(),
            py_func_for_cold=kernel.py_func,
            arg_types_for_cold=arg_types,
        )
"""


def cmd_compare(args: argparse.Namespace) -> int:
    ok, message = _require_metal("advisor compare")
    if not ok:
        print(message, file=sys.stderr)
        return 1

    import runpy

    from numba_metal.advisor.comparison import compare
    from numba_metal.advisor.correctness import compare_results
    from numba_metal.advisor.export import render_all_text_reports, save_report
    from numba_metal.advisor.models import Report, now_ns
    from numba_metal.advisor.profiler import build_device_info
    from numba_metal.advisor.recommendations import generate_recommendations
    from numba_metal.advisor.render import (
        render_comparison_summary,
        render_correctness,
        render_recommendation,
    )
    from numba_metal.advisor.scanner import scan_file

    old_argv = sys.argv
    sys.argv = [args.script, *args.script_args]
    try:
        module_globals = runpy.run_path(
            args.script, run_name="numba_metal_advisor_target"
        )
    finally:
        sys.argv = old_argv

    workload_factory = module_globals.get("advisor_workload")
    if workload_factory is None:
        print(_COMPARE_CONTRACT_HELP, file=sys.stderr)
        return 2

    workload = workload_factory()

    comparison = compare(
        qualified_name=workload.qualified_name,
        file=args.script,
        line_start=1,
        cpu_fn=workload.cpu_fn,
        metal_warm_fn=workload.metal_warm_fn,
        py_func_for_cold=workload.py_func_for_cold,
        arg_types_for_cold=workload.arg_types_for_cold,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
    )

    correctness = None
    if workload.get_cpu_result is not None and workload.get_metal_result is not None:
        cpu_result = workload.get_cpu_result()
        metal_result = workload.get_metal_result()
        correctness = compare_results(cpu_result, metal_result)

    from numba_metal.advisor.models import Candidate, Confidence

    file_candidates, _err = scan_file(Path(args.script))
    candidate = next(
        (c for c in file_candidates if c.qualified_name == workload.qualified_name),
        Candidate(
            file=args.script,
            line_start=1,
            line_end=1,
            qualified_name=workload.qualified_name,
            decorator=None,
            reasons=(),
            unknowns=(),
            blockers=(),
            parallel_dimension=None,
            inferred_dtypes=(),
            confidence=Confidence.MEDIUM,
            is_numba_decorated=False,
            is_metal_decorated=True,
        ),
    )
    recommendations = generate_recommendations(
        candidate, comparison=comparison, correctness=correctness
    )

    print(render_comparison_summary(comparison))
    print()
    if correctness is not None:
        print(render_correctness(correctness))
        print()
    for rec in recommendations:
        print(render_recommendation(rec))
        print()

    if args.output:
        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        report = Report(
            schema_version=1,
            generated_at_ns=now_ns(),
            device=build_device_info(),
            events=(),
            candidates=(candidate,),
            compatibility=(),
            comparisons=(comparison,),
            correctness=(correctness,) if correctness is not None else (),
            scores=(),
            recommendations=tuple(recommendations),
            sampling=None,
            source_paths_included=True,
        )
        save_report(report, outdir / "profile.json")
        for name, content in render_all_text_reports(
            report, width=args.width, color=not _no_color(args)
        ).items():
            (outdir / name).write_text(content)
        print(f"Wrote report to {outdir}/")

    if correctness is not None and not correctness.passed:
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from numba_metal.advisor.export import (
        load_report,
        render_all_text_reports,
        render_summary_text,
    )

    try:
        report = load_report(args.profile_json)
    except (ValueError, OSError) as exc:
        print(f"Could not load {args.profile_json}: {exc}", file=sys.stderr)
        return 1

    color = not _no_color(args)
    if args.format == "json":
        import json

        print(json.dumps(report.to_json_dict(), indent=2))
    else:
        print(render_summary_text(report))
        for name, content in render_all_text_reports(
            report, width=args.width, color=color
        ).items():
            print(f"--- {name} ---")
            print(content)

    if args.output:
        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, content in render_all_text_reports(
            report, width=args.width, color=color
        ).items():
            (outdir / name).write_text(content)
        print(f"Wrote report to {outdir}/")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from numba_metal.advisor.calibration import (
        delete_calibration,
        run_calibration,
        save_calibration,
    )

    if args.delete:
        deleted = delete_calibration()
        print("Calibration data deleted." if deleted else "No calibration data found.")
        return 0

    ok, message = _require_metal("advisor calibrate")
    if not ok:
        print(message, file=sys.stderr)
        return 1

    print("Running calibration against the local Metal device...")
    result = run_calibration()
    path = save_calibration(result)
    print(f"Device:                 {result.device_name}")
    print(f"Dispatch overhead:      {result.dispatch_overhead_ns/1000:.1f} us")
    print(f"Cold compile:           {result.cold_compile_ns/1e6:.2f} ms")
    print(f"Buffer allocation:      {result.buffer_alloc_ns/1000:.1f} us")
    print(f"Synchronization:        {result.sync_overhead_ns/1000:.2f} us")
    if result.float32_gflops is not None:
        print(f"float32 throughput:     {result.float32_gflops:.1f} GFLOPS")
    if result.memory_bandwidth_gbps is not None:
        print(f"Memory bandwidth:       {result.memory_bandwidth_gbps:.1f} GB/s")
    print()
    print(f"Saved to {path}")
    print(
        "Note: calibration results are hints, not promises -- "
        "project-specific measurements from `compare`/`profile` always "
        "take precedence."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Split on a literal "--" BEFORE argparse ever sees the tokens after
    # it -- see build_parser()'s comment on p_profile/p_compare for why
    # this is required (argparse.REMAINDER would swallow numba-metal's
    # own flags too, not just the target script's).
    script_args: list[str] = []
    if "--" in argv:
        sep = argv.index("--")
        script_args = argv[sep + 1 :]
        argv = argv[:sep]

    parser = build_parser()
    args = parser.parse_args(argv)
    args.script_args = script_args

    if args.command != "advisor" or not getattr(args, "advisor_command", None):
        parser.print_help()
        return 2

    handlers = {
        "scan": cmd_scan,
        "profile": cmd_profile,
        "compare": cmd_compare,
        "report": cmd_report,
        "calibrate": cmd_calibrate,
    }
    handler = handlers.get(args.advisor_command)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
