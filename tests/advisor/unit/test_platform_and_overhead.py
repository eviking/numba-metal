"""Test 14 (unsupported platform behavior) and Test 15 (profiling
disabled has no meaningful overhead) from the governing spec.

Neither test requires real Metal hardware: Test 14 checks the fallback
message contract and that scan/report never gate on device availability;
Test 15 checks the hook lists are empty by default and that the
guard-checked call sites in runtime/context.py are structured to skip
all hook-related work when no hooks are installed. A hardware-backed
before/after timing comparison additionally lives in
tests/advisor/integration/ under @pytest.mark.metal.
"""

from __future__ import annotations

import ast
import inspect

from numba_metal.advisor.cli import _require_metal, build_parser, cmd_report, cmd_scan
from numba_metal.runtime import context


def test_require_metal_never_raises_and_carries_the_spec_message():
    """Test 14: on any failure reason (wrong OS, wrong silicon, missing
    toolchain), _require_metal must return the exact spec-mandated
    fallback text, never let an exception escape."""
    ok, message = _require_metal("advisor profile")
    if ok:
        # This machine IS Metal-capable -- nothing to assert about the
        # fallback message itself, but the call must still not raise.
        assert message is None
        return
    assert message is not None
    assert "Metal profiling requires macOS on Apple Silicon." in message
    assert "Static project scan" in message
    assert "Existing profile rendering" in message
    assert "CPU-only profiling" in message


def test_require_metal_swallows_arbitrary_exceptions():
    """however check_capable() fails (ImportError, RuntimeError, a
    platform-specific exception type we've never seen), the CLI must
    still produce the same clean fallback message and exit path, not a
    raw traceback."""
    import numba_metal.advisor.cli as cli_module

    original = cli_module._require_metal
    try:

        def _boom():
            raise RuntimeError("simulated: no Metal toolchain on this host")

        # Patch at the call site used by _require_metal internally by
        # simulating check_capable failing -- verified via direct call
        # with a monkeypatched import target.
        import numba_metal.runtime.device as device_module

        real_check_capable = device_module.check_capable
        device_module.check_capable = _boom
        try:
            ok, message = cli_module._require_metal("advisor calibrate")
        finally:
            device_module.check_capable = real_check_capable

        assert ok is False
        assert "Metal profiling requires macOS on Apple Silicon." in message
        assert "simulated: no Metal toolchain" in message
    finally:
        cli_module._require_metal = original


def test_scan_and_report_never_call_require_metal():
    """Structural guard: cmd_scan and cmd_report must not reference
    _require_metal at all in their source, so static scanning and saved-
    report rendering work on any platform unconditionally."""
    assert "_require_metal" not in inspect.getsource(cmd_scan)
    assert "_require_metal" not in inspect.getsource(cmd_report)


def test_scan_subcommand_has_no_metal_capability_gate_in_parser_path():
    """The scan subcommand's argparse wiring must not require any Metal-
    specific argument that would imply a device check."""
    parser = build_parser()
    args = parser.parse_args(["advisor", "scan", "."])
    assert args.path == "."


def test_hooks_are_empty_by_default():
    """Test 15 (structural half): with no profiling session active, both
    hook lists are empty -- the guarded call sites in context.py
    (`if _submission_hooks:` / `if _sync_hooks and pending:`) therefore
    do zero extra work on the normal, non-profiled code path."""
    context.clear_hooks()
    assert context._submission_hooks == []
    assert context._sync_hooks == []


def test_hook_call_sites_are_guarded_not_unconditional():
    """Structural guard: register_submission and synchronize must check
    the hook lists' truthiness before doing any hook-related work, so an
    empty list short-circuits before any per-hook overhead (attribute
    lookups, function calls) is paid."""
    source = inspect.getsource(context)
    tree = ast.parse(source)

    found_submission_guard = False
    found_sync_guard = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_src = ast.dump(node.test)
            if "_submission_hooks" in test_src:
                found_submission_guard = True
            if "_sync_hooks" in test_src:
                found_sync_guard = True
    assert found_submission_guard, "register_submission must guard on _submission_hooks"
    assert found_sync_guard, "synchronize must guard on _sync_hooks"


def test_clear_hooks_removes_all_installed_hooks():
    calls = []
    context.add_submission_hook(lambda record: calls.append(("submit", record)))
    context.add_sync_hook(lambda *a: calls.append(("sync", a)))
    assert len(context._submission_hooks) == 1
    assert len(context._sync_hooks) == 1
    context.clear_hooks()
    assert context._submission_hooks == []
    assert context._sync_hooks == []


def test_collector_reports_its_own_dropped_event_count():
    """Bounded event limits (spec section 18) must report exactly how
    many events were dropped, never silently discard them. Uses a plain
    fake record (kernel_name/sequence/resources), not a real Metal
    SubmissionRecord -- this only exercises EventCollector's own
    bounded-buffer bookkeeping, no hardware needed."""
    from numba_metal.advisor import metal_events

    collector = metal_events.EventCollector(max_events=2)
    for i in range(5):
        collector._on_submission(
            type(
                "FakeRecord",
                (),
                {"kernel_name": f"k{i}", "sequence": i, "resources": ()},
            )()
        )
    assert len(collector.events()) <= 2
    assert collector.dropped_count() == 3
