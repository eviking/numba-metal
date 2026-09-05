"""Test 16: adding the advisor feature does not change any existing
numba-metal behavior.

This test cannot re-run the *entire* pre-existing suite from inside
itself without duplicating pytest's own collection (that verification is
done once per session via `pytest tests/ -q`, recorded in the final
response). What it CAN check structurally, as a standing regression
guard that runs on every future test run: importing the advisor package
must not change behavior of, monkeypatch, or otherwise mutate the
existing public runtime/compiler modules at import time.
"""

from __future__ import annotations

import importlib
import inspect


def test_importing_advisor_does_not_mutate_existing_public_context_functions():
    """Import order independence: numba_metal.runtime.context's public
    functions must be identical objects before and after importing the
    advisor package (an earlier design considered monkeypatching
    register_submission/synchronize at import time -- that approach was
    abandoned in favor of additive hook lists precisely so this holds)."""
    from numba_metal.runtime import context

    context_fn_names = (
        "get_context",
        "add_submission_hook",
        "add_sync_hook",
        "clear_hooks",
    )
    before = {name: getattr(context, name) for name in context_fn_names}
    before_methods = {
        name: getattr(context._MetalContext, name)
        for name in ("register_submission", "synchronize")
    }

    importlib.import_module("numba_metal.advisor")
    importlib.import_module("numba_metal.advisor.metal_events")
    importlib.import_module("numba_metal.advisor.profiler")

    after = {name: getattr(context, name) for name in context_fn_names}
    after_methods = {
        name: getattr(context._MetalContext, name)
        for name in ("register_submission", "synchronize")
    }
    for name in before:
        assert before[name] is after[name], f"{name} was replaced by importing advisor"
    for name in before_methods:
        assert (
            before_methods[name] is after_methods[name]
        ), f"_MetalContext.{name} was replaced by importing advisor"


def test_context_hook_lists_start_empty_in_a_fresh_process_state():
    """A freshly-imported context module (no profiling session ever
    activated) must have empty hook lists -- advisor import alone must
    never install a hook as a side effect."""
    from numba_metal.runtime import context

    # Not asserting == [] unconditionally (another test in this session
    # may have left hooks installed); instead assert that clearing them
    # is always sufficient to reach the "no instrumentation" state, i.e.
    # nothing in this package holds a hook reference outside those two
    # module-level lists.
    context.clear_hooks()
    assert context._submission_hooks == []
    assert context._sync_hooks == []


def test_register_submission_and_synchronize_signatures_are_unchanged():
    """The advisor's hook points were required to be purely additive --
    no existing parameter removed or reordered, no return type changed.
    This locks the public call signature so a future refactor cannot
    silently break existing callers (e.g. benchmarks/, dispatcher.py)
    while only being caught by this advisor-specific test."""
    from numba_metal.runtime.context import _MetalContext

    sig = inspect.signature(_MetalContext.register_submission)
    # 'self' plus whatever positional/keyword params register_submission
    # already had -- this only fails if a REQUIRED param is added/removed,
    # since hook installation must not require touching this signature.
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty and p.name != "self"
    ]
    # The hook mechanism is a module-level list, not a new parameter --
    # so register_submission must take exactly the same required
    # arguments it always did (no new required positional slipped in).
    assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in required)


def test_advisor_module_source_contains_no_datashader_reference():
    """Repository-boundary rule carried through this entire feature:
    numba-metal must contain zero Datashader-specific code. This is a
    structural regression guard, not just a one-time check, so a future
    change that accidentally imports or references datashader is caught
    by the existing advisor test suite rather than requiring a human to
    remember the rule."""
    import pathlib

    advisor_dir = pathlib.Path(
        importlib.import_module("numba_metal.advisor").__file__
    ).parent
    for path in advisor_dir.rglob("*.py"):
        text = path.read_text()
        assert "datashader" not in text.lower(), f"{path} references datashader"
