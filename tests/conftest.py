"""Shared pytest configuration: the `metal` marker and a session-scoped
availability check used to skip GPU-dependent tests with a precise reason
on non-Apple-silicon / non-Metal environments.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "metal: requires a working Apple-silicon Metal device"
    )


def _metal_unavailable_reason() -> str | None:
    try:
        from numba_metal.runtime.device import check_capable
    except ImportError as exc:
        return f"numba_metal could not be imported: {exc}"
    try:
        check_capable()
    except Exception as exc:  # noqa: BLE001 - reporting only, re-raised nowhere
        return f"Metal is not available on this machine: {exc}"
    return None


@pytest.fixture(scope="session")
def metal_unavailable_reason() -> str | None:
    return _metal_unavailable_reason()


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    reason = _metal_unavailable_reason()
    if reason is None:
        return
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if "metal" in item.keywords:
            item.add_marker(skip_marker)
