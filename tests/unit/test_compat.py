"""Unit tests for the Numba version compatibility gate
(numba_metal.compat). No Metal device required -- these test pure
version-string logic."""

from __future__ import annotations

import pytest

from numba_metal.compat import (
    UnsupportedNumbaVersionError,
    _parse_version,
    check_numba_compatible,
)


def test_parse_version_simple() -> None:
    assert _parse_version("0.67.0") == (0, 67, 0)


def test_parse_version_with_suffix() -> None:
    # e.g. a dev/rc build string; only the leading digits of each
    # dot-separated component are used.
    assert _parse_version("0.67.1rc1") == (0, 67, 1)


def test_parse_version_missing_patch() -> None:
    assert _parse_version("0.67") == (0, 67, 0)


def test_exact_supported_version_passes(monkeypatch) -> None:
    import numba

    monkeypatch.setattr(numba, "__version__", "0.67.0")
    assert check_numba_compatible() == "0.67.0"


def test_untested_patch_in_supported_series_passes(monkeypatch) -> None:
    """0.67.x for x != 0 hasn't been explicitly tested but is within the
    dependency-bound series (numba>=0.67,<0.68); patch releases don't
    change the typed-IR shape numba-metal depends on, so this is
    accepted rather than rejected."""
    import numba

    monkeypatch.setattr(numba, "__version__", "0.67.3")
    assert check_numba_compatible() == "0.67.3"


def test_older_major_minor_rejected(monkeypatch) -> None:
    import numba

    monkeypatch.setattr(numba, "__version__", "0.59.0")
    with pytest.raises(UnsupportedNumbaVersionError) as exc_info:
        check_numba_compatible()
    message = str(exc_info.value)
    assert "0.59.0" in message
    assert "numba>=0.67,<0.68" in message


def test_newer_major_minor_rejected(monkeypatch) -> None:
    import numba

    monkeypatch.setattr(numba, "__version__", "0.68.0")
    with pytest.raises(UnsupportedNumbaVersionError) as exc_info:
        check_numba_compatible()
    assert "0.68.0" in str(exc_info.value)


def test_error_message_names_the_internal_api_risk(monkeypatch) -> None:
    """The error must explain *why*, not just *that*, an unvalidated
    version is rejected -- pointing at the internal-API dependency, not
    a generic version mismatch message."""
    import numba

    monkeypatch.setattr(numba, "__version__", "1.0.0")
    with pytest.raises(UnsupportedNumbaVersionError) as exc_info:
        check_numba_compatible()
    message = str(exc_info.value)
    assert "typed IR" in message or "internal" in message
