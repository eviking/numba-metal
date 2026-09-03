"""Numba version compatibility gate.

numba-metal's frontend (`numba_metal.compiler.frontend`) and MSL backend
(`numba_metal.compiler.msl_backend`) read Numba's typed intermediate
representation directly -- internal/semi-private APIs (`ir.Expr.phi`'s
`incoming_values`/`incoming_blocks`, `numba.core.types` internals, the
typed-IR block/CFG structure produced by `numba.core.compiler`) that are
not part of Numba's stable public API and have changed shape across
releases in the past (see docs/numba-rfc.md for the specific APIs relied
upon and why). Running against an untested Numba version is not a
"probably fine" situation: a shape change in any of these internals can
silently produce wrong MSL rather than a clean failure, which conflicts
directly with this project's correctness-first design.

This module is the single point where that risk is converted into an
explicit, actionable error instead of an unpredictable one -- narrower
than the packaging dependency bound (`numba>=0.67,<0.68` in
pyproject.toml) so it still fires correctly even when numba-metal is
installed without dependency resolution enforcing that bound (e.g.
`pip install --no-deps`, an existing environment with a newer Numba
upgraded in place, or a lockfile pinning something outside the range).
"""

from __future__ import annotations

from numba_metal.errors import NumbaMetalError

# Only the exact Numba release(s) actually exercised by this project's
# test suite (unit, integration, and differential) are listed as
# supported. See docs/numba-rfc.md's compatibility table for how this
# maps to "unit tested" vs. "Metal tested."
_SUPPORTED_NUMBA_VERSIONS = frozenset({"0.67.0"})

# The (major, minor) series covered by pyproject.toml's dependency bound
# (numba>=0.67,<0.68) -- used to phrase the error message accurately even
# if a future patch release (0.67.1, 0.67.2, ...) hasn't been tested yet
# but falls within the intended series.
_SUPPORTED_SERIES = (0, 67)


class UnsupportedNumbaVersionError(NumbaMetalError):
    """Raised when the installed Numba version has not been validated
    against numba-metal's internal-API dependencies."""


def _parse_version(version_string: str) -> tuple[int, int, int]:
    parts = version_string.split(".")[:3]
    numeric: list[int] = []
    for part in parts:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        numeric.append(int(digits) if digits else 0)
    while len(numeric) < 3:
        numeric.append(0)
    return numeric[0], numeric[1], numeric[2]


def check_numba_compatible() -> str:
    """Raise UnsupportedNumbaVersionError with an actionable message if
    the installed Numba version is not one numba-metal's internal-API
    dependencies have been validated against; otherwise return the
    installed version string.
    """
    import numba

    installed = numba.__version__
    if installed in _SUPPORTED_NUMBA_VERSIONS:
        return installed

    major, minor, _patch = _parse_version(installed)
    if (major, minor) == _SUPPORTED_SERIES:
        # An untested patch release within the supported series (e.g.
        # 0.67.1) -- allowed, since Numba patch releases do not change
        # the typed-IR/CFG shape numba-metal depends on, but this is
        # still worth being able to see distinctly from a fully-tested
        # version if a bug report comes in against one.
        return installed

    raise UnsupportedNumbaVersionError(
        f"numba-metal has only been validated against Numba "
        f"{sorted(_SUPPORTED_NUMBA_VERSIONS)}; found Numba {installed!r} "
        f"installed. numba-metal's frontend reads Numba's typed IR "
        f"directly, including internal/semi-private structures (see "
        f"docs/numba-rfc.md) that can change shape across releases "
        f"without a deprecation period -- running against an "
        f"unvalidated version risks silently incorrect GPU results "
        f"rather than a clean failure. Install a supported Numba "
        f"version (`pip install 'numba>=0.67,<0.68'`), or -- if you "
        f"have validated numba-metal's test suite against your Numba "
        f"version yourself -- see docs/numba-rfc.md for how to extend "
        f"this compatibility gate."
    )
