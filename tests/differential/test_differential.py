"""Hypothesis-driven differential test suite (Workstream 4): generates
kernels from the closed grammar in grammar.py/kernel_strategy.py and
compares real Metal GPU execution against real Numba CPU (@njit)
execution.

Two profiles:

- `quick` (default, always runs in a normal test session): a small
  number of examples, suitable for every local `pytest` run.
- `exhaustive` (opt-in via `NUMBA_METAL_DIFFERENTIAL_EXHAUSTIVE=1` or
  `pytest -m differential_exhaustive`): several hundred generated
  kernel/data combinations, intended for a full verification pass on
  Apple-silicon hardware. Writes a machine-readable summary to
  artifacts/differential/.

On any failure, the assertion message includes the full diagnostic
report (Hypothesis example values are shown separately by Hypothesis
itself in the standard falsifying-example block): generated Python
kernel source, generated CPU reference source, inputs, expected vs.
actual output, and -- when the failure was a GPU compilation/runtime
error -- the generated MSL.

Requires a working Metal device; run with `pytest -m metal`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings

from .harness import run_differential
from .kernel_strategy import generated_kernel

pytestmark = pytest.mark.metal

_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "artifacts" / "differential"
)

_EXHAUSTIVE_ENABLED = os.environ.get("NUMBA_METAL_DIFFERENTIAL_EXHAUSTIVE") == "1"

settings.register_profile(
    "quick",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile(
    "exhaustive",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _run_and_assert(gk) -> None:
    result = run_differential(
        gk.spec, gk.gpu_source, gk.cpu_source, n_elements=48, seed=7
    )
    assert result.ok, result.format_report()


@settings.get_profile("quick")
@given(generated_kernel(n_arrays=2, max_stmts=3))
def test_quick_differential(gk) -> None:
    """Always-on profile: a small number of generated kernel cases,
    fast enough for every local test run."""
    _run_and_assert(gk)


@pytest.mark.skipif(
    not _EXHAUSTIVE_ENABLED,
    reason=(
        "Exhaustive differential profile is opt-in: set "
        "NUMBA_METAL_DIFFERENTIAL_EXHAUSTIVE=1 to run 500 generated cases "
        "(takes several minutes). Use the quick profile "
        "(test_quick_differential) for routine runs."
    ),
)
def test_exhaustive_differential() -> None:
    """Opt-in profile: 500 generated kernel/data combinations (per the
    assignment's minimum), using only Hypothesis's public API
    (`@given`/`settings`). Individual case failures are caught and
    recorded rather than raised immediately, so a full 500-case run
    continues past them and reports a complete summary -- the overall
    test only fails (via `pytest.fail`, once, after every case has run)
    if at least one case failed, with the full per-case results written
    to artifacts/differential/ as a machine-readable summary regardless
    of outcome."""
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    n_cases = 500

    results: list[dict] = []
    case_counter = {"n": 0}
    start = time.perf_counter()

    @settings(
        max_examples=n_cases,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        report_multiple_bugs=False,
    )
    @given(generated_kernel(n_arrays=2, max_stmts=3))
    def _run_one(gk) -> None:
        case_idx = case_counter["n"]
        case_counter["n"] += 1
        result = run_differential(
            gk.spec, gk.gpu_source, gk.cpu_source, n_elements=48, seed=case_idx
        )
        if result.ok:
            results.append({"case": case_idx, "status": "pass"})
        else:
            results.append(
                {
                    "case": case_idx,
                    "status": "fail",
                    "error_message": result.error_message,
                    "max_abs_error": result.max_abs_error,
                    "gpu_source": result.gpu_source,
                    "cpu_source": result.cpu_source,
                    "cpu_reference_report": result.format_report(),
                }
            )

    _run_one()  # never raises: every case is caught and recorded above

    elapsed = time.perf_counter() - start
    n_passed = sum(1 for r in results if r["status"] == "pass")
    n_failed = sum(1 for r in results if r["status"] == "fail")
    summary = {
        "n_cases": len(results),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    with open(_ARTIFACT_DIR / "exhaustive_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if n_failed:
        first_failure = next(r for r in results if r["status"] == "fail")
        pytest.fail(
            f"{n_failed}/{len(results)} exhaustive differential cases failed "
            f"(first failure at case {first_failure['case']}); full summary "
            f"written to {_ARTIFACT_DIR / 'exhaustive_run_summary.json'}.\n"
            + first_failure["cpu_reference_report"]
        )
