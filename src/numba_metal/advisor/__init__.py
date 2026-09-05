"""Numba-Metal Advisor: a terminal-based analyzer and profiler for deciding
which Python/Numba functions are worth running on numba-metal's Metal
backend, whether they are actually supported today, and whether Metal is
measurably faster on the developer's own Apple Silicon machine.

This package is intentionally separate from numba-metal's compiler and
runtime: it never modifies compilation/dispatch behavior when profiling is
disabled, and it never claims a performance result it did not measure. See
docs/advisor.md for the full user-facing documentation, and
docs/architecture.md for how this package's few, additive instrumentation
hooks fit into the existing compiler/runtime pipeline.

Pipeline (collection -> analysis -> presentation, kept strictly separate):

    scanner.py / compatibility.py / profiler.py / sampling.py
        -> normalized profile model (models.py)
        -> comparison.py / correctness.py / scoring.py / recommendations.py
        -> render.py / flamegraph.py / timeline.py (ASCII only, no measurement logic)

Nothing in this package sends network requests, uploads source code or
telemetry, rewrites user source, or executes LLM-generated code.
"""

from __future__ import annotations
