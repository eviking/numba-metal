"""Differential/property-based testing infrastructure (Workstream 4):
generates small kernels from a closed grammar (grammar.py,
kernel_strategy.py) and compares real Metal GPU execution against real
Numba CPU (@njit) execution -- see harness.py for the comparison/failure
-reporting logic and test_differential.py for the pytest entry points
(quick/exhaustive profiles).
"""
