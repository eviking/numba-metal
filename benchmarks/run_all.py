"""Run all five numba-metal benchmarks and report results.

Usage:
    python benchmarks/run_all.py                 # text table
    python benchmarks/run_all.py --json out.json # also write machine-readable JSON
    python benchmarks/run_all.py --quick         # smaller problem sizes, for CI/dev

Never hard-codes expected speedups; every number printed is measured on
this run, on this machine. See docs/benchmarking.md for methodology.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import heat_diffusion
import mandelbrot
import monte_carlo_paths
import pairwise_distance
import vector_polynomial
from common import BenchmarkResult, get_environment_info, print_table, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=str, default=None, help="Write JSON results to this path"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use smaller problem sizes for a fast run"
    )
    args = parser.parse_args()

    all_results: list[BenchmarkResult] = []

    env = get_environment_info()
    print("numba-metal benchmark suite")
    print(f"  Mac: {env.get('model_name')} ({env.get('chip')})")
    print(
        f"  macOS {env.get('macos_version')}, Python {env.get('python_version')}, "
        f"Numba {env.get('numba_version')}, "
        f"numba-metal {env.get('numba_metal_version')}"
    )
    gpu = env.get("gpu", {})
    if "error" in gpu:
        print(f"  GPU: unavailable ({gpu['error']})")
    else:
        print(f"  GPU: {gpu.get('name')}")
    print()

    benchmarks = [
        (
            "Vector polynomial",
            vector_polynomial,
            dict(sizes=[10_000, 200_000] if args.quick else None),
        ),
        ("Mandelbrot", mandelbrot, dict(sizes=[256, 512] if args.quick else None)),
        (
            "Heat diffusion",
            heat_diffusion,
            dict(sizes=[64, 128] if args.quick else None),
        ),
        (
            "Monte Carlo paths",
            monte_carlo_paths,
            dict(sizes=[5_000, 50_000] if args.quick else None),
        ),
        (
            "Pairwise distance",
            pairwise_distance,
            dict(sizes=[(100, 100, 8), (500, 500, 8)] if args.quick else None),
        ),
    ]

    for name, module, kwargs in benchmarks:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        print(f"--- {name} ---")
        start = time.perf_counter()
        try:
            results = module.run(**kwargs)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - report and continue with other benchmarks
            print(f"  FAILED: {exc}")
            continue
        elapsed = time.perf_counter() - start
        print(f"  ({elapsed:.1f}s)")
        all_results.extend(results)
        print()

    print("=" * 100)
    print_table(all_results)

    n_correct = sum(1 for r in all_results if r.correctness_ok)
    print()
    print(f"Correctness: {n_correct}/{len(all_results)} results passed.")
    if n_correct < len(all_results):
        print("Failures:")
        for r in all_results:
            if not r.correctness_ok:
                print(f"  - {r.benchmark} {r.size_label}: {r.correctness_note}")

    if args.json:
        write_json(all_results, args.json)
        print(f"\nWrote JSON results to {args.json}")

    return 0 if n_correct == len(all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
