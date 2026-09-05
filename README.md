# numba-metal

An out-of-tree GPU backend that lets you write a constrained subset of
Numba-style Python kernels and run them on the GPU of Apple-silicon Macs
through Metal.

```python
import numpy as np
from numba_metal import metal

@metal.jit
def vector_add(a, b, output):
    i = metal.grid(1)
    if i < output.size:
        output[i] = a[i] + b[i]

a = np.arange(1_000_000, dtype=np.float32)
b = np.arange(1_000_000, dtype=np.float32)
d_a = metal.to_device(a)
d_b = metal.to_device(b)
d_output = metal.device_array_like(a)

threads = 256
blocks = (a.size + threads - 1) // threads
vector_add[blocks, threads](d_a, d_b, d_output)
metal.synchronize()

output = d_output.copy_to_host()
```

This is a real compiler backend, not a wrapper: kernels are compiled from
Numba's typed intermediate representation into Metal Shading Language
(MSL), compiled by Apple's Metal shader compiler, and executed by the GPU.
See `docs/architecture.md` for how the pipeline works.

## Status: alpha / MVP

This project is an early, deliberately narrow-scope MVP. It supports one
dialect of one language subset, on one platform, for the operations
documented in `docs/supported-features.md`. It is not a drop-in
replacement for `numba.cuda`, and it does not aim for broad Python or
NumPy compatibility. Read `docs/limitations.md` before relying on it for
anything beyond experimentation.

## Supported platform

- Apple-silicon Macs (arm64) only -- Intel Macs are not supported.
- macOS 14 or newer.
- Python: see `docs/installation.md` for the exact supported range.
- Numba: pinned to a bounded, tested version range (`pyproject.toml`).
- Requires Xcode command-line tools and Apple's Metal Toolchain component
  (a one-time download; see `docs/installation.md`).

On any other platform or OS version, `numba-metal` fails immediately with
a clear error -- it never silently falls back to the CPU.

## Installation

See `docs/installation.md` for full instructions, including creating a
virtual environment, installing the Metal toolchain, and verifying your
machine is capable before writing any kernels.

## What's supported

See `docs/supported-features.md` for the full compatibility matrix
(Python syntax, scalar types, operators, math functions, memory
operations, launch dimensions). In short: scalar arithmetic/comparison/
boolean operators, `if`/`if-else`, `for x in range(...)` (with `break`/
`continue`), 1D and 2D grids, 1D arrays with flattened multidimensional
indexing, and a documented subset of `abs`/`min`/`max`/`math.sqrt`/
`math.exp`/`math.log`/`math.sin`/`math.cos`, over `float32`/`int32`/
`uint32`/`bool` (with `float16`/`int64` also supported).

## Benchmarks

Five benchmark programs (`benchmarks/*.py`) compare plain Python, NumPy,
Numba CPU (`@njit`), and numba-metal, at multiple problem sizes, with
correctness checks and cold/warm/transfer-inclusive timing:

```bash
python benchmarks/run_all.py            # text report
python benchmarks/run_all.py --json out.json   # + machine-readable JSON
python benchmarks/run_all.py --quick    # smaller sizes, for a fast check
```

See `docs/benchmarking.md` for methodology and how to interpret results.
No speedup numbers are hard-coded anywhere in this repository; every
number reported by these scripts is measured on the machine you run them
on.

## Find the Python worth putting on the Metal

The `numba-metal advisor` CLI answers "which of my functions would
actually benefit from this?" -- as a terminal-only static scanner and
profiler, never a browser or notebook UI.

```bash
numba-metal advisor scan .                           # find candidates, runs nothing
numba-metal advisor compare my_workload.py            # measure CPU vs. Metal, with correctness checks
```

```
[HIGH POTENTIAL] mandelbrot.py:97 _make_metal_kernel.<locals>.metal_kernel
Why:
  - Already running on numba-metal (@metal.jit)

Recommendation (USE_METAL): measured 15.38x faster (steady_state), and
results match the CPU reference within tolerance.
```

Static analysis never claims a speedup by itself -- only `compare`,
which actually runs your code and checks correctness first, can say
that. See `docs/advisor.md` for the full command reference, how
compatibility and opportunity scoring work, and how to read the ASCII
flame graphs and CPU/GPU timeline.

## Safety and correctness caveats

- **No silent fallback.** If a kernel uses an unsupported Python
  construct, operator, or type, compilation raises a specific
  `UnsupportedFeatureError` naming the offending code. numba-metal never
  quietly executes your kernel through NumPy, Numba's CPU target, a
  Python loop, or any other substitute.
- **float64 is not supported for kernel arguments or array dtypes.**
  Local float64-typed intermediate values (e.g. from Python float
  literals or true division) are narrowed to float32 for MSL codegen;
  see `docs/limitations.md` for exactly what this means numerically.
- Metal's default fast-math shader compilation mode is disabled by
  numba-metal to keep floating-point results closer to standard
  (non-fused) semantics; see `docs/architecture.md`.
- This is alpha software. Read `docs/limitations.md` before using it for
  anything where numerical correctness matters beyond experimentation.

## Contributing

See `CONTRIBUTING.md`.

## License

BSD 2-Clause. See `LICENSE`.
