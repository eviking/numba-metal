# Changelog

## 0.1.0.dev0 (unreleased)

Initial MVP.

### Added

- `@metal.jit` kernel decorator and `kernel[blocks, threads](*args)`
  launch syntax (1D and 2D grids).
- `metal.grid(ndim)` / `metal.gridsize(ndim)` intrinsics.
- Device memory API: `metal.to_device`, `metal.device_array`,
  `metal.device_array_like`, `DeviceNDArray.copy_to_host`/`copy_to_device`.
- `metal.synchronize()`.
- Typed-Numba-IR-to-MSL compiler backend supporting: scalar arithmetic,
  comparison and boolean operators, assignment, 1D array reads/writes,
  flattened multidimensional indexing, `if`/`if-else`, `for x in
  range(...)` with `break`/`continue`, `abs`/`min`/`max`, and
  `math.sqrt`/`exp`/`log`/`sin`/`cos`.
- Support for `float32`, `int32`, `uint32`, `bool` (required) and
  `float16`, `int64` (optional) dtypes.
- In-process compilation cache keyed by kernel source, argument
  signature, and device identity.
- `NUMBA_METAL_DUMP_MSL` / `metal.config.dump_msl` debug MSL dumping.
- Explicit platform/device/toolchain capability checks
  (`UnsupportedPlatformError`, `MetalToolchainError`) that fail fast on
  unsupported hardware.
- Five benchmark programs (vector polynomial, Mandelbrot, heat diffusion,
  Monte Carlo paths, pairwise distance) comparing Python/NumPy/Numba
  CPU/Metal with correctness checks and cold/warm/transfer timing
  breakdown, plus `benchmarks/run_all.py` text + JSON reporting.
- Unit tests (compiler/frontend/codegen, no GPU required) and integration
  tests (`pytest -m metal`, real GPU execution), all passing on an Apple
  M4 Pro during development.
- Full documentation set: installation, quickstart, supported-features
  matrix, architecture (with Mermaid diagram), benchmarking methodology,
  limitations, troubleshooting, and a prioritized roadmap.

### Known limitations

See `docs/limitations.md`. Notably: no `while` loops, no device
functions/recursion, no float64 array support, no zero-copy host<->device
transfer yet, and only one Numba version (0.67.0) has been exercised
end-to-end against real hardware so far.
