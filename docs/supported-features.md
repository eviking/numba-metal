# Supported features

Compatibility matrix for the current MVP. Status values:

- **Supported** -- implemented, tested, and used by at least one of the
  five benchmark kernels or the test suite.
- **Partially supported** -- works with a documented restriction or
  caveat.
- **Unsupported** -- explicitly rejected at compile time with a specific
  error naming the construct.
- **Planned** -- not implemented; see `docs/roadmap.md`.

Nothing in this document describes behavior that hasn't actually been
exercised by the test suite in this repository.

## Python syntax

| Feature | Status | Notes |
|---|---|---|
| Scalar local variables | Supported | |
| Kernel arguments (arrays, scalars) | Supported | |
| Assignment | Supported | |
| Arithmetic operators `+ - * / // % **` | Supported | `**` lowers to `pow()`; `//` on two integers uses MSL's native truncating division (differs from Python's floor semantics for negative operands -- see `docs/limitations.md`) |
| Comparison operators `< <= > >= == !=` | Supported | |
| Boolean operators `and or not` | Supported | Short-circuit evaluation, as compiled by Numba's own frontend |
| Ternary expressions (`a if cond else b`) | Supported | Compiles through the same phi-elimination path as `if/else` |
| `if` / `if-else` | Supported | Arbitrarily nested |
| `for x in range(...)` | Supported | 1, 2, or 3-argument `range`; compiles to a native MSL `for` loop |
| `break` / `continue` in loops | Supported | |
| `return` (no value) | Supported | Kernels must not return a value |
| `while` loops | Unsupported | Only `for x in range(...)` loop shapes are recognized; a `while` raises `UnsupportedFeatureError` |
| Recursion | Unsupported | Not attempted; no call-graph support at all beyond the single kernel function |
| Exceptions (`try`/`except`/`raise`) | Unsupported | |
| Classes | Unsupported | |
| Strings | Unsupported | |
| Dictionaries, lists, sets | Unsupported | |
| f-strings, `print()` | Unsupported | |
| Closures over outer-scope non-constant variables | Unsupported | Only module-level globals/functions (e.g. `metal`, `math`) are resolved |
| Device functions (helper functions called from a kernel) | Planned | See `docs/roadmap.md` Phase 3 |

## Scalar types

| Type | Status | Notes |
|---|---|---|
| `float32` | Supported | Required by spec |
| `int32` | Supported | Required by spec |
| `uint32` | Supported | Required by spec |
| `bool` | Supported | Required by spec |
| `float16` | Supported | Optional; verified via the unit/integration tests, not extensively benchmarked |
| `int64` | Supported | Optional; used internally for `metal.grid()`'s return type and loop counters |
| `uint64` | Partially supported | Type-mapping exists (`ulong`); not exercised by any kernel argument in the test suite |
| `float64` | Unsupported (as an array dtype or kernel argument) | Rejected with a specific error. Local *intermediate* float64 values (e.g. from a Python float literal or `/` true division) are automatically narrowed to MSL's 32-bit `float`, which is a deliberate, documented precision decision -- see `docs/limitations.md` -- not a claim of float64 support |
| `int8`, `int16`, `uint8`, `uint16` | Unsupported | Not in the mapping table; rejected |

## Array dimensions

| Feature | Status | Notes |
|---|---|---|
| 1D arrays | Supported | The only array rank accepted as a kernel argument |
| Flattened multidimensional indexing | Supported | Used by `heat_diffusion.py` and `pairwise_distance.py` benchmarks (manual `row*width+col` arithmetic) |
| Native 2D/3D `numpy.ndarray` kernel arguments | Unsupported | A `.reshape(-1)` / flatten is required before `to_device()`; passing a 2D-or-higher array type raises `UnsupportedFeatureError` naming the dimensionality |
| `.shape` attribute | Unsupported | Only `.size` is supported |
| Local array allocation inside a kernel | Unsupported | e.g. `np.zeros(4, dtype=np.float32)` inside a kernel body |
| Broadcasting | Unsupported | |

## Operators and builtins

| Feature | Status | Notes |
|---|---|---|
| `abs()` | Supported | |
| `min()`, `max()` (2-argument) | Supported | |
| `float()`, `int()` casts | Supported | |
| `math.sqrt` | Supported | |
| `math.exp` | Supported | |
| `math.log` | Supported | |
| `math.sin` | Supported | |
| `math.cos` | Supported | |
| Any other `math.*` function (`tan`, `atan2`, `pow`, ...) | Unsupported | Raises `UnsupportedFeatureError` naming the specific function |
| NumPy ufuncs called inside a kernel (`np.sqrt(x)`, etc.) | Unsupported | Use the `math` module equivalents instead |

## Memory operations

| Feature | Status | Notes |
|---|---|---|
| `metal.to_device(array)` | Supported | |
| `metal.device_array(shape, dtype)` | Supported | |
| `metal.device_array_like(array)` | Supported | |
| `device_array.copy_to_host()` | Supported | |
| `device_array.copy_to_device(host_array)` | Supported | |
| GPU-resident arrays across multiple kernel launches | Supported | Demonstrated by `benchmarks/heat_diffusion.py` |
| Zero-copy host<->device (no memcpy at all) | Unsupported | Current implementation performs a host-side memcpy into/out of a shared-storage-mode `MTLBuffer` on every `to_device`/`copy_to_host` call, even though the underlying memory is unified -- see `docs/architecture.md` and `docs/limitations.md` |
| Threadgroup (shared) memory | Planned | See `docs/roadmap.md` Phase 2 |
| Atomics | Unsupported (MVP scope) | Not required by any of the five benchmark kernels |

## Launch dimensions

| Feature | Status | Notes |
|---|---|---|
| 1D launch: `kernel[blocks, threads](...)` with `metal.grid(1)` | Supported | |
| 2D launch: `kernel[(bx,by), (tx,ty)](...)` with `metal.grid(2)` | Supported | |
| 3D launch / `metal.grid(3)` | Unsupported | Raises `UnsupportedFeatureError` naming the requested ndim |
| `metal.gridsize(ndim)` | Supported | Total dispatched thread count along each dimension |
| Invalid launch geometry (zero/negative blocks or threads, threads exceeding device limit) | Rejected | Raises `KernelLaunchError` before any GPU work is submitted |

## Synchronization and execution

| Feature | Status | Notes |
|---|---|---|
| `metal.synchronize()` | Supported | Blocks on a barrier command buffer |
| Repeated kernel launches | Supported | |
| Compilation cache (source + signature + device) | Supported | In-process only; not persisted to disk (see `docs/roadmap.md`) |
| `NUMBA_METAL_DUMP_MSL=1` / `metal.config.dump_msl` | Supported | |
| Streams / multiple command queues / events | Unsupported | Single process-wide serial command queue |
| Device functions / `@vectorize` / ufunc support | Planned | See `docs/roadmap.md` Phase 3 |

## Error behavior

| Scenario | Behavior |
|---|---|
| Unsupported Python construct | `UnsupportedFeatureError` at first launch (compile time), naming the specific IR node/operator/type |
| Unsupported dtype (array or scalar) | `UnsupportedFeatureError` / `KernelLaunchError` naming the dtype |
| Metal shader compilation failure | `KernelCompilationError` including the Metal compiler's own diagnostic text and the generated MSL source |
| Invalid launch configuration | `KernelLaunchError` before any GPU work is submitted |
| Passing a host `numpy.ndarray` where a device array is required | `KernelLaunchError` -- never silently uploaded |
| Running on unsupported hardware/OS | `UnsupportedPlatformError` or `MetalToolchainError` immediately, with a specific remediation hint |
| Any of the above | Never a silent fallback to NumPy, Numba's CPU target, or a Python loop |
