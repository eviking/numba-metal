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
exercised by the test suite in this repository. See
`docs/feature-traceability.md` for the underlying audit: what specific
test provides evidence for each "Supported" row, the evidence-level
scale used, and the gaps found and closed while producing that audit.

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
| `while` loops | Partially supported | Straight-line body only (no nested `if`/`else`, `break`, or `continue`) -- e.g. a compare-and-swap retry loop or simple accumulation; a `while` with a nested conditional raises `UnsupportedFeatureError` rather than risking the confirmed-possible silent-wrong-result failure mode found during development. See `docs/limitations.md` and `tests/integration/test_while_loops.py`. |
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
| Local array allocation inside a kernel (`metal.local_array(shape, dtype)`) | Supported | Fixed-size, compile-time-constant shape; private to the calling thread. General NumPy allocation syntax (e.g. `np.zeros(4, dtype=np.float32)`) remains unsupported -- use `metal.local_array` explicitly |
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
| NumPy scalar dtype constructors inside a kernel body (`np.float32(x)`, `np.uint32(x)`, etc.) | Unsupported | Discovered while adding Workstream 6 test coverage: `out[i] = a[i] + np.uint32(1)` raises `UnsupportedFeatureError`, distinct from the dtype itself being unsupported as an *array* type (see Scalar types above) -- use a plain Python literal (`a[i] + 1`) instead |

## Memory operations

| Feature | Status | Notes |
|---|---|---|
| `metal.to_device(array)` | Supported | |
| `metal.device_array(shape, dtype)` | Supported | |
| `metal.device_array_like(array)` | Supported | |
| `device_array.copy_to_host()` | Supported | Synchronizes with all outstanding GPU work first -- see docs/architecture.md, "Host/device synchronization model" |
| `device_array.copy_to_device(host_array)` | Supported | Synchronizes with all outstanding GPU work first, for the same reason (prevents a host write racing an in-flight kernel's reads of the same shared-memory buffer -- verified by a regression test, see docs/architecture.md) |
| GPU-resident arrays across multiple kernel launches | Supported | Demonstrated by `benchmarks/heat_diffusion.py` |
| Zero-copy host<->device (no memcpy at all) | Unsupported | Current implementation performs a host-side memcpy into/out of a shared-storage-mode `MTLBuffer` on every `to_device`/`copy_to_host` call, even though the underlying memory is unified -- see `docs/architecture.md` and `docs/limitations.md` |
| Threadgroup (shared) memory (`metal.shared_array(shape, dtype)`) | Supported | Fixed-size, compile-time-constant shape; shared by every thread in a threadgroup; requires `metal.barrier()` between a write and a cross-thread read (see below). Backing memory is allocated at dispatch time via `setThreadgroupMemoryLength:atIndex:` -- omitting this (an internal detail, not user-visible) was found to silently leave the memory unallocated with no compile/runtime error, reading back as zero |
| `metal.barrier()` | Supported | `threadgroup_barrier(mem_flags::mem_threadgroup)`; synchronizes execution and memory ordering for every thread in the current threadgroup. Must be reached uniformly by every thread in the threadgroup (same restriction as CUDA's `syncthreads()`/MSL's own compiler; not statically checked) |
| Atomic add/sub/min/max/exchange (`metal.atomic_add` etc.) | Supported | int32, uint32, float32. min/max are native MSL ops for int32/uint32; float32 has no native MSL atomic min/max on any Apple GPU family (a real, permanent MSL-language limitation, verified directly) and is lowered to a compare-and-swap retry loop instead -- still genuinely race-free, verified under real multi-hundred-thousand-thread contention, see `tests/integration/test_atomics.py` |
| Atomic compare-and-swap (`metal.atomic_compare_exchange`) | Supported | Returns `(old_value, success)`; uses MSL's *weak* compare-exchange (may spuriously fail even on a true match -- callers building a retry loop are unaffected, see the intrinsic's docstring) |

## Launch dimensions

| Feature | Status | Notes |
|---|---|---|
| 1D launch: `kernel[blocks, threads](...)` with `metal.grid(1)` | Supported | |
| 2D launch: `kernel[(bx,by), (tx,ty)](...)` with `metal.grid(2)` | Supported | |
| 3D launch: `kernel[(bx,by,bz), (tx,ty,tz)](...)` with `metal.grid(3)` | Supported | |
| `metal.gridsize(ndim)` | Supported | Total dispatched thread count along each dimension; ndim in (1, 2, 3) |
| `metal.threadgroup_position(ndim)` | Supported | MSL's `threadgroup_position_in_grid`; ndim in (1, 2, 3) |
| `metal.thread_in_threadgroup(ndim)` | Supported | MSL's `thread_position_in_threadgroup`; ndim in (1, 2, 3) |
| `metal.threads_per_threadgroup(ndim)` | Supported | MSL's `threads_per_threadgroup`; always exactly the threadgroup shape the kernel was launched with -- the dispatcher previously could silently launch a different threadgroup shape than requested if it exceeded the device's per-threadgroup thread limit (unreachable in practice since launch validation already rejects that case first, but a real latent bug now that kernels can observe their own launch geometry); fixed to always dispatch exactly the requested shape |
| Invalid launch geometry (zero/negative blocks or threads, threads exceeding device limit) | Rejected | Raises `KernelLaunchError` before any GPU work is submitted |
| Zero-length device arrays | Partially supported | `metal.to_device()`, `device_array()`, `device_array_like()`, and `copy_to_host()` all handle a zero-length array correctly; there is no zero-block launch path, so `kernel[0, threads](...)` is rejected by the same `KernelLaunchError` as any other zero/negative launch dimension -- a caller with a zero-length array must skip the launch entirely rather than pass 0 blocks |

## Synchronization and execution

| Feature | Status | Notes |
|---|---|---|
| `metal.synchronize()` | Supported | Waits on and inspects the status/error of every outstanding tracked command buffer (not a single barrier); raises `MetalRuntimeError` naming the specific failing kernel and submission number -- see docs/architecture.md, "Command-buffer tracking and error propagation" |
| Repeated kernel launches | Supported | |
| Compilation cache (source + signature + device) | Supported | In-process only; not persisted to disk (see `docs/roadmap.md`) |
| `NUMBA_METAL_DUMP_MSL=1` / `metal.config.dump_msl` | Supported | |
| Command-buffer failure propagation | Supported | Every submitted command buffer is tracked through completion; a GPU-side Metal error on any of them is raised (not silently discarded) at the next `synchronize()`/`copy_to_host()`/`copy_to_device()` call |
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
