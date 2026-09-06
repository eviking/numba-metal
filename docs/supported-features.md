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
| `while` loops | Partially supported | Straight-line body, or a body with a nested `if`/`else` containing no `break`/`continue`, are both supported -- e.g. a compare-and-swap retry loop, a simple accumulation, or an iterative numerical method (Newton-Raphson, etc.) that branches differently per iteration. `break`/`continue` NESTED INSIDE that if/else remain unsupported and raise `UnsupportedFeatureError` rather than risking the confirmed-possible silent-wrong-result failure mode found during development. See `docs/limitations.md` and `tests/integration/test_while_loops.py`. |
| Recursion | Unsupported | Direct recursion in a `@metal.device_func` is rejected by Numba's own frontend at typing time; mutual/transitive recursion between two device functions is rejected by numba-metal's own in-progress-compilation cycle detection (see `_compile_device_function`) -- both fail with a clear compile-time error, not a stack overflow or hang |
| Exceptions (`try`/`except`/`raise`) | Unsupported | |
| Classes | Unsupported | |
| Strings | Unsupported | |
| Dictionaries, lists, sets | Unsupported | |
| f-strings, `print()` | Unsupported | |
| Closures over outer-scope non-constant variables | Unsupported | Only module-level globals/functions (e.g. `metal`, `math`) are resolved |
| Device functions (`@metal.device_func`, helper functions called from a kernel or another device function) | Supported | Scalar arguments, and 1D/2D/3D arrays of a supported dtype (same constraints as kernel array arguments -- a multi-dim array argument gets its own `_dimN` companion parameters, threaded through from the caller's own binding at every call site, including transitively through nested device-function calls); return type is scalar-only (no array return, and no array allocation inside a device function -- it can only read/write a caller-provided array). `metal.local_array`/`shared_array` are not supported as device-function arguments or locals. `metal.atomic_add`/`atomic_sub`/`atomic_min`/`atomic_max`/`atomic_exchange`/`atomic_compare_exchange` are usable inside a device function body, including in a `while`-loop compare-and-swap retry pattern, and are correct under real multi-thread contention when called this way. Other kernel-only intrinsics (`metal.grid`/`gridsize`/thread-position, `metal.local_array`/`shared_array`, `metal.barrier`) remain unsupported inside a device function body -- calling one raises a clear compile-time error rather than producing incorrect results. Compiled to a real, separate MSL function per distinct call-site signature -- not inlined; measured directly on `benchmarks/heat_diffusion.py`'s stencil (see that file's "Round 3"), the non-inlined call costs a real 2.3-2.6x per-iteration slowdown for a small, hot-loop 4-operand helper -- factor code out for organization/reuse, not assuming it is free. Compiling the exact same call twice with different Numba-level type tuples that narrow to the same MSL types (e.g. one call site's literal argument types as float64, another's as float32-after-narrowing) currently emits two functionally-identical MSL functions rather than deduplicating by post-narrowing MSL signature -- a real inefficiency, not a correctness issue (each is independently correct), documented in `docs/limitations.md` |

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
| 1D arrays | Supported | Kernel and `@metal.device_func` arguments |
| Native 2D/3D `numpy.ndarray` kernel arguments (`arr[x, y]`/`arr[x, y, z]`) | Supported | Requires a matching `metal.grid(2)`/`metal.grid(3)` launch on the kernel side; also supported as `@metal.device_func` array arguments (see "Python syntax" above). 4D and beyond are rejected with `UnsupportedFeatureError` naming the dimensionality -- flatten those manually. See `tests/integration/test_multidim_arrays.py`, `benchmarks/heat_diffusion.py`, `benchmarks/mandelbrot.py`, `benchmarks/pairwise_distance.py` |
| Flattened multidimensional indexing | Supported | Manual `row*width+col` arithmetic on a 1D array -- still necessary for 4D+ data. Measured (see `docs/performance-guidance.md`) to meaningfully slow down both Numba's own CPU codegen and Metal's performance relative to real multi-dimensional indexing at the same problem when native 2D/3D support already covers the case |
| `.shape` attribute | Unsupported | Only `.size` is supported (the total flattened element count) -- read a dimension's size from a separately-passed scalar argument if a kernel needs it directly |
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
| Whole-array reductions (`metal.reduce_sum`/`reduce_min`/`reduce_max`) | Supported | float32/int32/uint32 1D device arrays only (matches the atomic dtype set these are built on). A host-side helper, not a kernel-body intrinsic -- composes `shared_array`+`barrier` (per-threadgroup partial) with one `atomic_add`/`atomic_min`/`atomic_max` per threadgroup (final combine), avoiding the O(n) atomic contention a naive one-atomic-per-thread reduction would hit. Returns a 1-element `DeviceNDArray`, not a Python scalar, so a reduction result can feed a later kernel launch without a host round trip. See `numba_metal/reductions.py` and `tests/integration/test_reductions.py`. `argmax` is not implemented. |

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
| Scalar-argument buffer reuse pool | Supported | Small scalar/array-size constant `MTLBuffer`s are pooled by byte size and reused across dispatches instead of allocated fresh every launch; a buffer only re-enters the pool once `metal.synchronize()` confirms the command buffer that last used it has completed with no error -- see docs/architecture.md |
| `metal.batch()` | Supported | Encodes multiple kernel launches onto one shared `MTLCommandBuffer`, committed once as a single tracked submission when the `with` block exits, instead of one command buffer per launch. Not nestable; an exception inside the block discards the batch's uncommitted work entirely. See docs/architecture.md |
| `NUMBA_METAL_DUMP_MSL=1` / `metal.config.dump_msl` | Supported | |
| Command-buffer failure propagation | Supported | Every submitted command buffer is tracked through completion; a GPU-side Metal error on any of them is raised (not silently discarded) at the next `synchronize()`/`copy_to_host()`/`copy_to_device()` call. For a `metal.batch()` submission, a failure is attributed to the whole batch (naming every kernel encoded onto it), since Metal reports one status/error per command buffer, not per individual dispatch within it |
| Streams / multiple command queues / events | Unsupported | Single process-wide serial command queue (`metal.batch()` reduces per-launch submission overhead within that one queue, but does not provide concurrent/overlapping execution across independent command queues) |
| Device functions (`@metal.device_func`) | Supported | See "Python syntax" above |
| `@vectorize` / ufunc support | Planned | See `docs/roadmap.md` Phase 3 |

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
