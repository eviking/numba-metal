# Quickstart

This walks through writing, launching, and correctly timing a kernel.
Assumes you've completed `docs/installation.md`.

## Writing a kernel

A kernel is a Python function decorated with `@metal.jit`. It must return
nothing (`return` with no value, or fall off the end) and its arguments
are either 1D device arrays or scalars:

```python
import numpy as np
from numba_metal import metal

@metal.jit
def scale(a, out, factor):
    i = metal.grid(1)
    if i < out.size:
        out[i] = a[i] * factor
```

`metal.grid(1)` returns this thread's absolute position in the launch
grid, as an `int64`. Always guard array accesses with a bounds check
(`if i < out.size:`) -- the grid is usually launched with more threads
than there are elements, and out-of-bounds access is undefined behavior
on the GPU, not a Python-style exception.

See `docs/supported-features.md` for the full list of supported syntax
and operations. Anything outside that list raises a specific
`UnsupportedFeatureError` at compile time (on first launch) rather than
silently doing something else.

## Allocating device arrays

```python
a = np.arange(1000, dtype=np.float32)

d_a = metal.to_device(a)              # allocate + copy host -> device
d_out = metal.device_array_like(a)    # allocate, uninitialized, same shape/dtype
d_scratch = metal.device_array(1000, np.float32)  # allocate by shape/dtype directly
```

These are GPU-resident (`DeviceNDArray`) objects backed by
`MTLResourceStorageModeShared` buffers. On Apple silicon's unified
memory, `to_device`/`copy_to_host` are a host-side memcpy into/out of
that buffer's own memory, not a PCIe-style transfer -- see
`docs/architecture.md` for why this isn't quite "zero-copy" yet and what
would make it so.

## Calculating grid dimensions

Launch geometry is `kernel[blocks, threads](*args)`, CUDA-style. Pick
`threads` (threads per block, commonly 128-256) and compute `blocks` to
cover your array:

```python
threads = 256
blocks = (a.size + threads - 1) // threads   # ceiling division
scale[blocks, threads](d_a, d_out, np.float32(2.0))
```

2D launches use `(x, y)` tuples for both `blocks` and `threads`, paired
with `metal.grid(2)` inside the kernel (returns an `(x, y)` tuple):

```python
@metal.jit
def fill_2d(out, width, height):
    x, y = metal.grid(2)
    if x < width and y < height:
        out[y * width + x] = x + y

fill_2d[(1, 1), (width, height)](d_out, np.int32(width), np.int32(height))
```

## Launching a kernel

```python
scale[blocks, threads](d_a, d_out, np.float32(2.0))
```

This is asynchronous: it returns as soon as the work is *encoded and
submitted*, not when it's *finished*. Scalar arguments must be NumPy
scalars (or Python `int`/`float`/`bool`) of a supported dtype; passing a
raw host `numpy.ndarray` instead of a `DeviceNDArray` is a compile-time
error (numba-metal never silently uploads it for you).

## Synchronizing

```python
metal.synchronize()
```

Blocks until all previously submitted kernels have completed. Call this
before reading results with `copy_to_host()` in a context where you need
a guarantee the GPU is done -- `copy_to_host()` itself calls
`synchronize()` internally, so results read through it are always
correct, but if you are timing kernel execution, you must call
`synchronize()` yourself before stopping the clock (see below).

## Copying results back

```python
result = d_out.copy_to_host()          # new NumPy array
d_out.copy_to_host(out=existing_array) # write into an existing array
```

## Measuring execution correctly

**Never time an unsynchronized launch** -- the launch call returns almost
immediately regardless of how long the GPU work actually takes, because
it's asynchronous:

```python
import time

# WRONG: measures encode/submit time, not GPU execution time
start = time.perf_counter_ns()
scale[blocks, threads](d_a, d_out, np.float32(2.0))
elapsed = time.perf_counter_ns() - start   # meaningless, tiny number

# RIGHT: synchronize before stopping the clock
start = time.perf_counter_ns()
scale[blocks, threads](d_a, d_out, np.float32(2.0))
metal.synchronize()
elapsed = time.perf_counter_ns() - start
```

Also separate **cold** timing (includes first-time MSL compilation, which
can take tens of milliseconds) from **warm** timing (compiled kernel
already cached):

```python
scale[blocks, threads](d_a, d_out, np.float32(2.0))  # first call: compiles + runs
metal.synchronize()

start = time.perf_counter_ns()
scale[blocks, threads](d_a, d_out, np.float32(2.0))  # warm: cache hit
metal.synchronize()
warm_ns = time.perf_counter_ns() - start
```

See `benchmarks/common.py` for the reusable warm-up/repeat/median
helpers used throughout this project's own benchmarks, and
`docs/benchmarking.md` for the full methodology.

## Reusing resident data

Avoid re-uploading data for every kernel launch. Allocate device arrays
once and launch multiple kernels against them:

```python
d_a = metal.to_device(a)
d_b = metal.to_device(b)
d_tmp = metal.device_array_like(a)
d_out = metal.device_array_like(a)

kernel_one[blocks, threads](d_a, d_b, d_tmp)
kernel_two[blocks, threads](d_tmp, d_out)
metal.synchronize()
result = d_out.copy_to_host()   # only one host<->device round trip
```

`benchmarks/heat_diffusion.py` demonstrates this explicitly: it compares
keeping two ping-pong buffers resident on the GPU for an entire
multi-iteration simulation against re-uploading/downloading on every
single iteration, and reports the difference in wall time.

## Debugging generated MSL

```bash
NUMBA_METAL_DUMP_MSL=1 python your_script.py
```

or at runtime:

```python
from numba_metal import metal
metal.config.dump_msl = True
```

Prints the generated MSL source for every kernel compiled from that point
on, before it's handed to the Metal compiler.
