# Troubleshooting

## "No Metal device found"

```
UnsupportedPlatformError: No Metal device found (MTLCreateSystemDefaultDevice returned None).
```

This means `Metal.MTLCreateSystemDefaultDevice()` returned `None`. On a
real Mac this normally only happens in unusual sandboxed/virtualized
environments without GPU access. Verify:

```bash
system_profiler SPDisplaysDataType | grep -A2 "Metal Support"
```

should show a `Metal Support: Metal ...` line for your GPU.

## "Unsupported Intel Mac"

```
UnsupportedPlatformError: numba-metal requires Apple-silicon (arm64) Macs; detected machine architecture 'x86_64'.
```

Intel Macs are not supported (no Apple GPU / different Metal compute
feature set). Check `uname -m`; if you're on genuinely Apple-silicon
hardware but still see `x86_64`, your terminal/shell may be running under
Rosetta 2 translation -- open a fresh terminal window/tab, or check
Terminal.app's "Open using Rosetta" setting (should be unchecked).

## "Missing Xcode tools" / Metal Toolchain

```
MetalToolchainError: The Metal shader compiler is not available ...
```

Two distinct causes:

1. **No Xcode command-line tools at all**: run `xcode-select --install`.
2. **CLT/Xcode present but the Metal Toolchain component isn't**: this
   was the actual situation encountered on a fresh Xcode install while
   building this package. Run:
   ```bash
   xcodebuild -downloadComponent MetalToolchain
   ```
   This is a large (~700MB) download; it needs network access. Re-check
   with `xcrun -sdk macosx metal --version` afterward.

## MSL compilation failures

If a kernel's *Numba typing* succeeds but the generated MSL fails to
compile, you'll see a `KernelCompilationError` that includes both Apple's
Metal compiler diagnostic text and the full generated MSL source inline
in the exception message -- you don't need to separately dump MSL to
debug this. If you want to see the MSL for a kernel that compiles
successfully too (e.g. to inspect codegen), see "How to dump generated
MSL" below.

If you believe the generated MSL is invalid but numba-metal accepted the
kernel's Python source without complaint, that's a codegen bug in
numba-metal, not something to work around at the kernel-source level --
please file an issue including the printed MSL and the Python kernel
source.

## Numba-version mismatch

```
UnsupportedNumbaVersionError: numba-metal has only been validated against Numba ['0.67.0']; found Numba '...' installed. ...
```

This is the expected, deliberate failure mode for an unvalidated Numba
version: `numba_metal.compat.check_numba_compatible()` (invoked by
`metal.jit`/`metal.get_device_info()` via `check_capable()`) checks
`numba.__version__` before any kernel is compiled, and raises this
explicitly rather than letting an untested version silently produce
wrong MSL from a changed internal-IR shape. Install a supported version:

```bash
pip install 'numba>=0.67,<0.68'
```

If you've independently validated numba-metal's test suite against a
different Numba version, see `docs/numba-rfc.md` for how to extend the
compatibility gate.

```
KernelCompilationError: Internal error: numba-metal's typed-IR-only compiler pipeline did not produce a TypedKernelIR ...
```

This is a different, rarer symptom of the same underlying risk: it can
occur if the compatibility gate above accepted an untested *patch*
release within the `0.67.x` series (patch releases are allowed through
without an exact-version match -- see `docs/numba-rfc.md`) that turned
out to still change compiler-pipeline internals (`CompilerBase`,
`compiler_machinery`, or `typed_passes`) in a way numba-metal's frontend
adapter (`numba_metal/compiler/frontend.py`) doesn't handle. Check
`numba.__version__`; if it's not exactly `0.67.0` (the only version this
project's test suite has actually been run against), try pinning to
`0.67.0` exactly. If you're already on `0.67.0` and still see this,
please file an issue with your exact Numba version.

## Unsupported dtype

```
UnsupportedFeatureError: Unsupported array dtype dtype('float64') ...
```
or
```
MetalRuntimeError: Unsupported dtype dtype('float64') for device array ...
```

See `docs/supported-features.md` for the exact list of supported dtypes,
and `docs/limitations.md` for why float64 specifically is rejected (not
yet verified on Apple GPU hardware) rather than silently downcast for
array data (local float64 *intermediates*, as opposed to array dtypes,
are handled differently -- see that same limitations entry).

## Invalid launch geometry

```
KernelLaunchError: Invalid launch configuration: block count must be positive, got 0.
```
or
```
KernelLaunchError: Invalid launch configuration: threads-per-block (2048) exceeds this device's maximum (1024).
```

`blocks`/`threads` in `kernel[blocks, threads](...)` must each be a
positive `int` (1D) or a `(x, y)` tuple of positive `int` (2D), and the
threads-per-threadgroup product must not exceed the device's
`maxThreadsPerThreadgroup` (query it via
`metal.get_device_info().max_threads_per_threadgroup`).

## Incorrect timing caused by asynchronous execution

If a benchmark or timing measurement of yours reports suspiciously tiny
GPU times (microseconds for what should be a substantial computation),
you are almost certainly timing an unsynchronized launch. Kernel launches
are asynchronous -- `kernel[blocks, threads](*args)` returns as soon as
the work is *submitted*, not when it's *finished*. Always call
`metal.synchronize()` before stopping your clock:

```python
start = time.perf_counter_ns()
kernel[blocks, threads](*args)
metal.synchronize()          # <-- required for a meaningful measurement
elapsed = time.perf_counter_ns() - start
```

See `docs/quickstart.md` ("Measuring execution correctly") and
`docs/benchmarking.md` for the full methodology this project's own
benchmarks follow.

## How to dump generated MSL

Environment variable (affects the whole process, set before it starts):

```bash
NUMBA_METAL_DUMP_MSL=1 python your_script.py
```

Or at runtime, before compiling the kernels you want to inspect:

```python
from numba_metal import metal
metal.config.dump_msl = True
```

Both print the full generated MSL source (with a `// ---- numba-metal
generated MSL: <kernel_name> ----` / `// ---- end ----` banner) for every
kernel compiled from that point on, before it's handed to the Metal
compiler -- including kernels that go on to fail compilation, so this is
also the first thing to try when debugging a `KernelCompilationError`
(though that error already includes the MSL inline).
