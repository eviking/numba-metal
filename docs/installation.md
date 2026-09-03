# Installation

These instructions were followed and verified on a real machine (Apple
M4 Pro, macOS 26.5.1, Python 3.13.5) while building this package; commands
are exact, not illustrative.

## Supported versions

| Component | Supported range |
|---|---|
| macOS | 14 or newer |
| Architecture | Apple silicon (arm64) only -- Intel Macs are rejected at runtime |
| Python | 3.10 - 3.13 |
| Numba | 0.59 - 0.67 (pinned bound in `pyproject.toml`; only 0.67.0 has been exercised so far) |
| NumPy | 1.24 - 2.x |

## 1. Verify you're on Apple silicon

```bash
uname -m
```

Must print `arm64`. If it prints `x86_64`, this package will refuse to
run (Intel Macs have no Apple GPU and are not supported).

## 2. Install Xcode command-line tools

```bash
xcode-select --install
```

If already installed, this prints an error saying so -- that's fine.
Verify with:

```bash
xcode-select -p
```

This should print a path such as `/Applications/Xcode.app/Contents/Developer`
(a full Xcode.app install) or `/Library/Developer/CommandLineTools` (the
standalone CLT package). Either works, but a fresh CLT-only install may
still be missing the Metal Toolchain component (next step).

## 3. Install the Metal Toolchain component

The `metal`/`metallib` shader compilers are a separate downloadable
component, not always bundled with a fresh Xcode install. Check whether
it's already present:

```bash
xcrun -sdk macosx metal --version
```

If this fails with a message mentioning "missing Metal Toolchain", run:

```bash
xcodebuild -downloadComponent MetalToolchain
```

This is a large (~700MB) one-time download requiring network access and
(depending on your Xcode license state) may prompt for `sudo` or Xcode
license acceptance. Re-run the `xcrun -sdk macosx metal --version` check
above afterward to confirm it now prints a version string.

## 4. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Homebrew-installed Python on macOS treats the system install as
"externally managed" (PEP 668) and will refuse a bare `pip install`
outside a venv -- using a venv avoids that entirely and is the
recommended approach regardless.

## 5. Install numba-metal

From PyPI (once published):

```bash
pip install numba-metal
```

From source (editable, for development):

```bash
git clone https://github.com/numba-metal/numba-metal.git
cd numba-metal
pip install -e ".[dev]"
```

This installs `numba`, `numpy`, `pyobjc-framework-Metal`, and
`pyobjc-framework-libdispatch` as declared dependencies (see
`pyproject.toml` for exact bounds), plus `pytest`/`ruff`/`black` for the
`dev` extra.

## 6. Run a Metal capability check

```bash
python3 -c "from numba_metal import metal; print(metal.get_device_info())"
```

On a supported machine this prints a `DeviceInfo(...)` with your GPU's
name, unified-memory flag, and thread limits. On an unsupported machine
(wrong architecture, macOS too old, no Metal device, or missing Metal
Toolchain) it raises `UnsupportedPlatformError` or `MetalToolchainError`
with a specific, actionable message -- it does not silently report
"available" on a machine that can't actually run kernels.

You can also check without raising:

```bash
python3 -c "from numba_metal import metal; print(metal.is_available())"
```

## 7. Run the unit tests

Tests that don't require a GPU (compiler/frontend/codegen logic):

```bash
pytest -m "not metal"
```

## 8. Run the Metal integration tests

Requires a real Apple-silicon Mac with a working Metal device (these were
run and passed on the M4 Pro used to build this package):

```bash
pytest -m metal
```

Run everything:

```bash
pytest
```

## 9. Run the benchmarks

```bash
python benchmarks/run_all.py --quick     # fast sanity run, smaller sizes
python benchmarks/run_all.py             # full sizes (Mandelbrot 4096^2 takes ~20-30s)
python benchmarks/run_all.py --json results.json
```

See `docs/benchmarking.md` for what each column means and how to
interpret the numbers.

## Common installation failures

**`xcrun: error: ... cannot execute tool 'metal' due to missing Metal Toolchain`**
Run `xcodebuild -downloadComponent MetalToolchain` (step 3).

**`ModuleNotFoundError: No module named 'Metal'`**
`pyobjc-framework-Metal` did not install, or you're not inside the venv
where it was installed. Re-run `pip install -e ".[dev]"` inside an
activated venv.

**`error: externally-managed-environment` from pip**
You're trying to `pip install` into the system/Homebrew Python directly.
Create and activate a venv first (step 4).

**`UnsupportedPlatformError: numba-metal requires Apple-silicon (arm64) Macs`**
You're on an Intel Mac, or running under Rosetta translation. Check
`uname -m`; if it prints `x86_64` under what you believe is an
Apple-silicon Mac, you may be in a Rosetta-translated terminal/shell --
relaunch your terminal natively.

**Numba version mismatch / `KernelCompilationError` mentioning "did not
produce a TypedKernelIR"**
Your installed Numba version changed compiler-pipeline internals in a way
numba-metal's frontend adapter doesn't yet handle. Check the pinned range
in `pyproject.toml`; file an issue with your exact `numba.__version__`.

See `docs/troubleshooting.md` for runtime (post-install) issues.
