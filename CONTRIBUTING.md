# Contributing

## Setup

```bash
git clone https://github.com/numba-metal/numba-metal.git
cd numba-metal
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

See `docs/installation.md` for platform prerequisites (Xcode tools, Metal
Toolchain).

## Running tests

```bash
pytest -m "not metal"   # compiler/frontend/codegen tests, no GPU required
pytest -m metal          # requires a real Apple-silicon Metal device
pytest                   # everything
```

## Formatting and linting

```bash
black src tests benchmarks
ruff check src tests benchmarks
```

## Code organization

- `src/numba_metal/errors.py` -- all exception types. New failure modes
  get a new type here, not a bare `raise Exception(...)`.
- `src/numba_metal/types/` -- the single source of truth for Numba<->MSL
  type mapping. Don't hard-code an MSL type name anywhere else.
- `src/numba_metal/compiler/` -- everything from Python source to MSL
  text. `frontend.py` is the only module allowed to touch Numba
  compiler-internal APIs (`CompilerBase`, `compiler_machinery`,
  `typed_passes`) -- keep that isolation; if you need something from
  Numba's internals elsewhere, route it through `frontend.py`'s output
  (`TypedKernelIR`) instead of importing Numba internals directly in
  `msl_backend.py`/`structuring.py`.
- `src/numba_metal/runtime/` -- device discovery, buffers, dispatch. No
  Numba imports belong here.
- No unconstrained string-substitution code generation, anywhere. MSL is
  emitted via the structured tree-walk in `msl_backend.py`/
  `structuring.py`; if you're tempted to `f"...{some_text}..."` a chunk
  of MSL together from source-text fragments, that's the wrong layer --
  it belongs in the IR walker.

## Adding support for a new Python construct

1. Confirm it type-checks through Numba's own frontend first (write a
   small script using `numba_metal.compiler.frontend.compile_to_typed_ir`
   and inspect the resulting `func_ir.blocks` -- see
   `docs/architecture.md` for how the typed IR is shaped).
2. Add codegen for it in `msl_backend.py` (or `structuring.py` if it's a
   new control-flow shape), following the existing dispatch-table
   pattern.
3. Verify the generated MSL actually compiles via
   `Metal.MTLCreateSystemDefaultDevice().newLibraryWithSource_options_error_`
   before considering it done -- a change that only "looks right" in
   generated text is not verified.
4. Add both a unit test (MSL text assertion, no GPU needed --
   `tests/unit/test_msl_backend.py`) and, if it's new enough to warrant
   one, a `pytest -m metal` integration test that actually runs it and
   checks the numeric result (`tests/integration/test_end_to_end.py`).
5. Update `docs/supported-features.md`.

## Adding an unsupported-construct rejection

If you're intentionally adding a new "this raises `UnsupportedFeatureError`"
case (rather than support), add a test in `tests/test_unsupported.py` and
a row in `docs/supported-features.md`'s "Unsupported" column -- don't let
a rejection go undocumented.

## Honesty requirements (non-negotiable, see the project's own README)

- No silent fallback to NumPy/CPU/a Python loop for anything.
- No benchmark numbers invented from theory -- if you add or change a
  benchmark, actually run it on real Metal hardware and report what you
  measured.
- No claims of support for something that hasn't been exercised by an
  automated test on real GPU hardware.
- Dedicated exception types, not bare `except Exception` that swallows
  context.

## Pull requests

Include what you tested and how (unit tests, `pytest -m metal` on what
hardware, benchmark numbers if performance-relevant). If you touched
`msl_backend.py`/`structuring.py`, mention whether you verified the
change against real Metal compilation, not just unit-test MSL-text
assertions.
