# Feature traceability matrix (Workstream 6)

`docs/supported-features.md` states conclusions ("Supported" /
"Partially supported" / "Unsupported"). This document records the audit
that backs those conclusions: for every row marked **Supported**, what
specific evidence exists and where.

## Evidence levels

Ordered from weakest to strongest; a feature is only labeled "Supported"
in `docs/supported-features.md` if it reaches at least **Executed on
Metal** with a differential/reference check (the fourth level below):

1. **Type-mapped only** -- a dtype/type appears in a mapping table
   (e.g. `numba_scalar_to_msl`) with a unit test checking the string
   output, but no kernel anywhere actually uses it.
2. **MSL generated** -- a unit test compiles a kernel through the
   frontend and checks the generated MSL text contains an expected
   substring/pattern, without ever executing it on a GPU.
3. **MSL successfully compiled** -- the generated MSL is handed to
   Apple's real `MTLDevice.newLibraryWithSource_options_error_` and a
   pipeline state is created, but the kernel is never dispatched.
4. **Executed on Metal** -- the kernel is actually dispatched on real
   Metal hardware (`pytest -m metal`) and its output is checked, either
   against a hand-computed expected value or a reference implementation.
5. **Differentially verified against Numba CPU** -- Metal output is
   compared against an independent `@njit` (or NumPy) computation of the
   same algorithm, not just an inline expected-value assertion.
6. **Benchmarked** -- also exercised by one of the five
   `benchmarks/*.py` kernels with a correctness gate
   (`assert_allclose`/`assert_near_integer_match`), which additionally
   proves the feature survives at realistic problem sizes, not just
   4-16-element unit-test inputs.

## Audit method

Every "Supported" row in `docs/supported-features.md` (as of this
Workstream) was cross-referenced against:

- `tests/unit/test_types.py`, `test_msl_backend.py`, `test_frontend.py`,
  `test_dispatcher_launch_config.py`, `test_command_buffer_tracking.py`
  (levels 1-2 only -- no Metal hardware involved).
- `tests/integration/*.py` (levels 4-5 -- real Metal execution).
- `benchmarks/*.py` kernels, insofar as they are also exercised by
  `tests/integration/test_benchmark_correctness.py` (level 6).

The audit found **eight rows** claiming "Supported" on evidence weaker
than level 4 (some backed only by a type-mapping unit test, one --
`metal.gridsize` -- with no test at all). `tests/integration/
test_feature_matrix_gaps.py` was added specifically to close each of
these to level 4 or 5 with real Metal execution. One genuine new
limitation was discovered in the process (NumPy scalar dtype
constructors, e.g. `np.uint32(x)`, are rejected inside a kernel body --
now documented in `docs/supported-features.md`).

## Gaps found and closed

| Feature | Prior evidence level | Gap | Closed by |
|---|---|---|---|
| `metal.gridsize(1)` | None found | Zero test coverage of any kind | `test_gridsize_1d_matches_launch_geometry` |
| `metal.gridsize(2)` | None found | Zero test coverage of any kind | `test_gridsize_2d_matches_launch_geometry` |
| `min()`, `max()` | MSL generated | Substring check only (`test_msl_backend.py::test_abs_min_max_supported`), never executed | `test_min_max_builtins_on_real_hardware` |
| `float()`, `int()` casts | None found | No test at any level | `test_float_and_int_casts_on_real_hardware` |
| `uint32` array argument | Type-mapped only | Only `test_types.py`'s dtype-string mapping test; no kernel used it as an array dtype | `test_uint32_array_input_and_output` |
| `float16` array argument | Type-mapped only | Doc's own Notes column claimed integration coverage that did not exist | `test_float16_array_input_and_output` |
| `int64` array argument (input) | Executed, output-only | `test_grid_2d_matches_reference` used int64 only as an output dtype | `test_int64_array_input` |
| `bool` array argument (input) | Executed, output-only | `test_boolean_operators` used bool only as an output dtype | `test_bool_array_input` |
| Comparison operators `<=`, `==`, `!=` | Executed (partial) | Only `<`, `>`, `>=` had appeared inside any tested kernel body; the doc row claims all six | `test_all_comparison_operators_on_real_hardware` |
| Boolean operators `or`, `not` | Executed (partial) | Only `and` had appeared inside any tested kernel body; the doc row claims all three | `test_or_and_not_boolean_operators_on_real_hardware` |
| `NUMBA_METAL_DUMP_MSL` / `metal.config.dump_msl` | None found | Zero test coverage | `test_dump_msl_env_var_prints_generated_source` |

## Renamed/added compiler-error tests

`test_end_to_end.py::test_metal_compilation_error_reports_msl_and_diagnostic`
was renamed to
`test_unsupported_construct_rejected_before_metal_compilation`: its own
docstring already admitted it tested a Numba frontend rejection (a dict
literal), not a real Metal shader compiler failure -- the name
overstated what it covered. A new test,
`test_real_metal_compilation_error_reports_diagnostic_and_msl`, was
added that genuinely exercises Apple's Metal compiler: it monkeypatches
`MSLKernelLowerer.lower` to emit deliberately invalid MSL body text
(numba-metal's own codegen never emits invalid MSL for any currently
supported construct, so there is no way to trigger this path with an
ordinary kernel), confirms the real
`device.newLibraryWithSource_options_error_` call rejects it, and
asserts the resulting `KernelCompilationError` contains both Metal's
own diagnostic text and the generated MSL.

## Boundary-value coverage

The mandatory end-to-end test list also requires boundary values:
signed/unsigned dtype limits, zero, negative values, bool arrays, odd
and non-power-of-two sizes, non-divisible grid sizes, zero-length
arrays, and NaN/inf. None of this existed anywhere in the test suite
before this Workstream; `tests/integration/test_boundary_values.py`
(10 tests, all real Metal execution) adds it:

- `test_int32_signed_limits`, `test_uint32_limits_including_wraparound`
  (including verifying uint32 wraparound matches NumPy's modular
  semantics).
- `test_float32_nan_and_inf_propagation`,
  `test_float32_nan_comparisons_are_false` (IEEE-754 ordered-comparison
  semantics, including `NaN == NaN` being `False`).
- `test_bool_array_all_true_all_false_mixed`.
- `test_odd_and_prime_sized_arrays`,
  `test_non_divisible_grid_launch_does_not_overrun`,
  `test_single_element_array`.
- `test_zero_length_array_allocation_and_transfer` and
  `test_zero_blocks_launch_is_rejected_not_silently_skipped` -- the
  latter discovered a genuine, previously-undocumented behavior: there
  is no zero-block no-op launch path, so a caller with a zero-length
  array must skip the kernel launch entirely rather than pass `0` for
  `blocks` (which raises `KernelLaunchError`, the same as any other
  invalid launch geometry). Now documented in
  `docs/supported-features.md` under "Launch dimensions."

## Full per-row evidence summary

See `docs/supported-features.md` for the current status of every
feature. Every row there marked "Supported" now has at least
**Executed on Metal** evidence (level 4) via one or more of:
`tests/integration/test_end_to_end.py`,
`tests/integration/test_phi_dessa.py`,
`tests/integration/test_host_device_sync.py`,
`tests/integration/test_command_buffer_tracking_metal.py`,
`tests/integration/test_feature_matrix_gaps.py`,
`tests/integration/test_benchmark_correctness.py`, or direct benchmark
execution under `benchmarks/`.
