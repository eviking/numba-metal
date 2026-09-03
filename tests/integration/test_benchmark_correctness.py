"""Deterministic correctness tests for the five benchmark kernels, on
small inputs. Complements benchmarks/*.py (which focus on performance
measurement) by giving each kernel a fast, CI-friendly correctness check.

Requires a working Metal device; run with `pytest -m metal`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))

pytestmark = pytest.mark.metal


def test_vector_polynomial_small() -> None:
    import vector_polynomial as vp

    from numba_metal import metal

    a = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    b = np.array([2.0, 1.0, -1.5, 0.0], dtype=np.float32)
    expected = vp.numpy_impl(a, b)

    kernel = vp._make_metal_kernel()
    d_a, d_b = metal.to_device(a), metal.to_device(b)
    d_out = metal.device_array_like(a)
    kernel[1, 8](d_a, d_b, d_out)
    metal.synchronize()
    assert np.allclose(d_out.copy_to_host(), expected, rtol=1e-4, atol=1e-4)


def test_mandelbrot_small() -> None:
    import mandelbrot as mb

    from numba_metal import metal

    width, height, max_iter = 16, 16, 30
    kernel = mb._make_metal_kernel()
    d_out = metal.device_array(width * height, np.int32)
    kernel[1, 256](d_out, np.int32(width), np.int32(height), np.int32(max_iter))
    metal.synchronize()
    result = d_out.copy_to_host()

    cpu_impl = mb._make_numba_cpu_impl(parallel=True)
    out_cpu = np.empty(width * height, dtype=np.int32)
    cpu_impl(out_cpu, width, height, max_iter)
    assert np.array_equal(result, out_cpu)
    assert result.min() >= 1
    assert result.max() <= max_iter


def test_heat_diffusion_small() -> None:
    import heat_diffusion as hd

    grid = hd._initial_grid(16)
    result = hd._metal_resident(grid, iterations=20)
    expected = hd.numpy_impl(grid, iterations=20)
    assert np.allclose(result, expected, rtol=1e-2, atol=1e-2)
    # Boundary should remain unchanged (Dirichlet-like fixed edges in this
    # simple update rule, since the kernel copies cur[i] through at the
    # border rather than updating it).
    assert np.array_equal(result[0, :], grid[0, :])
    assert np.array_equal(result[:, 0], grid[:, 0])


def test_monte_carlo_paths_small() -> None:
    import monte_carlo_paths as mc

    from numba_metal import metal

    n_paths, n_steps = 64, 10
    rng = np.random.default_rng(1)
    z = rng.standard_normal((n_paths, n_steps)).astype(np.float32)
    z_flat = z.reshape(-1)

    dt = mc.T / n_steps
    drift = np.float32((mc.R - 0.5 * mc.SIGMA * mc.SIGMA) * dt)
    diffusion = np.float32(mc.SIGMA * np.sqrt(dt))

    kernel = mc._make_metal_kernel()
    d_z = metal.to_device(z_flat)
    d_out = metal.device_array(n_paths, np.float32)
    kernel[1, 128](d_z, d_out, np.int32(n_steps), np.float32(mc.S0), drift, diffusion)
    metal.synchronize()
    gpu_terminal = d_out.copy_to_host()

    cpu_impl = mc._make_numba_cpu_impl(parallel=True)
    out_cpu = np.empty(n_paths, dtype=np.float32)
    cpu_impl(z_flat, out_cpu, n_paths, n_steps, mc.S0, mc.K, mc.R, mc.SIGMA, mc.T)

    assert np.allclose(gpu_terminal, out_cpu, rtol=1e-3, atol=1e-3)
    assert np.all(gpu_terminal > 0)  # GBM prices are always positive


def test_pairwise_distance_small() -> None:
    import pairwise_distance as pd

    from numba_metal import metal

    n_a, n_b, k = 5, 7, 3
    rng = np.random.default_rng(2)
    a = rng.standard_normal((n_a, k)).astype(np.float32)
    b = rng.standard_normal((n_b, k)).astype(np.float32)
    expected = pd.numpy_impl(a, b)

    kernel = pd._make_metal_kernel()
    d_a = metal.to_device(a.reshape(-1))
    d_b = metal.to_device(b.reshape(-1))
    d_out = metal.device_array(n_a * n_b, np.float32)
    kernel[1, 64](d_a, d_b, d_out, np.int32(n_a), np.int32(n_b), np.int32(k))
    metal.synchronize()
    result = d_out.copy_to_host().reshape(n_a, n_b)

    assert np.allclose(result, expected, rtol=1e-4, atol=1e-4)
    # Self-distance sanity check: distance(a[i], a[i]) via a==b should be ~0.
    d_out_self = metal.device_array(n_a * n_a, np.float32)
    d_a2 = metal.to_device(a.reshape(-1))
    kernel[1, 64](d_a, d_a2, d_out_self, np.int32(n_a), np.int32(n_a), np.int32(k))
    metal.synchronize()
    diag = d_out_self.copy_to_host().reshape(n_a, n_a).diagonal()
    assert np.allclose(diag, 0.0, atol=1e-5)
