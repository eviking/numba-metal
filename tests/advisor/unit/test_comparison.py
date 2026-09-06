"""Unit tests for `classify_roofline` -- deterministic, synthetic
calibration data only, no real Metal hardware or machine-dependent
timing, matching the spec's "do not make timing-sensitive unit tests
depend on real machine performance" instruction. The calibration file
is written to a temp XDG_CACHE_HOME so no real `~/.cache/numba-metal/`
calibration is read or clobbered.
"""

from __future__ import annotations

import json

from numba_metal.advisor.calibration import calibration_path
from numba_metal.advisor.comparison import classify_roofline
from numba_metal.advisor.models import RooflinePerformanceRegime

_FAKE_CALIBRATION = {
    "dispatch_overhead_ns": 180_000,
    "memory_bandwidth_gbps": 221.0,
    "float32_gflops_memory_bound": 37.0,
    "float32_gflops_compute_bound": 683.0,
    "roofline_ridge_flops_per_byte": 3.09,
    "device_name": "Apple M-series (fake, test-only)",
}


def _write_fake_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_FAKE_CALIBRATION))


def test_no_bytes_or_flops_returns_none(tmp_path, monkeypatch):
    _write_fake_calibration(tmp_path, monkeypatch)
    assert (
        classify_roofline(
            metal_median_ns=1000.0, bytes_per_call=None, flops_per_call=100
        )
        is None
    )
    assert (
        classify_roofline(
            metal_median_ns=1000.0, bytes_per_call=100, flops_per_call=None
        )
        is None
    )


def test_no_calibration_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert not calibration_path().exists()
    result = classify_roofline(
        metal_median_ns=1000.0, bytes_per_call=1024, flops_per_call=2048
    )
    assert result is None


def test_dispatch_bound_when_overhead_dominates(tmp_path, monkeypatch):
    _write_fake_calibration(tmp_path, monkeypatch)
    # 180us calibrated dispatch overhead against a 200us measured call:
    # 90% of the time is dispatch overhead, regardless of intensity.
    result = classify_roofline(
        metal_median_ns=200_000.0, bytes_per_call=1024, flops_per_call=2048
    )
    assert result is not None
    assert result.regime == RooflinePerformanceRegime.DISPATCH_BOUND
    assert result.dispatch_overhead_fraction is not None
    assert result.dispatch_overhead_fraction > 0.3


def test_compute_bound_when_intensity_above_ridge(tmp_path, monkeypatch):
    _write_fake_calibration(tmp_path, monkeypatch)
    # High arithmetic intensity (100 FLOPs/byte, well above the 3.09
    # ridge) and a long enough runtime that dispatch overhead is a
    # small fraction of the total.
    result = classify_roofline(
        metal_median_ns=10_000_000.0,
        bytes_per_call=1_000_000,
        flops_per_call=100_000_000,
    )
    assert result is not None
    assert result.regime == RooflinePerformanceRegime.COMPUTE_BOUND


def test_bandwidth_bound_when_intensity_below_ridge(tmp_path, monkeypatch):
    _write_fake_calibration(tmp_path, monkeypatch)
    # Low arithmetic intensity (1 FLOP/byte, below the 3.09 ridge), and
    # achieved bandwidth well under the 221 GB/s ceiling.
    result = classify_roofline(
        metal_median_ns=10_000_000.0, bytes_per_call=1_000_000, flops_per_call=1_000_000
    )
    assert result is not None
    assert result.regime == RooflinePerformanceRegime.BANDWIDTH_BOUND
    assert result.bandwidth_ceiling_fraction is not None
    assert result.bandwidth_ceiling_fraction < 1.0


def test_cache_bound_when_achieved_bandwidth_exceeds_dram_ceiling(
    tmp_path, monkeypatch
):
    _write_fake_calibration(tmp_path, monkeypatch)
    # Runtime long enough that the 180us calibrated dispatch overhead is
    # a small fraction of the total (avoiding DISPATCH_BOUND), but the
    # bytes moved per call are still large enough to push achieved
    # bandwidth above the calibrated 221 GB/s DRAM ceiling, at an
    # intensity still below the ridge point.
    result = classify_roofline(
        metal_median_ns=1_000_000.0,
        bytes_per_call=300_000_000,
        flops_per_call=300_000_000,
    )
    assert result is not None
    assert result.regime == RooflinePerformanceRegime.CACHE_BOUND
    assert result.bandwidth_ceiling_fraction is not None
    assert result.bandwidth_ceiling_fraction > 1.0
