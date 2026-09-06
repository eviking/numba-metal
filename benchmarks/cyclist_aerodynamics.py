"""Cyclist aerodynamic drag-area (CdA) position sweep -- and the fix for
when it doesn't win.

Two kernels, same physical model, deliberately different arithmetic
intensity, to make one point concrete: numba-metal's speedup on Apple
Silicon depends on FLOPs-per-byte-moved, not on how "parallel" a
workload looks.

`sweep`: for each of N (torso_angle, arm_width, head_drop, yaw, speed)
combinations, compute CdA (m^2) and the resulting aero drag power --
one instantaneous calculation per output element. Structurally
realistic relationships from published cycling-aerodynamics literature
(Blocken et al., "CFD simulation of the aerodynamic drag of two
drafting cyclists"; wind-tunnel studies from A2 Wind Tunnel/
Specialized's Win Tunnel program): CdA depends on torso angle and arm
width (narrower is not always better -- airflow separation increases
drag again below a real, documented minimum spacing), and CdA has a
non-monotonic dependence on yaw (many published wind-tunnel curves show
CdA DECREASING at small positive yaw before rising again, because
crosswind can partially "sail" a well-shaped rider). Not claimed as
validated wind-tunnel data -- the right shape and right order of
magnitude, fit to sit in the real published range (~0.19 m^2 aggressive
time-trial tuck to ~0.35 m^2 upright road position).

`course_energy`: the same per-position CdA model, but integrated over a
simulated course -- varying gradient and wind direction at each of
`n_segments` points along the route -- instead of one instantaneous
condition. This is a real technique used in time-trial pacing and
position-optimization tools: find the rider position that minimizes
total energy cost over a SPECIFIC course profile, not just at one
speed/yaw condition.

Why the second kernel exists: `sweep` plateaus at roughly parity with
CPU even at 50,000,000 elements (measured on an M4 Pro: 0.35x-1.08x
across problem sizes) -- one instantaneous drag calculation is too
little arithmetic per element to be anything but memory-bandwidth-
bound on Apple Silicon's shared CPU/GPU memory subsystem, where the CPU
already has the same bandwidth available. `course_energy` reads the
same amount of memory per output value (the course profile is small,
read-only, and shared across every thread -- not re-loaded per
parameter combination) but does `n_segments` times more arithmetic per
output value, which is exactly the lever that moves a workload from
bandwidth-bound to compute-bound. Measured on the same machine: a real
win from small sizes up through 10,000,000 riders (see `if __name__`
below for a fresh run).

See docs/performance-guidance.md for the general version of this
lesson, and how to estimate which regime YOUR workload falls into
before porting it.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from numba_metal import metal

AIR_DENSITY = np.float32(1.225)  # kg/m^3, sea level, 15C
RIDER_MASS_KG = np.float32(75.0)  # rider + bike
G = np.float32(9.81)
CRR = np.float32(0.004)  # rolling resistance coefficient, typical road tire

# Base CdA model coefficients (see module docstring for the literature
# this shape is drawn from).
BASE_CDA_UPRIGHT = np.float32(0.35)
BASE_CDA_AGGRESSIVE = np.float32(0.19)

N_SEGMENTS = 200  # course_energy's course-profile resolution
WARMUP = 2
REPEATS = 7


@njit(fastmath=False)
def _cda_from_position(torso_angle_deg, arm_width_cm, head_drop_cm):
    # torso_angle: 0 = fully upright, 90 = fully horizontal (TT-tuck-like).
    torso_frac = torso_angle_deg / np.float32(90.0)
    base = BASE_CDA_UPRIGHT + (BASE_CDA_AGGRESSIVE - BASE_CDA_UPRIGHT) * torso_frac

    # Arm width: real TT-position studies show a minimum around
    # 25-30cm hand spacing -- narrower than that increases drag again
    # as airflow can no longer attach cleanly around the forearms.
    optimal_width = np.float32(27.0)
    width_penalty = np.float32(0.00015) * (arm_width_cm - optimal_width) ** 2

    # Head drop: lowering the head reduces frontal area, with
    # diminishing returns and a small penalty past a natural limit
    # (modeled as a soft quadratic floor, not a hard cutoff).
    head_benefit = (
        -np.float32(0.0008) * head_drop_cm + np.float32(0.00002) * head_drop_cm**2
    )

    return base + width_penalty + head_benefit


@njit(fastmath=False)
def _yaw_multiplier(yaw_deg):
    # Non-monotonic yaw response: many published wind-tunnel CdA-vs-yaw
    # curves for a well-faired position show a slight CdA REDUCTION at
    # small positive yaw (5-10 degrees) before increasing at higher
    # yaw, rather than a simple monotonic increase.
    y = yaw_deg
    return np.float32(1.0) - np.float32(0.004) * y + np.float32(0.0009) * y * y


@njit(parallel=True, fastmath=False)
def cpu_sweep(torso, arm_width, head_drop, yaw, speed, out_power, n):
    for idx in prange(n):
        cda = _cda_from_position(torso[idx], arm_width[idx], head_drop[idx])
        cda = cda * _yaw_multiplier(yaw[idx])
        v = speed[idx]
        drag_force = np.float32(0.5) * AIR_DENSITY * v * v * cda
        out_power[idx] = drag_force * v  # watts to overcome aero drag alone


@metal.jit
def metal_sweep(torso, arm_width, head_drop, yaw, speed, out_power, n):
    i = metal.grid(1)
    if i < n:
        torso_frac = torso[i] / 90.0
        base = 0.35 + (0.19 - 0.35) * torso_frac
        optimal_width = 27.0
        width_penalty = (
            0.00015 * (arm_width[i] - optimal_width) * (arm_width[i] - optimal_width)
        )
        head_benefit = -0.0008 * head_drop[i] + 0.00002 * head_drop[i] * head_drop[i]
        cda = base + width_penalty + head_benefit

        y = yaw[i]
        yaw_mult = 1.0 - 0.004 * y + 0.0009 * y * y
        cda = cda * yaw_mult

        v = speed[i]
        drag_force = 0.5 * 1.225 * v * v * cda
        out_power[i] = drag_force * v


@njit(parallel=True, fastmath=False)
def cpu_course_energy(
    torso,
    arm_width,
    head_drop,
    base_yaw,
    speed,
    gradient,
    wind_yaw_offset,
    segment_length_m,
    out_energy,
    n,
    n_segments,
):
    for idx in prange(n):
        cda_base = _cda_from_position(torso[idx], arm_width[idx], head_drop[idx])
        v = speed[idx]
        total_energy = np.float32(0.0)
        for s in range(n_segments):
            yaw = base_yaw[idx] + wind_yaw_offset[s]
            cda = cda_base * _yaw_multiplier(yaw)
            aero_force = np.float32(0.5) * AIR_DENSITY * v * v * cda
            grade_force = RIDER_MASS_KG * G * gradient[s]
            rolling_force = RIDER_MASS_KG * G * CRR
            total_force = aero_force + grade_force + rolling_force
            total_energy += total_force * segment_length_m
        out_energy[idx] = total_energy


@metal.jit
def metal_course_energy(
    torso,
    arm_width,
    head_drop,
    base_yaw,
    speed,
    gradient,
    wind_yaw_offset,
    segment_length_m,
    out_energy,
    n,
    n_segments,
):
    idx = metal.grid(1)
    if idx < n:
        torso_frac = torso[idx] / 90.0
        cda_base = 0.35 + (0.19 - 0.35) * torso_frac
        optimal_width = 27.0
        width_penalty = (
            0.00015
            * (arm_width[idx] - optimal_width)
            * (arm_width[idx] - optimal_width)
        )
        head_benefit = (
            -0.0008 * head_drop[idx] + 0.00002 * head_drop[idx] * head_drop[idx]
        )
        cda_base = cda_base + width_penalty + head_benefit

        v = speed[idx]
        total_energy = 0.0
        for s in range(n_segments):
            yaw = base_yaw[idx] + wind_yaw_offset[s]
            yaw_mult = 1.0 - 0.004 * yaw + 0.0009 * yaw * yaw
            cda = cda_base * yaw_mult
            aero_force = 0.5 * 1.225 * v * v * cda
            grade_force = 75.0 * 9.81 * gradient[s]
            rolling_force = 75.0 * 9.81 * 0.004
            total_force = aero_force + grade_force + rolling_force
            total_energy = total_energy + total_force * segment_length_m
        out_energy[idx] = total_energy


def make_course(n_segments, seed=7):
    """A synthetic but structurally realistic course: rolling gradient
    profile (some climbing, some descending, net zero) plus a wind
    direction that varies along the route (crosswind sections, a
    headwind stretch, a tailwind stretch)."""
    rng = np.random.default_rng(seed)
    gradient = np.cumsum(rng.normal(0, 0.003, n_segments)).astype(np.float32)
    gradient -= gradient.mean()  # net-zero elevation change
    wind_yaw_offset = (10.0 * np.sin(np.linspace(0, 3 * np.pi, n_segments))).astype(
        np.float32
    )
    segment_length_m = np.float32(50.0)  # 50m per segment
    return gradient, wind_yaw_offset, segment_length_m


def make_params(n, seed=42):
    rng = np.random.default_rng(seed)
    torso = rng.uniform(0.0, 90.0, n).astype(np.float32)
    arm_width = rng.uniform(15.0, 45.0, n).astype(np.float32)
    head_drop = rng.uniform(0.0, 20.0, n).astype(np.float32)
    yaw = rng.uniform(-15.0, 15.0, n).astype(np.float32)
    speed = rng.uniform(8.0, 20.0, n).astype(np.float32)  # m/s, ~29-72 km/h
    return torso, arm_width, head_drop, yaw, speed


def _median_ms(fn, *, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def bench_sweep(n):
    torso, arm_width, head_drop, yaw, speed = make_params(n)
    out_cpu = np.zeros(n, dtype=np.float32)
    cpu_ms = _median_ms(
        lambda: cpu_sweep(torso, arm_width, head_drop, yaw, speed, out_cpu, n)
    )

    d_torso = metal.to_device(torso)
    d_arm = metal.to_device(arm_width)
    d_head = metal.to_device(head_drop)
    d_yaw = metal.to_device(yaw)
    d_speed = metal.to_device(speed)
    d_out = metal.device_array_like(torso)
    threads = 256
    blocks = (n + threads - 1) // threads

    def launch():
        metal_sweep[blocks, threads](d_torso, d_arm, d_head, d_yaw, d_speed, d_out, n)
        metal.synchronize()

    metal_ms = _median_ms(launch)
    metal_out = d_out.copy_to_host()
    max_rel_err = np.max(
        np.abs((out_cpu - metal_out) / np.maximum(np.abs(out_cpu), 1e-9))
    )
    return cpu_ms, metal_ms, max_rel_err


def bench_course_energy(n, n_segments=N_SEGMENTS):
    gradient, wind_yaw_offset, seg_len = make_course(n_segments)
    torso, arm_width, head_drop, base_yaw, speed = make_params(n)
    out_cpu = np.zeros(n, dtype=np.float32)
    cpu_ms = _median_ms(
        lambda: cpu_course_energy(
            torso,
            arm_width,
            head_drop,
            base_yaw,
            speed,
            gradient,
            wind_yaw_offset,
            seg_len,
            out_cpu,
            n,
            n_segments,
        )
    )

    d_torso = metal.to_device(torso)
    d_arm = metal.to_device(arm_width)
    d_head = metal.to_device(head_drop)
    d_yaw = metal.to_device(base_yaw)
    d_speed = metal.to_device(speed)
    d_gradient = metal.to_device(gradient)
    d_wind = metal.to_device(wind_yaw_offset)
    d_out = metal.device_array_like(torso)
    threads = 256
    blocks = (n + threads - 1) // threads

    def launch():
        metal_course_energy[blocks, threads](
            d_torso,
            d_arm,
            d_head,
            d_yaw,
            d_speed,
            d_gradient,
            d_wind,
            seg_len,
            d_out,
            n,
            n_segments,
        )
        metal.synchronize()

    metal_ms = _median_ms(launch)
    metal_out = d_out.copy_to_host()
    max_rel_err = np.max(
        np.abs((out_cpu - metal_out) / np.maximum(np.abs(out_cpu), 1e-6))
    )
    return cpu_ms, metal_ms, max_rel_err


if __name__ == "__main__":
    print("sweep (one instantaneous drag calc per rider -- low arithmetic intensity)")
    for n in (10_000, 100_000, 1_000_000, 10_000_000, 50_000_000):
        cpu_ms, metal_ms, err = bench_sweep(n)
        print(
            f"  n={n:>10,}: CPU={cpu_ms:9.3f}ms  Metal={metal_ms:9.3f}ms  "
            f"speedup={cpu_ms / metal_ms:6.2f}x  max_rel_err={err:.2e}"
        )

    print()
    print(
        f"course_energy ({N_SEGMENTS}-segment course integral per rider "
        "-- high arithmetic intensity)"
    )
    for n in (10_000, 100_000, 1_000_000, 5_000_000, 10_000_000):
        cpu_ms, metal_ms, err = bench_course_energy(n)
        print(
            f"  n={n:>10,}: CPU={cpu_ms:9.3f}ms  Metal={metal_ms:9.3f}ms  "
            f"speedup={cpu_ms / metal_ms:6.2f}x  max_rel_err={err:.2e}"
        )
