"""Metal device discovery and platform capability checks.

This module is the only place that decides whether numba-metal can run at
all on the current machine. It fails fast and loud on unsupported hardware
or operating systems, per the project's "no silent fallback" requirement.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from numba_metal.errors import MetalToolchainError, UnsupportedPlatformError

_MIN_MACOS_MAJOR = 14


@dataclass(frozen=True)
class DeviceInfo:
    """Static information about the selected Metal device and host."""

    name: str
    has_unified_memory: bool
    max_threads_per_threadgroup: int
    registry_id: int
    macos_version: str
    machine: str


def _check_platform() -> None:
    system = platform.system()
    if system != "Darwin":
        raise UnsupportedPlatformError(
            f"numba-metal requires macOS; detected platform {system!r}."
        )
    machine = platform.machine()
    if machine != "arm64":
        raise UnsupportedPlatformError(
            "numba-metal requires Apple-silicon (arm64) Macs; detected "
            f"machine architecture {machine!r}. Intel Macs are not "
            "supported (no Apple GPU / limited Metal compute feature set)."
        )
    mac_ver = platform.mac_ver()[0]
    try:
        major = int(mac_ver.split(".")[0])
    except (ValueError, IndexError) as exc:
        raise UnsupportedPlatformError(
            f"Could not parse macOS version {mac_ver!r}."
        ) from exc
    if major < _MIN_MACOS_MAJOR:
        raise UnsupportedPlatformError(
            f"numba-metal requires macOS {_MIN_MACOS_MAJOR} or newer; "
            f"detected macOS {mac_ver}."
        )


@lru_cache(maxsize=1)
def get_device_info() -> DeviceInfo:
    """Return static info about the default Metal device.

    Raises UnsupportedPlatformError if the platform is not a supported
    Apple-silicon Mac, or if no Metal device is found.
    """
    _check_platform()

    try:
        import Metal
    except ImportError as exc:
        raise UnsupportedPlatformError(
            "pyobjc-framework-Metal is not installed. Install numba-metal's "
            "declared dependencies (`pip install numba-metal`) or, in a "
            "development checkout, `pip install -e .`."
        ) from exc

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise UnsupportedPlatformError(
            "No Metal device found (MTLCreateSystemDefaultDevice returned "
            "None). numba-metal requires a Mac with an Apple-silicon GPU."
        )

    return DeviceInfo(
        name=str(device.name()),
        has_unified_memory=bool(device.hasUnifiedMemory()),
        max_threads_per_threadgroup=int(device.maxThreadsPerThreadgroup().width),
        registry_id=int(device.registryID()),
        macos_version=platform.mac_ver()[0],
        machine=platform.machine(),
    )


@lru_cache(maxsize=1)
def check_metal_compiler() -> str:
    """Verify the Metal shader compiler (metal/metallib) is usable.

    Returns the compiler version string on success. Raises
    MetalToolchainError with remediation instructions otherwise. This is a
    separate, explicit check from device discovery because on a freshly
    installed Xcode, the Metal Toolchain component may not yet be
    downloaded even though a GPU device is present.
    """
    try:
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MetalToolchainError(
            "Could not invoke `xcrun -sdk macosx metal --version`. Install "
            "Xcode command-line tools with `xcode-select --install`."
        ) from exc

    if result.returncode != 0:
        raise MetalToolchainError(
            "The Metal shader compiler is not available "
            f"(xcrun exit code {result.returncode}).\n"
            f"{result.stderr.strip()}\n"
            "If this mentions a missing Metal Toolchain component, run:\n"
            "    xcodebuild -downloadComponent MetalToolchain"
        )
    return result.stdout.strip() or result.stderr.strip()


def check_capable() -> DeviceInfo:
    """Run all platform/device/toolchain/Numba-version checks; return
    DeviceInfo on success.

    This is the single entry point `metal.jit` and `metal` module import
    should use to fail fast with a clear error on unsupported machines
    or an unvalidated Numba version.
    """
    from numba_metal.compat import check_numba_compatible

    check_numba_compatible()
    info = get_device_info()
    check_metal_compiler()
    return info
