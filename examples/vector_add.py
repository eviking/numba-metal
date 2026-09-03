"""Minimal end-to-end example: the exact program from the project README,
runnable directly on a supported Apple-silicon Mac.

    python examples/vector_add.py
"""

from __future__ import annotations

import numpy as np

from numba_metal import metal


@metal.jit
def vector_add(a, b, output):
    i = metal.grid(1)
    if i < output.size:
        output[i] = a[i] + b[i]


def main() -> None:
    a = np.arange(1_000_000, dtype=np.float32)
    b = np.arange(1_000_000, dtype=np.float32)

    d_a = metal.to_device(a)
    d_b = metal.to_device(b)
    d_output = metal.device_array_like(a)

    threads = 256
    blocks = (a.size + threads - 1) // threads
    vector_add[blocks, threads](d_a, d_b, d_output)
    metal.synchronize()

    output = d_output.copy_to_host()

    expected = a + b
    assert np.allclose(output, expected)
    print(f"vector_add({a.size:,} elements): OK")
    print(f"output[:5]  = {output[:5]}")
    print(f"output[-5:] = {output[-5:]}")


if __name__ == "__main__":
    main()
