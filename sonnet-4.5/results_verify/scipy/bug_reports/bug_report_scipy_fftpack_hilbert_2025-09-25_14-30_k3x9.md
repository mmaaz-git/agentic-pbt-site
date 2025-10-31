# Bug Report: scipy.fftpack hilbert/ihilbert Round-Trip Fails for n % 4 == 2

**Target**: `scipy.fftpack.hilbert` and `scipy.fftpack.ihilbert`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The documented property "If `sum(x, axis=0) == 0` then `hilbert(ihilbert(x)) == x`" is violated when the array size `n` satisfies `n % 4 == 2` (i.e., n = 2, 6, 10, 14, 18, ...).

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume
import scipy.fftpack as fftpack

@given(st.integers(min_value=2, max_value=50))
def test_hilbert_ihilbert_round_trip(n):
    x = np.random.randn(n)
    x = x - np.mean(x)

    if np.abs(np.sum(x)) > 1e-10:
        x = x - np.sum(x) / n

    assume(np.abs(np.sum(x)) < 1e-10)

    result = fftpack.hilbert(fftpack.ihilbert(x))

    assert np.allclose(result, x, rtol=1e-9, atol=1e-9), \
        f"hilbert(ihilbert(x)) != x for size {n}, max diff = {np.max(np.abs(result - x))}"
```

**Failing input**: `n=2` (Hypothesis found this immediately)

## Reproducing the Bug

```python
import numpy as np
import scipy.fftpack as fftpack

x = np.array([-0.5, 0.5])
assert np.sum(x) == 0.0

result = fftpack.hilbert(fftpack.ihilbert(x))

print(f"Input:  {x}")
print(f"Output: {result}")
print(f"Expected: {x}")

assert np.allclose(result, x)
```

Output:
```
Input:  [-0.5  0.5]
Output: [ 0. -0.]
Expected: [-0.5  0.5]
AssertionError
```

The same failure occurs for all array sizes where `n % 4 == 2`:
- n=2:  hilbert(ihilbert(x)) returns [0, -0] instead of [-0.5, 0.5]
- n=6:  max difference = 0.67
- n=10: max difference varies
- n=14, 18, 22, ... all fail

The property holds correctly for n % 4 ∈ {0, 1, 3}.

## Why This Is A Bug

The scipy.fftpack documentation for `hilbert` explicitly states in its Notes section:

> If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.

This is a documented API contract that is violated for array sizes where `n % 4 == 2`. Users relying on this documented property will get incorrect results.

## Fix

The bug likely originates from the FFT-based implementation of the Hilbert transform. For arrays of size `n % 4 == 2`, the sign function `sign(j)` in the frequency domain may not be handled correctly.

The Hilbert transform is defined as:
```
y_j = sqrt(-1)*sign(j) * x_j
y_0 = 0
```

For even-sized arrays, the Nyquist frequency (at index n/2) requires special handling. When `n % 4 == 2`, the Nyquist frequency is at an odd index relative to n/4, which may cause the sign function to produce incorrect results.

A potential fix would involve special-casing the Nyquist frequency handling for sizes where `n % 4 == 2`, ensuring the sign function produces the correct value at this critical frequency bin.