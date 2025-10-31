# Bug Report: numpy.fft hfft/ihfft Even Case Round-Trip Failure

**Target**: `numpy.fft.hfft` and `numpy.fft.ihfft`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The documented round-trip property for the even case `ihfft(hfft(a, 2*len(a) - 2)) == a` fails, with the last element of the Hermitian array losing its imaginary part.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings
from hypothesis.extra import numpy as npst

float_strategy = npst.arrays(
    dtype=np.float64,
    shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
    elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100)
)

@given(float_strategy)
@settings(max_examples=1000)
def test_hfft_ihfft_even_roundtrip(real_signal):
    herm = np.fft.ifft(real_signal)
    n_even = 2 * len(herm) - 2
    result = np.fft.ihfft(np.fft.hfft(herm, n_even))
    np.testing.assert_allclose(result, herm, rtol=1e-7, atol=1e-8)
```

**Failing input**: Hermitian array `[ 2.+0.j, -0.5-0.289j, -0.5+0.289j]` (from `ifft([1, 2, 3])`)

## Reproducing the Bug

```python
import numpy as np

a = np.fft.ifft([1.0, 2.0, 3.0])
n = 2 * len(a) - 2

result = np.fft.ihfft(np.fft.hfft(a, n))

print(f"Expected: {a}")
print(f"Got:      {result}")
print(f"Match: {np.allclose(result, a)}")
```

Output:
```
Expected: [ 2. +0.j         -0.5-0.28867513j -0.5+0.28867513j]
Got:      [ 2. -0.j         -0.5-0.28867513j -0.5-0.j        ]
Match: False
```

## Why This Is A Bug

The `numpy.fft.hfft` docstring explicitly states:

> even: `ihfft(hfft(a, 2*len(a) - 2)) == a`, within roundoff error

However, the last element of the Hermitian array loses its imaginary part (0.289 imaginary component becomes 0), which is far beyond roundoff error. This violates the documented contract.

The odd case (`ihfft(hfft(a, 2*len(a) - 1)) == a`) works correctly with only roundoff-level errors (~1e-16).

## Fix

This appears to be a documentation bug rather than an implementation bug. The Notes section mentions that for even-length outputs, the Nyquist frequency component is "treated as purely real" which causes information loss. The even case formula in the docstring should either:

1. Be removed (only the odd case truly preserves information)
2. Be qualified with additional constraints on the input (e.g., "when a[-1] is real")
3. The implementation should be fixed to preserve the imaginary component

The most appropriate fix would be to update the documentation to clarify that the even-case round-trip only holds when the Hermitian input has a real last element, or to remove the even case claim entirely.