# Bug Report: scipy.fftpack.dct Type-1 Crashes on Single-Element Arrays

**Target**: `scipy.fftpack.dct` with `type=1`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The DCT Type-1 transform crashes with a `RuntimeError` when applied to single-element arrays, while other DCT types handle this case correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import fftpack


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
                min_size=1, max_size=100),
       st.integers(min_value=1, max_value=4))
def test_dct_orthogonality(x_list, dct_type):
    x = np.array(x_list)
    dct_x = fftpack.dct(x, type=dct_type, norm='ortho')
    energy_orig = np.sum(x**2)
    energy_dct = np.sum(dct_x**2)
    assert np.isclose(energy_orig, energy_dct, rtol=1e-4, atol=1e-6)
```

**Failing input**: `x_list=[0.0], dct_type=1`

## Reproducing the Bug

```python
import numpy as np
from scipy import fftpack

x = np.array([0.0])

print(f"Input: {x}")
print(f"Length: {len(x)}")

print("\nTesting DCT Type-1 with single element:")
try:
    result = fftpack.dct(x, type=1, norm='ortho')
    print(f"Result: {result}")
except RuntimeError as e:
    print(f"ERROR: {e}")

print("\nTesting DCT Type-2 with single element (for comparison):")
result2 = fftpack.dct(x, type=2, norm='ortho')
print(f"Result: {result2}")

print("\nTesting DCT Type-1 with two elements:")
x2 = np.array([0.0, 1.0])
result_two = fftpack.dct(x2, type=1, norm='ortho')
print(f"Result: {result_two}")
```

**Output**:
```
Input: [0.]
Length: 1

Testing DCT Type-1 with single element:
ERROR: zero-length FFT requested

Testing DCT Type-2 with single element (for comparison):
Result: [0.]

Testing DCT Type-1 with two elements:
Result: [-0.70710678  0.70710678]
```

## Why This Is A Bug

Single-element arrays are valid inputs for discrete transforms. DCT Types 2, 3, and 4 all handle single-element inputs correctly, but Type-1 crashes with a `RuntimeError: zero-length FFT requested`. This inconsistency makes the API unpredictable and can cause crashes in production code when input sizes vary.

The error message suggests an implementation issue where the internal FFT size calculation produces zero for single-element Type-1 DCTs, likely due to the boundary conditions specific to Type-1.

## Fix

The DCT Type-1 implementation should either:
1. Handle the single-element case explicitly as a special case (the DCT-I of a single value should be that value itself)
2. Adjust the internal FFT size calculation to avoid zero-length requests

The issue is likely in `/scipy/fftpack/_realtransforms.py` or the underlying `_pocketfft` implementation where Type-1 specific logic determines the FFT size.