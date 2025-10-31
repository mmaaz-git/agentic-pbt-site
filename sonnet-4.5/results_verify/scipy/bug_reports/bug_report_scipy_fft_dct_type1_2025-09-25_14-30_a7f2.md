# Bug Report: scipy.fft.dct Type 1 Single Element Array Crash

**Target**: `scipy.fft.dct` (type 1)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `scipy.fft.dct` with `type=1` on a single-element array raises a RuntimeError, despite single-element arrays being valid mathematical inputs for the Discrete Cosine Transform.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.fft


@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        min_size=1,
        max_size=100
    ),
    st.sampled_from([1, 2, 3, 4])
)
def test_dct_idct_roundtrip(x, dct_type):
    x_arr = np.array(x)
    result = scipy.fft.idct(scipy.fft.dct(x_arr, type=dct_type), type=dct_type)
    assert np.allclose(result, x_arr, rtol=1e-9, atol=1e-8)
```

**Failing input**: `x=[0.0], dct_type=1`

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

x = np.array([1.0])
result = scipy.fft.dct(x, type=1)
```

**Output:**
```
RuntimeError: zero-length FFT requested
```

## Why This Is A Bug

The DCT-I (Discrete Cosine Transform of type 1) is mathematically well-defined for arrays of any size, including single-element arrays. Other DCT types (2, 3, 4) handle single-element arrays correctly, returning valid results. This inconsistency makes type 1 unusable for generic code that processes arrays of varying sizes.

Additionally, the inverse operation `scipy.fft.idct` with `type=1` has the same issue, preventing the fundamental round-trip property `idct(dct(x)) == x` from working.

## Fix

The issue appears to stem from DCT-I's implementation internally using an FFT of size `2*(n-1)`, which becomes 0 for `n=1`. The fix should special-case single-element inputs:

For DCT-I with a single element `[x]`, the output should be `[x]` (identity transform).
For IDCT-I with a single element `[x]`, the output should also be `[x]`.

This matches the mathematical definition of DCT-I and ensures consistency with the other DCT types.