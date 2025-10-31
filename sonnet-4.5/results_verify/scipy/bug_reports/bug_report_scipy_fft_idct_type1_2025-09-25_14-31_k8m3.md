# Bug Report: scipy.fft.idct Type 1 Single Element Array Crash

**Target**: `scipy.fft.idct` (type 1)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `scipy.fft.idct` with `type=1` on a single-element array raises a RuntimeError, breaking the inverse Discrete Cosine Transform for this edge case.

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
result = scipy.fft.idct(x, type=1)
```

**Output:**
```
RuntimeError: zero-length FFT requested
```

## Why This Is A Bug

The IDCT-I (Inverse Discrete Cosine Transform of type 1) should be mathematically well-defined for arrays of any size, including single-element arrays. This bug is the inverse counterpart of the DCT-I bug and prevents the round-trip property `idct(dct(x)) == x` from working. Other IDCT types (2, 3, 4) handle single-element arrays correctly.

## Fix

Similar to DCT-I, IDCT-I should special-case single-element inputs to return the identity transform. For a single element `[x]`, IDCT-I should return `[x]`.