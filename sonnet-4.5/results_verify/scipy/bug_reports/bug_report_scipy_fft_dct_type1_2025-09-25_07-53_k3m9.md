# Bug Report: scipy.fft.dct Type 1 Crashes on Single-Element Arrays

**Target**: `scipy.fft.dct`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.fft.dct` with `type=1` raises `RuntimeError: zero-length FFT requested` when given a single-element array, while other DCT types and DST type 1 handle single-element arrays correctly.

## Property-Based Test

```python
import numpy as np
import scipy.fft
from hypothesis import given, strategies as st
from numpy.testing import assert_allclose

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                min_size=1, max_size=1000),
       st.sampled_from([1, 4]))
def test_dct_self_inverse(data, dct_type):
    x = np.array(data)
    transformed = scipy.fft.dct(x, type=dct_type, norm='ortho')
    result = scipy.fft.dct(transformed, type=dct_type, norm='ortho')
    assert_allclose(result, x, rtol=1e-10, atol=1e-10)
```

**Failing input**: `data=[0.0], dct_type=1`

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

x = np.array([0.0])
result = scipy.fft.dct(x, type=1, norm='ortho')
```

**Output:**
```
RuntimeError: zero-length FFT requested
```

**Expected behavior:** Either:
1. Successfully compute the DCT (like DST type 1 does), or
2. Raise a clear error message like "DCT type 1 requires at least 2 elements"

## Why This Is A Bug

1. **Inconsistent behavior**: DCT types 2, 3, 4 all handle single-element arrays correctly, but type 1 crashes
2. **Inconsistent with DST**: `scipy.fft.dst(x, type=1)` successfully processes single-element arrays
3. **Misleading error**: The error message "zero-length FFT requested" is incorrect - the array has length 1
4. **Violates documented API**: The documentation does not mention any minimum size restriction for DCT type 1

Examples of the inconsistency:
```python
import numpy as np
import scipy.fft

x = np.array([1.0])

scipy.fft.dct(x, type=2, norm='ortho')
scipy.fft.dct(x, type=3, norm='ortho')
scipy.fft.dct(x, type=4, norm='ortho')
scipy.fft.dst(x, type=1, norm='ortho')
```

## Fix

The issue appears to be in the implementation of DCT type 1, which likely assumes at least 2 points. The fix should either:

1. **Handle single-element arrays gracefully**: Mathematically, the DCT-I of a single point could be defined as the identity or zero, depending on convention
2. **Add proper input validation**: Check array size and raise a clear `ValueError` with message "DCT type 1 requires at least 2 elements"

Recommended approach: Add input validation for consistency and clarity:

```python
if type == 1 and tmp.shape[axis] < 2:
    raise ValueError(f"DCT type 1 requires at least 2 elements, got {tmp.shape[axis]}")
```

This should be added in `/scipy/fft/_pocketfft/realtransforms.py` in the `_r2r` function before the transform is called.