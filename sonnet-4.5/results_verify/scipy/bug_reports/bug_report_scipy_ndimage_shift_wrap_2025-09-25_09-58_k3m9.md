# Bug Report: scipy.ndimage.shift with mode='wrap' is not invertible

**Target**: `scipy.ndimage.shift`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When using `scipy.ndimage.shift` with `mode='wrap'` (periodic boundary conditions), shifting by n then -n should be the identity operation, but it produces incorrect values at array boundaries even with `order=0` (nearest neighbor interpolation).

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import scipy.ndimage

@given(
    npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=1, max_dims=2, min_side=5, max_side=15),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    ),
    st.integers(min_value=1, max_value=4)
)
@settings(max_examples=100)
def test_shift_wrap_invertible(arr, shift_amount):
    shifted = scipy.ndimage.shift(arr, shift_amount, order=0, mode='wrap')
    shifted_back = scipy.ndimage.shift(shifted, -shift_amount, order=0, mode='wrap')
    assert np.array_equal(arr, shifted_back), \
        "Shift with mode='wrap' should be invertible"
```

**Failing input**: 1D array `[0., 1., 2., 3., 4.]` with `shift_amount=2`

## Reproducing the Bug

```python
import numpy as np
import scipy.ndimage

arr = np.array([0., 1., 2., 3., 4.])
shifted = scipy.ndimage.shift(arr, 2, order=0, mode='wrap')
shifted_back = scipy.ndimage.shift(shifted, -2, order=0, mode='wrap')

print("Original:     ", arr)
print("After roundtrip:", shifted_back)
print("Expected:     ", arr)
print("Match:", np.array_equal(arr, shifted_back))
```

Output:
```
Original:      [0. 1. 2. 3. 4.]
After roundtrip: [0. 1. 2. 3. 0.]
Expected:      [0. 1. 2. 3. 4.]
Match: False
```

## Why This Is A Bug

With `mode='wrap'`, the array is treated as periodic. Shifting by n positions then shifting back by -n positions should be the identity operation, especially with `order=0` (nearest neighbor) where no interpolation occurs. The last element should be `4.` but is incorrectly `0.`.

This violates the fundamental property of periodic boundary conditions and makes the shift operation unexpectedly non-invertible for users who rely on wrap mode for periodic data (e.g., signals, textures, simulations with periodic boundaries).

## Fix

The bug appears to be in how scipy.ndimage handles the boundary wrapping logic in the shift function. The fix would require examining the C implementation of the shift function to ensure that wrap mode correctly handles the periodic boundary conditions in both forward and backward directions.

A high-level fix would ensure that for integer shifts with order=0 and mode='wrap', the operation `shift(shift(arr, n), -n)` is exactly equal to `arr` for all valid shift values.