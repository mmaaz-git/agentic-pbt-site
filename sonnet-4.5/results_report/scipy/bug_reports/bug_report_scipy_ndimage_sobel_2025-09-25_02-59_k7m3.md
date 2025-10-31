# Bug Report: scipy.ndimage.sobel Incorrect Gradient on Constant Arrays

**Target**: `scipy.ndimage.sobel`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.ndimage.sobel` function incorrectly returns non-zero values on constant arrays when using `mode='constant'`, even when `cval` matches the constant value. The gradient of a constant function should always be zero.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.ndimage as ndi

@given(
    value=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    rows=st.integers(5, 15),
    cols=st.integers(5, 15)
)
@settings(max_examples=50, deadline=None)
def test_sobel_constant_zero(value, rows, cols):
    """
    Property: sobel(constant_image) = 0

    Sobel filter detects edges. A constant image has no edges,
    so the result should be zero everywhere.
    """
    x = np.full((rows, cols), value, dtype=np.float64)
    result = ndi.sobel(x, mode='constant', cval=value)

    assert np.allclose(result, 0.0, atol=1e-10), \
        f"Sobel on constant not zero: max = {np.max(np.abs(result))}"
```

**Failing input**: `value=5.0`, any array shape (e.g., 10x10)

## Reproducing the Bug

```python
import numpy as np
import scipy.ndimage as ndi

arr = np.full((10, 10), 5.0, dtype=np.float64)

result = ndi.sobel(arr, mode='constant', cval=5.0)

print("Input: constant 5.0 array")
print(f"Result:\n{result}")

print(f"\nExpected: All zeros")
print(f"Actual: Top row = {result[0, :]}")
print(f"        Bottom row = {result[-1, :]}")
print(f"Max value: {np.max(np.abs(result))}")

# Output shows:
# Top row = [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
# Bottom row = [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
# Max value: 5.0
#
# But with mode='nearest', it works correctly:
result_nearest = ndi.sobel(arr, mode='nearest')
print(f"\nWith mode='nearest': max = {np.max(np.abs(result_nearest)):.2e}")
# Output: 0.00e+00 ✓
```

## Why This Is A Bug

The Sobel filter approximates the gradient magnitude of an image. For a constant function f(x,y) = c:

- ∂f/∂x = 0 (derivative of constant is zero)
- ∂f/∂y = 0
- Sobel should return √(0² + 0²) = 0 everywhere

This is a fundamental mathematical property. When the user provides `mode='constant', cval=5.0` for a constant 5.0 array, they are explicitly telling scipy that the padding should also be 5.0, creating a uniform field with no edges.

**Current behavior**: Returns the constant value along the entire boundary (top and bottom rows, left and right columns)

**Expected behavior**: Returns all zeros

**Impact**:
- False edge detection in image processing pipelines
- Incorrect gradient computations for uniform regions
- mode='nearest', 'reflect', and 'mirror' work correctly; only mode='constant' is affected

## Fix

The issue likely stems from how the Sobel filter handles boundaries with `mode='constant'`. Looking at the pattern where the boundary rows/columns equal the constant value, it appears the implementation may be incorrectly handling the case where the image value matches `cval`.

Investigation shows:
- `sobel(axis=0)` returns non-zero at left and right columns (corners)
- `sobel(axis=1)` returns non-zero at top and bottom rows (corners)
- When combined (default axis=-1), the entire boundary gets non-zero values

The fix should ensure that when computing gradients at boundaries:
1. If the image is constant and `cval` matches that constant, the gradient is correctly computed as zero
2. The boundary handling logic should recognize that there's no actual edge when padding with the same value as the image

A workaround for users: Use `mode='nearest'`, `mode='reflect'`, or `mode='mirror'` instead of `mode='constant'` when processing images with uniform regions.