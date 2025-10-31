# Bug Report: scipy.ndimage.rotate - Data Loss on Non-Square Arrays

**Target**: `scipy.ndimage.rotate`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Rotating a non-square array by 90 degrees four times with `reshape=False` causes significant data loss instead of returning to the original array, violating the mathematical property that 360-degree rotation should be identity.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy import ndimage
from hypothesis.extra.numpy import arrays

@given(
    arr=arrays(dtype=np.float64, shape=st.tuples(
        st.integers(2, 10),
        st.integers(2, 10)
    ), elements=st.floats(
        min_value=-1e6, max_value=1e6,
        allow_nan=False, allow_infinity=False
    ))
)
@settings(max_examples=200)
def test_rotate_90_four_times_identity(arr):
    """
    Property: Rotating by 90 degrees 4 times should return the original array

    Evidence: Four 90-degree rotations equal a 360-degree rotation, which
    should be the identity transformation.
    """
    result = arr
    for _ in range(4):
        result = ndimage.rotate(result, 90, reshape=False)

    assert np.allclose(arr, result)
```

**Failing input**: `array([[1., 1., 1.], [1., 1., 1.]])`

## Reproducing the Bug

```python
import numpy as np
from scipy import ndimage

arr = np.array([[1., 1., 1.],
                [1., 1., 1.]])

result = arr.copy()
for i in range(4):
    result = ndimage.rotate(result, 90, reshape=False)

print("Original:", arr)
print("After 4x 90° rotations:", result)
```

**Output:**
```
Original: [[1. 1. 1.]
 [1. 1. 1.]]
After 4x 90° rotations: [[0.    0.125 0.   ]
 [0.    0.125 0.   ]]
```

## Why This Is A Bug

1. **Violates mathematical properties**: Four 90-degree rotations equal a 360-degree rotation, which should be the identity transformation. However, a single 360-degree rotation DOES work correctly, showing internal inconsistency.

2. **Significant data corruption**: Values degrade exponentially with each rotation (1.0 → 0.5 → 0.25 → 0.125), causing complete loss of information.

3. **Affects all non-square arrays**: The bug occurs for ANY non-square array shape (2x3, 3x2, 3x4, etc.) when using `reshape=False`.

4. **Silent failure**: The function doesn't warn users that their data is being corrupted, making this a dangerous silent bug.

5. **Works correctly in analogous cases**:
   - Square arrays: 4x 90° rotations preserve the array ✓
   - Non-square with `reshape=True`: 4x 90° rotations preserve the array ✓
   - Single 360° rotation: Preserves the array ✓
   - **Non-square with `reshape=False`: FAILS** ✗

## Fix

The issue occurs because with `reshape=False` on non-square arrays, each 90-degree rotation tries to fit a rotated array into the original shape, causing clipping/padding that accumulates error. The function should either:

1. **Preferred**: Make reshape=False work correctly by properly handling the coordinate transformation
2. **Alternative**: Raise a warning or error when `reshape=False` is used with non-square arrays and rotation angles that would change dimensions (90, 270 degrees, etc.)

The root cause appears to be in how the function handles boundary conditions when the rotated array doesn't fit the output shape. Each rotation compounds the interpolation error from boundary handling, leading to exponential degradation.