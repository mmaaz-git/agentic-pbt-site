# Bug Report: scipy.datasets.face() Grayscale Conversion Incorrect Due to Weight Sum

**Target**: `scipy.datasets.face(gray=True)`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The grayscale conversion in `scipy.datasets.face(gray=True)` uses weights (0.21, 0.71, 0.07) that sum to 0.99 instead of 1.0, causing systematic underestimation of brightness, particularly visible for dark pixels where RGB=(1,1,1) incorrectly converts to gray=0 instead of gray=1.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings

@given(
    r=st.integers(min_value=0, max_value=255),
    g=st.integers(min_value=0, max_value=255),
    b=st.integers(min_value=0, max_value=255)
)
@settings(max_examples=1000)
def test_grayscale_formula_overflow(r, g, b):
    """
    Grayscale value should be bounded by min and max of RGB values.
    Fails for RGB=(1,1,1) which produces gray=0 instead of gray=1.
    """
    expected_gray_float = 0.21 * r + 0.71 * g + 0.07 * b

    rgb_array = np.array([[[r, g, b]]], dtype='uint8')

    gray = (0.21 * rgb_array[:, :, 0] +
            0.71 * rgb_array[:, :, 1] +
            0.07 * rgb_array[:, :, 2]).astype('uint8')

    actual_gray = gray[0, 0]

    min_val = min(r, g, b)
    max_val = max(r, g, b)

    assert min_val <= actual_gray <= max_val, \
        f"Gray {actual_gray} outside [{min_val}, {max_val}] " + \
        f"for RGB=({r}, {g}, {b}), expected_float={expected_gray_float:.2f}"
```

**Failing input**: `r=1, g=1, b=1`

## Reproducing the Bug

```python
import numpy as np

r, g, b = 1, 1, 1

rgb_array = np.array([[[r, g, b]]], dtype='uint8')

gray = (0.21 * rgb_array[:, :, 0] +
        0.71 * rgb_array[:, :, 1] +
        0.07 * rgb_array[:, :, 2]).astype('uint8')

actual_gray = gray[0, 0]

print(f"RGB = ({r}, {g}, {b})")
print(f"Expected grayscale (float) = 0.21 * {r} + 0.71 * {g} + 0.07 * {b} = {0.21 * r + 0.71 * g + 0.07 * b}")
print(f"Actual grayscale (uint8) = {actual_gray}")

for val in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 255]:
    r, g, b = val, val, val
    rgb_array = np.array([[[r, g, b]]], dtype='uint8')
    gray = (0.21 * rgb_array[:, :, 0] +
            0.71 * rgb_array[:, :, 1] +
            0.07 * rgb_array[:, :, 2]).astype('uint8')
    actual_gray = gray[0, 0]
    expected_float = 0.21 * r + 0.71 * g + 0.07 * b
    print(f"RGB=({val:3d}, {val:3d}, {val:3d}) -> gray={actual_gray:3d}, expected_float={expected_float:6.2f}, diff={val - actual_gray}")
```

Output:
```
RGB = (1, 1, 1)
Expected grayscale (float) = 0.21 * 1 + 0.71 * 1 + 0.07 * 1 = 0.99
Actual grayscale (uint8) = 0

RGB=(  1,   1,   1) -> gray=  0, expected_float=  0.99, diff=1
RGB=(  2,   2,   2) -> gray=  1, expected_float=  1.98, diff=1
RGB=(  3,   3,   3) -> gray=  2, expected_float=  2.97, diff=1
RGB=(  4,   4,   4) -> gray=  3, expected_float=  3.96, diff=1
RGB=(  5,   5,   5) -> gray=  4, expected_float=  4.95, diff=1
RGB=( 10,  10,  10) -> gray=  9, expected_float=  9.90, diff=1
RGB=( 20,  20,  20) -> gray= 19, expected_float= 19.80, diff=1
RGB=( 50,  50,  50) -> gray= 49, expected_float= 49.50, diff=1
RGB=(100, 100, 100) -> gray= 99, expected_float= 99.00, diff=1
RGB=(200, 200, 200) -> gray=198, expected_float=198.00, diff=2
RGB=(255, 255, 255) -> gray=252, expected_float=252.45, diff=3
```

## Why This Is A Bug

The grayscale conversion uses the ITU-R BT.601 luminance formula, but with incorrect weights. The standard formula uses weights (0.299, 0.587, 0.114) which sum to 1.0. However, scipy uses (0.21, 0.71, 0.07) which sum to 0.99.

This causes several issues:
1. **Brightness underestimation**: All grayscale values are systematically 1-3 units lower than expected
2. **Violates mathematical property**: For grayscale images (R=G=B), the conversion should preserve the value, but it doesn't
3. **Extreme case failure**: RGB=(1,1,1) produces gray=0, which is incorrect (should be 1)
4. **Loss of dynamic range**: White (255,255,255) becomes (252) instead of (255), losing 3 levels of brightness

## Fix

The weights should sum to 1.0. The standard ITU-R BT.601 weights are (0.299, 0.587, 0.114), or if the current weights are intentional, they should be normalized to sum to 1.0:

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -213,8 +213,8 @@ def face(gray=False):
     face_data = bz2.decompress(rawdata)
     face = frombuffer(face_data, dtype='uint8')
     face.shape = (768, 1024, 3)
     if gray is True:
-        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+        face = (0.299 * face[:, :, 0] + 0.587 * face[:, :, 1] +
+                0.114 * face[:, :, 2]).astype('uint8')
     return face
```

Alternatively, if the current weights are intentional, normalize them:

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -213,8 +213,9 @@ def face(gray=False):
     face_data = bz2.decompress(rawdata)
     face = frombuffer(face_data, dtype='uint8')
     face.shape = (768, 1024, 3)
     if gray is True:
-        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+        w = [0.21, 0.71, 0.07]
+        w = [x/sum(w) for x in w]  # Normalize to sum to 1.0
+        face = (w[0] * face[:, :, 0] + w[1] * face[:, :, 1] +
+                w[2] * face[:, :, 2]).astype('uint8')
     return face
```