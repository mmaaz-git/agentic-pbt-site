# Bug Report: scipy.datasets.face() Gray Conversion Coefficients

**Target**: `scipy.datasets.face(gray=True)`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The grayscale conversion in `scipy.datasets.face(gray=True)` uses RGB coefficients (0.21, 0.71, 0.07) that sum to 0.99 instead of 1.0, causing loss of dynamic range.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(
    r=st.integers(min_value=0, max_value=255),
    g=st.integers(min_value=0, max_value=255),
    b=st.integers(min_value=0, max_value=255)
)
def test_face_gray_conversion_bounds(r, g, b):
    result = (0.21 * r + 0.71 * g + 0.07 * b)
    assert result <= 255, f"Gray value {result} exceeds 255 for RGB({r}, {g}, {b})"

def test_gray_conversion_coefficients_sum():
    coefficients = [0.21, 0.71, 0.07]
    total = sum(coefficients)
    assert abs(total - 1.0) < 1e-10, f"Coefficients sum to {total}, not 1.0"
```

**Failing input**: All RGB values at maximum (255, 255, 255)

## Reproducing the Bug

```python
coefficients = [0.21, 0.71, 0.07]
print(f"Sum: {sum(coefficients)}")

r, g, b = 255, 255, 255
gray = 0.21 * r + 0.71 * g + 0.07 * b
print(f"White RGB(255,255,255) -> {int(gray)}")
```

Output:
```
Sum: 0.99
White RGB(255,255,255) -> 252
```

## Why This Is A Bug

All standard RGB-to-grayscale conversion formulas use coefficients that sum to exactly 1.0:
- Rec. 601: 0.299 + 0.587 + 0.114 = 1.0
- Rec. 709/sRGB: 0.2126 + 0.7152 + 0.0722 = 1.0

The current coefficients appear to be incorrectly rounded from sRGB values. This causes:
1. Pure white (255, 255, 255) converts to 252 instead of 255
2. Loss of 3 gray levels from the upper range
3. Reduced dynamic range in the output

## Fix

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -220,7 +220,7 @@ def face(gray=False):
     face.shape = (768, 1024, 3)
     if gray is True:
         face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+                0.08 * face[:, :, 2]).astype('uint8')
     return face
```

Alternatively, use the more accurate sRGB coefficients:
```diff
-        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+        face = (0.2126 * face[:, :, 0] + 0.7152 * face[:, :, 1] +
+                0.0722 * face[:, :, 2]).astype('uint8')
```