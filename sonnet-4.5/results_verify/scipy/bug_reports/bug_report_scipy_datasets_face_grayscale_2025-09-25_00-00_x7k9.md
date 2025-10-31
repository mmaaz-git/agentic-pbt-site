# Bug Report: scipy.datasets.face Grayscale Conversion Weights Sum to 0.99

**Target**: `scipy.datasets.face`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The grayscale conversion in `scipy.datasets.face(gray=True)` uses weights (0.21, 0.71, 0.07) that sum to 0.99 instead of 1.0, causing the grayscale image to be systematically darker than expected. This violates the standard luminance preservation property where white (255, 255, 255) should remain white (255) after grayscale conversion.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=255),
       st.integers(min_value=0, max_value=255),
       st.integers(min_value=0, max_value=255))
def test_grayscale_preserves_white(r, g, b):
    """Property: Pure white should convert to pure white in grayscale."""
    weights = [0.21, 0.71, 0.07]
    gray_value = int(weights[0] * r + weights[1] * g + weights[2] * b)

    if r == 255 and g == 255 and b == 255:
        assert gray_value == 255, f"White (255,255,255) converted to {gray_value}, not 255"
```

**Failing input**: `r=255, g=255, b=255`

## Reproducing the Bug

```python
import numpy as np

r, g, b = 255, 255, 255

scipy_weights = [0.21, 0.71, 0.07]
gray_value_scipy = int(scipy_weights[0] * r + scipy_weights[1] * g + scipy_weights[2] * b)

print(f"Input RGB: ({r}, {g}, {b})")
print(f"SciPy grayscale weights: {scipy_weights}")
print(f"Sum of weights: {sum(scipy_weights)}")
print(f"Expected grayscale value: 255")
print(f"Actual grayscale value: {gray_value_scipy}")
print(f"Brightness loss: {255 - gray_value_scipy}")

correct_weights = [0.2126, 0.7152, 0.0722]
gray_value_correct = int(correct_weights[0] * r + correct_weights[1] * g + correct_weights[2] * b)
print(f"\nWith ITU-R BT.709 weights {correct_weights}: {gray_value_correct}")
```

Output:
```
Input RGB: (255, 255, 255)
SciPy grayscale weights: [0.21, 0.71, 0.07]
Sum of weights: 0.99
Expected grayscale value: 255
Actual grayscale value: 252
Brightness loss: 3

With ITU-R BT.709 weights [0.2126, 0.7152, 0.0722]: 255
```

## Why This Is A Bug

Standard grayscale conversion formulas require weights that sum to 1.0 to preserve luminance:
- ITU-R BT.709: Y = 0.2126R + 0.7152G + 0.0722B (sum = 1.0)
- ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B (sum = 1.0)

The current weights (0.21, 0.71, 0.07) sum to only 0.99, violating this fundamental property. This causes:
1. Pure white (255, 255, 255) converts to 252 instead of 255
2. Systematic ~1% brightness reduction across all grayscale values
3. Inconsistent behavior with standard image processing tools

## Fix

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -220,8 +220,8 @@ def face(gray=False):
     face = frombuffer(face_data, dtype='uint8')
     face.shape = (768, 1024, 3)
     if gray is True:
-        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+        face = (0.2126 * face[:, :, 0] + 0.7152 * face[:, :, 1] +
+                0.0722 * face[:, :, 2]).astype('uint8')
     return face
```

The fix uses the ITU-R BT.709 standard weights which sum to exactly 1.0, ensuring proper luminance preservation.