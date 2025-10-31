# Bug Report: scipy.datasets.face() Grayscale Conversion Weights

**Target**: `scipy.datasets.face(gray=True)`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The RGB-to-grayscale conversion weights in `scipy.datasets.face()` sum to 0.99 instead of 1.0, causing grayscale images to be 1% darker than they should be.

## Property-Based Test

```python
def test_grayscale_weights_sum_to_one():
    weights = [0.21, 0.71, 0.07]
    weight_sum = sum(weights)

    assert weight_sum == 1.0, (
        f"Grayscale conversion weights {weights} sum to {weight_sum}, not 1.0. "
        f"This causes grayscale images to be {abs(1.0 - weight_sum) * 100:.1f}% "
        f"{'darker' if weight_sum < 1.0 else 'brighter'} than expected."
    )
```

**Failing input**: The hardcoded weights `[0.21, 0.71, 0.07]` in the source code

## Reproducing the Bug

```python
weights = [0.21, 0.71, 0.07]
print(f"Sum of grayscale weights: {sum(weights)}")
print(f"Expected: 1.0")
print(f"Difference: {1.0 - sum(weights)}")
```

Output:
```
Sum of grayscale weights: 0.99
Expected: 1.0
Difference: 0.010000000000000009
```

## Why This Is A Bug

Standard RGB-to-grayscale conversion formulas require weights that sum to exactly 1.0 to preserve luminance. The current implementation uses weights (0.21, 0.71, 0.07) which sum to 0.99, causing the output to be systematically darker than it should be.

Common standard weights:
- ITU-R BT.601: (0.299, 0.587, 0.114)
- ITU-R BT.709: (0.2126, 0.7152, 0.0722)

The current weights appear to be incorrectly rounded versions of BT.709.

## Fix

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -220,7 +220,7 @@ def face(gray=False):
     face = frombuffer(face_data, dtype='uint8')
     face.shape = (768, 1024, 3)
     if gray is True:
-        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+        face = (0.2126 * face[:, :, 0] + 0.7152 * face[:, :, 1] +
+                0.0722 * face[:, :, 2]).astype('uint8')
     return face
```

This uses the ITU-R BT.709 standard weights which sum to exactly 1.0.