# Bug Report: scipy.datasets.face Grayscale Conversion Coefficients

**Target**: `scipy.datasets.face`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The grayscale conversion coefficients in `scipy.datasets.face(gray=True)` sum to 0.99 instead of 1.0, causing a 1% loss in overall image brightness.

## Property-Based Test

```python
import inspect
import scipy.datasets


def test_grayscale_coefficients_sum():
    """Grayscale coefficients should sum to 1.0 for brightness preservation"""
    source = inspect.getsource(scipy.datasets.face)

    # Extract coefficients: 0.21, 0.71, 0.07
    import re
    pattern = r'(\d+\.\d+)\s*\*\s*face\[:,\s*:,\s*0\]\s*\+\s*(\d+\.\d+)\s*\*\s*face\[:,\s*:,\s*1\]\s*\+\s*(\d+\.\d+)\s*\*\s*face\[:,\s*:,\s*2\]'
    match = re.search(pattern, source)

    r_coeff, g_coeff, b_coeff = float(match.group(1)), float(match.group(2)), float(match.group(3))
    total = r_coeff + g_coeff + b_coeff

    assert abs(total - 1.0) < 0.001, \
        f"Coefficients ({r_coeff}, {g_coeff}, {b_coeff}) sum to {total}, expected 1.0"
```

**Failing input**: The source code itself contains the bug.

## Reproducing the Bug

```python
import inspect
import scipy.datasets

source = inspect.getsource(scipy.datasets.face)
print(source)

# Extract the grayscale conversion line:
# face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
#         0.07 * face[:, :, 2]).astype('uint8')

coefficients = [0.21, 0.71, 0.07]
total = sum(coefficients)
print(f"Coefficient sum: {total}")  # Outputs: 0.99

# Standard grayscale conversions:
# Rec. 601: 0.299 + 0.587 + 0.114 = 1.0
# Rec. 709: 0.2126 + 0.7152 + 0.0722 = 1.0

# The current implementation loses 1% brightness
```

## Why This Is A Bug

Standard RGB to grayscale conversions use coefficients that sum to exactly 1.0 to preserve overall image brightness. The two most common standards are:

1. **Rec. 601 (NTSC)**: 0.299R + 0.587G + 0.114B = 1.0
2. **Rec. 709 (HDTV)**: 0.2126R + 0.7152G + 0.0722B = 1.0

The current implementation uses 0.21R + 0.71G + 0.07B = **0.99**, which causes a 1% reduction in overall brightness when converting to grayscale.

Looking at the coefficients more closely:
- 0.21 ≈ 0.2126 (close to Rec. 709 red coefficient)
- 0.71 ≈ 0.7152 (close to Rec. 709 green coefficient)
- 0.07 ≠ 0.0722 (should be ~0.08 to match the pattern)

This suggests the last coefficient should be **0.08** instead of 0.07. The pattern 0.21 + 0.71 + 0.08 = 1.00 maintains brightness preservation while using simplified (rounded) coefficients similar to Rec. 709.

## Fix

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -220,5 +220,5 @@ def face(gray=False):
     face.shape = (768, 1024, 3)
     if gray is True:
         face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+                0.08 * face[:, :, 2]).astype('uint8')
     return face
```

This changes the blue coefficient from 0.07 to 0.08, making the coefficients sum to 1.0 and preserving image brightness during grayscale conversion.