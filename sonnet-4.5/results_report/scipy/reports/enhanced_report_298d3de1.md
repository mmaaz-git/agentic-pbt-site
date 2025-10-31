# Bug Report: scipy.datasets.face() Incorrect Grayscale Conversion Coefficients

**Target**: `scipy.datasets.face(gray=True)`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The grayscale conversion in `scipy.datasets.face(gray=True)` uses RGB coefficients that sum to 0.99 instead of 1.0, causing a 1.2% loss of dynamic range and incorrect grayscale values.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis-based property test for scipy.datasets.face() gray conversion coefficients"""

from hypothesis import given, strategies as st
import hypothesis

# Set up hypothesis settings for better output
hypothesis.settings.register_profile("debug", max_examples=100, verbosity=hypothesis.Verbosity.verbose)
hypothesis.settings.load_profile("debug")

@given(
    r=st.integers(min_value=0, max_value=255),
    g=st.integers(min_value=0, max_value=255),
    b=st.integers(min_value=0, max_value=255)
)
def test_face_gray_conversion_bounds(r, g, b):
    """Test that gray conversion never exceeds 255 for valid RGB inputs"""
    result = (0.21 * r + 0.71 * g + 0.07 * b)
    assert result <= 255, f"Gray value {result} exceeds 255 for RGB({r}, {g}, {b})"

def test_gray_conversion_coefficients_sum():
    """Test that grayscale conversion coefficients sum to 1.0"""
    coefficients = [0.21, 0.71, 0.07]
    total = sum(coefficients)
    assert abs(total - 1.0) < 1e-10, f"Coefficients sum to {total}, not 1.0"

if __name__ == "__main__":
    print("Running Hypothesis property-based tests for gray conversion...")
    print("=" * 60)

    # Test 1: Coefficients should sum to 1.0
    print("\nTest 1: Gray conversion coefficients sum")
    try:
        test_gray_conversion_coefficients_sum()
        print("✓ Coefficients sum test PASSED")
    except AssertionError as e:
        print(f"✗ Coefficients sum test FAILED: {e}")

    # Test 2: Gray values should never exceed 255
    print("\nTest 2: Gray conversion bounds check")
    try:
        test_face_gray_conversion_bounds()
        print("✓ Bounds test PASSED")
    except AssertionError as e:
        print(f"✗ Bounds test FAILED on first counterexample")
        print(f"  Error: {e}")

        # Find the minimal failing example
        print("\n  Looking for minimal failing example...")
        # The minimal case is when all RGB components are at maximum
        r, g, b = 255, 255, 255
        result = 0.21 * r + 0.71 * g + 0.07 * b
        print(f"  Minimal failing input: RGB({r}, {g}, {b})")
        print(f"  Result: {result} (exceeds 255 by {result - 255:.2f})")

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("The grayscale conversion coefficients (0.21, 0.71, 0.07) are flawed:")
    print(f"1. They sum to {sum([0.21, 0.71, 0.07]):.2f} instead of 1.0")
    print("2. This causes pure white (255, 255, 255) to convert to 252.45 instead of 255")
```

<details>

<summary>
**Failing input**: `Coefficients sum to 0.99, not 1.0`
</summary>
```
Running Hypothesis property-based tests for gray conversion...
============================================================

Test 1: Gray conversion coefficients sum
✗ Coefficients sum test FAILED: Coefficients sum to 0.99, not 1.0

Test 2: Gray conversion bounds check
Trying example: test_face_gray_conversion_bounds(
    r=0,
    g=0,
    b=0,
)
Trying example: test_face_gray_conversion_bounds(
    r=225,
    g=0,
    b=0,
)
Trying example: test_face_gray_conversion_bounds(
    r=225,
    g=88,
    b=49,
)
Trying example: test_face_gray_conversion_bounds(
    r=152,
    g=0,
    b=0,
)
Trying example: test_face_gray_conversion_bounds(
    r=152,
    g=58,
    b=203,
)
Trying example: test_face_gray_conversion_bounds(
    r=200,
    g=0,
    b=0,
)
Trying example: test_face_gray_conversion_bounds(
    r=200,
    g=111,
    b=34,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=0,
    b=0,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=110,
    b=77,
)
Trying example: test_face_gray_conversion_bounds(
    r=25,
    g=0,
    b=0,
)
Trying example: test_face_gray_conversion_bounds(
    r=25,
    g=247,
    b=238,
)
Trying example: test_face_gray_conversion_bounds(
    r=247,
    g=247,
    b=238,
)
Trying example: test_face_gray_conversion_bounds(
    r=247,
    g=247,
    b=247,
)
Trying example: test_face_gray_conversion_bounds(
    r=39,
    g=100,
    b=68,
)
Trying example: test_face_gray_conversion_bounds(
    r=68,
    g=100,
    b=68,
)
Trying example: test_face_gray_conversion_bounds(
    r=68,
    g=100,
    b=100,
)
Trying example: test_face_gray_conversion_bounds(
    r=100,
    g=100,
    b=100,
)
Trying example: test_face_gray_conversion_bounds(
    r=149,
    g=114,
    b=202,
)
Trying example: test_face_gray_conversion_bounds(
    r=149,
    g=114,
    b=149,
)
Trying example: test_face_gray_conversion_bounds(
    r=149,
    g=149,
    b=149,
)
Trying example: test_face_gray_conversion_bounds(
    r=42,
    g=67,
    b=100,
)
Trying example: test_face_gray_conversion_bounds(
    r=42,
    g=100,
    b=100,
)
Trying example: test_face_gray_conversion_bounds(
    r=43,
    g=177,
    b=107,
)
Trying example: test_face_gray_conversion_bounds(
    r=43,
    g=43,
    b=107,
)
Trying example: test_face_gray_conversion_bounds(
    r=43,
    g=43,
    b=43,
)
Trying example: test_face_gray_conversion_bounds(
    r=68,
    g=83,
    b=100,
)
Trying example: test_face_gray_conversion_bounds(
    r=68,
    g=68,
    b=100,
)
Trying example: test_face_gray_conversion_bounds(
    r=100,
    g=68,
    b=100,
)
Trying example: test_face_gray_conversion_bounds(
    r=128,
    g=203,
    b=204,
)
Trying example: test_face_gray_conversion_bounds(
    r=128,
    g=204,
    b=204,
)
Trying example: test_face_gray_conversion_bounds(
    r=204,
    g=204,
    b=204,
)
Trying example: test_face_gray_conversion_bounds(
    r=108,
    g=4,
    b=185,
)
Trying example: test_face_gray_conversion_bounds(
    r=185,
    g=4,
    b=185,
)
Trying example: test_face_gray_conversion_bounds(
    r=185,
    g=4,
    b=4,
)
Trying example: test_face_gray_conversion_bounds(
    r=4,
    g=4,
    b=185,
)
Trying example: test_face_gray_conversion_bounds(
    r=4,
    g=4,
    b=4,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=177,
    b=171,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=177,
    b=177,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=177,
    b=133,
)
Trying example: test_face_gray_conversion_bounds(
    r=177,
    g=177,
    b=133,
)
Trying example: test_face_gray_conversion_bounds(
    r=177,
    g=177,
    b=177,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=185,
    b=22,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=185,
    b=185,
)
Trying example: test_face_gray_conversion_bounds(
    r=185,
    g=185,
    b=185,
)
Trying example: test_face_gray_conversion_bounds(
    r=54,
    g=198,
    b=193,
)
Trying example: test_face_gray_conversion_bounds(
    r=54,
    g=54,
    b=193,
)
Trying example: test_face_gray_conversion_bounds(
    r=54,
    g=54,
    b=54,
)
Trying example: test_face_gray_conversion_bounds(
    r=183,
    g=204,
    b=68,
)
Trying example: test_face_gray_conversion_bounds(
    r=68,
    g=204,
    b=68,
)
Trying example: test_face_gray_conversion_bounds(
    r=68,
    g=204,
    b=204,
)
Trying example: test_face_gray_conversion_bounds(
    r=92,
    g=76,
    b=196,
)
Trying example: test_face_gray_conversion_bounds(
    r=92,
    g=76,
    b=76,
)
Trying example: test_face_gray_conversion_bounds(
    r=92,
    g=92,
    b=76,
)
Trying example: test_face_gray_conversion_bounds(
    r=92,
    g=92,
    b=92,
)
Trying example: test_face_gray_conversion_bounds(
    r=81,
    g=236,
    b=255,
)
Trying example: test_face_gray_conversion_bounds(
    r=81,
    g=255,
    b=255,
)
Trying example: test_face_gray_conversion_bounds(
    r=81,
    g=255,
    b=81,
)
Trying example: test_face_gray_conversion_bounds(
    r=255,
    g=255,
    b=81,
)
Trying example: test_face_gray_conversion_bounds(
    r=255,
    g=255,
    b=255,
)
Trying example: test_face_gray_conversion_bounds(
    r=208,
    g=210,
    b=255,
)
Trying example: test_face_gray_conversion_bounds(
    r=208,
    g=208,
    b=255,
)
Trying example: test_face_gray_conversion_bounds(
    r=208,
    g=208,
    b=208,
)
Trying example: test_face_gray_conversion_bounds(
    r=8,
    g=12,
    b=1,
)
Trying example: test_face_gray_conversion_bounds(
    r=8,
    g=12,
    b=12,
)
Trying example: test_face_gray_conversion_bounds(
    r=12,
    g=12,
    b=12,
)
Trying example: test_face_gray_conversion_bounds(
    r=161,
    g=178,
    b=81,
)
Trying example: test_face_gray_conversion_bounds(
    r=161,
    g=178,
    b=178,
)
Trying example: test_face_gray_conversion_bounds(
    r=178,
    g=178,
    b=178,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=84,
    b=6,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=84,
    b=84,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=84,
    b=133,
)
Trying example: test_face_gray_conversion_bounds(
    r=84,
    g=84,
    b=133,
)
Trying example: test_face_gray_conversion_bounds(
    r=84,
    g=133,
    b=133,
)
Trying example: test_face_gray_conversion_bounds(
    r=133,
    g=133,
    b=133,
)
Trying example: test_face_gray_conversion_bounds(
    r=126,
    g=155,
    b=81,
)
Trying example: test_face_gray_conversion_bounds(
    r=126,
    g=155,
    b=155,
)
Trying example: test_face_gray_conversion_bounds(
    r=155,
    g=155,
    b=155,
)
Trying example: test_face_gray_conversion_bounds(
    r=100,
    g=43,
    b=216,
)
Trying example: test_face_gray_conversion_bounds(
    r=100,
    g=100,
    b=216,
)
Trying example: test_face_gray_conversion_bounds(
    r=100,
    g=59,
    b=255,
)
Trying example: test_face_gray_conversion_bounds(
    r=255,
    g=59,
    b=255,
)
Trying example: test_face_gray_conversion_bounds(
    r=59,
    g=59,
    b=255,
)
Trying example: test_face_gray_conversion_bounds(
    r=59,
    g=59,
    b=59,
)
Trying example: test_face_gray_conversion_bounds(
    r=81,
    g=125,
    b=126,
)
Trying example: test_face_gray_conversion_bounds(
    r=81,
    g=125,
    b=125,
)
Trying example: test_face_gray_conversion_bounds(
    r=81,
    g=125,
    b=81,
)
Trying example: test_face_gray_conversion_bounds(
    r=81,
    g=81,
    b=125,
)
Trying example: test_face_gray_conversion_bounds(
    r=81,
    g=81,
    b=81,
)
Trying example: test_face_gray_conversion_bounds(
    r=100,
    g=218,
    b=231,
)
Trying example: test_face_gray_conversion_bounds(
    r=100,
    g=100,
    b=231,
)
Trying example: test_face_gray_conversion_bounds(
    r=5,
    g=63,
    b=89,
)
Trying example: test_face_gray_conversion_bounds(
    r=89,
    g=63,
    b=89,
)
Trying example: test_face_gray_conversion_bounds(
    r=89,
    g=63,
    b=63,
)
Trying example: test_face_gray_conversion_bounds(
    r=89,
    g=89,
    b=89,
)
Trying example: test_face_gray_conversion_bounds(
    r=56,
    g=179,
    b=240,
)
Trying example: test_face_gray_conversion_bounds(
    r=179,
    g=179,
    b=240,
)
Trying example: test_face_gray_conversion_bounds(
    r=240,
    g=179,
    b=240,
)
Trying example: test_face_gray_conversion_bounds(
    r=179,
    g=179,
    b=179,
)
Trying example: test_face_gray_conversion_bounds(
    r=155,
    g=135,
    b=34,
)
Trying example: test_face_gray_conversion_bounds(
    r=34,
    g=135,
    b=34,
)
✓ Bounds test PASSED

============================================================
Test Summary:
The grayscale conversion coefficients (0.21, 0.71, 0.07) are flawed:
1. They sum to 0.99 instead of 1.0
2. This causes pure white (255, 255, 255) to convert to 252.45 instead of 255
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstrate the grayscale conversion coefficient bug in scipy.datasets.face()"""

import numpy as np

# The coefficients used in scipy.datasets.face(gray=True)
coefficients = [0.21, 0.71, 0.07]
print(f"Current coefficients: {coefficients}")
print(f"Sum of coefficients: {sum(coefficients)}")
print(f"Deviation from 1.0: {1.0 - sum(coefficients)}")
print()

# Test with pure white (maximum RGB values)
r, g, b = 255, 255, 255
gray = 0.21 * r + 0.71 * g + 0.07 * b
print(f"Pure white RGB({r},{g},{b}) converts to gray value: {gray}")
print(f"As uint8: {int(gray)}")
print(f"Expected: 255")
print(f"Loss: {255 - int(gray)} gray levels")
print()

# Test with other test cases
test_cases = [
    (255, 255, 255, "Pure white"),
    (0, 0, 0, "Pure black"),
    (255, 0, 0, "Pure red"),
    (0, 255, 0, "Pure green"),
    (0, 0, 255, "Pure blue"),
    (128, 128, 128, "Mid gray"),
]

print("Test cases:")
for r, g, b, name in test_cases:
    gray = 0.21 * r + 0.71 * g + 0.07 * b
    print(f"{name:15} RGB({r:3},{g:3},{b:3}) -> Gray: {gray:7.2f} (uint8: {int(gray):3})")
print()

# Show what standard coefficients would give
print("Comparison with standard coefficients:")
print("Rec. 601: 0.299 + 0.587 + 0.114 = ", 0.299 + 0.587 + 0.114)
print("Rec. 709: 0.2126 + 0.7152 + 0.0722 = ", 0.2126 + 0.7152 + 0.0722)
print()

# Demonstrate the actual function behavior
try:
    import scipy.datasets
    print("Testing actual scipy.datasets.face() function:")

    # Get the color face image
    face_color = scipy.datasets.face(gray=False)
    print(f"Color image shape: {face_color.shape}")
    print(f"Color image dtype: {face_color.dtype}")

    # Get the grayscale face image
    face_gray = scipy.datasets.face(gray=True)
    print(f"Gray image shape: {face_gray.shape}")
    print(f"Gray image dtype: {face_gray.dtype}")

    # Find any pure white pixels in the color image
    white_pixels = np.all(face_color == 255, axis=2)
    if np.any(white_pixels):
        # Get the corresponding gray values
        gray_values_at_white = face_gray[white_pixels]
        print(f"\nFound {np.sum(white_pixels)} pure white pixels in color image")
        print(f"Their grayscale values: min={np.min(gray_values_at_white)}, max={np.max(gray_values_at_white)}")
        print("Expected: 255 for all pure white pixels")
    else:
        print("\nNo pure white pixels found in the test image")

    # Check max gray value
    print(f"\nMaximum gray value in converted image: {np.max(face_gray)}")
    print("Expected maximum (for any white pixels): 255")

except ImportError:
    print("\nNote: scipy not installed or face data not available")
except Exception as e:
    print(f"\nError testing actual function: {e}")
```

<details>

<summary>
Grayscale conversion produces incorrect values for white pixels
</summary>
```
Current coefficients: [0.21, 0.71, 0.07]
Sum of coefficients: 0.99
Deviation from 1.0: 0.010000000000000009

Pure white RGB(255,255,255) converts to gray value: 252.44999999999996
As uint8: 252
Expected: 255
Loss: 3 gray levels

Test cases:
Pure white      RGB(255,255,255) -> Gray:  252.45 (uint8: 252)
Pure black      RGB(  0,  0,  0) -> Gray:    0.00 (uint8:   0)
Pure red        RGB(255,  0,  0) -> Gray:   53.55 (uint8:  53)
Pure green      RGB(  0,255,  0) -> Gray:  181.05 (uint8: 181)
Pure blue       RGB(  0,  0,255) -> Gray:   17.85 (uint8:  17)
Mid gray        RGB(128,128,128) -> Gray:  126.72 (uint8: 126)

Comparison with standard coefficients:
Rec. 601: 0.299 + 0.587 + 0.114 =  0.9999999999999999
Rec. 709: 0.2126 + 0.7152 + 0.0722 =  1.0

Testing actual scipy.datasets.face() function:
Color image shape: (768, 1024, 3)
Color image dtype: uint8
Gray image shape: (768, 1024)
Gray image dtype: uint8

No pure white pixels found in the test image

Maximum gray value in converted image: 250
Expected maximum (for any white pixels): 255
```
</details>

## Why This Is A Bug

The grayscale conversion violates the fundamental mathematical principle that RGB-to-grayscale conversion coefficients must sum to exactly 1.0. This ensures that pure white (255, 255, 255) maps to pure white (255) in grayscale, preserving the full dynamic range.

The current coefficients (0.21, 0.71, 0.07) sum to 0.99, which causes several issues:

1. **Loss of dynamic range**: The maximum possible grayscale value is 252 instead of 255, losing 3 gray levels (1.2% of the range)
2. **Incorrect luminance mapping**: All colors are systematically darker than they should be
3. **Deviation from standards**: All standard color space conversions use coefficients summing to 1.0:
   - ITU-R Rec. 601: 0.299 + 0.587 + 0.114 = 1.0
   - ITU-R Rec. 709/sRGB: 0.2126 + 0.7152 + 0.0722 = 1.0

The coefficients appear to be incorrectly rounded versions of the sRGB/Rec. 709 standard (0.2126 ≈ 0.21, 0.7152 ≈ 0.71, 0.0722 ≈ 0.07), but the rounding was done without ensuring the sum remains 1.0.

## Relevant Context

The `scipy.datasets.face()` function is part of SciPy's example datasets module, commonly used for testing and demonstration purposes in image processing tutorials and documentation. The function loads a raccoon face image, with an optional grayscale conversion.

The relevant code is in `/scipy/datasets/_fetchers.py` lines 222-224:
```python
if gray is True:
    face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
            0.07 * face[:, :, 2]).astype('uint8')
```

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.face.html

While the actual test image doesn't contain pure white pixels (max value is 250), the mathematical incorrectness affects all color conversions proportionally, making all grayscale values about 1% darker than they should be.

## Proposed Fix

The simplest fix is to adjust one coefficient so the sum equals 1.0:

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -221,7 +221,7 @@ def face(gray=False):
     face.shape = (768, 1024, 3)
     if gray is True:
         face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+                0.08 * face[:, :, 2]).astype('uint8')
     return face
```

A better fix would use the accurate sRGB/Rec. 709 coefficients:

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -221,7 +221,7 @@ def face(gray=False):
     face.shape = (768, 1024, 3)
     if gray is True:
-        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+        face = (0.2126 * face[:, :, 0] + 0.7152 * face[:, :, 1] +
+                0.0722 * face[:, :, 2]).astype('uint8')
     return face
```