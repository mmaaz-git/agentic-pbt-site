# Bug Report: scipy.datasets.face() Grayscale Conversion Uses Incorrect Weight Sum

**Target**: `scipy.datasets.face(gray=True)`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The grayscale conversion in `scipy.datasets.face(gray=True)` uses RGB weights (0.21, 0.71, 0.07) that sum to 0.99 instead of 1.0, causing systematic brightness underestimation where all grayscale values are 1-3 units darker than mathematically correct, with the extreme case of RGB=(1,1,1) incorrectly converting to gray=0.

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

# Run the test
if __name__ == "__main__":
    test_grayscale_formula_overflow()
```

<details>

<summary>
**Failing input**: `r=1, g=1, b=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 34, in <module>
    test_grayscale_formula_overflow()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 5, in test_grayscale_formula_overflow
    r=st.integers(min_value=0, max_value=255),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 28, in test_grayscale_formula_overflow
    assert min_val <= actual_gray <= max_val, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Gray 0 outside [1, 1] for RGB=(1, 1, 1), expected_float=0.99
Falsifying example: test_grayscale_formula_overflow(
    r=1,
    g=1,
    b=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/33/hypo.py:29
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Test case demonstrating the grayscale conversion bug
r, g, b = 1, 1, 1

# Create a single pixel RGB image
rgb_array = np.array([[[r, g, b]]], dtype='uint8')

# Apply the same formula used in scipy.datasets.face(gray=True)
gray = (0.21 * rgb_array[:, :, 0] +
        0.71 * rgb_array[:, :, 1] +
        0.07 * rgb_array[:, :, 2]).astype('uint8')

actual_gray = gray[0, 0]

print(f"RGB = ({r}, {g}, {b})")
print(f"Expected grayscale (float) = 0.21 * {r} + 0.71 * {g} + 0.07 * {b} = {0.21 * r + 0.71 * g + 0.07 * b}")
print(f"Actual grayscale (uint8) = {actual_gray}")
print(f"Weight sum = 0.21 + 0.71 + 0.07 = {0.21 + 0.71 + 0.07}")
print()

# Show the issue across multiple gray values
print("Demonstration of systematic underestimation:")
print("RGB Value -> Gray Value (Expected Float vs Actual uint8)")
print("-" * 50)
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

<details>

<summary>
Output demonstrating systematic brightness underestimation
</summary>
```
RGB = (1, 1, 1)
Expected grayscale (float) = 0.21 * 1 + 0.71 * 1 + 0.07 * 1 = 0.99
Actual grayscale (uint8) = 0
Weight sum = 0.21 + 0.71 + 0.07 = 0.99

Demonstration of systematic underestimation:
RGB Value -> Gray Value (Expected Float vs Actual uint8)
--------------------------------------------------
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
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Mathematical Incorrectness**: In any valid RGB to grayscale conversion, the weights must sum to 1.0 to preserve luminance. The current weights (0.21, 0.71, 0.07) sum to 0.99, violating this fundamental requirement.

2. **Violates Grayscale Identity Property**: For any grayscale image where R=G=B, the conversion should preserve the original value (gray = R = G = B). The current implementation fails this property, producing values that are 1-3 units lower than the input.

3. **Contradicts Standard Specifications**: The weights appear to be incorrectly rounded approximations of standard grayscale conversion coefficients. The ITU-R BT.601 standard specifies (0.299, 0.587, 0.114), and even the older NTSC standard uses (0.2126, 0.7152, 0.0722) - both sum to exactly 1.0.

4. **Extreme Edge Case Failure**: The most egregious case is RGB=(1,1,1) producing gray=0. This converts a very dark gray pixel to pure black, which is mathematically and visually incorrect.

5. **Loss of Dynamic Range**: Pure white (255, 255, 255) converts to 252 instead of 255, losing 3 levels of brightness at the high end of the spectrum. This reduces the effective dynamic range of the grayscale image.

## Relevant Context

The bug is located in `/home/npc/.local/lib/python3.13/site-packages/scipy/datasets/_fetchers.py` at lines 223-224. The function is part of SciPy's datasets module, which provides sample images for testing and demonstration purposes.

While this function is primarily used for demos and testing rather than production image processing, SciPy is a scientific computing library where mathematical correctness is expected. Users rely on SciPy for accurate computations, and even example/demo functions should maintain mathematical rigor.

The weights used (0.21, 0.71, 0.07) appear to be truncated/rounded versions of standard coefficients, suggesting this was likely an unintentional error rather than a deliberate design choice. No documentation indicates that non-standard weights are being used intentionally.

## Proposed Fix

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -220,8 +220,8 @@ def face(gray=False):
     face = frombuffer(face_data, dtype='uint8')
     face.shape = (768, 1024, 3)
     if gray is True:
-        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
-                0.07 * face[:, :, 2]).astype('uint8')
+        face = (0.299 * face[:, :, 0] + 0.587 * face[:, :, 1] +
+                0.114 * face[:, :, 2]).astype('uint8')
     return face
```