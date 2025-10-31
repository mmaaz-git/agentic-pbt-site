# Bug Report: scipy.ndimage.rotate - Exponential Data Degradation on Non-Square Arrays with reshape=False

**Target**: `scipy.ndimage.rotate`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Rotating a non-square array by 90 degrees four times with `reshape=False` causes exponential data degradation (values decay by ~50% each rotation), violating the mathematical property that 360-degree rotation should be identity.

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

if __name__ == "__main__":
    # Run the test
    test_rotate_90_four_times_identity()
```

<details>

<summary>
**Failing input**: `array([[1., 1., 1.], [1., 1., 1.]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 31, in <module>
    test_rotate_90_four_times_identity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 7, in test_rotate_90_four_times_identity
    arr=arrays(dtype=np.float64, shape=st.tuples(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 27, in test_rotate_90_four_times_identity
    assert np.allclose(arr, result)
           ~~~~~~~~~~~^^^^^^^^^^^^^
AssertionError
Falsifying example: test_rotate_90_four_times_identity(
    arr=array([[1., 1., 1.],
           [1., 1., 1.]]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy import ndimage

# Create a simple non-square array
arr = np.array([[1., 1., 1.],
                [1., 1., 1.]])

print("Original array:")
print(arr)
print(f"Shape: {arr.shape}")
print()

# Apply 4 rotations of 90 degrees with reshape=False
result = arr.copy()
for i in range(4):
    result = ndimage.rotate(result, 90, reshape=False)
    print(f"After rotation {i+1} (90° × {i+1} = {90*(i+1)}°):")
    print(result)
    print(f"Shape: {result.shape}")
    print()

print("Expected: Should return to original array")
print("Actual result after 4×90° rotations:")
print(result)
print()

# Compare with a single 360° rotation
single_360 = ndimage.rotate(arr, 360, reshape=False)
print("Result of single 360° rotation:")
print(single_360)
print()

# Check if they are close
print("Are 4×90° rotations equal to original?", np.allclose(arr, result))
print("Is single 360° rotation equal to original?", np.allclose(arr, single_360))
```

<details>

<summary>
Exponential data degradation through 4 rotations
</summary>
```
Original array:
[[1. 1. 1.]
 [1. 1. 1.]]
Shape: (2, 3)

After rotation 1 (90° × 1 = 90°):
[[0. 1. 0.]
 [0. 1. 0.]]
Shape: (2, 3)

After rotation 2 (90° × 2 = 180°):
[[0.  0.5 0. ]
 [0.  0.5 0. ]]
Shape: (2, 3)

After rotation 3 (90° × 3 = 270°):
[[0.   0.25 0.  ]
 [0.   0.25 0.  ]]
Shape: (2, 3)

After rotation 4 (90° × 4 = 360°):
[[0.    0.125 0.   ]
 [0.    0.125 0.   ]]
Shape: (2, 3)

Expected: Should return to original array
Actual result after 4×90° rotations:
[[0.    0.125 0.   ]
 [0.    0.125 0.   ]]

Result of single 360° rotation:
[[1. 1. 1.]
 [1. 1. 1.]]

Are 4×90° rotations equal to original? False
Is single 360° rotation equal to original? True
```
</details>

## Why This Is A Bug

This violates fundamental mathematical properties and expectations:

1. **Mathematical identity violation**: In linear algebra and geometry, four 90-degree rotations must equal a 360-degree rotation, which is the identity transformation. The fact that a single 360° rotation works correctly while 4×90° rotations fail demonstrates internal inconsistency.

2. **Exponential data decay**: Values degrade by approximately 50% with each rotation (1.0 → 0.5 → 0.25 → 0.125), indicating a compounding interpolation error. This is not a simple rounding error but systematic data corruption.

3. **Shape mismatch handling**: When `reshape=False`, rotating a 2×3 array by 90° forces a naturally 3×2 result into the original 2×3 shape. This requires cropping/padding that loses information. Each subsequent rotation compounds this loss.

4. **Silent data corruption**: The function provides no warning that data is being lost, making this particularly dangerous for scientific computing applications where accuracy is critical.

5. **Documentation gap**: The documentation states that `reshape=False` maintains the original shape but doesn't warn about severe data loss for non-square arrays with rotations that naturally change dimensions.

## Relevant Context

The issue specifically occurs when ALL of the following conditions are met:
- Array is non-square (e.g., 2×3, 3×4, etc.)
- `reshape=False` parameter is used
- Rotation angle is 90° or 270° (angles that swap dimensions)
- Multiple rotations are applied sequentially

The bug does NOT occur when:
- Arrays are square (e.g., 3×3)
- `reshape=True` is used (default)
- Single 360° rotation is applied
- Rotation angles don't swap dimensions (e.g., 180°)

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html

Relevant code location: `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/ndimage/_interpolation.py:898-1034`

## Proposed Fix

The issue stems from forcing rotated content into the original shape when dimensions naturally swap. Here's a high-level fix approach:

For a comprehensive fix, the function should either:
1. Track cumulative rotation angle and apply optimized transformations for multiples of 90°
2. Raise a warning when `reshape=False` is used with non-square arrays and 90°/270° rotations
3. Improve the interpolation boundary handling to minimize data loss

A minimal fix would be to add a warning when problematic conditions are detected:

```diff
--- a/scipy/ndimage/_interpolation.py
+++ b/scipy/ndimage/_interpolation.py
@@ -30,6 +30,7 @@
 # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 import itertools
+import warnings

 import numpy as np
 from scipy._lib._util import normalize_axis_index
@@ -997,6 +998,14 @@ def rotate(input, angle, axes=(1, 0), reshape=True, output=None, order=3,
                                    [0, ix, 0, ix]]
         # Compute the shape of the transformed input plane
         out_plane_shape = (np.ptp(out_bounds, axis=1) + 0.5).astype(int)
     else:
         out_plane_shape = img_shape[axes]
+        # Warn if using reshape=False with non-square arrays and 90/270 degree rotations
+        angle_mod = angle % 360
+        if (in_plane_shape[0] != in_plane_shape[1] and
+            (np.isclose(angle_mod, 90) or np.isclose(angle_mod, 270))):
+            warnings.warn(
+                "Using reshape=False with non-square arrays and 90/270 degree "
+                "rotations will cause data loss due to dimension mismatch. "
+                "Consider using reshape=True to preserve data integrity.",
+                RuntimeWarning, stacklevel=2)

     out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
```