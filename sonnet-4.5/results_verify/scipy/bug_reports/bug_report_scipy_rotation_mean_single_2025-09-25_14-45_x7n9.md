# Bug Report: scipy.spatial.transform.Rotation.mean() Fails on Single Rotations

**Target**: `scipy.spatial.transform.Rotation.mean()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `mean()` method crashes with `TypeError: Single rotation has no len()` when called on a single rotation object, despite this being a reasonable operation (the mean of one rotation should be itself).

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from scipy.spatial.transform import Rotation
import numpy as np


def rotation_strategy():
    return st.one_of(
        st.floats(min_value=-180, max_value=180).map(
            lambda angle: Rotation.from_euler('z', angle, degrees=True)
        ),
    )


@given(rotation_strategy())
@settings(max_examples=500)
def test_mean_of_single_rotation(r):
    mean = r.mean()
    assert mean.as_quat() == r.as_quat()
```

**Failing input**: Any single rotation, e.g., `Rotation.from_euler('z', 45, degrees=True)`

## Reproducing the Bug

```python
from scipy.spatial.transform import Rotation

r = Rotation.from_euler('z', 45, degrees=True)
mean = r.mean()
```

This raises:
```
TypeError: Single rotation has no len().
```

However, creating a sequence of one rotation works fine:
```python
r_seq = Rotation.from_euler('z', [45], degrees=True)
mean = r_seq.mean()
```

## Why This Is A Bug

1. **Inconsistent API**: A sequence of one rotation can have its mean computed, but a single rotation cannot
2. **Mathematical validity**: The mean of a single rotation is well-defined (it's the rotation itself)
3. **Unexpected error**: The error message "Single rotation has no len()" is confusing and doesn't explain why `mean()` isn't supported
4. **User expectations**: Users might reasonably expect `mean()` to work on any Rotation object, returning itself for single rotations
5. **Violates duck typing**: The method exists on the object but crashes based on internal state

## Fix

The `mean()` method should handle single rotations by returning a copy of the rotation:

```diff
diff --git a/scipy/spatial/transform/_rotation.pyx b/scipy/spatial/transform/_rotation.pyx
index abc1234..def5678 100644
--- a/scipy/spatial/transform/_rotation.pyx
+++ b/scipy/spatial/transform/_rotation.pyx
@@ -2960,6 +2960,10 @@ cdef class Rotation:
         -------
         mean : `Rotation` instance
             Object containing the mean of the rotations in the current instance.
+    """
+    # Handle single rotation case
+    if self.single:
+        return Rotation.from_quat(self.as_quat())
+
+    # Original code for sequences
     ...
```

Alternatively, raise a more descriptive error message explaining that `mean()` is only supported for rotation sequences, not single rotations - though this would still be inconsistent with sequences of length 1.