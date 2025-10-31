# Bug Report: pandas SparseArray density property returns NaN for empty arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.density`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `density` property of `SparseArray` returns `NaN` (or potentially raises `ZeroDivisionError`) when called on an empty array due to division by zero (`0 / 0`). This violates mathematical expectations and can cause issues in downstream code that expects a valid float or proper error handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst
import numpy as np
from pandas.arrays import SparseArray

@given(
    data=npst.arrays(
        dtype=npst.integer_dtypes() | npst.floating_dtypes(),
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=100)
    )
)
@settings(max_examples=1000)
def test_density_property(data):
    sparse = SparseArray(data)

    if len(sparse) == 0:
        density = sparse.density

        assert not np.isnan(density), (
            f"BUG: density={density} for empty array (length=0). "
            f"Should return 0.0 or raise informative error."
        )
        assert not np.isinf(density), f"density should not be Inf for empty array"

    else:
        expected_density = sparse.npoints / len(sparse)
        assert sparse.density == expected_density
```

**Failing input**: `SparseArray([])`

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

empty_sparse = SparseArray([])
print(f"len: {len(empty_sparse)}")
print(f"npoints: {empty_sparse.npoints}")

density = empty_sparse.density
print(f"density: {density}")
print(f"Is NaN: {np.isnan(density)}")
```

## Why This Is A Bug

1. **Mathematically undefined**: The expression `0 / 0` is undefined, yet the property returns `nan` without warning or documentation
2. **Undocumented behavior**: The docstring for `density` provides no guidance on empty array behavior
3. **Type contract violation**: The property claims to return `float` (a valid percentage), but `nan` is semantically not a "percent of non-fill_value points"
4. **Inconsistent with expectations**: An empty array logically has 0% density (no points at all), not undefined density
5. **Downstream errors**: Code using `.density` may not expect or handle `nan`, leading to silent failures or incorrect computations

## Fix

The `density` property should return `0.0` for empty arrays, as this is the most sensible interpretation (zero points in zero positions = 0% density). Here's the fix:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -696,7 +696,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
     @property
     def density(self) -> float:
         """
         The percent of non- ``fill_value`` points, as decimal.

+        For empty arrays, returns 0.0.
+
         Examples
         --------
         >>> from pandas.arrays import SparseArray
         >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
         >>> s.density
         0.6
+        >>> empty = SparseArray([])
+        >>> empty.density
+        0.0
         """
-        return self.sp_index.npoints / self.sp_index.length
+        if self.sp_index.length == 0:
+            return 0.0
+        return self.sp_index.npoints / self.sp_index.length
```

## Additional Context

- No existing pandas tests cover the `.density` property at all (verified via grep)
- No existing pandas tests create empty `SparseArray` instances
- This edge case has likely gone unnoticed due to lack of property-based testing
- The fix is minimal, backward-compatible (replacing `nan` with `0.0`), and mathematically sound