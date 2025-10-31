# Bug Report: SparseArray.map() Density Preservation

**Target**: `pandas.core.arrays.sparse.SparseArray.map`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `SparseArray.map()` method's docstring claims "The output array will have the same density as the input", but the implementation raises a `ValueError` when a mapping would naturally change the density by transforming a sparse value into the fill value.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st

@given(
    st.lists(st.integers(min_value=0, max_value=10), min_size=2, max_size=20),
    st.integers(min_value=0, max_value=10)
)
def test_map_density_claim(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)

    if arr.npoints == 0:
        return

    sparse_val = arr.sp_values[0]
    mapper = lambda x: fill_value if x == sparse_val else x + 100

    mapped = arr.map(mapper)
    assert mapped.density == arr.density
```

**Failing input**: `data=[0, 0, 1, 2], fill_value=0, mapper=lambda x: 0 if x == 1 else x`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr = SparseArray([0, 0, 1, 2], fill_value=0)
print(f"Original density: {arr.density}")

try:
    mapped = arr.map(lambda x: 0 if x == 1 else x)
    print(f"Mapped density: {mapped.density}")
except ValueError as e:
    print(f"Error: {e}")
```

**Output:**
```
Original density: 0.5
Error: fill value in the sparse values not supported
```

## Why This Is A Bug

The docstring at line 1331 states: "The output array will have the same density as the input."

This is a **contract violation** because:

1. The docstring presents density preservation as a guaranteed property
2. The implementation raises an error instead of preserving density
3. The error message doesn't explain the density preservation constraint
4. Users cannot freely map values even when it would be semantically valid

There are two ways this could be a bug:

1. **Docstring is wrong**: It should state "The output will have the same density (raises an error if mapping would change density)"
2. **Implementation is wrong**: It should recalculate the sparse index when mapped values equal the fill value

## Fix

The most user-friendly fix would be to update the implementation to handle the case where mapped values equal the fill value by recalculating the sparse index:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1360,14 +1360,9 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if na_action is None or notna(fill_val):
             fill_val = mapper.get(fill_val, fill_val) if is_map else mapper(fill_val)

-        def func(sp_val):
-            new_sp_val = mapper.get(sp_val, None) if is_map else mapper(sp_val)
-            # check identity and equality because nans are not equal to each other
-            if new_sp_val is fill_val or new_sp_val == fill_val:
-                msg = "fill value in the sparse values not supported"
-                raise ValueError(msg)
-            return new_sp_val
-
-        sp_values = [func(x) for x in self.sp_values]
-
-        return type(self)(sp_values, sparse_index=self.sp_index, fill_value=fill_val)
+        # Map all values in the dense representation
+        # This ensures correct handling when sparse values map to fill value
+        dense = self.to_dense()
+        mapped_dense = np.array([mapper.get(x, x) if is_map else mapper(x) for x in dense])
+        if na_action == 'ignore':
+            mapped_dense[self.isna()] = self.fill_value
+        return type(self)(mapped_dense, fill_value=fill_val)
```

Alternatively, update the docstring to accurately reflect the restriction.