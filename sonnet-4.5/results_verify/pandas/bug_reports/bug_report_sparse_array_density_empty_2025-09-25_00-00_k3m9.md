# Bug Report: SparseArray.density ZeroDivisionError on Empty Array

**Target**: `pandas.core.arrays.sparse.SparseArray.density`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `density` property of `SparseArray` crashes with a `ZeroDivisionError` when called on an empty array, rather than returning a sensible value like 0.0 or raising a more informative error.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.arrays import SparseArray

@st.composite
def sparse_arrays(draw, min_size=0, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    dtype_choice = draw(st.sampled_from(['int64', 'float64', 'bool']))

    if dtype_choice == 'int64':
        values = draw(st.lists(st.integers(min_value=-1000, max_value=1000),
                              min_size=size, max_size=size))
        fill_value = 0
    elif dtype_choice == 'float64':
        values = draw(st.lists(st.floats(min_value=-1e6, max_value=1e6,
                                        allow_nan=False, allow_infinity=False),
                              min_size=size, max_size=size))
        fill_value = 0.0
    else:
        values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
        fill_value = False

    kind = draw(st.sampled_from(['integer', 'block']))
    return SparseArray(values, fill_value=fill_value, kind=kind)

@given(sparse_arrays())
@settings(max_examples=100)
def test_density_in_range(arr):
    """Density should always be between 0 and 1"""
    density = arr.density
    assert 0 <= density <= 1, f"Density {density} not in [0, 1]"
```

**Failing input**: `SparseArray([])`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr = SparseArray([])
print(f"Array: {arr}")
print(f"Length: {len(arr)}")

density = arr.density
```

**Output:**
```
Array: []
Fill: nan
IntIndex
Indices: array([], dtype=int32)

Length: 0
ZeroDivisionError: division by zero
```

## Why This Is A Bug

The `density` property is documented to return "The percent of non-`fill_value` points, as decimal." For an empty array, a sensible return value would be 0.0 (no points means 0% density) or potentially NaN. Instead, the implementation performs division by zero:

```python
@property
def density(self) -> float:
    return self.sp_index.npoints / self.sp_index.length  # length is 0 for empty array!
```

This violates the reasonable expectation that properties of data structures should handle empty cases gracefully.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -705,7 +705,10 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
         >>> s.density
         0.6
         """
-        return self.sp_index.npoints / self.sp_index.length
+        if self.sp_index.length == 0:
+            return 0.0
+        return self.sp_index.npoints / self.sp_index.length
```