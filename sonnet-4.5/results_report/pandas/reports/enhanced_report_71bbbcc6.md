# Bug Report: SparseArray.density ZeroDivisionError on Empty Array

**Target**: `pandas.core.arrays.sparse.SparseArray.density`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `density` property of `SparseArray` raises an unhandled `ZeroDivisionError` when accessed on an empty array, instead of returning a sensible value like 0.0 or handling the edge case gracefully.

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

if __name__ == "__main__":
    test_density_in_range()
```

<details>

<summary>
**Failing input**: `SparseArray([], fill_value=0, kind='integer')`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 33, in <module>
    test_density_in_range()
    ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 26, in test_density_in_range
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 29, in test_density_in_range
    density = arr.density
              ^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 708, in density
    return self.sp_index.npoints / self.sp_index.length
           ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: division by zero
Falsifying example: test_density_in_range(
    arr=[]
    Fill: 0
    IntIndex
    Indices: array([], dtype=int32),
)
```
</details>

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

# Create an empty SparseArray
arr = SparseArray([])

# Print information about the array
print(f"Array: {arr}")
print(f"Length: {len(arr)}")
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")

# Try to access the density property - this should crash
try:
    density = arr.density
    print(f"Density: {density}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
```

<details>

<summary>
Output showing ZeroDivisionError when accessing density on empty SparseArray
</summary>
```
Array: []
Fill: nan
IntIndex
Indices: array([], dtype=int32)

Length: 0
Shape: (0,)
Dtype: Sparse[float64, nan]
ZeroDivisionError: division by zero
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Properties should not crash on valid objects**: Creating an empty SparseArray with `SparseArray([])` produces a valid object. Properties like `density` should handle all valid states of the object gracefully, not crash with unhandled exceptions.

2. **Documentation implies safe access**: The docstring states that density returns "The percent of non-fill_value points, as decimal" without mentioning any preconditions or exceptions. Users would reasonably expect this to work for any valid SparseArray.

3. **Mathematical interpretation**: For an empty array, the density (percentage of non-sparse elements) can reasonably be interpreted as 0.0, since there are zero non-sparse elements out of zero total elements. This follows the convention that an empty collection has 0% of any property.

4. **Inconsistent with pandas conventions**: Other pandas operations handle empty cases gracefully. For example, `Series([]).mean()` returns NaN rather than crashing. The density property should follow similar robustness principles.

5. **Unhandled division by zero**: The implementation directly performs `self.sp_index.npoints / self.sp_index.length` without checking if length is zero, which is a classic programming oversight that should be handled.

## Relevant Context

The bug occurs in `/pandas/core/arrays/sparse/array.py` at line 708. The implementation uses properties from the internal `sp_index` object (a SparseIndex from `pandas._libs.sparse`), where `npoints` represents the number of non-sparse points and `length` represents the total array length.

For empty arrays:
- `sp_index.npoints = 0` (no non-sparse elements)
- `sp_index.length = 0` (empty array)
- Division `0 / 0` causes the ZeroDivisionError

The SparseArray class is imported from `pandas.arrays` and is used to efficiently store arrays where most values are the same (the "fill value"). The density property is meant to indicate what fraction of the array contains non-fill values.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.density.html

## Proposed Fix

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