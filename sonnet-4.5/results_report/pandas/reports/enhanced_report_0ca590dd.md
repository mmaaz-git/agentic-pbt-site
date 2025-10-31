# Bug Report: pandas.core.arrays.SparseArray.density ZeroDivisionError on Empty Arrays

**Target**: `pandas.core.arrays.SparseArray.density`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `density` property of `SparseArray` raises a `ZeroDivisionError` when accessed on an empty array instead of returning a sensible value like 0.0 or NaN.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test that discovered the SparseArray.density ZeroDivisionError bug.
"""

from hypothesis import given, strategies as st, settings
import pandas.core.arrays as arr
import numpy as np

@given(st.lists(st.integers(min_value=0, max_value=1000), min_size=0, max_size=100))
@settings(max_examples=500)
def test_sparsearray_empty_and_edge_cases(values):
    if len(values) == 0:
        sparse = arr.SparseArray([], fill_value=0)
        assert len(sparse) == 0
        assert sparse.density == 0 or np.isnan(sparse.density)
    else:
        sparse = arr.SparseArray(values, fill_value=0)
        assert len(sparse) == len(values)

if __name__ == "__main__":
    # Run the test
    test_sparsearray_empty_and_edge_cases()
```

<details>

<summary>
**Failing input**: `values=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 23, in <module>
    test_sparsearray_empty_and_edge_cases()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 11, in test_sparsearray_empty_and_edge_cases
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 16, in test_sparsearray_empty_and_edge_cases
    assert sparse.density == 0 or np.isnan(sparse.density)
           ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 708, in density
    return self.sp_index.npoints / self.sp_index.length
           ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: division by zero
Falsifying example: test_sparsearray_empty_and_edge_cases(
    values=[],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/9/hypo.py:14
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/construction.py:645
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py:1876
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of SparseArray.density ZeroDivisionError on empty array.
"""

import pandas.core.arrays as arr

# Create an empty SparseArray with fill_value=0
sparse = arr.SparseArray([], fill_value=0)

# This should work but causes a ZeroDivisionError
density = sparse.density
print(f"Density of empty SparseArray: {density}")
```

<details>

<summary>
ZeroDivisionError when accessing density property
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/repo.py", line 12, in <module>
    density = sparse.density
              ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 708, in density
    return self.sp_index.npoints / self.sp_index.length
           ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: division by zero
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Valid object construction leads to invalid state**: The SparseArray constructor successfully creates an empty array (`arr.SparseArray([], fill_value=0)`), but accessing a public property on this valid object crashes.

2. **Inconsistent API behavior**: Other properties and methods work correctly on empty SparseArrays:
   - `len(sparse)` returns 0
   - `sparse.nbytes` returns 0
   - `sparse.npoints` returns 0
   - `sparse.fill_value` returns the fill value
   - `sparse.dtype` returns the correct dtype

3. **Violates property access expectations**: Properties in Python should not raise arithmetic exceptions. Users expect property access to be safe and should not need defensive `try/except` blocks around simple attribute access.

4. **Documentation doesn't warn about this**: The docstring for `density` states "The percent of non- fill_value points, as decimal" with a single example, but doesn't mention any exceptions or edge cases for empty arrays.

5. **Mathematical interpretation exists**: The density of an empty array has reasonable interpretations:
   - 0.0 (0 non-fill values out of 0 total = 0%)
   - NaN (undefined for empty sets)
   Either would be better than crashing.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py` at line 708:

```python
@property
def density(self) -> float:
    """
    The percent of non- ``fill_value`` points, as decimal.

    Examples
    --------
    >>> from pandas.arrays import SparseArray
    >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
    >>> s.density
    0.6
    """
    return self.sp_index.npoints / self.sp_index.length  # Line 708
```

For empty arrays, both `self.sp_index.npoints` and `self.sp_index.length` are 0, causing division by zero.

The `sp_index` is of type `pandas._libs.sparse.IntIndex`. When the array is empty:
- `sp_index.npoints = 0` (number of non-sparse points)
- `sp_index.length = 0` (total length of array)

This edge case works correctly for arrays with all sparse values (returns 0.0) but fails for completely empty arrays.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -705,7 +705,10 @@ class SparseArray(OpsMixin, ExtensionArray):
         >>> s.density
         0.6
         """
-        return self.sp_index.npoints / self.sp_index.length
+        if self.sp_index.length == 0:
+            # Empty array has 0% density (no non-fill values)
+            return 0.0
+        return self.sp_index.npoints / self.sp_index.length

     @property
     def npoints(self) -> int:
```