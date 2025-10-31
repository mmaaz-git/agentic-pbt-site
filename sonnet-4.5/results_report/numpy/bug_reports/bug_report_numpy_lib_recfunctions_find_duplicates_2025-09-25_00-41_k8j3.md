# Bug Report: numpy.lib.recfunctions.find_duplicates Fails on Regular Arrays

**Target**: `numpy.lib.recfunctions.find_duplicates`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `find_duplicates` function crashes when passed a regular (non-masked) structured array, despite the docstring describing the input as "array-like" with no requirement that it be masked.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from numpy.lib import recfunctions as rfn

@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=20)
)
def test_find_duplicates_returns_actual_duplicates(values):
    arr = np.array(list(zip(values)), dtype=[('value', 'i4')])
    duplicates = rfn.find_duplicates(arr, return_index=False)

    for dup in duplicates:
        count = np.sum(arr['value'] == dup['value'])
        assert count > 1
```

**Failing input**: `values=[0, 0]`

## Reproducing the Bug

```python
import numpy as np
from numpy.lib import recfunctions as rfn

arr = np.array([(1,), (1,), (2,), (3,)], dtype=[('value', 'i4')])
duplicates = rfn.find_duplicates(arr, return_index=False)
```

Output:
```
AttributeError: 'numpy.ndarray' object has no attribute 'filled'. Did you mean: 'fill'?
```

The function works correctly with masked arrays:
```python
masked_arr = np.ma.array([(1,), (1,), (2,), (3,)], dtype=[('value', 'i4')])
duplicates = rfn.find_duplicates(masked_arr, return_index=False)
```

## Why This Is A Bug

1. The docstring specifies `a : array-like` with no indication that masked arrays are required
2. Regular structured arrays are valid "array-like" inputs
3. The function should handle both masked and regular arrays, as is standard practice in NumPy
4. The error occurs because the code unconditionally calls `.filled()`, which only exists on masked arrays

## Fix

The bug is in `/numpy/lib/recfunctions.py` around line 1458. The code currently reads:

```python
sorteddata = sortedbase.filled()
```

This should be changed to use the module-level function which works for both array types:

```diff
--- a/numpy/lib/recfunctions.py
+++ b/numpy/lib/recfunctions.py
@@ -1455,7 +1455,7 @@ def find_duplicates(a, key=None, ignoremask=True, return_index=False):
     # Get the sorting indices and the sorted data
     sortidx = base.argsort()
     sortedbase = base[sortidx]
-    sorteddata = sortedbase.filled()
+    sorteddata = np.ma.filled(sortedbase, fill_value=0)
     # Get the comparison results
     flag = (sorteddata[:-1] == sorteddata[1:])
```

The `np.ma.filled()` function works for both regular arrays (returns them unchanged) and masked arrays (fills masked values with the fill_value). The fill_value of 0 is arbitrary for comparison purposes since masked values will be handled separately.

Alternatively, a more explicit fix would be:

```diff
-    sorteddata = sortedbase.filled()
+    sorteddata = sortedbase.filled() if np.ma.isMaskedArray(sortedbase) else sortedbase
```