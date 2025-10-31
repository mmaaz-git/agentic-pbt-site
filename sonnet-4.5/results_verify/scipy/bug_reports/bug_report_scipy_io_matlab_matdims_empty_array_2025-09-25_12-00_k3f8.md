# Bug Report: scipy.io.matlab Empty 1D Array Ignores oned_as Parameter

**Target**: `scipy.io.matlab.matdims` and `scipy.io.matlab.savemat/loadmat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When saving empty 1D NumPy arrays to MATLAB files, the `oned_as` parameter is ignored, resulting in inconsistent behavior compared to non-empty 1D arrays. Empty arrays always produce shape `(0, 0)` regardless of the `oned_as` setting, breaking the round-trip property.

## Property-Based Test

```python
import io
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=20))
@settings(max_examples=500)
def test_roundtrip_1d_int_array_row_orientation(lst):
    original = {'x': np.array(lst)}

    f = io.BytesIO()
    savemat(f, original, oned_as='row')
    f.seek(0)
    loaded = loadmat(f)

    np_arr = np.array(lst)
    if np_arr.ndim == 1 and np_arr.size > 0:
        expected_shape = (1, len(lst))
    elif np_arr.ndim == 1 and np_arr.size == 0:
        expected_shape = (1, 0)
    else:
        expected_shape = np_arr.shape

    assert loaded['x'].shape == expected_shape
```

**Failing input**: `lst=[]`

## Reproducing the Bug

```python
import io
import numpy as np
from scipy.io.matlab import loadmat, savemat

arr = np.array([])

f_row = io.BytesIO()
savemat(f_row, {'x': arr}, oned_as='row')
f_row.seek(0)
loaded_row = loadmat(f_row)

print(f"Empty array with oned_as='row': {loaded_row['x'].shape}")
print(f"Expected: (1, 0), Actual: {loaded_row['x'].shape}")

f_col = io.BytesIO()
savemat(f_col, {'x': arr}, oned_as='column')
f_col.seek(0)
loaded_col = loadmat(f_col)

print(f"Empty array with oned_as='column': {loaded_col['x'].shape}")
print(f"Expected: (0, 1), Actual: {loaded_col['x'].shape}")

non_empty = np.array([1, 2, 3])
f_row_ne = io.BytesIO()
savemat(f_row_ne, {'x': non_empty}, oned_as='row')
f_row_ne.seek(0)
loaded_row_ne = loadmat(f_row_ne)
print(f"\nNon-empty array with oned_as='row': {loaded_row_ne['x'].shape}")
```

Output:
```
Empty array with oned_as='row': (0, 0)
Expected: (1, 0), Actual: (0, 0)
Empty array with oned_as='column': (0, 0)
Expected: (0, 1), Actual: (0, 0)

Non-empty array with oned_as='row': (1, 3)
```

## Why This Is A Bug

The `oned_as` parameter controls whether 1D arrays should be saved as row vectors `(1, n)` or column vectors `(n, 1)`. This behavior works correctly for non-empty 1D arrays but is completely ignored for empty 1D arrays, which always get shape `(0, 0)`.

This violates the principle of uniform behavior across input domains and breaks the consistency property that users expect: if `oned_as='row'` produces `(1, n)` for any `n > 0`, it should produce `(1, 0)` for `n = 0`.

The bug is in the `matdims` function in `scipy/io/matlab/_miobase.py` at lines 326-327, which special-cases empty arrays before checking the `oned_as` parameter.

## Fix

```diff
--- a/scipy/io/matlab/_miobase.py
+++ b/scipy/io/matlab/_miobase.py
@@ -323,8 +323,6 @@ def matdims(arr, oned_as='column'):
     if shape == ():  # scalar
         return (1, 1)
     if len(shape) == 1:  # 1D
-        if shape[0] == 0:
-            return (0, 0)
-        elif oned_as == 'column':
+        if oned_as == 'column':
             return shape + (1,)
         elif oned_as == 'row':
```

This fix removes the special case for empty arrays and allows them to be handled consistently with the `oned_as` parameter like all other 1D arrays.