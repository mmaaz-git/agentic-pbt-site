# Bug Report: numpy.matrixlib.bmat Crashes When gdict Provided Without ldict

**Target**: `numpy.matrixlib.bmat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.bmat` crashes with `TypeError: 'NoneType' object is not subscriptable` when `gdict` is provided without `ldict`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst


@given(
    npst.arrays(dtype=np.float64, shape=(2, 2), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
    npst.arrays(dtype=np.float64, shape=(2, 2), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
)
def test_bmat_gdict_without_ldict_crashes(arr1, arr2):
    m1 = np.matrix(arr1)
    m2 = np.matrix(arr2)
    global_vars = {'X': m1, 'Y': m2}
    result = np.bmat('X,Y', gdict=global_vars)
    expected = np.bmat([[m1, m2]])
    assert np.array_equal(result, expected)
```

**Failing input**: Any valid matrices and variable names.

## Reproducing the Bug

```python
import numpy as np

A = np.matrix([[1, 2], [3, 4]])
B = np.matrix([[5, 6], [7, 8]])

np.bmat('A,B', gdict={'A': A, 'B': B})
```

Output:
```
TypeError: 'NoneType' object is not subscriptable
```

## Why This Is A Bug

When `gdict` is provided to specify the global namespace for string parsing, users should be able to omit `ldict` if all variables are in the global scope. However, the implementation sets `loc_dict = ldict` which is `None`, and then `_from_string()` tries to subscript it:

```python
try:
    thismat = ldict[col]  # Crashes when ldict is None
except KeyError:
    thismat = gdict[col]
```

## Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -1102,7 +1102,7 @@ def bmat(obj, ldict=None, gdict=None):
             loc_dict = frame.f_locals
         else:
             glob_dict = gdict
-            loc_dict = ldict
+            loc_dict = ldict if ldict is not None else {}

         return matrix(_from_string(obj, glob_dict, loc_dict))
```