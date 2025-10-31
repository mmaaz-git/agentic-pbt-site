# Bug Report: numpy.matrixlib.bmat TypeError with gdict but no ldict

**Target**: `numpy.matrixlib.bmat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`bmat()` crashes with TypeError when `gdict` is provided but `ldict` is `None` (or not provided), despite both parameters being documented as optional.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st


@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5)
)
def test_bmat_gdict_without_ldict(rows, cols):
    A = np.matrix(np.ones((rows, cols)))
    result = np.bmat('A', ldict=None, gdict={'A': A})
    assert np.array_equal(result, A)
```

**Failing input**: `rows=1, cols=1` (or any valid matrix dimensions)

## Reproducing the Bug

```python
import numpy as np

A = np.matrix([[1, 2], [3, 4]])
result = np.bmat('A', ldict=None, gdict={'A': A})
```

Output:
```
TypeError: 'NoneType' object is not subscriptable
```

## Why This Is A Bug

The `bmat` function documents both `ldict` and `gdict` as optional parameters (defaulting to `None`). The docstring states that `ldict` is "Ignored if `obj` is not a string or `gdict` is None", but does not state that `ldict` must be provided when `gdict` is provided.

When `gdict` is provided without `ldict`, the code sets `loc_dict = ldict` (which is `None`), then passes it to `_from_string()`, which tries to subscript `None`, causing a crash.

This violates the API contract that optional parameters with default values should be independently usable.

## Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -1097,7 +1097,7 @@ def bmat(obj, ldict=None, gdict=None):
             loc_dict = frame.f_locals
         else:
             glob_dict = gdict
-            loc_dict = ldict
+            loc_dict = ldict if ldict is not None else {}

         return matrix(_from_string(obj, glob_dict, loc_dict))
```