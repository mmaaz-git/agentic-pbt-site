# Bug Report: numpy.matrixlib.bmat gdict Parameter Crash

**Target**: `numpy.matrixlib.bmat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `bmat` function crashes with `TypeError: 'NoneType' object is not subscriptable` when `gdict` is provided without `ldict`, even though the API allows `ldict` to be optional.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from numpy.matrixlib import bmat, matrix
import hypothesis.extra.numpy as npst


@given(
    st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=10),
    npst.arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 3), st.integers(1, 3)))
)
def test_bmat_gdict_without_ldict(varname, arr):
    m = matrix(arr)
    gdict = {varname: m}
    result = bmat(varname, gdict=gdict)
    np.testing.assert_array_equal(result, m)
```

**Failing input**: `varname='A', gdict={'A': matrix([[0.]])}` (any variable name and matrix)

## Reproducing the Bug

```python
import numpy as np
from numpy.matrixlib import bmat, matrix

X = matrix([[1, 2]])
result = bmat('X', gdict={'X': X})
```

## Why This Is A Bug

The function signature declares `ldict` as optional with a default of `None`:
```python
def bmat(obj, ldict=None, gdict=None):
```

The docstring states that `ldict` is "Ignored if `obj` is not a string or `gdict` is None", implying that when `gdict` is provided (not None), `ldict` should be usable even if not provided.

However, the implementation in `_from_string` sets `loc_dict = ldict` when `gdict is not None`, without checking if `ldict` is `None`. This causes a crash when trying to access `ldict[col]`:

```python
if gdict is None:
    frame = sys._getframe().f_back
    glob_dict = frame.f_globals
    loc_dict = frame.f_locals
else:
    glob_dict = gdict
    loc_dict = ldict  # Bug: ldict can be None here
```

## Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -1098,7 +1098,7 @@ def _from_string(str, gdict, ldict):
             loc_dict = frame.f_locals
         else:
             glob_dict = gdict
-            loc_dict = ldict
+            loc_dict = ldict if ldict is not None else {}

         return matrix(_from_string(obj, glob_dict, loc_dict))
```