# Bug Report: numpy.matrixlib.bmat Crashes When gdict Provided Without ldict

**Target**: `numpy.matrixlib.bmat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `numpy.bmat` function crashes with a `TypeError` when called with string input and `gdict` parameter but without `ldict`, even though `ldict` has a default value of `None` in the function signature.

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

if __name__ == "__main__":
    test_bmat_gdict_without_ldict_crashes()
```

<details>

<summary>
**Failing input**: `arr1=array([[0., 0.], [0., 0.]]), arr2=array([[0., 0.], [0., 0.]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 19, in <module>
    test_bmat_gdict_without_ldict_crashes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 7, in test_bmat_gdict_without_ldict_crashes
    npst.arrays(dtype=np.float64, shape=(2, 2), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 14, in test_bmat_gdict_without_ldict_crashes
    result = np.bmat('X,Y', gdict=global_vars)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1107, in bmat
    return matrix(_from_string(obj, glob_dict, loc_dict))
                  ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1030, in _from_string
    thismat = ldict[col]
              ~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
Falsifying example: test_bmat_gdict_without_ldict_crashes(
    # The test always failed when commented parts were varied together.
    arr1=array([[0., 0.],
           [0., 0.]]),  # or any other generated value
    arr2=array([[0., 0.],
           [0., 0.]]),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Create simple 2x2 matrices
A = np.matrix([[1, 2], [3, 4]])
B = np.matrix([[5, 6], [7, 8]])

# This should work but crashes when gdict is provided without ldict
# The function signature allows ldict=None as default, so this should be valid
print("Attempting to call np.bmat('A,B', gdict={'A': A, 'B': B})")
result = np.bmat('A,B', gdict={'A': A, 'B': B})
print("Result:", result)
```

<details>

<summary>
TypeError: 'NoneType' object is not subscriptable
</summary>
```
Attempting to call np.bmat('A,B', gdict={'A': A, 'B': B})
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/repo.py", line 10, in <module>
    result = np.bmat('A,B', gdict={'A': A, 'B': B})
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1107, in bmat
    return matrix(_from_string(obj, glob_dict, loc_dict))
                  ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1030, in _from_string
    thismat = ldict[col]
              ~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
```
</details>

## Why This Is A Bug

This violates expected behavior because the function signature explicitly declares `ldict=None` as the default parameter value, indicating it should be optional. The documentation confirms this by describing `ldict` as "optional". When users provide `gdict` to specify the global namespace for string parsing, they should reasonably expect to omit `ldict` if all variables are global.

The crash occurs due to improper handling of `None` in the implementation. At line 1105 in `bmat()`, when `gdict` is provided, the code assigns `loc_dict = ldict` without checking if `ldict` is `None`. This `None` value is then passed to `_from_string()`, which at line 1030 attempts to subscript it with `ldict[col]` before checking `gdict`, causing a `TypeError` because `None` is not subscriptable.

The code tries to look up variables first in the local dictionary, then falls back to the global dictionary. However, it doesn't handle the case where the local dictionary is `None`, which is the default value according to the function signature.

## Relevant Context

The numpy.bmat function is used to build block matrices from string expressions, nested sequences, or arrays. When using string input (like 'A,B'), it needs to resolve variable names from dictionaries. The function supports three modes:

1. No dictionaries provided: uses the caller's frame locals and globals
2. Both ldict and gdict provided: uses the provided dictionaries
3. Only gdict provided with ldict=None: **crashes** (this bug)

The documentation states that `ldict` is "Ignored if obj is not a string or gdict is None", but doesn't explicitly state what happens when gdict is provided and ldict is None. However, since ldict has a default value of None, this combination should be valid.

Workaround: Users can pass an empty dict for ldict: `np.bmat('A,B', ldict={}, gdict={'A': A, 'B': B})`

Note: The numpy.matrix module is deprecated in favor of regular numpy arrays, but it's still part of the numpy API and should work correctly.

## Proposed Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -1027,11 +1027,14 @@ def _from_string(str, gdict, ldict):
         coltup = []
         for col in trow:
             col = col.strip()
-            try:
-                thismat = ldict[col]
-            except KeyError:
+            if ldict is not None:
+                try:
+                    thismat = ldict[col]
+                except KeyError:
+                    thismat = gdict[col]
+            else:
                 try:
                     thismat = gdict[col]
                 except KeyError as e:
                     raise NameError(f"name {col!r} is not defined") from None

             coltup.append(thismat)
```