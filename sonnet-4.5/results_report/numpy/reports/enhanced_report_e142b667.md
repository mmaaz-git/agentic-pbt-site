# Bug Report: numpy.matrixlib.bmat Crashes When gdict is Provided Without ldict

**Target**: `numpy.matrixlib.bmat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `numpy.bmat` function crashes with `TypeError: 'NoneType' object is not subscriptable` when the optional parameter `gdict` is provided without `ldict`, despite both parameters being marked as optional in the function signature.

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

# Run the test
if __name__ == "__main__":
    test_bmat_gdict_without_ldict_crashes()
```

<details>

<summary>
**Failing input**: `arr1=array([[0., 0.], [0., 0.]]), arr2=array([[0., 0.], [0., 0.]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 20, in <module>
    test_bmat_gdict_without_ldict_crashes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 7, in test_bmat_gdict_without_ldict_crashes
    npst.arrays(dtype=np.float64, shape=(2, 2), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 14, in test_bmat_gdict_without_ldict_crashes
    result = np.bmat('X,Y', gdict=global_vars)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1107, in bmat
    return matrix(_from_string(obj, glob_dict, loc_dict))
                  ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1030, in _from_string
    thismat = ldict[col]
              ~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
Falsifying example: test_bmat_gdict_without_ldict_crashes(
    arr1=array([[0., 0.],
           [0., 0.]]),
    arr2=array([[0., 0.],
           [0., 0.]]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Create test matrices
A = np.matrix([[1, 2], [3, 4]])
B = np.matrix([[5, 6], [7, 8]])

# Try to call bmat with only gdict (no ldict)
print("Attempting to call np.bmat('A,B', gdict={'A': A, 'B': B})")
try:
    result = np.bmat('A,B', gdict={'A': A, 'B': B})
    print("Result:")
    print(result)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError: 'NoneType' object is not subscriptable
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/repo.py", line 10, in <module>
    result = np.bmat('A,B', gdict={'A': A, 'B': B})
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1107, in bmat
    return matrix(_from_string(obj, glob_dict, loc_dict))
                  ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1030, in _from_string
    thismat = ldict[col]
              ~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
Attempting to call np.bmat('A,B', gdict={'A': A, 'B': B})
Error occurred: TypeError: 'NoneType' object is not subscriptable
```
</details>

## Why This Is A Bug

This violates expected behavior for optional parameters in Python. The `numpy.bmat` function signature declares both `ldict` and `gdict` as optional parameters with default values of `None`. According to standard Python conventions and the documentation, optional parameters should be independently usable unless explicitly documented otherwise.

The bug occurs because when `gdict` is provided but `ldict` is not, the code in `bmat` at line 1105 sets `loc_dict = ldict` (which is `None`), then passes this to `_from_string()`. The `_from_string()` function then attempts to subscript `ldict` at line 1030 (`thismat = ldict[col]`) without checking if it's `None`, causing a `TypeError`.

The documentation states that `ldict` is "Ignored if `obj` is not a string or `gdict` is None" but doesn't specify that `ldict` must be provided when `gdict` is provided. The fact that providing an empty dict (`ldict={}`) works correctly demonstrates that the intended behavior is to search the local dictionary first and fall back to the global dictionary, which the code already implements in the try/except block at lines 1029-1035.

## Relevant Context

The numpy.matrix module is deprecated and users are encouraged to use regular arrays, as evidenced by the PendingDeprecationWarning issued in the matrix constructor (lines 120-125 of defmatrix.py). However, the module is still part of the public API and should not crash on reasonable inputs.

The workaround is simple: users can pass an empty dictionary for `ldict` when providing `gdict`:
```python
np.bmat('A,B', ldict={}, gdict={'A': A, 'B': B})  # Works correctly
```

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.bmat.html

## Proposed Fix

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