# Bug Report: numpy.matrixlib.bmat TypeError when using gdict parameter without ldict

**Target**: `numpy.matrixlib.defmatrix.bmat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `np.bmat()` function crashes with a `TypeError` when called with a string argument and only the `gdict` parameter, despite both `ldict` and `gdict` being documented as optional parameters.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import warnings

@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5)
)
def test_bmat_string_with_globals(rows, cols):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)

        globals_dict = {
            'A': np.matrix(np.ones((rows, cols))),
            'B': np.matrix(np.zeros((rows, cols)))
        }

        result = np.bmat('A, B', gdict=globals_dict)

        assert result.shape == (rows, 2 * cols)

# Run the test
if __name__ == "__main__":
    test_bmat_string_with_globals()
```

<details>

<summary>
**Failing input**: `rows=1, cols=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 24, in <module>
    test_bmat_string_with_globals()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 6, in test_bmat_string_with_globals
    st.integers(min_value=1, max_value=5),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 18, in test_bmat_string_with_globals
    result = np.bmat('A, B', gdict=globals_dict)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1107, in bmat
    return matrix(_from_string(obj, glob_dict, loc_dict))
                  ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1030, in _from_string
    thismat = ldict[col]
              ~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
Falsifying example: test_bmat_string_with_globals(
    rows=1,  # or any other generated value
    cols=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import warnings
import numpy as np

# Suppress the deprecation warning for cleaner output
with warnings.catch_warnings():
    warnings.simplefilter("ignore", PendingDeprecationWarning)

    # Create a simple matrix
    A = np.matrix([[1, 2]])

    # Try to use bmat with only gdict parameter
    # This should work according to documentation, but crashes
    result = np.bmat('A', gdict={'A': A})
    print("Result:", result)
```

<details>

<summary>
TypeError when attempting to use gdict without ldict
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/repo.py", line 13, in <module>
    result = np.bmat('A', gdict={'A': A})
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

The `numpy.bmat()` function documentation clearly states that both `ldict` and `gdict` parameters are optional, with no indication that they must be used together. The function signature accepts default `None` values for both parameters. However, when `gdict` is provided without `ldict`, the internal `_from_string()` function attempts to subscript `ldict` (which is `None`) without checking, causing a `TypeError`.

The bug occurs because the exception handling in `_from_string()` only catches `KeyError` (expected when a key is missing from a dictionary) but not `TypeError` (raised when attempting to subscript `None`). The code's structure shows it was intended to fall back from `ldict` to `gdict` when a variable isn't found, but this fallback mechanism fails when `ldict` is `None`.

This violates the documented contract where both parameters are independently optional. Users reasonably expect to be able to provide only `gdict` for custom global variable lookups when processing string expressions.

## Relevant Context

The bug is located in `/numpy/matrixlib/defmatrix.py` at lines 1029-1035 in the `_from_string()` function. When `bmat()` is called with a string argument and only `gdict`, it passes `None` for `ldict` to `_from_string()`.

The numpy.matrix class and related functions are deprecated (hence the PendingDeprecationWarning), but they should still work as documented until removed.

Documentation link: https://numpy.org/doc/stable/reference/generated/numpy.bmat.html

Workaround: Users can pass an empty dictionary for `ldict` when using `gdict`:
```python
np.bmat('A', ldict={}, gdict={'A': A})  # This works
```

## Proposed Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -1027,7 +1027,7 @@ def _from_string(str, gdict, ldict):
         coltup = []
         for col in trow:
             col = col.strip()
             try:
                 thismat = ldict[col]
-            except KeyError:
+            except (KeyError, TypeError):
                 try:
                     thismat = gdict[col]
                 except KeyError as e:
                     raise NameError(f"name {col!r} is not defined") from None
```