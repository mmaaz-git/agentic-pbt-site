# Bug Report: numpy.matrixlib.bmat TypeError When Using gdict Without ldict

**Target**: `numpy.matrixlib.bmat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `bmat()` function crashes with a TypeError when providing `gdict` parameter while leaving `ldict` as None, despite both parameters being documented as optional and independent.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st

@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5)
)
def test_bmat_gdict_without_ldict(rows, cols):
    """Test that bmat works with gdict provided but ldict=None.

    According to documentation, both ldict and gdict are optional parameters.
    This test verifies that providing only gdict (with ldict=None) should work.
    """
    A = np.matrix(np.ones((rows, cols)))
    result = np.bmat('A', ldict=None, gdict={'A': A})
    assert np.array_equal(result, A), f"Expected result to equal A, got {result}"

if __name__ == "__main__":
    # Run the property-based test
    test_bmat_gdict_without_ldict()
```

<details>

<summary>
**Failing input**: `rows=1, cols=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 20, in <module>
    test_bmat_gdict_without_ldict()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 5, in test_bmat_gdict_without_ldict
    st.integers(min_value=1, max_value=5),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 15, in test_bmat_gdict_without_ldict
    result = np.bmat('A', ldict=None, gdict={'A': A})
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1107, in bmat
    return matrix(_from_string(obj, glob_dict, loc_dict))
                  ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1030, in _from_string
    thismat = ldict[col]
              ~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
Falsifying example: test_bmat_gdict_without_ldict(
    rows=1,
    cols=1,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Create a simple matrix
A = np.matrix([[1, 2], [3, 4]])

# Try to use bmat with gdict but no ldict (ldict=None)
# According to the documentation, both are optional parameters
try:
    result = np.bmat('A', ldict=None, gdict={'A': A})
    print(f"Success: Result is\n{result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError: 'NoneType' object is not subscriptable
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/repo.py", line 9, in <module>
    result = np.bmat('A', ldict=None, gdict={'A': A})
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1107, in bmat
    return matrix(_from_string(obj, glob_dict, loc_dict))
                  ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1030, in _from_string
    thismat = ldict[col]
              ~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
Error: TypeError: 'NoneType' object is not subscriptable
```
</details>

## Why This Is A Bug

This violates the documented API contract where both `ldict` and `gdict` are marked as optional parameters with default values of `None`. The function signature `bmat(obj, ldict=None, gdict=None)` indicates these parameters should be independently optional.

The documentation states:
- `ldict`: "dict, optional - A dictionary that replaces local operands in current frame. Ignored if `obj` is not a string or `gdict` is None."
- `gdict`: "dict, optional - A dictionary that replaces global operands in current frame. Ignored if `obj` is not a string."

The phrase "Ignored if `obj` is not a string or `gdict` is None" for `ldict` does NOT imply that `ldict` must be non-None when `gdict` is provided. It only describes when `ldict` is ignored (i.e., when `obj` is not a string OR when `gdict` is None).

The crash occurs because when `gdict` is provided but `ldict` is None, the code at line 1105 in defmatrix.py sets `loc_dict = ldict` (which is None), then passes it to `_from_string()`. The `_from_string()` function at line 1030 attempts to subscript `ldict[col]` without checking if `ldict` is None, causing the TypeError.

## Relevant Context

- The bug is in `/numpy/matrixlib/defmatrix.py` at lines 1103-1105 where `loc_dict` is set to `ldict` without handling the None case
- The `_from_string()` function expects non-None dictionaries and tries local lookup first (line 1030) before falling back to global (line 1033)
- Workaround exists: users can pass an empty dict `{}` for `ldict` instead of None
- The numpy.matrixlib module is deprecated in favor of regular numpy arrays, but the bug still violates the documented API
- Other parameter combinations work correctly:
  - `bmat('A')` - uses caller's frame (works)
  - `bmat('A', ldict={'A': matrix})` - uses provided ldict (works)
  - `bmat('A', ldict={}, gdict={'A': matrix})` - uses both (works)

Code locations:
- Bug location: `/numpy/matrixlib/defmatrix.py:1105`
- Crash location: `/numpy/matrixlib/defmatrix.py:1030`

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