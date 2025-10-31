# Bug Report: numpy.matrixlib.bmat Crashes When Using gdict Without ldict

**Target**: `numpy.matrixlib.bmat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `bmat` function crashes with `TypeError: 'NoneType' object is not subscriptable` when `gdict` is provided without `ldict`, despite `ldict` being documented as optional with a default value of `None`.

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
    """Test that bmat works with gdict parameter but without ldict.

    Since ldict has a default value of None in the function signature,
    it should be optional when using gdict.
    """
    m = matrix(arr)
    gdict = {varname: m}
    # This should work but crashes with TypeError
    result = bmat(varname, gdict=gdict)
    np.testing.assert_array_equal(result, m)

if __name__ == "__main__":
    # Run the test
    test_bmat_gdict_without_ldict()
```

<details>

<summary>
**Failing input**: `varname='A', arr=array([[0.]])`
</summary>
```
Falsifying example: test_bmat_gdict_without_ldict(
    # The test always failed when commented parts were varied together.
    varname='A',  # or any other generated value
    arr=array([[0.]]),  # or any other generated value
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 25, in <module>
    test_bmat_gdict_without_ldict()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 8, in test_bmat_gdict_without_ldict
    st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 20, in test_bmat_gdict_without_ldict
    result = bmat(varname, gdict=gdict)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1107, in bmat
    return matrix(_from_string(obj, glob_dict, loc_dict))
                  ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py", line 1030, in _from_string
    thismat = ldict[col]
              ~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
```
</details>

## Reproducing the Bug

```python
import numpy as np
from numpy.matrixlib import bmat, matrix

# Create a simple matrix
X = matrix([[1, 2]])

# Try to use bmat with gdict parameter but without ldict
# According to the function signature, ldict has a default value of None
# so this should work
try:
    result = bmat('X', gdict={'X': X})
    print(f"Success: result = {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
Error: TypeError: 'NoneType' object is not subscriptable
</summary>
```
Error: TypeError: 'NoneType' object is not subscriptable
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/repo.py", line 11, in <module>
    result = bmat('X', gdict={'X': X})
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

This violates expected behavior because:

1. **API Contract Violation**: The function signature explicitly defines `ldict=None` as a default parameter:
   ```python
   def bmat(obj, ldict=None, gdict=None):
   ```
   This clearly indicates that `ldict` should be optional.

2. **Documentation States Optional**: The docstring explicitly marks `ldict` as "optional" with the description: "A dictionary that replaces local operands in current frame. Ignored if `obj` is not a string or `gdict` is None."

3. **Reasonable User Expectation**: Users would reasonably expect to be able to provide `gdict` alone to override global variables while leaving locals empty/unused, especially since both parameters are marked as optional.

4. **Inconsistent Implementation**: The implementation incorrectly assumes `ldict` is always a dictionary when `gdict` is provided, leading to a crash when trying to subscript `None` at line 1030 in `_from_string`:
   ```python
   thismat = ldict[col]  # Crashes when ldict is None
   ```

5. **Unintentional Error**: The crash produces a low-level `TypeError` about NoneType subscription rather than a meaningful error message, indicating this is an unhandled edge case rather than intentional behavior.

## Relevant Context

The bug occurs in the implementation at lines 1103-1105 of `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py`:

```python
else:  # when gdict is not None
    glob_dict = gdict
    loc_dict = ldict  # Bug: ldict can be None here
```

When `gdict` is provided but `ldict` is not (defaults to None), `loc_dict` gets set to `None`. Later, in `_from_string` at line 1030, the code attempts to access `ldict[col]` which causes the TypeError.

**Workaround**: Users can work around this bug by explicitly passing an empty dictionary for `ldict`:
```python
result = bmat('X', ldict={}, gdict={'X': X})  # Works correctly
```

**Note**: While `numpy.matrix` is deprecated in favor of regular arrays (as indicated by the PendingDeprecationWarning), it remains part of NumPy's public API and crashes in public APIs should still be fixed.

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