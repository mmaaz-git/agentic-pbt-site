# Bug Report: scipy.io.matlab Empty 1D Array Ignores oned_as Parameter

**Target**: `scipy.io.matlab._miobase.matdims`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `oned_as` parameter in `scipy.io.matlab.savemat` is completely ignored for empty 1D NumPy arrays, always producing shape `(0, 0)` regardless of whether `oned_as='row'` or `oned_as='column'` is specified, breaking consistency with non-empty 1D arrays.

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

    assert loaded['x'].shape == expected_shape, f"Expected shape {expected_shape} but got {loaded['x'].shape} for lst={lst}"


if __name__ == "__main__":
    test_roundtrip_1d_int_array_row_orientation()
```

<details>

<summary>
**Failing input**: `lst=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 29, in <module>
    test_roundtrip_1d_int_array_row_orientation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 8, in test_roundtrip_1d_int_array_row_orientation
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 25, in test_roundtrip_1d_int_array_row_orientation
    assert loaded['x'].shape == expected_shape, f"Expected shape {expected_shape} but got {loaded['x'].shape} for lst={lst}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected shape (1, 0) but got (0, 0) for lst=[]
Falsifying example: test_roundtrip_1d_int_array_row_orientation(
    lst=[],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_miobase.py:327
        /home/npc/pbt/agentic-pbt/worker_/16/hypo.py:20
```
</details>

## Reproducing the Bug

```python
import io
import numpy as np
from scipy.io.matlab import loadmat, savemat

# Test empty array with oned_as='row'
arr = np.array([])

f_row = io.BytesIO()
savemat(f_row, {'x': arr}, oned_as='row')
f_row.seek(0)
loaded_row = loadmat(f_row)

print(f"Empty array with oned_as='row': {loaded_row['x'].shape}")
print(f"Expected: (1, 0), Actual: {loaded_row['x'].shape}")

# Test empty array with oned_as='column'
f_col = io.BytesIO()
savemat(f_col, {'x': arr}, oned_as='column')
f_col.seek(0)
loaded_col = loadmat(f_col)

print(f"\nEmpty array with oned_as='column': {loaded_col['x'].shape}")
print(f"Expected: (0, 1), Actual: {loaded_col['x'].shape}")

# Test non-empty array for comparison
non_empty = np.array([1, 2, 3])
f_row_ne = io.BytesIO()
savemat(f_row_ne, {'x': non_empty}, oned_as='row')
f_row_ne.seek(0)
loaded_row_ne = loadmat(f_row_ne)
print(f"\nNon-empty array with oned_as='row': {loaded_row_ne['x'].shape}")
print(f"Expected: (1, 3), Actual: {loaded_row_ne['x'].shape}")

# Also test 2D empty arrays to show they work correctly
print("\n2D empty arrays (for comparison):")
for shape in [(1, 0), (0, 1), (0, 0)]:
    arr_2d = np.zeros(shape)
    f = io.BytesIO()
    savemat(f, {'x': arr_2d})
    f.seek(0)
    loaded = loadmat(f)
    print(f"Shape {shape} -> {loaded['x'].shape} (should match)")
```

<details>

<summary>
Output showing inconsistent behavior for empty 1D arrays
</summary>
```
Empty array with oned_as='row': (0, 0)
Expected: (1, 0), Actual: (0, 0)

Empty array with oned_as='column': (0, 0)
Expected: (0, 1), Actual: (0, 0)

Non-empty array with oned_as='row': (1, 3)
Expected: (1, 3), Actual: (1, 3)

2D empty arrays (for comparison):
Shape (1, 0) -> (1, 0) (should match)
Shape (0, 1) -> (0, 1) (should match)
Shape (0, 0) -> (0, 0) (should match)
```
</details>

## Why This Is A Bug

The `oned_as` parameter is explicitly documented in `scipy.io.matlab.savemat` to control how 1D NumPy arrays are written: "If 'column', write 1-D NumPy arrays as column vectors. If 'row', write 1-D NumPy arrays as row vectors." This documentation makes no exception for empty arrays.

The bug violates expected behavior in several ways:

1. **Inconsistency**: The `oned_as` parameter works correctly for all non-empty 1D arrays but is completely ignored for empty 1D arrays. A 1D array with 3 elements respects `oned_as='row'` producing `(1, 3)`, but a 1D array with 0 elements ignores it, producing `(0, 0)` instead of `(1, 0)`.

2. **MATLAB Compatibility**: MATLAB explicitly supports and distinguishes between different empty array orientations. For example, MATLAB's `find()` function can return "1×0 empty double row vector" or "0×1 empty double column vector". The current behavior prevents proper MATLAB compatibility.

3. **Round-trip Property Violation**: The bug breaks the fundamental property that data should survive a save/load cycle with the same parameters. Users cannot reliably preserve the orientation of empty 1D arrays.

4. **Principle of Least Surprise**: The system already correctly handles 2D empty arrays with different shapes `(1, 0)`, `(0, 1)`, and `(0, 0)`, proving there's no technical limitation. Users reasonably expect 1D empty arrays to follow the same `oned_as` rules as non-empty ones.

## Relevant Context

The bug is located in `scipy/io/matlab/_miobase.py` in the `matdims` function at lines 326-327:

```python
if len(shape) == 1:  # 1D
    if shape[0] == 0:
        return (0, 0)  # This special case executes BEFORE checking oned_as
    elif oned_as == 'column':
        return shape + (1,)
    elif oned_as == 'row':
        return (1,) + shape
```

The special case for empty arrays (`shape[0] == 0`) returns `(0, 0)` immediately without ever checking the `oned_as` parameter. This appears to be an oversight rather than intentional design, as:
- The function's docstring shows `matdims(np.array([]))` returns `(0, 0)` but doesn't show examples with `oned_as` for empty arrays
- No documentation explains why empty arrays should ignore `oned_as`
- The behavior contradicts the parameter's documented purpose

Documentation links:
- SciPy savemat documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html
- Source code: https://github.com/scipy/scipy/blob/main/scipy/io/matlab/_miobase.py

## Proposed Fix

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