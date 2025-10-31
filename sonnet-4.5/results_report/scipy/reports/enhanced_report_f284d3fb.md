# Bug Report: scipy.io.matlab.savemat oned_as Parameter Ignored for Empty 1D Arrays

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `oned_as` parameter in `scipy.io.matlab.savemat` is completely ignored when saving empty 1D arrays, causing them to always be saved as (0,0) shape regardless of whether 'row' or 'column' is specified, breaking consistency with non-empty arrays.

## Property-Based Test

```python
import numpy as np
from io import BytesIO
from hypothesis import given, strategies as st
import scipy.io.matlab as matlab


@given(oned_as=st.sampled_from(['row', 'column']))
def test_oned_as_consistency_empty_arrays(oned_as):
    arr = np.array([])

    f = BytesIO()
    matlab.savemat(f, {'arr': arr}, oned_as=oned_as)
    f.seek(0)
    loaded = matlab.loadmat(f)
    result = loaded['arr']

    if oned_as == 'row':
        expected_shape = (1, 0)
    else:
        expected_shape = (0, 1)

    assert result.shape == expected_shape, (
        f"oned_as='{oned_as}' should produce shape {expected_shape}, "
        f"but got {result.shape}"
    )

# Run the test
if __name__ == "__main__":
    test_oned_as_consistency_empty_arrays()
```

<details>

<summary>
**Failing input**: `oned_as='row'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 29, in <module>
    test_oned_as_consistency_empty_arrays()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 8, in test_oned_as_consistency_empty_arrays
    def test_oned_as_consistency_empty_arrays(oned_as):
                  ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 22, in test_oned_as_consistency_empty_arrays
    assert result.shape == expected_shape, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: oned_as='row' should produce shape (1, 0), but got (0, 0)
Falsifying example: test_oned_as_consistency_empty_arrays(
    oned_as='row',
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from io import BytesIO
import scipy.io.matlab as matlab

# Test empty 1D arrays with different oned_as settings
arr_empty = np.array([])
print(f"Original empty array shape: {arr_empty.shape}")

for oned_as in ['row', 'column']:
    print(f"\nTesting with oned_as='{oned_as}':")

    # Save and load empty array
    f_empty = BytesIO()
    matlab.savemat(f_empty, {'arr': arr_empty}, oned_as=oned_as)
    f_empty.seek(0)
    loaded_empty = matlab.loadmat(f_empty)['arr']

    print(f"  Input shape:  {arr_empty.shape}")
    print(f"  Output shape: {loaded_empty.shape}")

    if oned_as == 'row':
        expected = (1, 0)
        print(f"  Expected:     {expected}")
        print(f"  Match: {loaded_empty.shape == expected}")
    else:
        expected = (0, 1)
        print(f"  Expected:     {expected}")
        print(f"  Match: {loaded_empty.shape == expected}")

# Compare with non-empty arrays for consistency
print("\n--- Comparison with non-empty arrays ---")
arr_nonempty = np.array([1, 2, 3])
print(f"Original non-empty array shape: {arr_nonempty.shape}")

for oned_as in ['row', 'column']:
    print(f"\nTesting with oned_as='{oned_as}':")

    # Save and load non-empty array
    f_nonempty = BytesIO()
    matlab.savemat(f_nonempty, {'arr': arr_nonempty}, oned_as=oned_as)
    f_nonempty.seek(0)
    loaded_nonempty = matlab.loadmat(f_nonempty)['arr']

    print(f"  Input shape:  {arr_nonempty.shape}")
    print(f"  Output shape: {loaded_nonempty.shape}")
```

<details>

<summary>
Empty arrays incorrectly saved as (0,0) regardless of oned_as parameter
</summary>
```
Original empty array shape: (0,)

Testing with oned_as='row':
  Input shape:  (0,)
  Output shape: (0, 0)
  Expected:     (1, 0)
  Match: False

Testing with oned_as='column':
  Input shape:  (0,)
  Output shape: (0, 0)
  Expected:     (0, 1)
  Match: False

--- Comparison with non-empty arrays ---
Original non-empty array shape: (3,)

Testing with oned_as='row':
  Input shape:  (3,)
  Output shape: (1, 3)

Testing with oned_as='column':
  Input shape:  (3,)
  Output shape: (3, 1)
```
</details>

## Why This Is A Bug

The `oned_as` parameter is documented to control how "1-D NumPy arrays" are written to MATLAB files, with no exceptions or special cases mentioned for empty arrays. According to the documentation:
- `oned_as='row'` should write 1-D arrays as row vectors (shape `(1, N)`)
- `oned_as='column'` should write 1-D arrays as column vectors (shape `(N, 1)`)

An empty array with shape `(0,)` is a valid 1-D NumPy array, and the transformation pattern is clear from non-empty arrays:
- Non-empty: `(3,)` with 'row' → `(1, 3)` (prepend dimension of size 1)
- Non-empty: `(3,)` with 'column' → `(3, 1)` (append dimension of size 1)

Following this same pattern:
- Empty: `(0,)` with 'row' should → `(1, 0)` (prepend dimension of size 1)
- Empty: `(0,)` with 'column' should → `(0, 1)` (append dimension of size 1)

The current behavior violates this expected consistency, making it impossible to correctly save and distinguish between empty row vectors `(1, 0)` and empty column vectors `(0, 1)` when starting from a 1D array. This breaks round-trip data integrity and MATLAB interoperability for empty vectors.

## Relevant Context

The bug originates in the `matdims` function in `scipy/io/matlab/_miobase.py` (lines 326-327). The function has a special case that returns `(0, 0)` for any empty 1D array, completely bypassing the `oned_as` parameter logic:

```python
def matdims(arr, oned_as='column'):
    # ...
    if len(shape) == 1:  # 1D
        if shape[0] == 0:
            return (0, 0)  # Bug: ignores oned_as parameter
        elif oned_as == 'column':
            return shape + (1,)
        elif oned_as == 'row':
            return (1,) + shape
```

MATLAB does support empty matrices with shapes `(1, 0)` and `(0, 1)`, which represent empty row and column vectors respectively. These have semantic meaning in MATLAB and are distinct from `(0, 0)` empty matrices.

Documentation reference: The `matdims` function's own docstring shows examples but notably demonstrates the bug in line 302-303, showing that `np.array([])` returns `(0, 0)` without any qualification about the `oned_as` parameter being ignored.

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
         if oned_as == 'column':
             return shape + (1,)
         elif oned_as == 'row':
```