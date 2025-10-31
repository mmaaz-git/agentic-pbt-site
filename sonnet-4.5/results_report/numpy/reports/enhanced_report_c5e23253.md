# Bug Report: numpy.matrixlib.defmatrix Dead Code Due to Logic Error

**Target**: `numpy.matrixlib.defmatrix.matrix.__new__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

A logic error in numpy.matrixlib.defmatrix.py line 166 makes the condition always evaluate to False, rendering line 167 (`arr = arr.copy()`) as unreachable dead code that can never execute.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, example
from hypothesis.extra import numpy as hnp


@given(hnp.arrays(dtype=np.float64, shape=(4, 4)))
@example(np.array([[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]]))
def test_dead_code_bug_noncontiguous_copy(arr):
    non_contiguous = arr[::2, ::2]

    if not non_contiguous.flags.contiguous:
        m = np.matrix(non_contiguous, copy=False)

        if not m.flags.c_contiguous and not m.flags.f_contiguous:
            assert False, "Matrix from non-contiguous array is non-contiguous"

if __name__ == "__main__":
    test_dead_code_bug_noncontiguous_copy()
```

<details>

<summary>
**Failing input**: `np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 21, in <module>
    test_dead_code_bug_noncontiguous_copy()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 7, in test_dead_code_bug_noncontiguous_copy
    @example(np.array([[1.0, 2.0, 3.0, 4.0],
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 18, in test_dead_code_bug_noncontiguous_copy
    assert False, "Matrix from non-contiguous array is non-contiguous"
           ^^^^^
AssertionError: Matrix from non-contiguous array is non-contiguous
Falsifying explicit example: test_dead_code_bug_noncontiguous_copy(
    arr=array([[ 1.,  2.,  3.,  4.],
           [ 5.,  6.,  7.,  8.],
           [ 9., 10., 11., 12.],
           [13., 14., 15., 16.]]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Create a 4x4 array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])

# Create a non-contiguous view by slicing every other row and column
non_contiguous = arr[::2, ::2]

print(f"Original array shape: {arr.shape}")
print(f"Non-contiguous view shape: {non_contiguous.shape}")
print(f"Non-contiguous view:\n{non_contiguous}")
print(f"Is input contiguous? {non_contiguous.flags.contiguous}")
print(f"Is input C-contiguous? {non_contiguous.flags.c_contiguous}")
print(f"Is input F-contiguous? {non_contiguous.flags.f_contiguous}")

# Create a matrix from the non-contiguous array with copy=False
m = np.matrix(non_contiguous, copy=False)

print(f"\nMatrix created with copy=False:")
print(f"Matrix:\n{m}")
print(f"Is matrix contiguous? {m.flags.contiguous}")
print(f"Is matrix C-contiguous? {m.flags.c_contiguous}")
print(f"Is matrix F-contiguous? {m.flags.f_contiguous}")

# Demonstrate the logic error
print("\n--- Logic Error Analysis ---")
order = 'C'  # This is always set to 'C' or 'F'
print(f"order = '{order}'")
print(f"bool(order) = {bool(order)}")
print(f"If order='C': (order or arr.flags.contiguous) = {bool(order or non_contiguous.flags.contiguous)}")
print(f"Therefore: not (order or arr.flags.contiguous) = {not (order or non_contiguous.flags.contiguous)}")
print("The condition is ALWAYS False, so line 167 (arr = arr.copy()) is NEVER executed!")
```

<details>

<summary>
Demonstrates that matrices created from non-contiguous arrays remain non-contiguous
</summary>
```
Original array shape: (4, 4)
Non-contiguous view shape: (2, 2)
Non-contiguous view:
[[ 1  3]
 [ 9 11]]
Is input contiguous? False
Is input C-contiguous? False
Is input F-contiguous? False

Matrix created with copy=False:
Matrix:
[[ 1  3]
 [ 9 11]]
Is matrix contiguous? False
Is matrix C-contiguous? False
Is matrix F-contiguous? False

--- Logic Error Analysis ---
order = 'C'
bool(order) = True
If order='C': (order or arr.flags.contiguous) = True
Therefore: not (order or arr.flags.contiguous) = False
The condition is ALWAYS False, so line 167 (arr = arr.copy()) is NEVER executed!
```
</details>

## Why This Is A Bug

The code at lines 162-167 in numpy/matrixlib/defmatrix.py contains a logic error that makes line 167 unreachable:

1. **The variable `order` is always a truthy string**: At lines 162-164, `order` is set to either 'C' or 'F', both of which are non-empty strings that evaluate to `True` in boolean context.

2. **Boolean OR with a truthy value always returns True**: The expression `(order or arr.flags.contiguous)` uses the OR operator. Since `order` is always truthy, this expression always evaluates to `True` regardless of the value of `arr.flags.contiguous`.

3. **The negation always yields False**: Therefore, `not (order or arr.flags.contiguous)` always evaluates to `False`, making the if-block at line 167 unreachable dead code.

4. **Intent vs. implementation mismatch**: The apparent intent was to copy non-contiguous arrays to make them contiguous (similar to how `numpy.array` behaves with `copy=True`). However, due to this logic error, non-contiguous arrays are never copied and remain non-contiguous when creating matrices with `copy=False`.

## Relevant Context

- **numpy.matrix is deprecated**: As of NumPy 1.15, the matrix class is deprecated and users are encouraged to use regular ndarray objects instead. This may explain why this bug has gone unnoticed.

- **Performance implications**: Non-contiguous arrays can have slower access patterns due to memory layout. The dead code was likely intended to optimize performance by ensuring matrices are always contiguous.

- **Code location**: The bug is in `/numpy/matrixlib/defmatrix.py` at lines 166-167, within the `matrix.__new__` method.

- **Similar behavior in numpy.array**: When using `numpy.array()`, the `copy` parameter works correctly:
  - `np.array(non_contiguous, copy=False)` preserves non-contiguity
  - `np.array(non_contiguous, copy=True)` creates a contiguous copy

## Proposed Fix

The fix is straightforward - remove `order` from the boolean expression since it's always truthy:

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -163,7 +163,7 @@ class matrix(N.ndarray):
         if (ndim == 2) and arr.flags.fortran:
             order = 'F'

-        if not (order or arr.flags.contiguous):
+        if not arr.flags.contiguous:
             arr = arr.copy()

         ret = N.ndarray.__new__(subtype, shape, arr.dtype,
```