# Bug Report: numpy.ctypeslib.ndpointer Accepts Non-Integer ndim Values

**Target**: `numpy.ctypeslib.ndpointer`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ndpointer` function accepts non-integer `ndim` values (specifically floats) without proper validation, violating its documented contract that requires `ndim` to be an integer. Floats that equal integers (like 2.0) are silently accepted, while other values cause various errors.

## Property-Based Test

```python
import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, strategies as st, settings, assume

@given(ndim_value=st.one_of(st.floats(), st.text(), st.lists(st.integers())))
@settings(max_examples=200)
def test_ndpointer_ndim_type_validation(ndim_value):
    assume(not isinstance(ndim_value, (int, type(None))))

    try:
        ptr = npc.ndpointer(ndim=ndim_value)
        arr = np.zeros((2, 3), dtype=np.int32)
        result = ptr.from_param(arr)
        assert False, f"Should reject non-integer ndim: {ndim_value}"
    except (TypeError, ValueError, OverflowError) as e:
        pass  # Expected - these exceptions are correct

if __name__ == "__main__":
    # Run the test to find failing examples
    test_ndpointer_ndim_type_validation()
```

<details>

<summary>
**Failing input**: `ndim=2.0`
</summary>
```
Testing ndim=2.0 (a float that equals integer 2):
ERROR: ndim=2.0 was accepted when it should have been rejected!
  Created pointer with _ndim_=2.0
  from_param returned: <numpy._core._internal._ctypes object at 0x7b6972b5c980>

============================================================
Running hypothesis test:

Failing input: `ndim=2.0`

Failing input: `ndim=2.0`

Failing input: `ndim=2.0`

Assertion Error: Should reject non-integer ndim: 2.0

This confirms the bug: float values that happen to equal integers (like 2.0) are incorrectly accepted.
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib as npc

# Test case that demonstrates the bug: float(2.0) is accepted but shouldn't be
print("Demonstrating the bug: ndim=2.0 (float) is incorrectly accepted")
print("="*60)

try:
    # This should raise TypeError because ndim should only accept integers
    ptr = npc.ndpointer(ndim=2.0)
    print(f"BUG: Float ndim=2.0 was accepted!")
    print(f"  Created pointer with _ndim_ = {ptr._ndim_} (type: {type(ptr._ndim_)})")

    # Show that it works with the matching array
    arr_2d = np.zeros((2, 3), dtype=np.int32)
    result = ptr.from_param(arr_2d)
    print(f"  from_param with (2,3) array succeeded: {result}")

    # But fails with wrong dimensions (as expected)
    arr_3d = np.zeros((2, 3, 4), dtype=np.int32)
    try:
        result = ptr.from_param(arr_3d)
    except TypeError as e:
        print(f"  from_param with (2,3,4) array failed correctly: {e}")

except (TypeError, ValueError) as e:
    print(f"Good: Correctly rejected with {type(e).__name__}: {e}")
```

<details>

<summary>
BUG: Float ndim=2.0 was accepted when it should have been rejected
</summary>
```
Demonstrating the bug: ndim=2.0 (float) is incorrectly accepted
============================================================
BUG: Float ndim=2.0 was accepted!
  Created pointer with _ndim_ = 2.0 (type: <class 'float'>)
  from_param with (2,3) array succeeded: <numpy._core._internal._ctypes object at 0x717762ecea50>
  from_param with (2,3,4) array failed correctly: array must have 2 dimension(s)
```
</details>

## Why This Is A Bug

The NumPy documentation explicitly states that `ndim` should be "int, optional" in the ndpointer function signature. However, the function currently accepts any value without validation, leading to multiple issues:

1. **Contract Violation**: Float values like 2.0 are accepted when only integers should be allowed according to the documentation
2. **Inconsistent Behavior**: Float values that equal integers (2.0) work silently, but fractional floats (2.5) cause confusing TypeErrors during array validation
3. **Poor Error Messages**: When ndim=2.5, the error message "array must have 2 dimension(s)" is misleading because no array can have 2.5 dimensions
4. **Type Storage Issue**: The _ndim_ attribute stores the float directly (e.g., 2.0 instead of converting to 2), breaking type expectations
5. **Name Generation Failures**: Special float values cause crashes during type name generation (inf causes OverflowError, nan causes ValueError)

This violates the principle of fail-fast design - the function should validate input types at the point of creation, not during later usage.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py` at line 239-353 in the `ndpointer` function. The function validates and normalizes other parameters (`dtype`, `flags`, `shape`) but skips validation for `ndim`.

Line 336 shows the problematic usage:
```python
name += "_%dd" % ndim
```
This string formatting assumes `ndim` is an integer-like value, which fails for inf/nan.

Line 195-196 shows where the stored `ndim` value is used for validation:
```python
if cls._ndim_ is not None and obj.ndim != cls._ndim_:
    raise TypeError("array must have %d dimension(s)" % cls._ndim_)
```
This comparison uses the float value directly, which can work for floats like 2.0 but creates misleading error messages for values like 2.5.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.ctypeslib.ndpointer.html

## Proposed Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -290,6 +290,11 @@ def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
     # normalize dtype to dtype | None
     if dtype is not None:
         dtype = np.dtype(dtype)
+
+    # validate ndim to be integer | None
+    if ndim is not None and not isinstance(ndim, (int, np.integer)):
+        raise TypeError(f"ndim must be an integer or None, not {type(ndim).__name__}")

     # normalize flags to int | None
     num = None
```