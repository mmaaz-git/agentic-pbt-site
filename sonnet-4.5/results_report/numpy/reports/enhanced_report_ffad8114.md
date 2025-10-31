# Bug Report: numpy.ctypeslib.ndpointer Accepts Invalid Negative ndim Values

**Target**: `numpy.ctypeslib.ndpointer`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ndpointer` function accepts negative `ndim` values without validation, creating pointer types with semantically invalid dimension requirements that produce confusing error messages when used.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.ctypeslib
import numpy as np


@given(st.integers(min_value=-1000, max_value=-1))
@settings(max_examples=200)
def test_ndpointer_negative_ndim(ndim):
    """Test that ndpointer rejects negative ndim values."""
    try:
        ptr = numpy.ctypeslib.ndpointer(ndim=ndim)
        assert False, f"Should reject negative ndim {ndim}"
    except (TypeError, ValueError):
        pass

# Run the test
if __name__ == "__main__":
    test_ndpointer_negative_ndim()
```

<details>

<summary>
**Failing input**: `ndim=-1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 18, in <module>
    test_ndpointer_negative_ndim()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 7, in test_ndpointer_negative_ndim
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 12, in test_ndpointer_negative_ndim
    assert False, f"Should reject negative ndim {ndim}"
           ^^^^^
AssertionError: Should reject negative ndim -1
Falsifying example: test_ndpointer_negative_ndim(
    ndim=-1,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib

# Test with negative ndim value
ptr = numpy.ctypeslib.ndpointer(ndim=-1)
print(f"Created pointer class: {ptr}")
print(f"Class name: {ptr.__name__}")
print(f"ndim attribute: {ptr._ndim_}")

# Try to use the pointer with an actual array
arr = np.array([1, 2, 3])
print(f"\nAttempting to validate array with shape {arr.shape} and ndim {arr.ndim}")
try:
    result = ptr.from_param(arr)
    print(f"Validation succeeded: {result}")
except TypeError as e:
    print(f"Validation failed with error: {e}")

# Test with another negative value
print("\n--- Testing with ndim=-10 ---")
ptr2 = numpy.ctypeslib.ndpointer(ndim=-10)
print(f"Created pointer class: {ptr2}")
print(f"Class name: {ptr2.__name__}")
print(f"ndim attribute: {ptr2._ndim_}")

try:
    result = ptr2.from_param(arr)
    print(f"Validation succeeded: {result}")
except TypeError as e:
    print(f"Validation failed with error: {e}")
```

<details>

<summary>
Output showing invalid negative dimension error messages
</summary>
```
Created pointer class: <class 'numpy.ctypeslib._ctypeslib.ndpointer_any_-1d'>
Class name: ndpointer_any_-1d
ndim attribute: -1

Attempting to validate array with shape (3,) and ndim 1
Validation failed with error: array must have -1 dimension(s)

--- Testing with ndim=-10 ---
Created pointer class: <class 'numpy.ctypeslib._ctypeslib.ndpointer_any_-10d'>
Class name: ndpointer_any_-10d
ndim attribute: -10
Validation failed with error: array must have -10 dimension(s)
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Semantic invalidity**: Arrays fundamentally cannot have negative dimensions. The concept of "-1 dimensions" or "-10 dimensions" is mathematically meaningless in the context of multi-dimensional arrays.

2. **Documentation contradiction**: The `ndim` parameter is documented as "Number of array dimensions" which inherently implies non-negative values. In NumPy, the `ndim` attribute of arrays is always ≥ 0 (0-dimensional arrays are scalars, 1-dimensional are vectors, etc.).

3. **Delayed validation**: The function violates the principle of early validation by accepting invalid input at creation time and only failing during use. This makes debugging harder as the error occurs far from where the invalid value was provided.

4. **Confusing error messages**: When validation fails, users see nonsensical messages like "array must have -1 dimension(s)" which provides no useful diagnostic information and may confuse users about what went wrong.

5. **API consistency**: Other NumPy functions that accept dimension parameters typically validate that dimensions are non-negative. This inconsistency breaks user expectations.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py` at the `ndpointer` function (lines 239-353). The function creates a pointer type class dynamically but never validates that the `ndim` parameter is non-negative.

The validation only happens later in the `from_param` class method (line 196) which checks `obj.ndim != cls._ndim_`, producing the confusing error message.

Key code locations:
- `ndpointer` function: `_ctypeslib.py:239-353`
- `from_param` validation: `_ctypeslib.py:194-196`
- Class name generation with negative ndim: `_ctypeslib.py:336` (formats as `"_%dd" % ndim`)

NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.ctypeslib.ndpointer.html

## Proposed Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -289,6 +289,10 @@ def ndpointer(dtype=None, ndim=None, shape=None, flags=None):

     # normalize dtype to dtype | None
     if dtype is not None:
         dtype = np.dtype(dtype)
+
+    # validate ndim is non-negative
+    if ndim is not None and ndim < 0:
+        raise ValueError(f"ndim must be non-negative, got {ndim}")

     # normalize flags to int | None
```