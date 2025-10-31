# Bug Report: numpy.ma.mask_or Crashes When Given Python Lists as Inputs

**Target**: `numpy.ma.mask_or`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.ma.mask_or` crashes with `AttributeError: 'NoneType' object has no attribute 'names'` when given Python lists as inputs, despite its docstring explicitly stating it accepts "array_like" inputs which should include lists according to NumPy conventions.

## Property-Based Test

```python
import numpy.ma as ma
from hypothesis import given, settings, strategies as st, assume


@settings(max_examples=1000)
@given(st.lists(st.booleans(), min_size=1, max_size=20))
def test_mask_or_accepts_lists(mask1):
    assume(len(mask1) > 0)
    mask2 = [not m for m in mask1]
    result = ma.mask_or(mask1, mask2)

# Run the test
if __name__ == "__main__":
    test_mask_or_accepts_lists()
```

<details>

<summary>
**Failing input**: `mask1=[False]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 14, in <module>
    test_mask_or_accepts_lists()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 6, in test_mask_or_accepts_lists
    @given(st.lists(st.booleans(), min_size=1, max_size=20))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 10, in test_mask_or_accepts_lists
    result = ma.mask_or(mask1, mask2)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py", line 1808, in mask_or
    if dtype1.names is not None:
       ^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'names'
Falsifying example: test_mask_or_accepts_lists(
    mask1=[False],
)
```
</details>

## Reproducing the Bug

```python
import numpy.ma as ma

# Test case that should work according to documentation (array_like inputs)
mask1 = [False, True, False]
mask2 = [True, False, False]

print("Testing numpy.ma.mask_or with Python lists as inputs:")
print(f"mask1 = {mask1}")
print(f"mask2 = {mask2}")
print()

try:
    result = ma.mask_or(mask1, mask2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
AttributeError when calling ma.mask_or with list inputs
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/repo.py", line 13, in <module>
    result = ma.mask_or(mask1, mask2)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py", line 1808, in mask_or
    if dtype1.names is not None:
       ^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'names'
Testing numpy.ma.mask_or with Python lists as inputs:
mask1 = [False, True, False]
mask2 = [True, False, False]

Error: AttributeError: 'NoneType' object has no attribute 'names'
```
</details>

## Why This Is A Bug

This violates the documented behavior of `numpy.ma.mask_or` in multiple ways:

1. **Documentation Contract Violation**: The function's docstring explicitly states it accepts "array_like" parameters:
   ```python
   Parameters
   ----------
   m1, m2 : array_like
       Input masks.
   ```

2. **NumPy Convention Violation**: According to NumPy's official glossary, "array_like" refers to "Any scalar or sequence that can be interpreted as an ndarray", which explicitly includes Python lists, tuples, and other sequences. Any argument accepted by `numpy.array` should be considered array_like.

3. **Inconsistency Within Module**: Other mask utility functions in the same `numpy.ma` module correctly handle list inputs:
   - `ma.make_mask([True, False, True])` works correctly and returns a proper mask array
   - `ma.getmask([1, 2, 3])` works correctly and returns `False`

4. **Inconsistency with Core NumPy**: The underlying NumPy function `np.logical_or([False, True], [True, False])` correctly handles list inputs and returns the expected result.

5. **Poor Error Handling**: The function crashes with an uninformative `AttributeError` rather than either:
   - Converting the lists to arrays as expected for array_like inputs
   - Raising a clear error message about input requirements

The error occurs at line 1808 in `numpy/ma/core.py` where the code attempts to access `dtype1.names` without checking if `dtype1` is `None`. When Python lists are passed as inputs, `getattr(m1, 'dtype', None)` returns `None` (since lists don't have a dtype attribute), causing the subsequent attribute access to fail.

## Relevant Context

The bug occurs in the control flow of `mask_or` after handling special cases (nomask inputs). The problematic code section:

```python
(dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
if dtype1 != dtype2:
    raise ValueError(f"Incompatible dtypes '{dtype1}'<>'{dtype2}'")
if dtype1.names is not None:  # Line 1808 - crashes here when dtype1 is None
    # ... structured array handling
```

The function works correctly when given NumPy arrays:
- `ma.mask_or(np.array([False, True, False]), np.array([True, False, False]))` returns `array([True, True, False])`

This indicates the core logic is sound but the function fails to properly handle array_like inputs as promised in its documentation.

**Documentation link**: https://numpy.org/doc/stable/reference/generated/numpy.ma.mask_or.html
**Source code**: numpy/ma/core.py:1759-1813

## Proposed Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -1803,6 +1803,10 @@ def mask_or(m1, m2, copy=False, shrink=True):
         return make_mask(m1, copy=copy, shrink=shrink, dtype=dtype)
     if m1 is m2 and is_mask(m1):
         return _shrink_mask(m1) if shrink else m1
+    # Convert array_like inputs to arrays if needed
+    if not hasattr(m1, 'dtype'):
+        m1 = np.asarray(m1)
+    if not hasattr(m2, 'dtype'):
+        m2 = np.asarray(m2)
     (dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
     if dtype1 != dtype2:
         raise ValueError(f"Incompatible dtypes '{dtype1}'<>'{dtype2}'")
```