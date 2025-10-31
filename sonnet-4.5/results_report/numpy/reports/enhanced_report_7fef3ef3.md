# Bug Report: numpy.ma.array Shrink Parameter Does Not Compress Python False Masks

**Target**: `numpy.ma.array`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When creating a masked array with `mask=False` (Python bool) and `shrink=True`, the mask is incorrectly expanded to a boolean array instead of being compressed to `nomask`, unlike the behavior with `mask=np.False_` which correctly compresses.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis that discovers the numpy.ma shrink bug.
Tests that mask=False and mask=np.False_ should behave identically when shrink=True.
"""

import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=20)
)
@settings(max_examples=500)
def test_shrink_with_scalar_false_mask(data_list):
    """
    Test that both Python False and np.False_ shrink to nomask when shrink=True.

    The shrink parameter is documented to compress empty masks to nomask.
    Since both False and np.False_ represent empty masks (no masked elements),
    they should both be compressed to nomask when shrink=True.
    """
    data = np.array(data_list)

    # Create masked arrays with both types of False
    arr_with_npfalse = ma.array(data, mask=np.False_, shrink=True)
    arr_with_pyfalse = ma.array(data, mask=False, shrink=True)

    # Get the masks
    mask_npfalse = ma.getmask(arr_with_npfalse)
    mask_pyfalse = ma.getmask(arr_with_pyfalse)

    # Both should be nomask (compressed) since they represent empty masks
    assert mask_npfalse is ma.nomask, \
        f"np.False_ with shrink=True should give nomask, got {mask_npfalse}"
    assert mask_pyfalse is ma.nomask, \
        f"Python False with shrink=True should give nomask, got {mask_pyfalse}"

if __name__ == "__main__":
    # Run the test
    test_shrink_with_scalar_false_mask()
```

<details>

<summary>
**Failing input**: `data_list=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 41, in <module>
    test_shrink_with_scalar_false_mask()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 12, in test_shrink_with_scalar_false_mask
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=20)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 36, in test_shrink_with_scalar_false_mask
    assert mask_pyfalse is ma.nomask, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Python False with shrink=True should give nomask, got [False]
Falsifying example: test_shrink_with_scalar_false_mask(
    data_list=[0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction demonstrating the numpy.ma shrink parameter bug.
When creating a masked array with mask=False (Python bool) and shrink=True,
the mask is incorrectly expanded to a boolean array instead of being compressed
to nomask, unlike the behavior with mask=np.False_.
"""

import numpy as np
import numpy.ma as ma

# Test with np.False_ (numpy scalar) - works correctly
arr_correct = ma.array([1, 2, 3], mask=np.False_, shrink=True)
mask_correct = ma.getmask(arr_correct)
print(f"mask=np.False_ with shrink=True:")
print(f"  Mask: {mask_correct}")
print(f"  Type: {type(mask_correct)}")
print(f"  Is nomask? {mask_correct is ma.nomask}")
print()

# Test with Python False - demonstrates the bug
arr_buggy = ma.array([1, 2, 3], mask=False, shrink=True)
mask_buggy = ma.getmask(arr_buggy)
print(f"mask=False with shrink=True:")
print(f"  Mask: {mask_buggy}")
print(f"  Type: {type(mask_buggy)}")
print(f"  Is nomask? {mask_buggy is ma.nomask}")
print()

# Show that both should be equivalent (all False values, no masked elements)
print("Comparison:")
print(f"  Both masks have no True values: {not np.any(mask_correct) and not np.any(mask_buggy)}")
print(f"  Expected behavior: Both should compress to nomask when shrink=True")
print(f"  Actual behavior: Only np.False_ compresses, Python False expands to array")
print()

# Assert to demonstrate the bug
try:
    assert mask_correct is ma.nomask, "np.False_ should shrink to nomask"
    print("✓ np.False_ correctly shrinks to nomask")
except AssertionError as e:
    print(f"✗ {e}")

try:
    assert mask_buggy is ma.nomask, "Python False should also shrink to nomask"
    print("✓ Python False correctly shrinks to nomask")
except AssertionError as e:
    print(f"✗ {e}")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
mask=np.False_ with shrink=True:
  Mask: False
  Type: <class 'numpy.bool'>
  Is nomask? True

mask=False with shrink=True:
  Mask: [False False False]
  Type: <class 'numpy.ndarray'>
  Is nomask? False

Comparison:
  Both masks have no True values: True
  Expected behavior: Both should compress to nomask when shrink=True
  Actual behavior: Only np.False_ compresses, Python False expands to array

✓ np.False_ correctly shrinks to nomask
✗ Python False should also shrink to nomask
```
</details>

## Why This Is A Bug

The `shrink` parameter in `numpy.ma.array` is documented to "force compression of an empty mask." An empty mask is one with no `True` values (no masked elements). Both `mask=False` (Python bool) and `mask=np.False_` (NumPy scalar) semantically represent empty masks.

The bug arises from an implementation detail in the MaskedArray `__new__` method. The code uses identity comparison (`mask is nomask`) to check if the mask is `nomask` (which is `np.False_`). Since Python `False` and `np.False_` are not the same object (even though they have the same value), Python `False` fails this identity check and is treated differently:

1. When `mask=np.False_` is passed, it matches `mask is nomask` and the shrink logic is properly applied
2. When `mask=False` is passed, it doesn't match the identity check, so it's converted to `np.zeros(_data.shape, dtype=mdtype)` without applying the shrink logic

This inconsistency violates the documented behavior and causes:
- **Memory inefficiency**: Unnecessary boolean arrays are stored instead of using the singleton `nomask`
- **API inconsistency**: Semantically equivalent inputs produce different results
- **Documentation violation**: The shrink parameter doesn't work as documented for Python bool values

## Relevant Context

The root cause is in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py` in the `MaskedArray.__new__` method around lines 2919-2968:

- Line 2919 checks `if mask is nomask:` using identity comparison
- Lines 2967-2968 handle `mask is False` by creating a zeros array but don't apply shrink
- The shrink logic is only applied in the `mask is nomask` branch (lines 2924-2928)

NumPy's documentation actually recommends using Python `True`/`False` for mask values, making this bug particularly relevant. The `nomask` constant is defined as `np.False_` (see line 3 in numpy/ma/core.py: `nomask = np.False_`).

Related documentation:
- numpy.ma.array documentation: https://numpy.org/doc/stable/reference/generated/numpy.ma.array.html
- numpy.ma.make_mask documentation shows shrink behavior: https://numpy.org/doc/stable/reference/generated/numpy.ma.make_mask.html

## Proposed Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -2965,8 +2965,14 @@ class MaskedArray(ndarray):
             if mask is True and mdtype == MaskType:
                 mask = np.ones(_data.shape, dtype=mdtype)
             elif mask is False and mdtype == MaskType:
-                mask = np.zeros(_data.shape, dtype=mdtype)
+                # Apply shrink logic for Python False just like np.False_
+                if shrink:
+                    mask = nomask
+                else:
+                    mask = np.zeros(_data.shape, dtype=mdtype)
             else:
                 # Read the mask with the current mdtype
                 try:
                     mask = np.array(mask, copy=copy, dtype=mdtype)
                 # Or assume it's a sequence of bool/int
@@ -2988,7 +2994,9 @@ class MaskedArray(ndarray):
                 copy = True
             # Set the mask to the new value
             if _data._mask is nomask:
-                _data._mask = mask
+                # Apply shrink if we have an all-False mask array
+                if shrink and mask is not nomask and hasattr(mask, 'any') and not mask.any():
+                    _data._mask = nomask
+                else:
+                    _data._mask = mask
                 _data._sharedmask = not copy
```