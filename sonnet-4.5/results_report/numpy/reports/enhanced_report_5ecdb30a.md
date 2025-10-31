# Bug Report: numpy.ma clump_masked/clump_unmasked IndexError on Empty Arrays

**Target**: `numpy.ma.clump_masked`, `numpy.ma.clump_unmasked`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The functions `numpy.ma.clump_masked()` and `numpy.ma.clump_unmasked()` crash with an IndexError when given empty masked arrays because the underlying `_ezclump()` helper function attempts to access the first element of the mask array without checking if the array is empty.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test that discovers the numpy.ma clump_masked bug."""

import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_arrays_1d(draw, dtype=np.int64, max_size=50):
    size = draw(st.integers(min_value=0, max_value=max_size))
    data = draw(npst.arrays(dtype=dtype, shape=(size,),
                           elements=st.integers(min_value=-1000, max_value=1000)))
    mask = draw(npst.arrays(dtype=bool, shape=(size,)))
    return ma.array(data, mask=mask)

@given(masked_arrays_1d())
@settings(max_examples=500)
def test_clump_masked_partition(arr):
    clumps = ma.clump_masked(arr)
    mask = ma.getmaskarray(arr)
    covered_indices = set()
    for clump in clumps:
        for i in range(clump.start, clump.stop):
            assert mask[i]
            covered_indices.add(i)
    for i in range(len(arr)):
        if mask[i]:
            assert i in covered_indices

if __name__ == "__main__":
    test_clump_masked_partition()
```

<details>

<summary>
**Failing input**: `masked_array(data=[], mask=[], fill_value=999999, dtype=int64)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 32, in <module>
    test_clump_masked_partition()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 18, in test_clump_masked_partition
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 20, in test_clump_masked_partition
    clumps = ma.clump_masked(arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py", line 2286, in clump_masked
    return _ezclump(mask)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py", line 2199, in _ezclump
    if mask[0]:
       ~~~~^^^
IndexError: index 0 is out of bounds for axis 0 with size 0
Falsifying example: test_clump_masked_partition(
    arr=masked_array(data=[],
                 mask=[],
           fill_value=999999,
                dtype=int64),
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of numpy.ma clump_masked/clump_unmasked bug with empty arrays."""

import numpy as np
import numpy.ma as ma

print("Testing numpy.ma clump functions with empty arrays")
print("=" * 60)

# Create an empty masked array
empty_arr = ma.array([], dtype=int, mask=[])
print(f"Input array: {empty_arr}")
print(f"Array shape: {empty_arr.shape}")
print(f"Array mask: {ma.getmask(empty_arr)}")
print()

# Test clump_masked
print("Testing ma.clump_masked() with empty array:")
print("-" * 40)
try:
    result = ma.clump_masked(empty_arr)
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    import traceback
    traceback.print_exc()

print()

# Test clump_unmasked
print("Testing ma.clump_unmasked() with empty array:")
print("-" * 40)
try:
    result = ma.clump_unmasked(empty_arr)
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
IndexError: index 0 is out of bounds for axis 0 with size 0
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/repo.py", line 21, in <module>
    result = ma.clump_masked(empty_arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py", line 2286, in clump_masked
    return _ezclump(mask)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py", line 2199, in _ezclump
    if mask[0]:
       ~~~~^^^
IndexError: index 0 is out of bounds for axis 0 with size 0
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/repo.py", line 34, in <module>
    result = ma.clump_unmasked(empty_arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py", line 2250, in clump_unmasked
    return _ezclump(~mask)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py", line 2199, in _ezclump
    if mask[0]:
       ~~~~^^^
IndexError: index 0 is out of bounds for axis 0 with size 0
Testing numpy.ma clump functions with empty arrays
============================================================
Input array: []
Array shape: (0,)
Array mask: []

Testing ma.clump_masked() with empty array:
----------------------------------------
IndexError raised: index 0 is out of bounds for axis 0 with size 0

Testing ma.clump_unmasked() with empty array:
----------------------------------------
IndexError raised: index 0 is out of bounds for axis 0 with size 0
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Empty arrays are valid masked arrays in NumPy**: The numpy.ma module consistently treats empty arrays as valid inputs. Other functions like `ma.count()`, `ma.mean()`, and `ma.sum()` handle empty arrays gracefully without crashing.

2. **The crash occurs due to unchecked array access**: The `_ezclump()` helper function at line 2199 of `/numpy/ma/extras.py` attempts to access `mask[0]` without first verifying that the mask array has any elements. This is a basic bounds-checking error.

3. **Semantic violation**: The purpose of these functions is to find contiguous regions (clumps) of masked/unmasked elements. For an empty array with no elements, the logical result should be an empty list of slices `[]`, not a crash.

4. **Inconsistent with numpy conventions**: NumPy generally follows the principle of handling edge cases gracefully. Functions that operate on arrays should handle the full range of valid array sizes, including size 0.

5. **Documentation doesn't warn about this limitation**: Neither function's documentation mentions that empty arrays will cause crashes or are unsupported inputs.

## Relevant Context

The bug affects both `clump_masked()` and `clump_unmasked()` functions because they both rely on the same underlying `_ezclump()` helper function. The issue is at line 2199 of the numpy.ma.extras module where the code checks `if mask[0]:` without first ensuring the mask has at least one element.

This type of bug can cause production failures in data processing pipelines where array sizes can vary dynamically. For example, when filtering data based on conditions, it's common to end up with empty arrays that then need to be processed by various numpy functions.

**Source code location**: `/numpy/ma/extras.py:2199` in the `_ezclump()` function

**Documentation**:
- [numpy.ma.clump_masked](https://numpy.org/doc/stable/reference/generated/numpy.ma.clump_masked.html)
- [numpy.ma.clump_unmasked](https://numpy.org/doc/stable/reference/generated/numpy.ma.clump_unmasked.html)

## Proposed Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -2196,6 +2196,10 @@ def _ezclump(mask):
     idx = (mask[1:] ^ mask[:-1]).nonzero()
     idx = idx[0] + 1

+    # Handle empty arrays
+    if mask.size == 0:
+        return []
+
     if mask[0]:
         if len(idx) == 0:
             return [slice(0, mask.size)]
```