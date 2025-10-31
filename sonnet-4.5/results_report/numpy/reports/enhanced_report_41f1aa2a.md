# Bug Report: numpy.ma.unique Returns Multiple Masked Values

**Target**: `numpy.ma.unique`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ma.unique()` violates its documented contract by returning multiple masked values when they have different underlying data, instead of treating all masked values as a single unique element as promised in the documentation.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_arrays(draw, dtype=np.int64, max_dims=1, max_side=20):
    shape = draw(npst.array_shapes(max_dims=max_dims, max_side=max_side))
    data = draw(npst.arrays(dtype=dtype, shape=shape))
    mask = draw(npst.arrays(dtype=bool, shape=shape))
    return ma.array(data, mask=mask)

@given(masked_arrays())
@settings(max_examples=500)
def test_unique_treats_all_masked_as_one(arr):
    unique_vals = ma.unique(arr)
    masked_count = sum(1 for val in unique_vals if ma.is_masked(val))
    if ma.getmaskarray(arr).any():
        assert masked_count <= 1, f"Expected at most 1 masked value, got {masked_count}. Input: {arr}, Unique: {unique_vals}"

if __name__ == "__main__":
    test_unique_treats_all_masked_as_one()
```

<details>

<summary>
**Failing input**: `masked_array(data=[--, --, 9223372036854775807, 0, 0, 0], mask=[True, True, False, False, False, False])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 22, in <module>
    test_unique_treats_all_masked_as_one()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 14, in test_unique_treats_all_masked_as_one
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 19, in test_unique_treats_all_masked_as_one
    assert masked_count <= 1, f"Expected at most 1 masked value, got {masked_count}. Input: {arr}, Unique: {unique_vals}"
           ^^^^^^^^^^^^^^^^^
AssertionError: Expected at most 1 masked value, got 2. Input: [-- -- 9223372036854775807 0 0 0], Unique: [0 -- 9223372036854775807 --]
Falsifying example: test_unique_treats_all_masked_as_one(
    arr=masked_array(data=[--, --, 9223372036854775807, 0, 0, 0],
                 mask=[ True,  True, False, False, False, False],
           fill_value=999999),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

# Create a masked array with multiple masked values having different underlying data
arr = ma.array([999, 9223372036854775807, 888], mask=[True, False, True])
unique_result = ma.unique(arr)

print(f"Input array: {arr}")
print(f"Input data: {arr.data}")
print(f"Input mask: {arr.mask}")
print()
print(f"Unique result: {unique_result}")
print(f"Unique data: {unique_result.data}")
print(f"Unique mask: {unique_result.mask}")
print()

# Count masked values in the result
masked_count = sum(1 for val in unique_result if ma.is_masked(val))
print(f"Number of masked values in unique result: {masked_count}")
print(f"Expected: At most 1 (per documentation: 'Masked values are considered the same element')")
print(f"Actual: {masked_count}")
print()

if masked_count > 1:
    print("BUG CONFIRMED: Multiple masked values returned instead of treating all masked as one element")
else:
    print("No bug detected")
```

<details>

<summary>
BUG CONFIRMED: Multiple masked values returned (2 instead of 1)
</summary>
```
Input array: [-- 9223372036854775807 --]
Input data: [                999 9223372036854775807                 888]
Input mask: [ True False  True]

Unique result: [-- 9223372036854775807 --]
Unique data: [                999 9223372036854775807                 888]
Unique mask: [ True False  True]

Number of masked values in unique result: 2
Expected: At most 1 (per documentation: 'Masked values are considered the same element')
Actual: 2

BUG CONFIRMED: Multiple masked values returned instead of treating all masked as one element
```
</details>

## Why This Is A Bug

This is a clear violation of the documented behavior. The numpy.ma.unique docstring explicitly states at line 1329 of `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py`:

> "Masked values are considered the same element (masked)."

The documentation example (lines 1339-1349) demonstrates that even when there's a masked value (1000) in the input array, the output contains only ONE masked element in the result. This establishes the expected behavior.

The bug occurs because the current implementation (lines 1363-1372) simply:
1. Calls `np.unique()` on the raw data array without considering masks
2. Converts the result to a MaskedArray afterward
3. This causes masked values with different underlying data (999 and 888 in our example) to be treated as distinct elements

This violates the fundamental principle of masked arrays where masked values represent unknown, invalid, or missing data that should not be distinguished based on their underlying values.

## Relevant Context

- **Documentation location**: `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py:1325-1372`
- **NumPy documentation**: https://numpy.org/doc/stable/reference/generated/numpy.ma.unique.html
- **Related function**: `numpy.unique` operates on raw data without mask awareness

The issue affects scientific computing workflows where masked arrays are used to handle missing data. For example:
- Climate data with missing measurements at different stations
- Sensor readings with invalid values at different time points
- Statistical analysis where NaN or invalid values need special handling

Users expect masked values to be treated as a single "unknown" category, not multiple distinct unknowns based on arbitrary underlying data.

## Proposed Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -1360,6 +1360,16 @@ def unique(ar1, return_index=False, return_inverse=False):
                 fill_value=999999), array([0, 1, 4, 2]), array([0, 1, 3, 1, 2]))
     """
+    ar1 = np.ma.asanyarray(ar1)
+    if np.ma.is_masked(ar1):
+        mask = np.ma.getmaskarray(ar1)
+        if mask.any():
+            # Normalize all masked values to have the same underlying data
+            # so np.unique treats them as one element
+            data = np.ma.getdata(ar1).copy()
+            data[mask] = ar1.fill_value if hasattr(ar1, 'fill_value') else 999999
+            ar1 = np.ma.array(data, mask=mask, fill_value=ar1.fill_value if hasattr(ar1, 'fill_value') else 999999)
+
     output = np.unique(ar1,
                        return_index=return_index,
                        return_inverse=return_inverse)
```