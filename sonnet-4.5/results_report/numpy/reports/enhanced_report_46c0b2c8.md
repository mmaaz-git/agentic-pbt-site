# Bug Report: numpy.ma.intersect1d Returns Multiple Masked Values

**Target**: `numpy.ma.intersect1d`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.ma.intersect1d` function violates its documented behavior by returning multiple masked elements in the result array when it should treat all masked values as equal and return at most one masked element.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.integer_dtypes(), shape=st.tuples(st.integers(2, 20))),
       npst.arrays(dtype=npst.integer_dtypes(), shape=st.tuples(st.integers(2, 20))),
       st.data())
def test_intersect1d_masked_handling(ar1, ar2, data):
    assume(ar1.size > 1 and ar2.size > 1)

    mask1 = data.draw(npst.arrays(dtype=np.bool_, shape=ar1.shape))
    mask2 = data.draw(npst.arrays(dtype=np.bool_, shape=ar2.shape))

    assume(np.sum(mask1) >= 1 and np.sum(mask2) >= 1)

    mar1 = ma.array(ar1, mask=mask1)
    mar2 = ma.array(ar2, mask=mask2)

    intersection = ma.intersect1d(mar1, mar2)

    masked_in_result = ma.getmaskarray(intersection)
    if masked_in_result is not ma.nomask:
        assert np.sum(masked_in_result) <= 1, f"Found {np.sum(masked_in_result)} masked values, expected at most 1"

if __name__ == "__main__":
    test_intersect1d_masked_handling()
```

<details>

<summary>
**Failing input**: `ar1=array([2147483647, 2147483647, 2147483647, 2147483647], dtype=int32)` with `mask=[False, True, False, True]`, `ar2=array([0, 0], dtype=int8)` with `mask=[False, True]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 27, in <module>
    test_intersect1d_masked_handling()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 7, in test_intersect1d_masked_handling
    npst.arrays(dtype=npst.integer_dtypes(), shape=st.tuples(st.integers(2, 20))),
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 24, in test_intersect1d_masked_handling
    assert np.sum(masked_in_result) <= 1, f"Found {np.sum(masked_in_result)} masked values, expected at most 1"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Found 2 masked values, expected at most 1
Falsifying example: test_intersect1d_masked_handling(
    ar1=array([2147483647, 2147483647, 2147483647, 2147483647], dtype=int32),
    ar2=array([0, 0], dtype=int8),  # or any other generated value
    data=data(...),
)
Draw 1: array([False,  True, False,  True])
Draw 2: array([False,  True])
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

ar1 = np.array([0, 0], dtype=np.int16)
mask1 = np.array([True, False])
mar1 = ma.array(ar1, mask=mask1)

ar2 = np.array([0, 127, 0], dtype=np.int8)
mask2 = np.array([True, False, True])
mar2 = ma.array(ar2, mask=mask2)

print('Input array 1:', mar1)
print('Input array 2:', mar2)

intersection = ma.intersect1d(mar1, mar2)
print('Result:', intersection)
print('Result mask:', ma.getmaskarray(intersection))
print('Number of masked values:', np.sum(ma.getmaskarray(intersection)))
print()
print('Expected: At most 1 masked value (doc says "Masked values are considered equal one to the other")')
print('Actual:', np.sum(ma.getmaskarray(intersection)), 'masked values')
```

<details>

<summary>
Result contains 2 masked values instead of at most 1
</summary>
```
Input array 1: [-- 0]
Input array 2: [-- 127 --]
Result: [-- --]
Result mask: [ True  True]
Number of masked values: 2

Expected: At most 1 masked value (doc says "Masked values are considered equal one to the other")
Actual: 2 masked values
```
</details>

## Why This Is A Bug

The numpy.ma.intersect1d documentation explicitly states: "Masked values are considered equal one to the other." This establishes a clear contract that all masked values should be treated as identical elements, regardless of their underlying data values.

From a set theory perspective, if all masked values are equal to each other, they represent a single unique element. Therefore, the intersection of two sets each containing one or more masked values should contain at most one masked element in the result - present only when both input arrays contain masked values.

The current implementation violates this contract by returning multiple masked values. In the reproduction case above:
- Input array 1 contains 1 masked value
- Input array 2 contains 2 masked values
- The result contains 2 masked values (should be at most 1)

This occurs because the implementation relies on `ma.unique()` which itself has a bug where it doesn't properly deduplicate masked values. When `intersect1d` calls `unique()` on each input array, the masked values aren't collapsed into a single element, and these duplicates propagate through the subsequent concatenation and sorting operations.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py` at lines 1375-1405. The implementation uses the following algorithm:
1. If not `assume_unique`, call `unique()` on both input arrays
2. Concatenate the results
3. Sort the concatenated array
4. Find consecutive equal elements

The root cause is that `ma.unique()` (lines 1325-1355) doesn't properly handle the documented behavior of treating all masked values as equal. This causes `intersect1d` to inherit the same incorrect behavior.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.ma.intersect1d.html

The example in the documentation (lines 1389-1396) shows the expected behavior with a single masked value in the result when both inputs have masked values.

## Proposed Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -1398,11 +1398,26 @@ def intersect1d(ar1, ar2, assume_unique=False):

     """
-    if assume_unique:
-        aux = ma.concatenate((ar1, ar2))
-    else:
-        # Might be faster than unique( intersect1d( ar1, ar2 ) )?
-        aux = ma.concatenate((unique(ar1), unique(ar2)))
-    aux.sort()
-    return aux[:-1][aux[1:] == aux[:-1]]
+    # Handle masked values explicitly to ensure at most one masked value in result
+    has_masked1 = np.any(ma.getmaskarray(ar1))
+    has_masked2 = np.any(ma.getmaskarray(ar2))
+
+    # Get intersection of unmasked values
+    unmasked1 = ar1.compressed() if has_masked1 else ar1
+    unmasked2 = ar2.compressed() if has_masked2 else ar2
+
+    if not assume_unique:
+        unmasked1 = np.unique(unmasked1)
+        unmasked2 = np.unique(unmasked2)
+
+    unmasked_intersect = np.intersect1d(unmasked1, unmasked2, assume_unique=True)
+
+    # Add single masked element if both inputs have masked values
+    if has_masked1 and has_masked2:
+        result_data = np.append(unmasked_intersect, ar1.fill_value)
+        result_mask = np.append(np.zeros(len(unmasked_intersect), dtype=bool), True)
+        return ma.array(result_data, mask=result_mask)
+    else:
+        return ma.array(unmasked_intersect)
```