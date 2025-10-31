# Bug Report: numpy.ma.unique Returns Multiple Masked Values

**Target**: `numpy.ma.unique`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.ma.unique()` function violates its documented behavior by returning multiple masked elements instead of treating all masked values as a single element when the same underlying data value appears both masked and unmasked in the input array.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.integer_dtypes(), shape=npst.array_shapes()),
       st.data())
@settings(max_examples=100)
def test_unique_treats_masked_as_equal(arr, data):
    assume(arr.size > 1)
    mask = data.draw(npst.arrays(dtype=np.bool_, shape=arr.shape))
    assume(np.sum(mask) >= 2)

    marr = ma.array(arr, mask=mask)

    unique_result = ma.unique(marr)

    masked_in_result = ma.getmaskarray(unique_result)
    num_masked = np.sum(masked_in_result)

    # According to documentation: "Masked values are considered the same element (masked)"
    # This means all masked values should collapse to at most 1 masked value in the output
    assert num_masked <= 1, f"Expected at most 1 masked value, but got {num_masked}. Input: arr={arr}, mask={mask}"
```

<details>

<summary>
**Failing input**: `arr=array([32767, 32767, 32767], dtype=int16)`, `mask=array([True, False, True])`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/0
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_unique_treats_masked_as_equal FAILED                       [100%]

=================================== FAILURES ===================================
______________________ test_unique_treats_masked_as_equal ______________________
hypo.py:7: in test_unique_treats_masked_as_equal
    st.data())
            ^^^
hypo.py:23: in test_unique_treats_masked_as_equal
    assert num_masked <= 1, f"Expected at most 1 masked value, but got {num_masked}. Input: arr={arr}, mask={mask}"
E   AssertionError: Expected at most 1 masked value, but got 2. Input: arr=[32767 32767 32767], mask=[ True False  True]
E   assert np.int64(2) <= 1
E   Falsifying example: test_unique_treats_masked_as_equal(
E       arr=array([32767, 32767, 32767], dtype=int16),
E       data=data(...),
E   )
E   Draw 1: array([ True, False,  True])
E   Explanation:
E       These lines were always and only run by failing examples:
E           /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1304
E           /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1305
=========================== short test summary info ============================
FAILED hypo.py::test_unique_treats_masked_as_equal - AssertionError: Expected...
============================== 1 failed in 0.84s ===============================
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

# Create the failing test case
arr = np.array([32767, 32767, 32767], dtype=np.int16)
mask = np.array([True, False, True])
marr = ma.array(arr, mask=mask)

print("Input array:", arr)
print("Input mask:", mask)
print("Masked array:", marr)
print()

# Call ma.unique
unique_result = ma.unique(marr)
print("Result from ma.unique():", unique_result)
print("Result data:", unique_result.data)
print("Result mask:", ma.getmaskarray(unique_result))
print("Number of masked values in result:", np.sum(ma.getmaskarray(unique_result)))
print()

# According to documentation, masked values should be considered the same element
# Expected: At most 1 masked value in the result
# Actual: Multiple masked values are returned
print("Expected: At most 1 masked value (per documentation)")
print("Actual:", np.sum(ma.getmaskarray(unique_result)), "masked values")
print()
print("BUG: ma.unique() returns multiple masked values when it should treat")
print("all masked values as the same element and return at most one masked value.")
```

<details>

<summary>
Output shows 2 masked values returned instead of at most 1
</summary>
```
Input array: [32767 32767 32767]
Input mask: [ True False  True]
Masked array: [-- 32767 --]

Result from ma.unique(): [-- 32767 --]
Result data: [32767 32767 32767]
Result mask: [ True False  True]
Number of masked values in result: 2

Expected: At most 1 masked value (per documentation)
Actual: 2 masked values

BUG: ma.unique() returns multiple masked values when it should treat
all masked values as the same element and return at most one masked value.
```
</details>

## Why This Is A Bug

The numpy.ma.unique function documentation explicitly states: **"Masked values are considered the same element (masked)."** This unambiguous statement establishes that:

1. **All masked values should be treated as identical**, regardless of their underlying data values
2. **The output should contain at most one masked element**, representing all input masked values collapsed together
3. **This is a core semantic of masked arrays** - masked values represent missing/invalid data that should be treated uniformly

The current implementation violates this contract by:
- Simply passing the masked array to `np.unique()` without special mask handling
- Returning the result as a MaskedArray view without consolidating masked values
- Treating masked and unmasked instances of the same underlying value as separate unique elements

This occurs because `np.unique()` operates on the raw data array and doesn't understand mask semantics. When value 32767 appears both masked and unmasked, `np.unique()` preserves all instances, and the mask is carried through unchanged, resulting in multiple masked values in the output.

## Relevant Context

The bug manifests specifically when:
- The same underlying data value appears in both masked and unmasked positions
- Multiple masked values exist in the input array
- The masked values have the same underlying data as unmasked values

The documentation's example works correctly because the masked value (1000) doesn't appear unmasked elsewhere. However, this is a special case, not the general behavior promised by the documentation.

Key source code location: `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/extras.py`
NumPy version tested: 2.3.0

The function's current implementation (simplified):
```python
def unique(ar1, return_index=False, return_inverse=False):
    output = np.unique(ar1, return_index=return_index, return_inverse=return_inverse)
    if isinstance(output, tuple):
        output = list(output)
        output[0] = output[0].view(MaskedArray)
        output = tuple(output)
    else:
        output = output.view(MaskedArray)
    return output
```

## Proposed Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -385,11 +385,51 @@ def unique(ar1, return_index=False, return_inverse=False):
     (masked_array(data=[1, 2, 3, --],
                 mask=[False, False, False,  True],
         fill_value=999999), array([0, 1, 4, 2]), array([0, 1, 3, 1, 2]))
     """
-    output = np.unique(ar1,
-                       return_index=return_index,
-                       return_inverse=return_inverse)
-    if isinstance(output, tuple):
-        output = list(output)
-        output[0] = output[0].view(MaskedArray)
-        output = tuple(output)
-    else:
-        output = output.view(MaskedArray)
-    return output
+    ar1 = ma.asarray(ar1)
+    mask = ma.getmaskarray(ar1)
+    has_masked = np.any(mask)
+
+    if not has_masked:
+        # No masked values, use standard unique
+        output = np.unique(ar1, return_index=return_index, return_inverse=return_inverse)
+        if isinstance(output, tuple):
+            output = list(output)
+            output[0] = ma.array(output[0])
+            output = tuple(output)
+        else:
+            output = ma.array(output)
+        return output
+
+    # Handle masked values properly
+    unmasked_indices = np.where(~mask)[0]
+    masked_indices = np.where(mask)[0]
+
+    if len(unmasked_indices) == 0:
+        # All values are masked - return single masked value
+        result = ma.array([ar1.fill_value], mask=[True])
+        if return_index and return_inverse:
+            return result, np.array([0]), np.zeros(len(ar1), dtype=int)
+        elif return_index:
+            return result, np.array([0])
+        elif return_inverse:
+            return result, np.zeros(len(ar1), dtype=int)
+        else:
+            return result
+
+    # Get unique of unmasked values only
+    unmasked_data = ar1[unmasked_indices]
+    unique_output = np.unique(unmasked_data, return_index=return_index, return_inverse=return_inverse)
+
+    if isinstance(unique_output, tuple):
+        unique_vals = unique_output[0]
+        # Add single masked value at the end
+        result_data = np.append(unique_vals, ar1.fill_value)
+        result_mask = np.append(np.zeros(len(unique_vals), dtype=bool), True)
+        result = ma.array(result_data, mask=result_mask)
+
+        # Adjust indices and inverse mapping if requested
+        if return_index and return_inverse:
+            orig_indices = unique_output[1]
+            # Map back to original array indices
+            actual_indices = unmasked_indices[orig_indices]
+            # Add first masked index
+            indices = np.append(actual_indices, masked_indices[0])
+
+            # Fix inverse mapping
+            inverse = np.empty(len(ar1), dtype=int)
+            unmasked_inverse = unique_output[2]
+            inverse[unmasked_indices] = unmasked_inverse
+            inverse[masked_indices] = len(unique_vals)  # Point to the single masked value
+
+            return result, indices, inverse
+        elif return_index:
+            orig_indices = unique_output[1]
+            actual_indices = unmasked_indices[orig_indices]
+            indices = np.append(actual_indices, masked_indices[0])
+            return result, indices
+        elif return_inverse:
+            inverse = np.empty(len(ar1), dtype=int)
+            unmasked_inverse = unique_output[1]
+            inverse[unmasked_indices] = unmasked_inverse
+            inverse[masked_indices] = len(unique_vals)
+            return result, inverse
+        else:
+            return result
+    else:
+        # Simple case - no index or inverse requested
+        result_data = np.append(unique_output, ar1.fill_value)
+        result_mask = np.append(np.zeros(len(unique_output), dtype=bool), True)
+        return ma.array(result_data, mask=result_mask)
```