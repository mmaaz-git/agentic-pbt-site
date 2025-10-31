# Bug Report: scipy.ndimage Binary Morphology Monotonicity Violation

**Target**: `scipy.ndimage.binary_erosion` and `scipy.ndimage.binary_dilation`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`binary_erosion` and `binary_dilation` violate fundamental monotonicity properties when given structures containing only False values. Erosion incorrectly adds True values (should only remove), and dilation incorrectly removes True values (should only add).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.ndimage
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst


@given(
    arr=npst.arrays(
        dtype=np.bool_,
        shape=npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=10),
    ),
    structure=st.one_of(
        st.none(),
        npst.arrays(
            dtype=np.bool_,
            shape=npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=5),
        )
    )
)
@settings(max_examples=200)
def test_erosion_subset_of_original(arr, structure):
    if structure is not None and arr.ndim != structure.ndim:
        structure = None

    eroded = scipy.ndimage.binary_erosion(arr, structure=structure)

    assert np.all(eroded <= arr), \
        f"Erosion result should be a subset of the original (eroded <= original)"
```

**Failing input**: `arr=array([[False]]), structure=array([[False]])`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.ndimage

print("Bug 1: binary_erosion violates monotonicity")
arr = np.array([[False]])
structure = np.array([[False]])
result = scipy.ndimage.binary_erosion(arr, structure=structure)
print(f"Input: {arr[0]}, Result: {result[0]}")
print(f"Expected: [False], Got: {result[0]}")
assert not np.all(result <= arr), "Erosion added True values!"

print("\nBug 2: binary_dilation violates monotonicity")
arr2 = np.array([[True]])
structure2 = np.array([[False]])
result2 = scipy.ndimage.binary_dilation(arr2, structure=structure2)
print(f"Input: {arr2[0]}, Result: {result2[0]}")
print(f"Expected: [True], Got: {result2[0]}")
assert not np.all(result2 >= arr2), "Dilation removed True values!"
```

## Why This Is A Bug

Binary morphology operations have fundamental mathematical properties:
- **Erosion** is a *shrinking* operation: result should always be a subset of input (eroded ⊆ input)
- **Dilation** is a *growing* operation: input should always be a subset of result (input ⊆ dilated)

These monotonicity properties are violated when structures contain only False values:
1. `binary_erosion([[False]], structure=[[False]])` returns `[[True]]` - erosion *added* True values
2. `binary_dilation([[True]], structure=[[False]])` returns `[[False]]` - dilation *removed* True values

This is caused by vacuous truth in the mathematical definition: when the structure has no True elements, the conditions are trivially satisfied in unexpected ways.

## Fix

Structures with only False values are degenerate edge cases. The fix should validate the structure and raise an error when it contains no True elements:

```diff
--- a/scipy/ndimage/_morphology.py
+++ b/scipy/ndimage/_morphology.py
@@ -100,6 +100,11 @@ def binary_erosion(input, structure=None, iterations=1, mask=None,
         structure = iterate_structure(structure, iterations)
     else:
         structure = generate_binary_structure(input.ndim, 1)
+
+    # Validate structure has at least one True element
+    if structure is not None and not np.any(structure):
+        raise ValueError(
+            "Structure must contain at least one True element")

     return _binary_erosion(input, structure, mask, border_value,
                           origin, 0, brute_force, axes)
@@ -200,6 +205,11 @@ def binary_dilation(input, structure=None, iterations=1, mask=None,
         structure = iterate_structure(structure, iterations)
     else:
         structure = generate_binary_structure(input.ndim, 1)
+
+    # Validate structure has at least one True element
+    if structure is not None and not np.any(structure):
+        raise ValueError(
+            "Structure must contain at least one True element")

     return _binary_dilation(input, structure, mask, border_value,
                            origin, 1, brute_force, axes)
```