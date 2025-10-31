# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Lengths for Out-of-Bounds Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative lengths for slices with out-of-bounds indices, violating the fundamental invariant that it should return the same value as `len(target[indexer])`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    target=st.lists(st.integers(), min_size=1, max_size=50),
    slice_start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_step=st.one_of(st.none(), st.integers(min_value=1, max_value=10))
)
@settings(max_examples=100)
def test_length_of_indexer_slice_consistency(target, slice_start, slice_stop, slice_step):
    """Test that length_of_indexer returns the same length as actual numpy slicing."""
    target_array = np.array(target)
    indexer = slice(slice_start, slice_stop, slice_step)

    actual_length = len(target_array[indexer])
    predicted_length = length_of_indexer(indexer, target_array)

    assert actual_length == predicted_length, (
        f"Mismatch for {indexer} on array of length {len(target_array)}: "
        f"actual={actual_length}, predicted={predicted_length}"
    )

if __name__ == "__main__":
    # Run the property-based test
    print("Running property-based test for length_of_indexer...")
    print("Testing the property: length_of_indexer(indexer, target) == len(target[indexer])")
    print()

    try:
        test_length_of_indexer_slice_consistency()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error: {e}")
    except Exception as e:
        print(f"Test failed with error: {e}")
```

<details>

<summary>
**Failing input**: `slice(2, None, None)` on array of length 1
</summary>
```
Running property-based test for length_of_indexer...
Testing the property: length_of_indexer(indexer, target) == len(target[indexer])

Test failed with assertion error: Mismatch for slice(2, None, None) on array of length 1: actual=0, predicted=-1
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case 1: slice(None, -2, None) on array of length 1
target = np.array([0])
indexer1 = slice(None, -2, None)
print(f"Case 1: {indexer1} on array {target}")
print(f"  target_len: {len(target)}")
print(f"  Actual length using numpy slicing: {len(target[indexer1])}")
print(f"  Predicted length using length_of_indexer: {length_of_indexer(indexer1, target)}")
print(f"  Match: {len(target[indexer1]) == length_of_indexer(indexer1, target)}")
print()

# Test case 2: slice(2, None) on array of length 1
target = np.array([0])
indexer2 = slice(2, None)
print(f"Case 2: {indexer2} on array {target}")
print(f"  target_len: {len(target)}")
print(f"  Actual length using numpy slicing: {len(target[indexer2])}")
print(f"  Predicted length using length_of_indexer: {length_of_indexer(indexer2, target)}")
print(f"  Match: {len(target[indexer2]) == length_of_indexer(indexer2, target)}")
print()

# Additional test cases to understand the bug better
print("Additional test cases:")
print("-" * 40)

# Case 3: slice(-3, -2, None) on array of length 1
target = np.array([0])
indexer3 = slice(-3, -2, None)
print(f"Case 3: {indexer3} on array {target}")
print(f"  target_len: {len(target)}")
print(f"  Actual length: {len(target[indexer3])}")
print(f"  Predicted length: {length_of_indexer(indexer3, target)}")
print()

# Case 4: slice(None, -5, None) on array of length 3
target = np.array([0, 1, 2])
indexer4 = slice(None, -5, None)
print(f"Case 4: {indexer4} on array {target}")
print(f"  target_len: {len(target)}")
print(f"  Actual length: {len(target[indexer4])}")
print(f"  Predicted length: {length_of_indexer(indexer4, target)}")
print()

# Case 5: slice(5, None) on array of length 3
target = np.array([0, 1, 2])
indexer5 = slice(5, None)
print(f"Case 5: {indexer5} on array {target}")
print(f"  target_len: {len(target)}")
print(f"  Actual length: {len(target[indexer5])}")
print(f"  Predicted length: {length_of_indexer(indexer5, target)}")
```

<details>

<summary>
Output showing negative lengths returned for valid empty slices
</summary>
```
Case 1: slice(None, -2, None) on array [0]
  target_len: 1
  Actual length using numpy slicing: 0
  Predicted length using length_of_indexer: -1
  Match: False

Case 2: slice(2, None, None) on array [0]
  target_len: 1
  Actual length using numpy slicing: 0
  Predicted length using length_of_indexer: -1
  Match: False

Additional test cases:
----------------------------------------
Case 3: slice(-3, -2, None) on array [0]
  target_len: 1
  Actual length: 0
  Predicted length: 1

Case 4: slice(None, -5, None) on array [0 1 2]
  target_len: 3
  Actual length: 0
  Predicted length: -2

Case 5: slice(5, None, None) on array [0 1 2]
  target_len: 3
  Actual length: 0
  Predicted length: -2
```
</details>

## Why This Is A Bug

The function's docstring explicitly states "Return the expected length of target[indexer]", establishing a clear contract that `length_of_indexer(indexer, target)` should equal `len(target[indexer])`. However, the function returns negative values when it should return 0 for empty slices resulting from out-of-bounds indices.

The bug manifests in two scenarios:

1. **Negative stop indices that remain negative after adjustment**: For `slice(None, -2, None)` with `target_len=1`, the stop value `-2` becomes `-1` after adding `target_len`, but this negative value is not clamped to 0. The calculation `(stop - start + step - 1) // step` becomes `(-1 - 0 + 1 - 1) // 1 = -1`.

2. **Positive start indices beyond array length**: For `slice(2, None)` with `target_len=1`, the start value `2` exceeds the array length but is not clamped. With `stop=1` (the target length), the calculation becomes `(1 - 2 + 1 - 1) // 1 = -1`.

This violates the documented behavior and breaks code that relies on `length_of_indexer` for validation, particularly `check_setitem_lengths` at line 175 of the same file, which uses this function to validate slice assignments.

## Relevant Context

The `length_of_indexer` function is located in `/pandas/core/indexers/utils.py` starting at line 290. It's a utility function used internally by pandas for indexing operations, particularly in validation scenarios.

The function is directly used by `check_setitem_lengths` (line 175) to validate that slice assignments have matching lengths. With this bug, the validation could incorrectly reject valid empty slice assignments or accept invalid ones.

NumPy's slicing behavior correctly handles out-of-bounds indices by returning empty arrays (length 0), which is the expected behavior that `length_of_indexer` should mirror.

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -303,11 +303,17 @@ def length_of_indexer(indexer, target=None) -> int:
         if start is None:
             start = 0
         elif start < 0:
             start += target_len
+            if start < 0:
+                start = 0
+        elif start > target_len:
+            start = target_len
         if stop is None or stop > target_len:
             stop = target_len
         elif stop < 0:
             stop += target_len
+            if stop < 0:
+                stop = 0
         if step is None:
             step = 1
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
         return (stop - start + step - 1) // step
```