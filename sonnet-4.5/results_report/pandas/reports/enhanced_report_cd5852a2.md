# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Values for Out-of-Bounds Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer()` function incorrectly returns negative values when computing the length of a slice where the start index is greater than or equal to the target array length, violating the fundamental invariant that lengths must be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.indexers import length_of_indexer


@given(
    start=st.integers(min_value=0, max_value=100) | st.none(),
    stop=st.integers(min_value=0, max_value=100) | st.none(),
    step=st.integers(min_value=1, max_value=10) | st.none(),
    target_len=st.integers(min_value=0, max_value=100),
)
def test_length_of_indexer_slice_matches_actual(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = np.arange(target_len)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, \
        f"Mismatch for slice({start}, {stop}, {step}) on target of length {target_len}: " \
        f"computed={computed_length}, actual={actual_length}"


if __name__ == "__main__":
    # Run the test
    test_length_of_indexer_slice_matches_actual()
```

<details>

<summary>
**Failing input**: `start=1, stop=None, step=None, target_len=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 26, in <module>
    test_length_of_indexer_slice_matches_actual()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 7, in test_length_of_indexer_slice_matches_actual
    start=st.integers(min_value=0, max_value=100) | st.none(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 19, in test_length_of_indexer_slice_matches_actual
    assert computed_length == actual_length, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Mismatch for slice(1, None, None) on target of length 0: computed=-1, actual=0
Falsifying example: test_length_of_indexer_slice_matches_actual(
    start=1,
    stop=None,
    step=None,
    target_len=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/14/hypo.py:20
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case 1: Empty target with start=1
target = np.arange(0)
slc = slice(1, None, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Empty target, slice(1, None): Computed={computed_length}, Actual={actual_length}")

# Test case 2: Target of length 5 with start=10
target = np.arange(5)
slc = slice(10, None, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Target[0-4], slice(10, None): Computed={computed_length}, Actual={actual_length}")

# Test case 3: Edge case - start equals length
target = np.arange(3)
slc = slice(3, None, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Target[0-2], slice(3, None): Computed={computed_length}, Actual={actual_length}")

# Test case 4: Another example with start > length
target = np.arange(2)
slc = slice(5, None, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Target[0-1], slice(5, None): Computed={computed_length}, Actual={actual_length}")
```

<details>

<summary>
Output showing negative length values returned
</summary>
```
Empty target, slice(1, None): Computed=-1, Actual=0
Target[0-4], slice(10, None): Computed=-5, Actual=0
Target[0-2], slice(3, None): Computed=0, Actual=0
Target[0-1], slice(5, None): Computed=-3, Actual=0
```
</details>

## Why This Is A Bug

This bug violates multiple fundamental principles:

1. **Mathematical Violation**: Lengths are by definition non-negative integers. The function returns negative values (e.g., -1, -5, -3) which are mathematically impossible for a length calculation.

2. **Contract Violation**: The function's docstring states it should "Return the expected length of target[indexer]". The invariant `length_of_indexer(slc, target) == len(target[slc])` must hold for all valid inputs. When NumPy correctly returns an empty array (length 0) for out-of-bounds slices, `length_of_indexer` returns negative values instead.

3. **Implementation Error**: The bug stems from the formula at line 316 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/utils.py`:
   ```python
   return (stop - start + step - 1) // step
   ```
   When `start > stop` (which happens when start exceeds array bounds), this produces negative results. For example, with `start=1, stop=0, step=1`: `(0 - 1 + 1 - 1) // 1 = -1`.

4. **Downstream Impact**: The `check_setitem_lengths()` function relies on `length_of_indexer()` for validation. Due to this bug, it incorrectly raises `ValueError` for valid no-op assignments like `arr[10:] = []` on a 5-element array, even though NumPy correctly handles this as a no-op operation.

## Relevant Context

The bug occurs specifically in slice handling within `length_of_indexer()` when:
- The slice has a `start` index >= the target array length
- The function normalizes `stop` to the array length (line 308)
- This makes `stop < start`, causing the formula to return negative values

Key code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/utils.py:290-316`

The function is used internally by pandas for index validation, particularly in:
- `check_setitem_lengths()` for validating assignment operations
- Various internal indexing operations that need to pre-calculate result sizes

Documentation: The function is an internal utility not exposed in the public pandas API.

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -313,7 +313,10 @@ def length_of_indexer(indexer, target=None) -> int:
     elif step < 0:
         start, stop = stop + 1, start + 1
         step = -step
-    return (stop - start + step - 1) // step
+    if start >= stop:
+        return 0
+    else:
+        return (stop - start + step - 1) // step
 elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
     if isinstance(indexer, list):
         indexer = np.array(indexer)
```