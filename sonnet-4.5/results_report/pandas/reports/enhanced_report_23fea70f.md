# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Length for Valid Empty Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`length_of_indexer` incorrectly returns negative values for slices that would produce empty arrays (e.g., when start > stop with positive step), violating the invariant that lengths must be non-negative.

## Property-Based Test

```python
import numpy as np
from pandas.core.indexers import length_of_indexer
from hypothesis import given, strategies as st, settings

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10),
    target_len=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_length_of_indexer_matches_actual_length(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = np.arange(target_len)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, f"Mismatch for slice({start}, {stop}, {step}) on array of length {target_len}: computed={computed_length}, actual={actual_length}"

if __name__ == "__main__":
    # Run the test
    test_length_of_indexer_matches_actual_length()
```

<details>

<summary>
**Failing input**: `start=1, stop=0, step=1, target_len=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 23, in <module>
    test_length_of_indexer_matches_actual_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 6, in test_length_of_indexer_matches_actual_length
    start=st.integers(min_value=0, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 19, in test_length_of_indexer_matches_actual_length
    assert computed_length == actual_length, f"Mismatch for slice({start}, {stop}, {step}) on array of length {target_len}: computed={computed_length}, actual={actual_length}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Mismatch for slice(1, 0, 1) on array of length 1: computed=-1, actual=0
Falsifying example: test_length_of_indexer_matches_actual_length(
    start=1,
    stop=0,
    step=1,
    target_len=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case from the bug report
target = np.array([0])
slc = slice(1, 0, 1)

computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])

print(f"Test case 1: slice(1, 0, 1) on array of length 1")
print(f"Computed length: {computed_length}")
print(f"Actual length: {actual_length}")
print(f"Match: {computed_length == actual_length}")
print()

# Additional test cases
test_cases = [
    (slice(5, 3, 1), np.arange(10)),
    (slice(10, 5, 1), np.arange(20)),
    (slice(3, 5, -1), np.arange(10)),
    (slice(2, 1, 1), np.arange(5)),
    (slice(100, 50, 1), np.arange(200))
]

for i, (slc, target) in enumerate(test_cases, 2):
    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    print(f"Test case {i}: {slc} on array of length {len(target)}")
    print(f"Computed length: {computed_length}")
    print(f"Actual length: {actual_length}")
    print(f"Match: {computed_length == actual_length}")
    print()

# Demonstrate the assertion error
print("Assertion error on the original test case:")
target = np.array([0])
slc = slice(1, 0, 1)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
assert computed_length == actual_length, f"Computed {computed_length} != Actual {actual_length}"
```

<details>

<summary>
AssertionError: Computed -1 != Actual 0
</summary>
```
Test case 1: slice(1, 0, 1) on array of length 1
Computed length: -1
Actual length: 0
Match: False

Test case 2: slice(5, 3, 1) on array of length 10
Computed length: -2
Actual length: 0
Match: False

Test case 3: slice(10, 5, 1) on array of length 20
Computed length: -5
Actual length: 0
Match: False

Test case 4: slice(3, 5, -1) on array of length 10
Computed length: -2
Actual length: 0
Match: False

Test case 5: slice(2, 1, 1) on array of length 5
Computed length: -1
Actual length: 0
Match: False

Test case 6: slice(100, 50, 1) on array of length 200
Computed length: -50
Actual length: 0
Match: False

Assertion error on the original test case:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/repo.py", line 42, in <module>
    assert computed_length == actual_length, f"Computed {computed_length} != Actual {actual_length}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Computed -1 != Actual 0
```
</details>

## Why This Is A Bug

The function `length_of_indexer` is documented to "Return the expected length of target[indexer]" (line 292 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/utils.py`).

When a slice has `start > stop` with a positive step (or `start < stop` with a negative step), Python's standard slicing behavior returns an empty sequence with length 0. For example, `np.array([0])[1:0:1]` returns an empty array `[]` with length 0.

However, `length_of_indexer` incorrectly returns negative values in these cases. This violates two fundamental principles:
1. **Mathematical invariant**: The length of any collection must be a non-negative integer
2. **Contract violation**: The function should match the actual behavior of slicing, which never produces negative lengths

This bug impacts `check_setitem_lengths` (line 175), which uses `length_of_indexer` to validate slice assignments in pandas DataFrames and Series. Negative lengths could cause incorrect validation logic, potentially leading to unexpected errors or accepting invalid operations.

## Relevant Context

The bug occurs in line 316 of the function where it computes:
```python
return (stop - start + step - 1) // step
```

When `start > stop` for positive steps, `(stop - start)` becomes negative, resulting in a negative return value. The function correctly handles negative steps by swapping start and stop (lines 313-315), but doesn't handle the case where the slice would be empty.

This function is part of pandas' core indexing infrastructure and is used throughout the codebase for validating and pre-computing the results of indexing operations.

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -313,7 +313,7 @@ def length_of_indexer(indexer, target=None) -> int:
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
-        return (stop - start + step - 1) // step
+        return max(0, (stop - start + step - 1) // step)
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
             indexer = np.array(indexer)
```