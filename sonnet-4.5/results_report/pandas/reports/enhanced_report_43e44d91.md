# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Values for Empty Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function in pandas returns negative integers when calculating the length of slices that would produce empty results, violating the mathematical invariant that lengths must be non-negative.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0) | st.none(),
    target_len=st.integers(min_value=0, max_value=200)
)
@settings(max_examples=500)
def test_length_of_indexer_slice(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = np.arange(target_len)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, f"Mismatch: computed={computed_length}, actual={actual_length} for slice({start}, {stop}, {step}) on array of length {target_len}"

# Run the test
if __name__ == "__main__":
    test_length_of_indexer_slice()
    print("Test passed!")  # This will only print if no assertion errors occur
```

<details>

<summary>
**Failing input**: `start=None, stop=-1, step=None, target_len=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 23, in <module>
    test_length_of_indexer_slice()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 6, in test_length_of_indexer_slice
    start=st.integers(min_value=-100, max_value=100) | st.none(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 19, in test_length_of_indexer_slice
    assert computed_length == actual_length, f"Mismatch: computed={computed_length}, actual={actual_length} for slice({start}, {stop}, {step}) on array of length {target_len}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Mismatch: computed=-1, actual=0 for slice(None, -1, None) on array of length 0
Falsifying example: test_length_of_indexer_slice(
    start=None,
    stop=-1,
    step=None,
    target_len=0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case from the bug report
target = np.arange(0)
slc = slice(1, None, None)

computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])

print(f"Computed length: {computed_length}")
print(f"Actual length: {actual_length}")
print(f"Bug: {computed_length} != {actual_length}")
print()

# Additional test cases
test_cases = [
    (0, slice(1, None), "Empty array, start=1"),
    (0, slice(5, 10), "Empty array, start=5, stop=10"),
    (3, slice(5, None), "Array[3], start=5"),
    (5, slice(3, 2), "Array[5], start=3, stop=2"),
]

print("Additional test cases:")
for target_len, slc, desc in test_cases:
    target = np.arange(target_len)
    computed = length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"{desc}: computed={computed}, actual={actual}, match={computed == actual}")
```

<details>

<summary>
Output showing negative length values
</summary>
```
Computed length: -1
Actual length: 0
Bug: -1 != 0

Additional test cases:
Empty array, start=1: computed=-1, actual=0, match=False
Empty array, start=5, stop=10: computed=-5, actual=0, match=False
Array[3], start=5: computed=-2, actual=0, match=False
Array[5], start=3, stop=2: computed=-1, actual=0, match=False
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical invariant that a length (count of elements) cannot be negative. The function's docstring at line 292 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/utils.py` states: "Return the expected length of target[indexer]".

The bug occurs because the formula at line 316 `(stop - start + step - 1) // step` produces negative values when `start >= stop` after normalization. This happens in several scenarios:
- When slicing an empty array with a positive start value
- When the slice start exceeds the array bounds
- When using backwards slices with positive steps (e.g., `slice(3, 2)`)

This contradicts the behavior of NumPy and Python, which correctly return empty sequences (length 0) for these cases. The discrepancy could cause issues in code that:
1. Allocates arrays based on the computed length (negative sizes would raise errors)
2. Uses the length in arithmetic operations without checking for negative values
3. Relies on the reasonable assumption that lengths are always non-negative

## Relevant Context

The `length_of_indexer` function is part of pandas' public API and is used internally for validation in operations like `check_setitem_lengths` (line 175 of the same file). The function handles various indexer types including slices, arrays, and ranges.

The issue specifically affects the slice handling branch starting at line 298. The function correctly handles negative indices and steps, but fails to ensure the final result is non-negative.

Source code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/utils.py:290-330`

Documentation: The function is documented in the pandas API reference under `pandas.api.indexers`

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