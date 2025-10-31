# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Length for Out-of-Bounds Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function incorrectly returns negative values for slices that produce empty results due to out-of-bounds indices, violating the mathematical principle that lengths must be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    target=st.lists(st.integers(), min_size=1, max_size=100),
    slice_start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_step=st.one_of(st.none(), st.integers(min_value=1, max_value=5))
)
@settings(max_examples=500)
def test_length_of_indexer_slice_positive_step_consistency(target, slice_start, slice_stop, slice_step):
    target_array = np.array(target)
    indexer = slice(slice_start, slice_stop, slice_step)

    actual_length = len(target_array[indexer])
    predicted_length = length_of_indexer(indexer, target_array)

    assert actual_length == predicted_length, (
        f"Mismatch for slice({slice_start}, {slice_stop}, {slice_step}) on array of length {len(target_array)}: "
        f"actual_length={actual_length}, predicted_length={predicted_length}"
    )

if __name__ == "__main__":
    test_length_of_indexer_slice_positive_step_consistency()
```

<details>

<summary>
**Failing input**: `slice(None, -2, None)` on array `[0]` (length 1)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 25, in <module>
    test_length_of_indexer_slice_positive_step_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 6, in test_length_of_indexer_slice_positive_step_consistency
    target=st.lists(st.integers(), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 19, in test_length_of_indexer_slice_positive_step_consistency
    assert actual_length == predicted_length, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Mismatch for slice(None, -2, None) on array of length 1: actual_length=0, predicted_length=-1
Falsifying example: test_length_of_indexer_slice_positive_step_consistency(
    target=[0],
    slice_start=None,  # or any other generated value
    slice_stop=-2,
    slice_step=None,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/29/hypo.py:20
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case that demonstrates the bug
array = np.array([0])
indexer = slice(2, None, None)

# Get the actual result from slicing
actual_result = array[indexer]
actual_length = len(actual_result)

# Get the predicted length from the function
predicted_length = length_of_indexer(indexer, array)

print(f"Array: {array}")
print(f"Indexer: {indexer}")
print(f"Actual result: {actual_result}")
print(f"Actual length: {actual_length}")
print(f"Predicted length: {predicted_length}")
print(f"Bug: {predicted_length < 0}")
print(f"Mismatch: actual_length={actual_length} != predicted_length={predicted_length}")
```

<details>

<summary>
Output showing negative length returned
</summary>
```
Array: [0]
Indexer: slice(2, None, None)
Actual result: []
Actual length: 0
Predicted length: -1
Bug: True
Mismatch: actual_length=0 != predicted_length=-1
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Mathematical Invariant**: Lengths must be non-negative by definition. There is no such thing as a collection with -1 elements.

2. **Function Contract**: The docstring states the function returns "the expected length of target[indexer]", which should match `len(target[indexer])`. Python's `len()` function never returns negative values.

3. **Downstream Impact**: The function is used internally by pandas (e.g., in `check_setitem_lengths` at line 175) where negative lengths could cause logical errors or unexpected behavior.

4. **Common Operation**: Out-of-bounds slicing is not an obscure edge case - it occurs frequently in data manipulation when slicing beyond array boundaries, and Python handles this gracefully by returning empty results.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/utils.py` at line 316:

```python
return (stop - start + step - 1) // step
```

When a slice has out-of-bounds indices that result in `start >= stop` after normalization, this formula produces negative values. For example:
- `slice(2, None)` on array of length 1 normalizes to start=2, stop=1, step=1
- Formula: `(1 - 2 + 1 - 1) // 1 = -1`
- But the actual slice `array[2:]` returns an empty array with length 0

The function is part of pandas' internal indexing utilities and while marked as private (in `pandas.core`), it still needs to maintain correct behavior as it's used by other pandas operations.

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