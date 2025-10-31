# Bug Report: pandas.core.indexers length_of_indexer Returns Negative Length for Empty Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function `length_of_indexer` returns negative values for empty slices (where start >= stop with positive step or start <= stop with negative step), violating the mathematical property that lengths must be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    start=st.none() | st.integers(min_value=-50, max_value=50),
    stop=st.none() | st.integers(min_value=-50, max_value=50),
    step=st.none() | st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0),
)
@settings(max_examples=500)
def test_length_of_indexer_matches_actual_slice(start, stop, step):
    target_len = 50
    target = np.arange(target_len)
    slc = slice(start, stop, step)
    expected_len = length_of_indexer(slc, target)
    actual_sliced = target[slc]
    actual_len = len(actual_sliced)
    assert expected_len == actual_len, f"slice({start}, {stop}, {step}): expected {expected_len}, got {actual_len}"

if __name__ == "__main__":
    test_length_of_indexer_matches_actual_slice()
```

<details>

<summary>
**Failing input**: `slice(1, 0, None)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 21, in <module>
    test_length_of_indexer_matches_actual_slice()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 6, in test_length_of_indexer_matches_actual_slice
    start=st.none() | st.integers(min_value=-50, max_value=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 18, in test_length_of_indexer_matches_actual_slice
    assert expected_len == actual_len, f"slice({start}, {stop}, {step}): expected {expected_len}, got {actual_len}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: slice(1, 0, None): expected -1, got 0
Falsifying example: test_length_of_indexer_matches_actual_slice(
    start=1,
    stop=0,
    step=None,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.arange(50)

# Test case 1: slice(1, 0, None) - start > stop with positive step
slc1 = slice(1, 0, None)
print(f"slice(1, 0, None):")
print(f"  length_of_indexer: {length_of_indexer(slc1, target)}")
print(f"  Actual length: {len(target[slc1])}")
print(f"  Actual sliced array: {target[slc1]}")

# Test case 2: slice(0, 1, -1) - start < stop with negative step
slc2 = slice(0, 1, -1)
print(f"\nslice(0, 1, -1):")
print(f"  length_of_indexer: {length_of_indexer(slc2, target)}")
print(f"  Actual length: {len(target[slc2])}")
print(f"  Actual sliced array: {target[slc2]}")

# Test case 3: slice(None, None, -1) - full negative slice
slc3 = slice(None, None, -1)
print(f"\nslice(None, None, -1):")
print(f"  length_of_indexer: {length_of_indexer(slc3, target)}")
print(f"  Actual length: {len(target[slc3])}")

# Test case 4: slice(-1, 9, None) - negative start index
slc4 = slice(-1, 9, None)
print(f"\nslice(-1, 9, None):")
print(f"  length_of_indexer: {length_of_indexer(slc4, target)}")
print(f"  Actual length: {len(target[slc4])}")

# Test case 5: slice(10, 5, None) - another start > stop case
slc5 = slice(10, 5, None)
print(f"\nslice(10, 5, None):")
print(f"  length_of_indexer: {length_of_indexer(slc5, target)}")
print(f"  Actual length: {len(target[slc5])}")
print(f"  Actual sliced array: {target[slc5]}")
```

<details>

<summary>
Output showing incorrect negative lengths returned
</summary>
```
slice(1, 0, None):
  length_of_indexer: -1
  Actual length: 0
  Actual sliced array: []

slice(0, 1, -1):
  length_of_indexer: -1
  Actual length: 0
  Actual sliced array: []

slice(None, None, -1):
  length_of_indexer: -50
  Actual length: 50

slice(-1, 9, None):
  length_of_indexer: -40
  Actual length: 0

slice(10, 5, None):
  length_of_indexer: -5
  Actual length: 0
  Actual sliced array: []
```
</details>

## Why This Is A Bug

The function violates its documented contract and fundamental mathematical properties:

1. **Docstring violation (line 292)**: States "Return the expected length of target[indexer]". The expected length should match what `len(target[indexer])` returns, which is always non-negative.

2. **Mathematical incorrectness**: Length is a non-negative measure by definition. Returning negative values (-1, -5, -40, -50) violates this fundamental property.

3. **Python/NumPy consistency**: Python and NumPy always return 0 for empty slices, never negative values. The function should match this standard behavior.

4. **Internal usage impact**: The function is used internally in `check_setitem_lengths` (line 175) for validation. Negative lengths could cause incorrect validation logic.

5. **Formula error**: The calculation `(stop - start + step - 1) // step` produces negative results when slices would be empty, particularly when start >= stop with positive step.

## Relevant Context

The bug occurs in the slice handling logic (lines 298-316 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/utils.py`). The function handles negative steps by transforming coordinates but fails to check if the resulting range would be empty.

Key observations:
- The function is part of pandas' internal indexing utilities, not exposed in the public API
- Used for validation in `check_setitem_lengths` where incorrect lengths could affect data assignment operations
- Python's built-in slice behavior: empty slices (where iteration produces no elements) always have length 0
- The bug affects multiple slice configurations that would produce empty sequences

Documentation reference: [Python Data Model - Slicing](https://docs.python.org/3/reference/datamodel.html#object.__getitem__)

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -313,6 +313,10 @@ def length_of_indexer(indexer, target=None) -> int:
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
+
+        # Empty slices should return 0, not negative values
+        if start >= stop:
+            return 0
+
         return (stop - start + step - 1) // step
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
```