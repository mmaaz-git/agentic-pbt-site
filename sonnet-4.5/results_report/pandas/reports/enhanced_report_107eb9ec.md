# Bug Report: FixedForwardWindowIndexer Accepts Negative window_size Producing Invalid Window Bounds

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer accepts negative window_size values and produces mathematically invalid window bounds where start[i] > end[i], causing rolling window operations to silently fail and return all NaN values.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-50, max_value=-1),
    step=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=500)
def test_fixed_forward_window_indexer_negative_window_size(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    for i in range(len(start)):
        assert start[i] <= end[i], \
            f"Invalid window: start[{i}]={start[i]} > end[{i}]={end[i]} with window_size={window_size}"

# Run the test
test_fixed_forward_window_indexer_negative_window_size()
```

<details>

<summary>
**Failing input**: `num_values=2, window_size=-1, step=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 19, in <module>
    test_fixed_forward_window_indexer_negative_window_size()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 5, in test_fixed_forward_window_indexer_negative_window_size
    num_values=st.integers(min_value=1, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 15, in test_fixed_forward_window_indexer_negative_window_size
    assert start[i] <= end[i], \
           ^^^^^^^^^^^^^^^^^^
AssertionError: Invalid window: start[1]=1 > end[1]=0 with window_size=-1
Falsifying example: test_fixed_forward_window_indexer_negative_window_size(
    num_values=2,
    window_size=-1,  # or any other generated value
    step=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/18/hypo.py:16
```
</details>

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

# Test case with negative window_size
indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2, step=1)

print(f"start: {start}")
print(f"end: {end}")
print(f"start[1] > end[1]: {start[1]} > {end[1]}")

# Test with rolling operation
df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
print(f"\nOriginal DataFrame:")
print(df)

result = df.rolling(window=indexer, min_periods=1).sum()
print(f"\nRolling result with window_size=-1:")
print(result)
```

<details>

<summary>
Invalid window bounds and all NaN rolling results
</summary>
```
start: [0 1]
end: [0 0]
start[1] > end[1]: 1 > 0

Original DataFrame:
     B
0  0.0
1  1.0
2  2.0
3  NaN
4  4.0

Rolling result with window_size=-1:
    B
0 NaN
1 NaN
2 NaN
3 NaN
4 NaN
```
</details>

## Why This Is A Bug

The FixedForwardWindowIndexer implementation fails to validate that window_size is non-negative, violating several fundamental principles:

1. **Semantic Correctness**: A window size represents "the number of rows in a window" (as documented in the BaseIndexer docstring at line 26 of objects.py). A negative number of rows is semantically meaningless and invalid.

2. **Mathematical Invariant Violation**: The implementation produces window bounds where start[i] > end[i], which is mathematically invalid. Window bounds must always satisfy 0 <= start[i] <= end[i] <= num_values for valid array slicing operations.

3. **Silent Failure**: When these invalid bounds are used in rolling operations, they silently produce all NaN values instead of raising an error, making debugging difficult for users who may not realize they've passed an invalid window size.

4. **Inconsistent Input Validation**: The same class already validates other parameters (raises ValueError for center=True at line 332 and for closed parameter at line 334-335), demonstrating that input validation is an expected pattern in this code.

## Relevant Context

Looking at the implementation in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py` lines 340-344:

```python
start = np.arange(0, num_values, step, dtype="int64")
end = start + self.window_size  # When window_size is negative, end < start
if self.window_size:
    end = np.clip(end, 0, num_values)  # Clipping negative values to 0
```

When window_size is -1:
- start becomes [0, 1] for num_values=2
- end becomes start + (-1) = [-1, 0]
- The clipping operation changes end to [0, 0]
- This results in start[1]=1 > end[1]=0, violating the window bounds invariant

The pandas documentation at https://pandas.pydata.org/docs/reference/api/pandas.api.indexers.FixedForwardWindowIndexer.html shows examples only with positive window_size values and describes creating "fixed-length windows", implying non-negative lengths.

## Proposed Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -329,6 +329,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
+        if self.window_size < 0:
+            raise ValueError(f"window_size must be non-negative, got {self.window_size}")
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
```