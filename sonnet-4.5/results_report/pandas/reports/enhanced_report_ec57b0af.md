# Bug Report: FixedForwardWindowIndexer Invalid Window Bounds with Negative window_size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds()` produces mathematically invalid window bounds where start[i] > end[i] when initialized with a negative `window_size`, violating the fundamental invariant that window boundaries must satisfy start[i] <= end[i] for all indices.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=50),
    window_size=st.integers(min_value=-10, max_value=0)
)
def test_fixed_forward_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invariant violated at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}, num_values={num_values}, window_size={window_size}"

if __name__ == "__main__":
    test_fixed_forward_negative_window_size()
```

<details>

<summary>
**Failing input**: `num_values=2, window_size=-1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 16, in <module>
    test_fixed_forward_negative_window_size()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 5, in test_fixed_forward_negative_window_size
    num_values=st.integers(min_value=1, max_value=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 13, in test_fixed_forward_negative_window_size
    assert start[i] <= end[i], f"Invariant violated at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}, num_values={num_values}, window_size={window_size}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Invariant violated at index 1: start[1]=1 > end[1]=0, num_values=2, window_size=-1
Falsifying example: test_fixed_forward_negative_window_size(
    num_values=2,
    window_size=-1,
)
```
</details>

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

# Create indexer with negative window_size
indexer = FixedForwardWindowIndexer(window_size=-1)

# Get window bounds for 2 values
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print()
print("Checking invariant start[i] <= end[i]:")
for i in range(len(start)):
    if start[i] > end[i]:
        print(f"  VIOLATION at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}")
    else:
        print(f"  Valid at index {i}: start[{i}]={start[i]} <= end[{i}]={end[i]}")
```

<details>

<summary>
Output showing invariant violation
</summary>
```
start: [0 1]
end: [0 0]

Checking invariant start[i] <= end[i]:
  Valid at index 0: start[0]=0 <= end[0]=0
  VIOLATION at index 1: start[1]=1 > end[1]=0
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Mathematical Invariant Violation**: Window bounds must always satisfy start[i] <= end[i] for all valid indices. This is a fundamental assumption that any code using these bounds would rely upon. When start > end, operations like array slicing (data[start:end]) produce empty results when data should be included, and iteration from start to end becomes impossible.

2. **Silent Data Corruption**: The code accepts negative `window_size` values without any validation or error, producing silently incorrect results. This is particularly dangerous because downstream operations using these invalid bounds may produce wrong results without any indication of error.

3. **Inconsistency with Pandas Conventions**: Other parts of pandas validate window parameters. For example, `DataFrame.rolling()` validates that "window must be an integer 0 or greater". The lack of similar validation in `FixedForwardWindowIndexer` creates an inconsistency in the API.

4. **Algorithm Assumption Violation**: The implementation in lines 340-343 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py` computes `end = start + self.window_size`, then clips the result. With negative window_size, this produces negative values that get clipped to 0, while start continues incrementing, resulting in the invalid start > end situation.

5. **Semantic Nonsensicality**: A "window size" represents a count of observations, which cannot logically be negative. The class name "FixedForwardWindowIndexer" implies forward-looking windows, making negative sizes semantically meaningless.

## Relevant Context

The bug occurs in the `FixedForwardWindowIndexer.get_window_bounds()` method at lines 340-343 of the implementation:

```python
start = np.arange(0, num_values, step, dtype="int64")
end = start + self.window_size
if self.window_size:
    end = np.clip(end, 0, num_values)
```

The problem is that when `window_size` is negative:
- Line 341 calculates `end = start + window_size`, producing negative values
- Line 343 clips these negative values to 0
- But `start` continues to increment (0, 1, 2, ...)
- This results in situations where `start[i] > 0` but `end[i] = 0`

The parent class `BaseIndexer` at line 72-79 accepts the `window_size` parameter without any validation, allowing negative values to propagate through the system.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.api.indexers.FixedForwardWindowIndexer.html

## Proposed Fix

Add validation in the `BaseIndexer.__init__` method to reject negative window sizes, consistent with other pandas window operations:

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -72,6 +72,8 @@ class BaseIndexer:
     def __init__(
         self, index_array: np.ndarray | None = None, window_size: int = 0, **kwargs
     ) -> None:
+        if window_size < 0:
+            raise ValueError("window_size must be non-negative")
         self.index_array = index_array
         self.window_size = window_size
         # Set user defined kwargs as attributes that can be used in get_window_bounds
```