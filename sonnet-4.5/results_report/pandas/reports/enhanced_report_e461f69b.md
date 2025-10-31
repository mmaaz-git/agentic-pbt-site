# Bug Report: pandas.api.indexers.FixedForwardWindowIndexer Invalid Window Bounds with Negative Window Size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedForwardWindowIndexer.get_window_bounds()` returns invalid window bounds where start[i] > end[i] when given a negative window_size, violating the fundamental array slicing invariant that window start positions must not exceed end positions.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=-100, max_value=-1))
def test_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)
    assert np.all(start <= end), f"Found start > end for num_values={num_values}, window_size={window_size}"

# Run the test
if __name__ == "__main__":
    test_negative_window_size()
```

<details>

<summary>
**Failing input**: `num_values=2, window_size=-1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 14, in <module>
    test_negative_window_size()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 7, in test_negative_window_size
    def test_negative_window_size(num_values, window_size):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 10, in test_negative_window_size
    assert np.all(start <= end), f"Found start > end for num_values={num_values}, window_size={window_size}"
           ~~~~~~^^^^^^^^^^^^^^
AssertionError: Found start > end for num_values=2, window_size=-1
Falsifying example: test_negative_window_size(
    num_values=2,
    window_size=-1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

# Create an indexer with negative window size
indexer = FixedForwardWindowIndexer(window_size=-1)

# Get window bounds for 2 values
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")

# Check each index
for i in range(len(start)):
    print(f"Index {i}: start[{i}] = {start[i]}, end[{i}] = {end[i]}")
    if start[i] > end[i]:
        print(f"ERROR: start[{i}] > end[{i}] (violates window bound invariant)")

# Verify the invariant
assert np.all(start <= end), f"Invariant violated: Found start > end at some indices"
```

<details>

<summary>
AssertionError: Invariant violated at index 1
</summary>
```
start: [0 1]
end: [0 0]
Index 0: start[0] = 0, end[0] = 0
Index 1: start[1] = 1, end[1] = 0
ERROR: start[1] > end[1] (violates window bound invariant)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/repo.py", line 20, in <module>
    assert np.all(start <= end), f"Invariant violated: Found start > end at some indices"
           ~~~~~~^^^^^^^^^^^^^^
AssertionError: Invariant violated: Found start > end at some indices
```
</details>

## Why This Is A Bug

Window bounds returned by `get_window_bounds()` must satisfy the invariant `start[i] <= end[i]` for all indices to represent valid array slices. This is a fundamental requirement for any windowing operation as these bounds are used for array slicing operations like `arr[start[i]:end[i]]`.

When `window_size` is negative, the implementation in lines 340-341 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py` computes:
```python
start = np.arange(0, num_values, step, dtype="int64")  # [0, 1, ...]
end = start + self.window_size  # [0-1, 1-1, ...] = [-1, 0, ...]
```

After clipping in line 343, `end` becomes `[0, 0, ...]` while `start` remains `[0, 1, ...]`, resulting in `start[i] > end[i]` for i > 0.

The class name "FixedForwardWindowIndexer" and its documentation "Creates window boundaries for fixed-length windows that include the current row" semantically implies forward-looking windows. A negative window size contradicts this forward-looking behavior and produces mathematically invalid bounds.

While the class validates other parameters (raising ValueError for `center=True` in line 332 and for non-None `closed` in lines 333-336), it fails to validate that `window_size` is non-negative.

## Relevant Context

The FixedForwardWindowIndexer class is part of pandas' windowing operations API, used extensively in rolling window calculations. Other indexer classes in the same module validate their parameters appropriately:

- `VariableOffsetWindowIndexer` validates that its index is a DatetimeIndex (line 195) and offset is a BaseOffset (line 198)
- The base class documentation shows example usage with positive window_size values only

The pandas documentation for this class (lines 298-320) shows an example with `window_size=2` but doesn't explicitly state that negative values are invalid. However, the semantic meaning of "forward-looking" windows makes negative values nonsensical.

Documentation link: https://pandas.pydata.org/docs/reference/api/pandas.api.indexers.FixedForwardWindowIndexer.html

## Proposed Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -329,6 +329,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
+        if self.window_size < 0:
+            raise ValueError("Forward-looking windows require non-negative window_size")
         if closed is not None:
             raise ValueError(
                 "Forward-looking windows don't support setting the closed argument"
```