# Bug Report: pandas.api.indexers.FixedForwardWindowIndexer Integer Overflow with Negative Window Sizes

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer crashes with OverflowError when given extremely large negative window_size values and produces mathematically invalid window bounds (start > end) for any negative window_size.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(max_value=-1))
def test_fixed_forward_window_negative_size_start_end_invariant(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values)

    for i in range(len(start)):
        assert start[i] <= end[i]

if __name__ == "__main__":
    test_fixed_forward_window_negative_size_start_end_invariant()
```

<details>

<summary>
**Failing input**: `num_values=1, window_size=-9_223_372_036_854_775_809`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 15, in <module>
  |     test_fixed_forward_window_negative_size_start_end_invariant()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 7, in test_fixed_forward_window_negative_size_start_end_invariant
  |     def test_fixed_forward_window_negative_size_start_end_invariant(num_values, window_size):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 12, in test_fixed_forward_window_negative_size_start_end_invariant
    |     assert start[i] <= end[i]
    |            ^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_fixed_forward_window_negative_size_start_end_invariant(
    |     num_values=2,
    |     window_size=-1,  # or any other generated value
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 9, in test_fixed_forward_window_negative_size_start_end_invariant
    |     start, end = indexer.get_window_bounds(num_values)
    |                  ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py", line 341, in get_window_bounds
    |     end = start + self.window_size
    |           ~~~~~~^~~~~~~~~~~~~~~~~~
    | OverflowError: Python int too large to convert to C long
    | Falsifying example: test_fixed_forward_window_negative_size_start_end_invariant(
    |     num_values=1,  # or any other generated value
    |     window_size=-9_223_372_036_854_775_809,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-9_223_372_036_854_775_809)
start, end = indexer.get_window_bounds(num_values=1)
print(f"start: {start}, end: {end}")
```

<details>

<summary>
OverflowError: Python int too large to convert to C long
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/repo.py", line 4, in <module>
    start, end = indexer.get_window_bounds(num_values=1)
                 ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py", line 341, in get_window_bounds
    end = start + self.window_size
          ~~~~~~^~~~~~~~~~~~~~~~~~
OverflowError: Python int too large to convert to C long
```
</details>

## Why This Is A Bug

This violates expected behavior in two ways:

1. **Integer Overflow**: The code crashes when attempting to compute `end = start + self.window_size` (line 341 in objects.py) where `start` is a numpy int64 array and `window_size` is an extremely large negative Python int. NumPy cannot convert the Python int to a C long, causing an OverflowError.

2. **Invalid Window Bounds**: For any negative `window_size`, the code produces mathematically invalid window bounds where `start[i] > end[i]`. This violates the fundamental invariant of window bounds that the start position must be less than or equal to the end position. For example, with `window_size=-1` and `num_values=3`, the code produces `start=[0,1,2]` and `end=[0,0,1]`, resulting in invalid windows where `start[1]=1 > end[1]=0` and `start[2]=2 > end[2]=1`.

The class name "FixedForwardWindowIndexer" semantically implies forward-looking windows, making negative window sizes conceptually invalid. While users would rarely use negative values intentionally, the code should validate inputs and provide clear error messages rather than crash or produce invalid results.

## Relevant Context

The FixedForwardWindowIndexer class is designed to create window boundaries for fixed-length windows that include the current row and look forward. The implementation in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/objects.py` shows:

- Lines 340-341: `start = np.arange(0, num_values, step, dtype="int64")` followed by `end = start + self.window_size`
- Line 343: The code only clips the end values when `window_size` is non-zero
- No validation exists to ensure `window_size` is non-negative

The pandas documentation shows examples using positive window_size values (e.g., `window_size=2`) but doesn't explicitly state that negative values are prohibited. However, the semantic meaning of "Forward" in the class name clearly indicates the intended direction.

## Proposed Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -330,6 +330,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
     ) -> tuple[np.ndarray, np.ndarray]:
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
             raise ValueError(
                 "Forward-looking windows don't support setting the closed argument"
             )
+        if self.window_size < 0:
+            raise ValueError("window_size must be non-negative for forward-looking windows")
         if step is None:
             step = 1
```