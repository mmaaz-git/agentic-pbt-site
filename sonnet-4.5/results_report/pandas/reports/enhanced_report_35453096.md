# Bug Report: FixedForwardWindowIndexer Invalid Window Bounds with Negative window_size

**Target**: `pandas.api.indexers.FixedForwardWindowIndexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FixedForwardWindowIndexer produces invalid window bounds where `start[i] > end[i]` when instantiated with a negative `window_size`, violating the fundamental invariant that window bounds must satisfy `start <= end`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-20, max_value=-1),
    step=st.integers(min_value=1, max_value=10)
)
def test_fixed_forward_indexer_negative_window_size(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invalid window: start[{i}]={start[i]} > end[{i}]={end[i]}"

# Run the test
if __name__ == "__main__":
    test_fixed_forward_indexer_negative_window_size()
```

<details>

<summary>
**Failing input**: `num_values=2, window_size=-1, step=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 18, in <module>
    test_fixed_forward_indexer_negative_window_size()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 5, in test_fixed_forward_indexer_negative_window_size
    num_values=st.integers(min_value=1, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 14, in test_fixed_forward_indexer_negative_window_size
    assert start[i] <= end[i], f"Invalid window: start[{i}]={start[i]} > end[{i}]={end[i]}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Invalid window: start[1]=1 > end[1]=0
Falsifying example: test_fixed_forward_indexer_negative_window_size(
    num_values=2,
    window_size=-1,  # or any other generated value
    step=1,
)
```
</details>

## Reproducing the Bug

```python
from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2, step=1)

print(f"start: {start}")
print(f"end: {end}")
print(f"\nInvariant violated: start[1]={start[1]} > end[1]={end[1]}")

# Check all windows
for i in range(len(start)):
    if start[i] > end[i]:
        print(f"Window {i}: start[{i}]={start[i]} > end[{i}]={end[i]} - INVALID")
    else:
        print(f"Window {i}: start[{i}]={start[i]} <= end[{i}]={end[i]} - OK")
```

<details>

<summary>
Output demonstrating invalid window bounds
</summary>
```
start: [0 1]
end: [0 0]

Invariant violated: start[1]=1 > end[1]=0
Window 0: start[0]=0 <= end[0]=0 - OK
Window 1: start[1]=1 > end[1]=0 - INVALID
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Mathematical Invariant Violation**: Window bounds must always satisfy `start[i] <= end[i]` for valid indexing operations. When this invariant is broken, any code that uses these bounds for slicing or iteration will produce incorrect results or potentially crash.

2. **Semantic Contradiction**: The class name "FixedForwardWindowIndexer" explicitly indicates "forward-looking" windows. A negative window size contradicts this semantic meaning - you cannot look "forward" by a negative amount. This is reinforced by the existing validation that rejects `center=True` with the error message "Forward-looking windows can't have center=True".

3. **Implementation Flaw**: The bug occurs in lines 340-343 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/objects.py`:
   ```python
   start = np.arange(0, num_values, step, dtype="int64")
   end = start + self.window_size  # Negative window_size makes end < start
   if self.window_size:
       end = np.clip(end, 0, num_values)  # Clipping to 0 can make end < start
   ```
   When `window_size` is negative, `end = start + window_size` produces values less than `start`. The subsequent clipping operation `np.clip(end, 0, num_values)` forces negative values to 0, creating situations where `start[i] > end[i]`.

4. **Documentation Expectations**: While not explicitly documented, the class docstring example shows `window_size=2` (positive), and all pandas documentation examples use positive window sizes for forward-looking windows.

## Relevant Context

The FixedForwardWindowIndexer is part of pandas' rolling window API, used for operations like moving averages and other windowed aggregations. The indexer determines which elements are included in each window during rolling operations.

Key observations from the source code:
- The class already validates semantic constraints (e.g., line 331-332 rejects `center=True`)
- Other indexer classes like FixedWindowIndexer properly handle their bounds with clipping
- The implementation assumes positive window_size without explicit validation
- Similar validation exists in VariableOffsetWindowIndexer which validates its offset parameter

Documentation reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.indexers.FixedForwardWindowIndexer.html

## Proposed Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -329,6 +329,8 @@ class FixedForwardWindowIndexer(BaseIndexer):
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
+        if self.window_size < 0:
+            raise ValueError("FixedForwardWindowIndexer requires window_size >= 0, got {self.window_size}")
         if center:
             raise ValueError("Forward-looking windows can't have center=True")
         if closed is not None:
```