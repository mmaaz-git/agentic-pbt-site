# Bug Report: FixedWindowIndexer Produces Invalid Window Bounds Where Start > End

**Target**: `pandas.core.indexers.objects.FixedWindowIndexer.get_window_bounds`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`FixedWindowIndexer.get_window_bounds` produces window bounds where `start[i] > end[i]`, violating the fundamental invariant that window start indices must be less than or equal to window end indices. This occurs specifically when `window_size=0` and `closed='neither'`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
from pandas.core.indexers.objects import FixedWindowIndexer


@given(
    num_values=st.integers(min_value=0, max_value=100),
    window_size=st.integers(min_value=0, max_value=50),
    center=st.booleans(),
    closed=st.sampled_from([None, "left", "right", "both", "neither"]),
    step=st.one_of(st.none(), st.integers(min_value=1, max_value=10))
)
@settings(max_examples=1000)
def test_fixed_window_indexer_invariants(num_values, window_size, center, closed, step):
    indexer = FixedWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(
        num_values=num_values,
        center=center,
        closed=closed,
        step=step
    )

    assert len(start) == len(end)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invariant violated at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}"
        assert 0 <= start[i] <= num_values
        assert 0 <= end[i] <= num_values

if __name__ == "__main__":
    test_fixed_window_indexer_invariants()
```

<details>

<summary>
**Failing input**: `num_values=2, window_size=0, center=False, closed='neither', step=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 33, in <module>
    test_fixed_window_indexer_invariants()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 9, in test_fixed_window_indexer_invariants
    num_values=st.integers(min_value=0, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 28, in test_fixed_window_indexer_invariants
    assert start[i] <= end[i], f"Invariant violated at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Invariant violated at index 1: start[1]=1 > end[1]=0
Falsifying example: test_fixed_window_indexer_invariants(
    num_values=2,
    window_size=0,
    center=False,
    closed='neither',
    step=None,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.indexers.objects import FixedWindowIndexer

# Create an indexer with window_size=0
indexer = FixedWindowIndexer(window_size=0)

# Get window bounds with problematic parameters
start, end = indexer.get_window_bounds(num_values=2, closed='neither')

print(f"Window size: 0")
print(f"Num values: 2")
print(f"Closed: 'neither'")
print(f"Start: {start}")
print(f"End: {end}")
print()

# Check if the invariant holds
for i in range(len(start)):
    print(f"Window {i}: start={start[i]}, end={end[i]}, valid={start[i] <= end[i]}")

# Show which windows violate the invariant
violations = []
for i in range(len(start)):
    if start[i] > end[i]:
        violations.append(i)

if violations:
    print(f"\nInvariant violations found at indices: {violations}")
    print("This means window start > window end, which is invalid for a window range.")
```

<details>

<summary>
Window bounds invariant violation: start[1]=1 > end[1]=0
</summary>
```
Window size: 0
Num values: 2
Closed: 'neither'
Start: [0 1]
End: [0 0]

Window 0: start=0, end=0, valid=True
Window 1: start=1, end=0, valid=False

Invariant violations found at indices: [1]
This means window start > window end, which is invalid for a window range.
```
</details>

## Why This Is A Bug

Window bounds represent a slice range `[start, end)` where `start` is the inclusive beginning index and `end` is the exclusive ending index. For this to be a valid range (even if empty), the mathematical invariant `start <= end` must hold. When `start > end`, the window bounds become nonsensical and cannot represent any valid slice of data.

The bug occurs due to an unintended interaction in the code at line 105 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/objects.py`. When `window_size=0`, the condition `if center or self.window_size == 0:` evaluates to True, triggering the centering offset calculation with `offset = (self.window_size - 1) // 2 = -1`. This negative offset, combined with the `closed='neither'` parameter which further decrements the end array, produces invalid window bounds where start exceeds end.

While Python's array slicing handles `array[1:0]` gracefully by returning an empty array, this doesn't make the bounds semantically correct. The window indexer is producing mathematically invalid ranges that violate the fundamental contract of window operations. This invariant is implicitly assumed throughout pandas' windowing system and by any code that uses these bounds for data processing.

## Relevant Context

The FixedWindowIndexer is part of pandas' rolling window operations API, used extensively for time series analysis and data smoothing operations. The class is designed to create fixed-length windows for rolling calculations.

Key observations from the source code:
- Line 105: The condition `if center or self.window_size == 0:` treats `window_size=0` as if centering was requested, even when `center=False`
- Line 106: For `window_size=0`, this calculates `offset = (0 - 1) // 2 = -1`
- Lines 114-115: When `closed='neither'`, the end array is decremented by 1
- This combination leads to `end` values becoming less than `start` values after clipping

Other window indexers in pandas (ExpandingIndexer, FixedForwardWindowIndexer) maintain the `start <= end` invariant correctly. The documentation for `get_window_bounds` states it returns "boundaries of each window" but doesn't explicitly document this invariant, though it's a fundamental property of any valid range specification.

## Proposed Fix

```diff
--- a/pandas/core/indexers/objects.py
+++ b/pandas/core/indexers/objects.py
@@ -102,7 +102,7 @@ class FixedWindowIndexer(BaseIndexer):
         closed: str | None = None,
         step: int | None = None,
     ) -> tuple[np.ndarray, np.ndarray]:
-        if center or self.window_size == 0:
+        if center:
             offset = (self.window_size - 1) // 2
         else:
             offset = 0
```