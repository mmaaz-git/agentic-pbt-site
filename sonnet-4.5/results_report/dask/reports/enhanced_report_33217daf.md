# Bug Report: dask.dataframe.dask_expr._repartition._clean_new_division_boundaries Produces Non-Monotonic and Invalid Partition Boundaries

**Target**: `dask.dataframe.dask_expr._repartition._clean_new_division_boundaries`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_clean_new_division_boundaries` function produces mathematically invalid partition boundaries that violate monotonicity requirements and fails to ensure boundaries properly span from 0 to frame_npartitions, potentially causing data corruption in distributed DataFrame operations.

## Property-Based Test

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, assume, strategies as st, settings
from dask.dataframe.dask_expr._repartition import _clean_new_division_boundaries

@given(
    boundaries=st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=10),
    frame_npartitions=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_clean_boundaries_invariants(boundaries, frame_npartitions):
    assume(len(boundaries) >= 2)
    assume(all(b >= 0 for b in boundaries))

    result = _clean_new_division_boundaries(boundaries, frame_npartitions)

    assert result[0] == 0, f"First boundary should always be 0, got {result[0]}"
    assert result[-1] == frame_npartitions, f"Last boundary should be frame_npartitions ({frame_npartitions}), got {result[-1]}"

    for i in range(len(result) - 1):
        assert result[i] <= result[i+1], f"Boundaries should be non-decreasing at index {i}: {result}"

if __name__ == "__main__":
    test_clean_boundaries_invariants()
```

<details>

<summary>
**Failing input**: `boundaries=[2, 0], frame_npartitions=1`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 26, in <module>
  |     test_clean_boundaries_invariants()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 9, in test_clean_boundaries_invariants
  |     boundaries=st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=10),
  |                ^^^
  |   File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 23, in test_clean_boundaries_invariants
    |     assert result[i] <= result[i+1], f"Boundaries should be non-decreasing at index {i}: {result}"
    |            ^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Boundaries should be non-decreasing at index 1: [0, 2, 1]
    | Falsifying example: test_clean_boundaries_invariants(
    |     boundaries=[2, 0],
    |     frame_npartitions=1,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 20, in test_clean_boundaries_invariants
    |     assert result[-1] == frame_npartitions, f"Last boundary should be frame_npartitions ({frame_npartitions}), got {result[-1]}"
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Last boundary should be frame_npartitions (1), got 2
    | Falsifying example: test_clean_boundaries_invariants(
    |     boundaries=[0, 2],
    |     frame_npartitions=1,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.dataframe.dask_expr._repartition import _clean_new_division_boundaries

print("Testing Bug 1: Non-monotonic boundaries")
boundaries1 = [2, 0]
print(f"Input: boundaries={boundaries1}, frame_npartitions=1")
result1 = _clean_new_division_boundaries(boundaries1.copy(), 1)
print(f"Output: {result1}")
print(f"Is non-monotonic? {result1[1] > result1[2]} (element at index 1 > element at index 2)")
print()

print("Testing Bug 2: Last boundary != frame_npartitions")
boundaries2 = [0, 2]
print(f"Input: boundaries={boundaries2}, frame_npartitions=1")
result2 = _clean_new_division_boundaries(boundaries2.copy(), 1)
print(f"Output: {result2}")
print(f"Last element equals frame_npartitions? {result2[-1] == 1} (last={result2[-1]}, expected=1)")
print()

print("Testing Bug 3: Lost intermediate boundary")
boundaries3 = [0, 5, 10]
print(f"Input: boundaries={boundaries3}, frame_npartitions=20")
result3 = _clean_new_division_boundaries(boundaries3.copy(), 20)
print(f"Output: {result3}")
print(f"Original boundary 10 lost? {10 not in result3}")
print(f"Original length: {len([0, 5, 10])}, Result length: {len(result3)}")
```

<details>

<summary>
Non-monotonic boundaries, incorrect last boundary, and lost intermediate values
</summary>
```
Testing Bug 1: Non-monotonic boundaries
Input: boundaries=[2, 0], frame_npartitions=1
Output: [0, 2, 1]
Is non-monotonic? True (element at index 1 > element at index 2)

Testing Bug 2: Last boundary != frame_npartitions
Input: boundaries=[0, 2], frame_npartitions=1
Output: [0, 2]
Last element equals frame_npartitions? False (last=2, expected=1)

Testing Bug 3: Lost intermediate boundary
Input: boundaries=[0, 5, 10], frame_npartitions=20
Output: [0, 5, 20]
Original boundary 10 lost? True
Original length: 3, Result length: 3
```
</details>

## Why This Is A Bug

Partition boundaries in distributed DataFrames must satisfy fundamental mathematical invariants to correctly define partition ranges. The current implementation violates three critical requirements:

1. **Monotonicity Violation**: The function produces non-monotonic sequences like `[0, 2, 1]` when given unsorted input `[2, 0]`. Non-decreasing boundaries are essential because they define partition ranges as [boundary[i], boundary[i+1]). A sequence like `[0, 2, 1]` would define partition 1 as the range [2, 1), which is mathematically invalid and would cause incorrect data partitioning.

2. **Coverage Violation**: When boundaries exceed frame_npartitions (e.g., `[0, 2]` with frame_npartitions=1), the function fails to cap the last boundary at frame_npartitions. This violates the invariant that partitions must span exactly from 0 to frame_npartitions, potentially causing out-of-bounds errors or missing data.

3. **Data Loss**: The function replaces the last boundary instead of appending when it's less than frame_npartitions. For example, `[0, 5, 10]` becomes `[0, 5, 20]` when frame_npartitions=20, silently dropping the boundary at 10. This loses important partition boundaries that may have been calculated based on data distribution.

The function has no documentation, but its usage context in lines 184-186 and 451-453 of the same file shows it's used to clean boundaries generated from partition calculations. The function name and implementation suggest it should ensure boundaries start at 0 and end at frame_npartitions while preserving validity.

## Relevant Context

The `_clean_new_division_boundaries` function is an internal utility in Dask's DataFrame repartitioning logic, located at `/dask/dataframe/dask_expr/_repartition.py` (lines 499-506). It's called in two places:

1. **Line 184-186**: Used with boundaries calculated from integer partition ratios
2. **Line 451-453**: Used with cumulative sums of new partition counts

The function lacks any documentation - no docstring, comments, type hints, or specification of expected behavior. Despite being internal (underscore prefix), it handles critical partition boundary management that affects how data is distributed across partitions in Dask's distributed computing framework.

The implementation attempts to enforce that boundaries start at 0 (by prepending if needed) and end at frame_npartitions (by modifying the last element), but fails to maintain sorted order and incorrectly replaces instead of appending values.

## Proposed Fix

```diff
--- a/dask/dataframe/dask_expr/_repartition.py
+++ b/dask/dataframe/dask_expr/_repartition.py
@@ -499,9 +499,18 @@ def _clean_new_division_boundaries(new_partitions_boundaries, frame_npartitions
 def _clean_new_division_boundaries(new_partitions_boundaries, frame_npartitions):
     if not isinstance(new_partitions_boundaries, list):
         new_partitions_boundaries = list(new_partitions_boundaries)
+
+    # Ensure boundaries are sorted and unique
+    new_partitions_boundaries = sorted(set(new_partitions_boundaries))
+
+    # Ensure first boundary is 0
     if new_partitions_boundaries[0] > 0:
         new_partitions_boundaries.insert(0, 0)
+
+    # Ensure last boundary is frame_npartitions
     if new_partitions_boundaries[-1] < frame_npartitions:
-        new_partitions_boundaries[-1] = frame_npartitions
+        new_partitions_boundaries.append(frame_npartitions)
+    elif new_partitions_boundaries[-1] > frame_npartitions:
+        new_partitions_boundaries[-1] = frame_npartitions
+
     return new_partitions_boundaries
```