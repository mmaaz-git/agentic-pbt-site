# Bug Report: dask.dataframe.dask_expr._repartition._clean_new_division_boundaries Multiple Logic Bugs

**Target**: `dask.dataframe.dask_expr._repartition._clean_new_division_boundaries`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_clean_new_division_boundaries` function has multiple logic bugs that violate fundamental invariants: (1) it can produce non-monotonic boundaries, (2) it doesn't ensure the last boundary equals `frame_npartitions` when inputs exceed it, and (3) it loses boundaries by replacing instead of appending.

## Property-Based Test

```python
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

    assert result[0] == 0, "First boundary should always be 0"
    assert result[-1] == frame_npartitions, "Last boundary should be frame_npartitions"

    for i in range(len(result) - 1):
        assert result[i] <= result[i+1], f"Boundaries should be non-decreasing: {result}"
```

**Failing input 1**: `boundaries=[2, 0], frame_npartitions=1` → produces `[0, 2, 1]` (non-monotonic)
**Failing input 2**: `boundaries=[0, 2], frame_npartitions=1` → produces `[0, 2]` (last != frame_npartitions)
**Failing input 3**: `boundaries=[0, 5, 10], frame_npartitions=20` → produces `[0, 5, 20]` (lost boundary 10)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/dask/env')

from dask.dataframe.dask_expr._repartition import _clean_new_division_boundaries

boundaries1 = [2, 0]
result1 = _clean_new_division_boundaries(boundaries1.copy(), 1)
print(f"Bug 1: {result1}")
assert result1 == [0, 2, 1]
assert result1[1] > result1[2]

boundaries2 = [0, 2]
result2 = _clean_new_division_boundaries(boundaries2.copy(), 1)
print(f"Bug 2: {result2}")
assert result2 == [0, 2]
assert result2[-1] != 1

boundaries3 = [0, 5, 10]
result3 = _clean_new_division_boundaries(boundaries3.copy(), 20)
print(f"Bug 3: {result3}")
assert len(result3) == 3
assert 10 not in result3
```

## Why This Is A Bug

Partition boundaries must satisfy critical invariants for correct repartitioning:
1. **Monotonicity**: Boundaries must be non-decreasing to define valid partition ranges
2. **Coverage**: Must start at 0 and end at `frame_npartitions` to cover all partitions
3. **Preservation**: Should not silently drop intermediate boundaries

The current implementation violates all three:
- Bug 1 creates non-monotonic boundaries `[0, 2, 1]`, making partition ranges invalid
- Bug 2 fails to enforce `last == frame_npartitions` when input exceeds it
- Bug 3 replaces the last element instead of appending, losing intermediate boundaries

## Fix

```diff
--- a/dask/dataframe/dask_expr/_repartition.py
+++ b/dask/dataframe/dask_expr/_repartition.py
@@ -499,9 +499,14 @@ def _clean_new_division_boundaries(new_partitions_boundaries, frame_npartitions
 def _clean_new_division_boundaries(new_partitions_boundaries, frame_npartitions):
     if not isinstance(new_partitions_boundaries, list):
         new_partitions_boundaries = list(new_partitions_boundaries)
+
+    # Ensure boundaries are sorted
+    new_partitions_boundaries = sorted(set(new_partitions_boundaries))
+
     if new_partitions_boundaries[0] > 0:
         new_partitions_boundaries.insert(0, 0)
     if new_partitions_boundaries[-1] < frame_npartitions:
-        new_partitions_boundaries[-1] = frame_npartitions
+        new_partitions_boundaries.append(frame_npartitions)
+    elif new_partitions_boundaries[-1] > frame_npartitions:
+        new_partitions_boundaries[-1] = frame_npartitions
     return new_partitions_boundaries
```