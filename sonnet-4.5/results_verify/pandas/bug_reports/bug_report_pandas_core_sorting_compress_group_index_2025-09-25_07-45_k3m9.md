# Bug Report: pandas.core.sorting.compress_group_index Uniqueness Violation

**Target**: `pandas.core.sorting.compress_group_index`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `compress_group_index` function violates its uniqueness invariant when processing unsorted arrays containing only distinct negative values. Different input values are incorrectly mapped to the same compressed ID.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.sorting import compress_group_index


@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=0, max_size=50))
@settings(max_examples=500)
def test_compress_group_index_preserves_uniqueness(group_index):
    if len(group_index) == 0:
        return

    group_index_arr = np.array(group_index, dtype=np.int64)
    comp_ids, obs_group_ids = compress_group_index(group_index_arr, sort=True)

    assert len(comp_ids) == len(group_index)

    mapping = {}
    for comp_id, orig_id in zip(comp_ids, group_index_arr):
        if comp_id not in mapping:
            mapping[comp_id] = orig_id
        else:
            assert mapping[comp_id] == orig_id
```

**Failing input**: `group_index=[-1, -2]`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.sorting import compress_group_index

group_index = np.array([-1, -2], dtype=np.int64)
comp_ids, obs_group_ids = compress_group_index(group_index, sort=True)

print(f"Input: {group_index}")
print(f"Output comp_ids: {comp_ids}")
print(f"Output obs_group_ids: {obs_group_ids}")

for i, (comp_id, orig_id) in enumerate(zip(comp_ids, group_index)):
    print(f"Position {i}: group_index[{i}]={orig_id} → comp_id={comp_id}")
```

Output:
```
Input: [-1 -2]
Output comp_ids: [-1 -1]
Output obs_group_ids: []
Position 0: group_index[0]=-1 → comp_id=-1
Position 1: group_index[1]=-2 → comp_id=-1
```

## Why This Is A Bug

The function's docstring states it "compresses [group_index], by computing offsets (comp_ids) into the list of unique labels (obs_group_ids)". The fundamental invariant is that equal input values should map to equal comp_ids, and different input values should map to different comp_ids (injective mapping).

In this case, two distinct input values (-1 and -2) are both mapped to comp_id=-1, violating this invariant.

The bug only occurs in the slow path (hashtable-based). When the input is sorted ascending (triggering the fast path), the function works correctly: `compress_group_index(np.array([-2, -1], dtype=np.int64))` returns `comp_ids=[-1, 0]`, which correctly maps different values to different comp_ids.

## Fix

The bug is in the slow path at line 710, where `table.get_labels_groupby(group_index)` from the pandas hashtable incorrectly handles arrays containing only negative values. When all values are negative, it returns empty `obs_group_ids` and maps everything to -1.

The hashtable's `get_labels_groupby` implementation appears to treat negative values as a special case, possibly filtering them out entirely when no positive values are present. This behavior is inconsistent with the fast path and violates the function's contract.

A potential fix would be to ensure the slow path handles negative values consistently, either by:
1. Preprocessing to offset all negative values before passing to the hashtable, then reversing the offset in the output
2. Fixing the hashtable's `get_labels_groupby` to handle negative values correctly
3. Falling back to a simpler unique-based approach when all values are negative

Without access to the hashtable C implementation, a Python-level workaround would be:

```diff
diff --git a/pandas/core/sorting.py b/pandas/core/sorting.py
index abc123..def456 100644
--- a/pandas/core/sorting.py
+++ b/pandas/core/sorting.py
@@ -703,6 +703,15 @@ def compress_group_index(
         obs_group_ids = group_index[unique_mask]
     else:
         size_hint = len(group_index)
+
+        # Handle all-negative values which hashtable doesn't handle correctly
+        min_val = group_index.min()
+        if min_val < 0 and group_index.max() < 0:
+            offset = -min_val
+            shifted = group_index + offset
+            comp_ids, obs_group_ids = compress_group_index(shifted, sort=sort)
+            return ensure_int64(comp_ids), ensure_int64(obs_group_ids - offset)
+
         table = hashtable.Int64HashTable(size_hint)

         group_index = ensure_int64(group_index)
```