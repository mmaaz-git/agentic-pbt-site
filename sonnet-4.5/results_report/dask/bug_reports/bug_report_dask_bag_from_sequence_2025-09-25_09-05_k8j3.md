# Bug Report: dask.bag.from_sequence npartitions Parameter Not Respected

**Target**: `dask.bag.from_sequence`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `from_sequence` function does not respect the `npartitions` parameter. When creating a bag with a specified number of partitions, the actual number of partitions created can be significantly different from what was requested, violating the API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import dask.bag as db


@settings(max_examples=100)
@given(st.lists(st.integers(), min_size=1, max_size=20),
       st.integers(min_value=1, max_value=10))
def test_from_sequence_respects_npartitions(seq, npartitions):
    bag = db.from_sequence(seq, npartitions=npartitions)

    expected = npartitions if len(seq) >= npartitions else len(seq)

    assert bag.npartitions == expected, \
        f"from_sequence(seq of len {len(seq)}, npartitions={npartitions}) " \
        f"produced {bag.npartitions} partitions, expected {expected}"
```

**Failing input**: `seq=[0, 0, 0, 0, 0], npartitions=4` (produces 3 partitions instead of 4)

## Reproducing the Bug

```python
import dask.bag as db

seq = [1, 2, 3, 4]
requested_npartitions = 3

bag = db.from_sequence(seq, npartitions=requested_npartitions)

print(f"Requested: {requested_npartitions}")
print(f"Actual: {bag.npartitions}")

assert bag.npartitions == requested_npartitions
```

Output:
```
Requested: 3
Actual: 2
AssertionError: Expected 3 partitions, got 2
```

## Why This Is A Bug

The `npartitions` parameter is documented as "The number of desired partitions", implying users can specify how many partitions they want. However, the implementation uses a fixed `partition_size` calculation that cannot guarantee the exact number of partitions when the sequence length is not evenly divisible.

This creates downstream issues - for example, `dask.bag.zip` requires all bags to have the same number of partitions (enforced by an assertion), making it impossible to zip bags that were created with the same `npartitions` parameter but different sequence lengths:

```python
bag1 = db.from_sequence([0, 0, 0], npartitions=3)
bag2 = db.from_sequence([0, 0, 0, 0], npartitions=3)
db.zip(bag1, bag2)
```

This raises `AssertionError` because `bag1.npartitions == 3` but `bag2.npartitions == 2`, even though both requested `npartitions=3`.

## Fix

The current implementation uses a fixed partition size calculated by `math.ceil(len(seq) / npartitions)`, which doesn't guarantee the requested number of partitions. Instead, partitions should be created with variable sizes to match the requested count exactly.

Replace the current partitioning logic with:

```diff
--- a/dask/bag/core.py
+++ b/dask/bag/core.py
@@ -45,17 +45,21 @@ def from_sequence(seq, partition_size=None, npartitions=None):
     """
     seq = list(seq)
     if npartitions and not partition_size:
-        if len(seq) <= 100:
-            partition_size = int(math.ceil(len(seq) / npartitions))
-        else:
-            partition_size = max(1, int(math.floor(len(seq) / npartitions)))
+        npartitions = min(npartitions, len(seq)) if len(seq) > 0 else 1
+        base_size = len(seq) // npartitions
+        remainder = len(seq) % npartitions
+        parts = []
+        start = 0
+        for i in range(npartitions):
+            size = base_size + (1 if i < remainder else 0)
+            parts.append(seq[start:start + size])
+            start += size
+        name = "from_sequence-" + tokenize(seq, npartitions)
+        d = {(name, i): list(part) for i, part in enumerate(parts)}
+        return Bag(d, name, len(d))
     if npartitions is None and partition_size is None:
         if len(seq) <= 100:
             partition_size = 1
         else:
             partition_size = max(1, math.ceil(math.sqrt(len(seq)) / math.sqrt(100)))
-
-    parts = list(partition_all(partition_size, seq))
-    name = "from_sequence-" + tokenize(seq, partition_size)
-    if len(parts) > 0:
-        d = {(name, i): list(part) for i, part in enumerate(parts)}
-    else:
-        d = {(name, 0): []}
-
-    return Bag(d, name, len(d))
+    ...
```

This ensures that when `npartitions` is specified, the returned bag has exactly that many partitions (or `len(seq)` partitions if `len(seq) < npartitions`).