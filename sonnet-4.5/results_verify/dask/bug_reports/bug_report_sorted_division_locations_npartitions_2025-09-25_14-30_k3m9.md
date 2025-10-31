# Bug Report: sorted_division_locations Fails to Return Requested Number of Partitions with Duplicate Values

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `sorted_division_locations` is called with a sequence containing only duplicate values and `npartitions` is specified, the function returns fewer divisions than `npartitions + 1`, violating the expected API contract that the result should have exactly `npartitions + 1` divisions.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    npartitions=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=1000)
def test_sorted_division_locations_with_npartitions(seq, npartitions):
    assume(npartitions <= len(seq))
    seq = np.array(sorted(seq))

    divisions, locations = sorted_division_locations(seq, npartitions=npartitions)

    assert len(divisions) == npartitions + 1, \
        f"divisions must have length npartitions + 1, got {len(divisions)}"
```

**Failing input**: `seq=[0, 0], npartitions=2`

## Reproducing the Bug

```python
import numpy as np
from dask.dataframe.io.io import sorted_division_locations

seq = np.array([0, 0])
npartitions = 2
divisions, locations = sorted_division_locations(seq, npartitions=npartitions)

print(f"Expected divisions length: {npartitions + 1}")
print(f"Actual divisions: {divisions} (length: {len(divisions)})")
```

**Output:**
```
Expected divisions length: 3
Actual divisions: [0, 0] (length: 2)
```

## Why This Is A Bug

The function's API contract, as established by its parameters and typical usage, is that when `npartitions=N` is specified, the function should return divisions of length `N + 1` (N partitions require N+1 boundary points). However, when the input sequence contains only duplicate values or has fewer unique values than requested partitions, the function silently returns fewer divisions than requested.

This violates the principle of least surprise - callers expect that specifying `npartitions=2` will result in 2 partitions (3 divisions), but instead get only 1 partition when the input has all duplicate values.

The function should either:
1. Raise a clear error when `npartitions` cannot be satisfied due to insufficient unique values
2. Document that it may return fewer partitions than requested when duplicates prevent the requested partitioning
3. Handle duplicates differently to still satisfy the partition count (e.g., by allowing duplicate division values at consecutive positions)

## Fix

The issue occurs because the function cannot create enough unique division points when the sequence has many duplicates. The function should validate that it can satisfy the requested `npartitions` and raise an informative error if not:

```diff
diff --git a/dask/dataframe/io/io.py b/dask/dataframe/io/io.py
index 1234567..abcdefg 100644
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -285,6 +285,12 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
     enforce_exact = False
+
+    if npartitions is not None and len(seq_unique) < npartitions:
+        raise ValueError(
+            f"Cannot create {npartitions} partitions from a sequence with only "
+            f"{len(seq_unique)} unique value(s). Maximum partitions possible: {len(seq_unique)}"
+        )

     if duplicates:
         offsets = [bisect.bisect_left(seq, x) for x in seq_unique]
```