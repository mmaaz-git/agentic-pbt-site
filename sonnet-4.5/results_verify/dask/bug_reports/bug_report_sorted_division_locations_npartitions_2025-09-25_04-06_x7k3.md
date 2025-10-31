# Bug Report: sorted_division_locations Returns Fewer Divisions Than Requested

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `sorted_division_locations` is called with `npartitions=N` on a sequence with fewer unique values than N, it returns fewer than `N+1` divisions, violating the invariant that `len(divisions) == npartitions + 1`. This causes downstream code to create fewer partitions than requested.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
    npartitions=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=200, deadline=None)
def test_npartitions_length(seq, npartitions):
    assume(len(seq) >= npartitions)
    seq_sorted = sorted(seq)
    idx = pd.Index(seq_sorted)
    divisions, locations = sorted_division_locations(idx, npartitions=npartitions)

    assert len(divisions) == npartitions + 1, \
        f"divisions should have length npartitions+1: expected {npartitions+1}, got {len(divisions)}"
```

**Failing input**: `seq=[0, 0]`, `npartitions=2`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

idx = pd.Index([0, 0])
divisions, locations = sorted_division_locations(idx, npartitions=2)

print(f"Requested npartitions: 2")
print(f"Expected len(divisions): 3")
print(f"Actual len(divisions): {len(divisions)}")
print(f"Divisions: {divisions}")
print(f"Locations: {locations}")
```

Output:
```
Requested npartitions: 2
Expected len(divisions): 3
Actual len(divisions): 2
Divisions: [0, 0]
Locations: [0, 2]
```

## Why This Is A Bug

The function violates its implicit contract that when `npartitions=N` is specified, it will return `N+1` divisions. This contract is relied upon by `FromPandas.npartitions` (line 527 of io.py):

```python
def npartitions(self):
    if self._filtered:
        return super().npartitions
    return len(self._divisions_and_locations[0]) - 1  # Assumes len(divisions) == npartitions + 1
```

When `sorted_division_locations` returns 2 divisions instead of 3 for `npartitions=2`, the calling code incorrectly computes `npartitions = 2 - 1 = 1` instead of the requested 2 partitions.

The bug occurs when the number of unique values in the sequence is less than the requested `npartitions`. The function sets `enforce_exact = False` (line 292-293) and doesn't attempt to create the requested number of partitions.

## Fix

The function should handle the case where unique values are insufficient by either:

1. **Raising an error** when it cannot satisfy the requested npartitions:

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -289,7 +289,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     if duplicates:
         offsets = [bisect.bisect_left(seq, x) for x in seq_unique]
         enforce_exact = npartitions and len(offsets) >= npartitions
+        if npartitions and len(offsets) < npartitions:
+            raise ValueError(
+                f"Cannot create {npartitions} partitions from sequence with only "
+                f"{len(offsets)} unique values. Reduce npartitions or provide more unique values."
+            )
     else:
         offsets = seq_unique = None
```

2. **Creating empty partitions** to satisfy the length invariant (more complex, may not be desirable).

Option 1 (raising an error) is recommended as it makes the limitation explicit rather than silently returning incorrect results.