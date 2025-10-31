# Bug Report: pandas.core.reshape.tile.qcut crashes with duplicates='drop' on skewed data

**Target**: `pandas.core.reshape.tile.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.qcut()` with `duplicates='drop'` crashes with `ValueError: missing values must be missing in the same location both left and right sides` when applied to data with many duplicate values and quantile bins that collapse to nearly identical edges.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import column, data_frames, range_indexes


@given(
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6), min_size=20, max_size=100),
    q=st.integers(2, 10),
)
@settings(max_examples=300)
def test_qcut_equal_sized_bins(x, q):
    s = pd.Series(x)
    assume(len(s.unique()) >= q)
    result = pd.qcut(s, q=q, duplicates="drop")
    counts = result.value_counts()
    max_count = counts.max()
    min_count = counts.min()
    assert max_count - min_count <= 2
```

**Failing input**:
```python
x=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2250738585072014e-308]
q=2
```

## Reproducing the Bug

```python
import pandas as pd

x = [0.0] * 19 + [2.2250738585072014e-308]
s = pd.Series(x)

result = pd.qcut(s, q=2, duplicates="drop")
```

**Output**:
```
ValueError: missing values must be missing in the same location both left and right sides
```

**Full traceback**:
```
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 340, in qcut
    fac, bins = _bins_to_cuts(
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 483, in _format_labels
    labels = _format_labels(
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 577, in _format_labels
    return IntervalIndex.from_breaks(breaks, closed=closed)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/interval.py", line 275, in from_breaks
    array = IntervalArray.from_breaks(
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py", line 464, in from_breaks
    return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py", line 552, in from_arrays
    cls._validate(left, right, dtype=dtype)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py", line 664, in _validate
    raise ValueError(msg)
ValueError: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

The documentation for `qcut` states that `duplicates='drop'` should "drop non-uniques" when bin edges are not unique. This is a valid use case - when quantiles produce duplicate bin edges (common with highly skewed data), users should be able to request that duplicate edges be dropped rather than raising an error.

However, the implementation fails in certain edge cases. When `algos.unique(bins)` is called in `_bins_to_cuts` (line 440-447 of tile.py), it can produce a bins array that, after being processed through `_format_labels` and `IntervalIndex.from_breaks`, results in misaligned NaN values in the left and right interval boundaries.

This violates the constraint checked by `IntervalArray._validate` that "missing values must be missing in the same location both left and right sides" (interval.py:659-664).

## Fix

The root cause is that when `duplicates='drop'` deduplicates bins, the resulting breaks array may not be suitable for creating valid intervals. The fix should ensure that after deduplication, the breaks array can form valid non-NaN intervals.

One approach is to add validation after line 447 in tile.py to ensure that the deduplicated bins don't create invalid interval boundaries:

```diff
diff --git a/pandas/core/reshape/tile.py b/pandas/core/reshape/tile.py
index abc123..def456 100644
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -444,7 +444,12 @@ def _bins_to_cuts(
                 f"Bin edges must be unique: {repr(bins)}.\n"
                 f"You can drop duplicate edges by setting the 'duplicates' kwarg"
             )
-        bins = unique_bins
+        bins = unique_bins
+        # Ensure we have at least 2 bins after deduplication
+        if len(bins) < 2:
+            raise ValueError(
+                f"After dropping duplicates, must have at least 2 bin edges, got {len(bins)}"
+            )

     side: Literal["left", "right"] = "left" if right else "right"
```

However, a more complete fix would require investigating why `algos.unique()` on bins with near-zero differences produces NaN values in inconsistent positions, or ensuring that `_format_labels` handles deduplicated bins more robustly.