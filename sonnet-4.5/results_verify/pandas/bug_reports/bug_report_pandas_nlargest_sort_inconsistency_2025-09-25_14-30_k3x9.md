# Bug Report: pandas DataFrame.nlargest() Index Ordering Inconsistency

**Target**: `pandas.core.methods.selectn.SelectNFrame.compute`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`DataFrame.nlargest(n, columns)` produces different index ordering than the semantically equivalent `DataFrame.sort_values(columns, ascending=False).head(n)` when there are tied values. This violates user expectations that these two operations should be equivalent.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd


@given(
    df=data_frames(
        columns=[
            column("a", dtype=int),
            column("b", dtype=int),
        ],
        index=range_indexes(min_size=1, max_size=100),
    ),
    n=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=300)
def test_nlargest_multicolumn_consistent_with_sort(df, n):
    assume(len(df) > 0)

    result_nlargest = df.nlargest(n, ['a', 'b'])
    result_sort = df.sort_values(['a', 'b'], ascending=False).head(n)

    pd.testing.assert_frame_equal(result_nlargest, result_sort)
```

**Failing input**:
```python
df = pd.DataFrame({
    'a': [0, 0, 1, 1],
    'b': [0, 0, 0, 0]
})
n = 4
```

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({
    'a': [0, 0, 1, 1],
    'b': [0, 0, 0, 0]
})

result_nlargest = df.nlargest(4, ['a', 'b'])
result_sort = df.sort_values(['a', 'b'], ascending=False).head(4)

print("nlargest result indices:", result_nlargest.index.tolist())
print("sort_values result indices:", result_sort.index.tolist())
print("Are they equal?", result_nlargest.index.tolist() == result_sort.index.tolist())
```

**Output**:
```
nlargest result indices: [3, 2, 1, 0]
sort_values result indices: [2, 3, 0, 1]
Are they equal? False
```

## Why This Is A Bug

Users expect `df.nlargest(n, cols)` to be equivalent to `df.sort_values(cols, ascending=False).head(n)`. The pandas documentation and common usage patterns reinforce this expectation. When there are tied values, a stable sort should preserve the original relative ordering of equal elements. However, `nlargest` returns indices in reverse order `[3, 2, 1, 0]` while `sort_values` returns them in forward order `[2, 3, 0, 1]`.

This inconsistency can lead to:
1. Unexpected results when users switch between the two equivalent-seeming operations
2. Non-deterministic behavior depending on which API is used
3. Difficulty in testing and debugging code that relies on stable ordering

## Root Cause

The bug is in `/pandas/core/methods/selectn.py` in the `SelectNFrame.compute` method. The issue stems from how the Series `.nlargest()` method (called at line 232) handles index ordering differently than how the final `sort_values()` call (line 269) orders indices.

Specifically:
1. When processing each column, `SelectNFrame` calls the Series `nlargest` method which may return indices in a certain order
2. For multi-column cases, indices are collected and then a final `sort_values(kind="mergesort")` is called at line 269
3. However, the stable sort order used by `sort_values` may be different from the order produced by the Series `nlargest` algorithm

## Fix

The fix should ensure that `DataFrame.nlargest()` produces the same index ordering as `DataFrame.sort_values(ascending=False).head(n)`. One approach:

```diff
--- a/pandas/core/methods/selectn.py
+++ b/pandas/core/methods/selectn.py
@@ -260,9 +260,7 @@ class SelectNFrame(SelectN):
         # Restore the index on frame
         frame.index = original_index.take(indexer)

-        # If there is only one column, the frame is already sorted.
-        if len(columns) == 1:
-            return frame
+        # Always sort to ensure consistent ordering with sort_values

         ascending = method == "nsmallest"
```

This ensures that both single-column and multi-column cases go through the same `sort_values` call, providing consistent behavior. However, this may have performance implications, and a more nuanced fix might be needed to ensure the Series `nlargest` produces the same ordering as `sort_values` would.