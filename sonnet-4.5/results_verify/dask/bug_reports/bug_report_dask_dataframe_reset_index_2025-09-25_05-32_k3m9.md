# Bug Report: dask.dataframe.DataFrame.reset_index Produces Duplicate Index Values

**Target**: `dask.dataframe.DataFrame.reset_index`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `reset_index(drop=True)` is called on a multi-partition Dask DataFrame, each partition independently resets its index to start from 0, resulting in duplicate index values. This violates pandas semantics where `reset_index(drop=True)` creates a continuous RangeIndex.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import dask.dataframe as dd

@settings(max_examples=300)
@given(
    data=st.lists(
        st.tuples(st.integers(-1000, 1000), st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
        min_size=1,
        max_size=100
    ),
    npartitions=st.integers(min_value=1, max_value=10)
)
def test_reset_index_matches_pandas(data, npartitions):
    pdf = pd.DataFrame(data, columns=['a', 'b'])
    assume(len(pdf) >= npartitions)

    ddf = dd.from_pandas(pdf, npartitions=npartitions)

    reset_ddf = ddf.reset_index(drop=True).compute()
    reset_pdf = pdf.reset_index(drop=True)

    pd.testing.assert_frame_equal(reset_ddf, reset_pdf, check_index_type=False)
```

**Failing input**: `data=[(0, 0.0), (0, 0.0)], npartitions=2`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

data = [(0, 0.0), (0, 0.0)]
pdf = pd.DataFrame(data, columns=['a', 'b'])
ddf = dd.from_pandas(pdf, npartitions=2)

reset_pdf = pdf.reset_index(drop=True)
reset_ddf = ddf.reset_index(drop=True).compute()

print("Pandas index:", reset_pdf.index.tolist())
print("Dask index:", reset_ddf.index.tolist())

assert reset_pdf.index.tolist() == [0, 1]
assert reset_ddf.index.tolist() == [0, 0]
```

## Why This Is A Bug

The pandas documentation for `DataFrame.reset_index(drop=True)` states it "resets the index to the default integer index", which is a continuous RangeIndex from 0 to n-1. Dask claims to replicate pandas behavior but instead produces duplicate index values [0, 0] when the DataFrame has multiple partitions.

This occurs because each partition independently resets its local index to start from 0, without coordinating to create a global continuous index. The correct behavior would be for partition i to have indices starting from the cumulative size of all previous partitions.

## Fix

The fix requires coordinating index offsets across partitions. Each partition should reset its index starting from the cumulative count of rows in all previous partitions:

```diff
--- a/dask/dataframe/core.py
+++ b/dask/dataframe/core.py
@@ -reset_index_implementation
-    def _reset_index_partition(df):
-        return df.reset_index(drop=drop)
+    def _reset_index_partition(df, offset=0):
+        result = df.reset_index(drop=drop)
+        if drop:
+            result.index = range(offset, offset + len(result))
+        return result

-    return map_partitions(_reset_index_partition, df)
+    partition_sizes = df.map_partitions(len).compute()
+    offsets = [0] + list(partition_sizes.cumsum()[:-1])
+
+    return map_partitions(_reset_index_partition, df, offsets=offsets)
```

Note: The exact implementation may differ based on dask's internal architecture, but the key insight is that offsets must be calculated and passed to each partition.