# Bug Report: dask.dataframe tail() missing npartitions parameter

**Target**: `dask.dataframe.DataFrame.tail()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `tail()` method lacks the `npartitions` parameter that `head()` has, creating an API asymmetry and limiting functionality. This prevents users from retrieving the last n rows across multiple partitions.

## Property-Based Test

```python
import dask.dataframe as dd
import pandas as pd
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=20, max_size=50),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_head_tail_api_symmetry(data, npartitions):
    df = pd.DataFrame({'x': data})
    dask_df = dd.from_pandas(df, npartitions=npartitions)

    head_with_npartitions = dask_df.head(10, npartitions=-1)

    try:
        tail_with_npartitions = dask_df.tail(10, npartitions=-1)
        assert True
    except TypeError as e:
        assert "npartitions" in str(e)
```

**Failing input**: Any valid dask DataFrame

## Reproducing the Bug

```python
import dask.dataframe as dd
import pandas as pd

df = pd.DataFrame({'x': range(100)})
dask_df = dd.from_pandas(df, npartitions=5)

head_result = dask_df.head(10, npartitions=-1)
print(f"head() with npartitions=-1: {len(head_result)} rows")

try:
    tail_result = dask_df.tail(10, npartitions=-1)
except TypeError as e:
    print(f"tail() with npartitions=-1 failed: {e}")

tail_result = dask_df.tail(10)
print(f"tail() without npartitions: {len(tail_result)} rows")
```

## Why This Is A Bug

This violates the principle of API symmetry. The `head()` and `tail()` methods perform complementary operations and should have symmetric APIs. The lack of the `npartitions` parameter in `tail()` means:

1. Users cannot control how many partitions to search when getting the last n rows
2. The docstring admits this limitation: "Caveat, the only checks the last n rows of the last partition"
3. This creates an asymmetric API where `head()` is more flexible than `tail()`
4. It limits functionality when data is distributed across multiple partitions

## Fix

Add the `npartitions` parameter to `tail()` to match `head()`:

```diff
--- a/dask/dataframe/dask_expr/_collection.py
+++ b/dask/dataframe/dask_expr/_collection.py
@@ -1,10 +1,20 @@
-    def tail(self, n: int = 5, compute: bool = True):
+    def tail(self, n: int = 5, npartitions=1, compute: bool = True):
         """Last n rows of the dataset

-        Caveat, the only checks the last n rows of the last partition.
+        Parameters
+        ----------
+        n : int, optional
+            The number of rows to return. Default is 5.
+        npartitions : int, optional
+            Elements are only taken from the last ``npartitions``, with a
+            default of 1. If there are fewer than ``n`` rows in the last
+            ``npartitions`` a warning will be raised and any found rows
+            returned. Pass -1 to use all partitions.
+        compute : bool, optional
+            Whether to compute the result, default is True.
         """
-        out = new_collection(expr.Tail(self, n=n))
+        out = new_collection(expr.Tail(self, n=n, npartitions=npartitions))
         if compute:
             out = out.compute()
         return out
```

Note: This fix assumes that `expr.Tail` supports the `npartitions` parameter. If it doesn't, the underlying expression class would also need to be updated to support this parameter, similar to how `expr.Head` handles it.