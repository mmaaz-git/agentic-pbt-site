# Bug Report: dask.dataframe NLargest/NSmallest Column Selection TypeError

**Target**: `dask.dataframe.dask_expr._reductions.NLargest` and `dask.dataframe.dask_expr._reductions.NSmallest`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Selecting a column from DataFrame.nlargest() or DataFrame.nsmallest() results causes a TypeError when computing, as dask incorrectly passes the DataFrame's 'columns' parameter to the underlying pandas Series methods which don't accept that argument.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import dask.dataframe as dd


@settings(max_examples=50)
@given(
    data=st.lists(st.integers(min_value=-100, max_value=100), min_size=5, max_size=20),
    n=st.integers(min_value=1, max_value=5)
)
def test_nlargest_nsmallest_disjoint(data, n):
    """nlargest and nsmallest should be disjoint"""
    assume(len(set(data)) >= 2 * n)

    df = pd.DataFrame({'x': data})
    ddf = dd.from_pandas(df, npartitions=2)

    largest = set(ddf.nlargest(n, 'x')['x'].compute())
    smallest = set(ddf.nsmallest(n, 'x')['x'].compute())

    assert len(largest & smallest) == 0

if __name__ == "__main__":
    test_nlargest_nsmallest_disjoint()
```

<details>

<summary>
**Failing input**: `data=[0, 0, 0, 0, 1], n=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 836, in __getattr__
    return object.__getattribute__(self, key)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
AttributeError: 'NLargest' object has no attribute 'sort'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_expr.py", line 156, in __getattr__
    return super().__getattr__(key)
           ~~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 854, in __getattr__
    raise AttributeError(
    ...<3 lines>...
    )
AttributeError: 'NLargest' object has no attribute 'sort'

This often means that you are attempting to use an unsupported API function..

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 24, in <module>
    test_nlargest_nsmallest_disjoint()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 7, in test_nlargest_nsmallest_disjoint
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 18, in test_nlargest_nsmallest_disjoint
    largest = set(ddf.nlargest(n, 'x')['x'].compute())
                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    (result,) = compute(self, traverse=False, **kwargs)
                ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 678, in compute
    expr = expr.optimize()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 440, in optimize
    return optimize_until(self, stage)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 944, in optimize_until
    expr = expr.lower_completely()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 523, in lower_completely
    new = expr.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 489, in lower_once
    new = operand.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 478, in lower_once
    out = expr._lower()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 526, in _lower
    if not self.should_shuffle:
           ^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 476, in should_shuffle
    sort = getattr(self, "sort", False)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_expr.py", line 159, in __getattr__
    if is_dataframe_like(self._meta) and key in self._meta.columns:
                         ^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/functools.py", line 1042, in __get__
    val = self.func(instance)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 450, in _meta
    meta = self._meta_chunk
           ^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/functools.py", line 1042, in __get__
    val = self.func(instance)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 446, in _meta_chunk
    return self.chunk(meta, **self.chunk_kwargs)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 1339, in chunk
    return cls.reduction_chunk(df, **kwargs)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1235, in __call__
    return getattr(__obj, self.method)(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
TypeError: Series.nlargest() got an unexpected keyword argument 'columns'
Falsifying example: test_nlargest_nsmallest_disjoint(
    data=[0, 0, 0, 0, 1],  # or any other generated value
    n=1,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({'x': [0, 0, 0, 0, 1]})
ddf = dd.from_pandas(df, npartitions=2)

result = ddf.nlargest(1, 'x')['x']
result.compute()
```

<details>

<summary>
TypeError: Series.nlargest() got an unexpected keyword argument 'columns'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 836, in __getattr__
    return object.__getattribute__(self, key)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
AttributeError: 'NLargest' object has no attribute 'sort'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_expr.py", line 156, in __getattr__
    return super().__getattr__(key)
           ~~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 854, in __getattr__
    raise AttributeError(
    ...<3 lines>...
    )
AttributeError: 'NLargest' object has no attribute 'sort'

This often means that you are attempting to use an unsupported API function..

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/repo.py", line 8, in <module>
    result.compute()
    ~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    (result,) = compute(self, traverse=False, **kwargs)
                ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 678, in compute
    expr = expr.optimize()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 440, in optimize
    return optimize_until(self, stage)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 944, in optimize_until
    expr = expr.lower_completely()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 523, in lower_completely
    new = expr.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 489, in lower_once
    new = operand.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 478, in lower_once
    out = expr._lower()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 526, in _lower
    if not self.should_shuffle:
           ^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 476, in should_shuffle
    sort = getattr(self, "sort", False)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_expr.py", line 159, in __getattr__
    if is_dataframe_like(self._meta) and key in self._meta.columns:
                         ^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/functools.py", line 1042, in __get__
    val = self.func(instance)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 450, in _meta
    meta = self._meta_chunk
           ^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/functools.py", line 1042, in __get__
    val = self.func(instance)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 446, in _meta_chunk
    return self.chunk(meta, **self.chunk_kwargs)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_reductions.py", line 1339, in chunk
    return cls.reduction_chunk(df, **kwargs)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1235, in __call__
    return getattr(__obj, self.method)(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
TypeError: Series.nlargest() got an unexpected keyword argument 'columns'
```
</details>

## Why This Is A Bug

This violates expected pandas API compatibility that dask aims to provide. The pattern `df.nlargest(n, columns)[column]` is standard pandas usage that works correctly:

```python
# In pandas (works):
df.nlargest(1, 'x')['x']  # Returns Series([1])

# In dask (crashes):
ddf.nlargest(1, 'x')['x'].compute()  # TypeError
```

The error occurs because:
1. When `ddf.nlargest(1, 'x')` is called, it creates an NLargest expression with `columns='x'`
2. When `['x']` is applied, it converts the result to a Series expression
3. During computation, the chunk method in NLargest passes all kwargs including `columns='x'` to the reduction
4. Since the data is now a Series, it calls `Series.nlargest(n=1, columns='x')`
5. pandas Series.nlargest() only accepts parameters `n` and `keep`, not `columns`, causing the TypeError

The same issue affects NSmallest as it shares the same implementation pattern.

## Relevant Context

The bug is located in `/dask/dataframe/dask_expr/_reductions.py`:
- Line 1339: `NLargest.chunk()` method unconditionally passes all kwargs to `reduction_chunk`
- Line 1355-1356: NLargest uses `M.nlargest` which maps to either DataFrame.nlargest or Series.nlargest
- Line 1400-1402: NSmallest inherits from NLargest with same issue

Key code locations:
- `dask/dataframe/dask_expr/_reductions.py:1339` - Where the error occurs
- `dask/dataframe/dask_expr/_reductions.py:1352-1373` - NLargest class definition
- `dask/dataframe/dask_expr/_reductions.py:1400-1402` - NSmallest class definition

## Proposed Fix

```diff
--- a/dask/dataframe/dask_expr/_reductions.py
+++ b/dask/dataframe/dask_expr/_reductions.py
@@ -1336,7 +1336,11 @@ class ReductionConstantDim(Reduction):

     @classmethod
     def chunk(cls, df, **kwargs):
-        return cls.reduction_chunk(df, **kwargs)
+        # Remove 'columns' parameter if operating on a Series
+        if hasattr(df, 'name') and not hasattr(df, 'columns') and 'columns' in kwargs:
+            kwargs = {k: v for k, v in kwargs.items() if k != 'columns'}
+
+        return cls.reduction_chunk(df, **kwargs)

     @classmethod
     def combine(cls, inputs: list, **kwargs):  # type: ignore
```