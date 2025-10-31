# Bug Report: dask.dataframe.tseries.resample - AssertionError when resampling to coarser frequency with multiple partitions

**Target**: `dask.dataframe.tseries.resample.Resampler`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Dask's `Resampler` crashes with an AssertionError when resampling time series data to a coarser frequency that results in fewer output bins than input partitions, violating the documented pandas compatibility indicated by `@derived_from(pd_Resampler)`.

## Property-Based Test

```python
import pandas as pd
import numpy as np
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings

@given(
    st.integers(min_value=10, max_value=100),
    st.sampled_from(['1H', '2H', '6H', '1D', '2D', '1W']),
)
@settings(max_examples=200)
def test_resample_matches_pandas(n_points, rule):
    """
    Metamorphic property: Dask resample results should match pandas resample.
    """
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')
    np.random.seed(42)
    data = np.random.randn(n_points)

    pandas_series = pd.Series(data, index=dates)
    dask_series = dd.from_pandas(pandas_series, npartitions=4)

    for method in ['sum', 'mean', 'min', 'max', 'count']:
        pandas_result = getattr(pandas_series.resample(rule), method)()
        dask_result = getattr(dask_series.resample(rule), method)().compute()

        pd.testing.assert_series_equal(pandas_result, dask_result, check_dtype=False, rtol=1e-10)

if __name__ == "__main__":
    test_resample_matches_pandas()
```

<details>

<summary>
**Failing input**: `n_points=10, rule='1D'`
</summary>
```
Falsifying example: test_resample_matches_pandas(
    n_points=10,
    rule='1D',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
import dask.dataframe as dd

dates = pd.date_range('2024-01-01', periods=10, freq='1h')
data = np.random.randn(10)

pandas_series = pd.Series(data, index=dates)
dask_series = dd.from_pandas(pandas_series, npartitions=4)

pandas_result = pandas_series.resample('1D').sum()
dask_result = dask_series.resample('1D').sum().compute()
```

<details>

<summary>
AssertionError in _partitions_boundaries when npartitions_input <= npartitions
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/repo.py", line 12, in <module>
    dask_result = dask_series.resample('1D').sum().compute()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    (result,) = compute(self, traverse=False, **kwargs)
                ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 681, in compute
    results = schedule(expr, keys, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 581, in __dask_graph__
    layers.append(expr._layer())
                  ~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 1205, in _layer
    return toolz.merge(op._layer() for op in self.operands)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/toolz/dicttoolz.py", line 38, in merge
    for d in dicts:
             ^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 1205, in <genexpr>
    return toolz.merge(op._layer() for op in self.operands)
                       ~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_repartition.py", line 196, in _layer
    new_partitions_boundaries = self._partitions_boundaries
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/functools.py", line 1042, in __get__
    val = self.func(instance)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_repartition.py", line 192, in _partitions_boundaries
    assert npartitions_input > npartitions
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **API Contract Violation**: All resample methods in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py` are decorated with `@derived_from(pd_Resampler)` (lines 308-373), explicitly promising pandas compatibility. Pandas handles this operation correctly while Dask crashes.

2. **Common Use Case Failure**: Resampling hourly data to daily/weekly/monthly aggregates is one of the most common time series operations. With 10 hours of data partitioned into 4 chunks, resampling to '1D' (one day) is a typical aggregation scenario that should work.

3. **Invalid Assertion**: The assertion at line 192 in `_repartition.py` (`assert npartitions_input > npartitions`) assumes repartitioning only happens when increasing partitions, but resampling to coarser frequencies naturally reduces partitions. When we have 4 input partitions but only 1 output time bin, this assertion incorrectly fails.

4. **Misleading Error**: Users get an opaque `AssertionError` with no guidance, even though this is a legitimate operation that should be supported.

## Relevant Context

The bug occurs in the data flow:
1. `ResampleReduction._lower()` (line 156-157 in `resample.py`) calls `Repartition` with `force=True`
2. `Repartition` assumes it's always expanding partitions, not reducing them
3. The assertion `npartitions_input > npartitions` fails when output has fewer partitions than input

The resample operation correctly calculates that only 1 output bin is needed for '1D' frequency when input spans 10 hours, but the repartitioning logic doesn't handle this reduction case.

Documentation link: https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.DataFrame.resample
Code location: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py:156-157`

## Proposed Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -153,10 +153,19 @@ class ResampleReduction(Expr):
             return plain_column_projection(self, parent, dependents)

     def _lower(self):
-        partitioned = Repartition(
-            self.frame, new_divisions=self._resample_divisions[0], force=True
-        )
         output_divisions = self._resample_divisions[1]
+        input_divisions = self._resample_divisions[0]
+
+        # Only repartition if we're not reducing to fewer partitions than input
+        # When output has fewer bins than input partitions, repartitioning fails
+        if len(output_divisions) - 1 < self.frame.npartitions:
+            # Skip repartitioning when reducing partitions
+            partitioned = self.frame
+        else:
+            partitioned = Repartition(
+                self.frame, new_divisions=input_divisions, force=True
+            )
+
         return ResampleAggregation(
             partitioned,
             BlockwiseDep(output_divisions[:-1]),
```