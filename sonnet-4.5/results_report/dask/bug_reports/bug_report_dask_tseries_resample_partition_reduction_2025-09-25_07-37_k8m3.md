# Bug Report: dask.dataframe.tseries.resample - AssertionError on Partition Reduction

**Target**: `dask.dataframe.tseries.resample.Resampler`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Resampler` class fails with an `AssertionError` when resampling a time series that has more partitions than the number of output time bins. This violates the documented behavior that Dask's `Resampler` should match pandas' `Resampler` (indicated by `@derived_from(pd_Resampler)` decorators).

## Property-Based Test

```python
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
```

**Failing input**: `n_points=10, rule='1D'`

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

This raises:
```
AssertionError
  File "dask/dataframe/dask_expr/_repartition.py", line 192, in _partitions_boundaries
    assert npartitions_input > npartitions
```

The bug occurs because:
1. Input has 4 partitions spanning 10 hours
2. Resampling to '1D' produces only 1 output time bin
3. The repartitioning logic asserts `npartitions_input > npartitions` which fails (4 > 1 is True, but the comparison fails in the reduction case)

## Why This Is A Bug

The `Resampler` class is decorated with `@derived_from(pd_Resampler)`, explicitly claiming to match pandas behavior. Pandas handles this case correctly, but Dask fails with an assertion error. This is a violation of the documented API contract.

Additionally, this affects real users who partition their data for parallel processing but then want to resample to coarser time granularity, which is a common use case in time series analysis.

## Fix

The issue is in `resample.py` at line 156-157 where `Repartition` is called with `force=True`. The downstream repartitioning code doesn't handle the case where the target number of partitions is less than the source.

A potential fix would be to check if repartitioning would reduce partitions and handle that case specially:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -154,8 +154,15 @@ class ResampleReduction(Expr):

     def _lower(self):
+        output_divisions = self._resample_divisions[1]
+        target_npartitions = len(output_divisions) - 1
+
+        if target_npartitions >= self.frame.npartitions:
+            partitioned = Repartition(
+                self.frame, new_divisions=self._resample_divisions[0], force=True
+            )
+        else:
+            partitioned = self.frame
-        partitioned = Repartition(
-            self.frame, new_divisions=self._resample_divisions[0], force=True
-        )
-        output_divisions = self._resample_divisions[1]
         return ResampleAggregation(
             partitioned,
             BlockwiseDep(output_divisions[:-1]),
```

Note: This is a suggested fix that avoids the assertion error, but a more thorough fix might be needed to handle the case properly in the repartitioning logic or to adjust divisions calculation.