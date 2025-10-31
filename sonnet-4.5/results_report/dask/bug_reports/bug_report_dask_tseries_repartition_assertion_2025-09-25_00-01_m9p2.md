# Bug Report: dask.dataframe.tseries Resample Repartition Assertion Error

**Target**: `dask.dataframe.dask_expr._repartition.RepartitionToFewer._partitions_boundaries`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When resampling a Dask DataFrame/Series to a larger frequency (e.g., from hourly to daily), the code attempts to use `RepartitionToFewer` which asserts that the number of input partitions must be greater than the number of output partitions. However, resampling can legitimately increase the number of partitions, causing an AssertionError.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import dask.dataframe as dd

@st.composite
def series_with_datetime_index_strategy(draw):
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    periods = draw(st.integers(min_value=10, max_value=100))
    freq = draw(st.sampled_from(['h', 'D', '30min', '2h']))
    index = pd.date_range(start, periods=periods, freq=freq)
    values = draw(st.lists(st.integers(min_value=0, max_value=100),
                          min_size=len(index), max_size=len(index)))
    return pd.Series(values, index=index)

@given(series=series_with_datetime_index_strategy(),
       resample_freq=st.sampled_from(['2h', 'D', '2D']))
@settings(max_examples=50)
def test_resample_count_sum_equals_length(series, resample_freq):
    assume(len(series) >= 2)
    npartitions = min(3, len(series) // 2)
    ds = dd.from_pandas(series, npartitions=npartitions)

    try:
        resampled_count = ds.resample(resample_freq).count().compute()
        total_count = resampled_count.sum()
        assert total_count == len(series)
    except ValueError as e:
        if "Index is not contained within new index" in str(e):
            assume(False)
        raise
```

**Failing input**: A pandas Series with 10 hourly timestamps, partitioned into 5 Dask partitions, resampled to daily frequency.

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

series = pd.Series(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    index=pd.date_range('2000-01-01 00:00:00', periods=10, freq='h')
)

npartitions = 5
ds = dd.from_pandas(series, npartitions=npartitions)

result = ds.resample('D').count().compute()
```

Output:
```
AssertionError at dask/dataframe/dask_expr/_repartition.py:192
    assert npartitions_input > npartitions
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

## Why This Is A Bug

The `RepartitionToFewer` class in `_repartition.py` assumes that repartitioning always reduces the number of partitions. However, when resampling from a fine-grained frequency (hourly) to a coarser frequency (daily), the resample operation can legitimately result in more partitions than the input if the input has many partitions spanning a short time range.

In this example:
- Input: 10 hours of data in 5 partitions (each partition ~2 hours)
- After resampling to daily: Only 1 day of data, which should result in 1 partition
- The code tries to use `RepartitionToFewer` to go from 1 partition (after initial repartition) to 1 partition
- But somewhere in the pipeline, it incorrectly tries to increase partitions using `RepartitionToFewer`

The assertion at line 192 of `_repartition.py` is too strict:

```python
@functools.cached_property
def _partitions_boundaries(self):
    npartitions = self.new_partitions
    npartitions_input = self.frame.npartitions
    assert npartitions_input > npartitions  # This assertion is too strict
    return self._compute_partition_boundaries(npartitions, npartitions_input)
```

This prevents valid use cases where resampling might result in the same number or more partitions.

## Fix

The fix should either:

1. Use a different repartitioning strategy when the number of partitions is not decreasing, or
2. Remove the strict assertion and handle the case where `npartitions_input <= npartitions` gracefully

A high-level fix would be to modify the resample logic to detect when repartitioning would not reduce partitions and skip the `RepartitionToFewer` step, or use a different repartitioning approach:

```diff
--- a/dask/dataframe/dask_expr/_repartition.py
+++ b/dask/dataframe/dask_expr/_repartition.py
@@ -189,7 +189,10 @@ class RepartitionToFewer(Repartition):
     def _partitions_boundaries(self):
         npartitions = self.new_partitions
         npartitions_input = self.frame.npartitions
-        assert npartitions_input > npartitions
+        if npartitions_input <= npartitions:
+            # Cannot reduce partitions if input has same or fewer partitions
+            # Return identity boundaries
+            return list(range(npartitions_input))
         return self._compute_partition_boundaries(npartitions, npartitions_input)
```

However, a better fix would be in the resample code to avoid using `RepartitionToFewer` when it's not appropriate.