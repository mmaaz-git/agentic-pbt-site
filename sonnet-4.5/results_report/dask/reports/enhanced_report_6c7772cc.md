# Bug Report: dask.dataframe.tseries RepartitionToFewer Assertion Failure on Resample Operation

**Target**: `dask.dataframe.dask_expr._repartition.RepartitionToFewer._partitions_boundaries`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When resampling a Dask DataFrame/Series with datetime index from a fine-grained frequency (e.g., hourly) to a coarser frequency (e.g., daily), an AssertionError occurs in RepartitionToFewer when the operation results in equal input and output partition counts (specifically 1 partition each).

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

# Run the test
if __name__ == "__main__":
    try:
        test_resample_count_sum_equals_length()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    else:
        print("All tests passed!")
```

<details>

<summary>
**Failing input**: `series=Series with 10 hourly values starting 2000-01-01, resample_freq='D'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 38, in <module>
    test_resample_count_sum_equals_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 19, in test_resample_count_sum_equals_length
    resample_freq=st.sampled_from(['2h', 'D', '2D']))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 27, in test_resample_count_sum_equals_length
    resampled_count = ds.resample(resample_freq).count().compute()
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
Falsifying example: test_resample_count_sum_equals_length(
    series=2000-01-01 00:00:00    0
    2000-01-01 01:00:00    0
    2000-01-01 02:00:00    0
    2000-01-01 03:00:00    0
    2000-01-01 04:00:00    0
    2000-01-01 05:00:00    0
    2000-01-01 06:00:00    0
    2000-01-01 07:00:00    0
    2000-01-01 08:00:00    0
    2000-01-01 09:00:00    0
    Freq: h, dtype: int64,
    resample_freq='D',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:2272
        /home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:2277
Test failed with error:
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

# Create a minimal test case that reproduces the bug
series = pd.Series(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    index=pd.date_range('2000-01-01 00:00:00', periods=10, freq='h')
)

npartitions = 5
ds = dd.from_pandas(series, npartitions=npartitions)

# This will trigger the AssertionError
result = ds.resample('D').count().compute()
print(f"Result: {result}")
```

<details>

<summary>
AssertionError in _partitions_boundaries at line 192
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/repo.py", line 14, in <module>
    result = ds.resample('D').count().compute()
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

This violates expected behavior because:

1. **Valid operation fails**: The resample operation is documented in Dask's public API and should work for all valid time series data and resample frequencies. The Pandas equivalent works without issue.

2. **Incorrect assertion**: The `RepartitionToFewer` class at line 192 has a strict assertion requiring `npartitions_input > npartitions`, but during resample operations, the code path can legitimately have equal partition counts (both 1) when:
   - Initial data spans less than one day (10 hours)
   - Data is partitioned into multiple partitions (5 partitions)
   - Resampling to daily frequency results in single logical partition
   - The resample pipeline attempts to use RepartitionToFewer(1→1) which violates the assertion

3. **Logic contradiction**: The parent `Repartition` class already handles the equal partition case at lines 91-93 by returning the original frame unchanged, but `RepartitionToFewer` is instantiated before this check can apply, causing the crash.

4. **No user error**: The input data is completely valid, the resample frequency is standard, and nothing in the documentation suggests this combination shouldn't work.

## Relevant Context

The bug occurs in the resample pipeline when:
- Original Series has datetime index with fine-grained frequency (hourly)
- Data is split across multiple Dask partitions (common for parallelism)
- Resampling to coarser frequency (daily) mathematically results in fewer time buckets than original partitions
- The internal repartitioning logic incorrectly attempts to use `RepartitionToFewer` when going from 1 partition to 1 partition

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_repartition.py:192`

The resample operation is a fundamental time series operation in data analysis. Users commonly resample high-frequency data (seconds, minutes, hours) to lower frequencies (hours, days, weeks) for aggregation and analysis.

## Proposed Fix

```diff
--- a/dask/dataframe/dask_expr/_repartition.py
+++ b/dask/dataframe/dask_expr/_repartition.py
@@ -189,7 +189,9 @@ class RepartitionToFewer(Repartition):
     def _partitions_boundaries(self):
         npartitions = self.new_partitions
         npartitions_input = self.frame.npartitions
-        assert npartitions_input > npartitions
+        if npartitions_input <= npartitions:
+            # Cannot reduce partitions; return identity mapping
+            return list(range(npartitions_input + 1))
         return self._compute_partition_boundaries(npartitions, npartitions_input)
```