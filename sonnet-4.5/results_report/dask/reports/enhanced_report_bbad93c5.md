# Bug Report: dask.dataframe.tseries.resample Division Length Mismatch Causes Runtime Crash

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function in dask's time series resampling module returns division tuples with mismatched lengths when `outdivs[-1] == divisions[-1]`, causing an AssertionError that prevents resample operations from completing.

## Property-Based Test

```python
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def date_range_divisions(draw):
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))

    periods = draw(st.integers(min_value=2, max_value=100))
    freq_choice = draw(st.sampled_from(['h', 'D', '2h', '6h', '12h', '2D', '3D']))

    start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    divisions = pd.date_range(start=start, periods=periods, freq=freq_choice)

    return list(divisions)


@st.composite
def resample_params(draw):
    divisions = draw(date_range_divisions())

    rule = draw(st.sampled_from(['h', '2h', '6h', '12h', 'D', '2D', '3D', 'W', 'ME']))
    closed = draw(st.sampled_from(['left', 'right']))
    label = draw(st.sampled_from(['left', 'right']))

    return divisions, rule, closed, label


@settings(max_examples=200)
@given(resample_params())
def test_resample_divisions_same_length(params):
    divisions, rule, closed, label = params

    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)
    except Exception:
        assume(False)

    assert len(newdivs) == len(outdivs), f"newdivs and outdivs have different lengths: {len(newdivs)} vs {len(outdivs)}"


if __name__ == "__main__":
    # Run the test
    print("Running hypothesis test to find failing cases...")
    test_resample_divisions_same_length()
```

<details>

<summary>
**Failing input**: `([Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-02 00:00:00')], 'W', 'right', 'right')`
</summary>
```
Running hypothesis test to find failing cases...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 51, in <module>
    test_resample_divisions_same_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 36, in test_resample_divisions_same_length
    @given(resample_params())
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 45, in test_resample_divisions_same_length
    assert len(newdivs) == len(outdivs), f"newdivs and outdivs have different lengths: {len(newdivs)} vs {len(outdivs)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: newdivs and outdivs have different lengths: 2 vs 1
Falsifying example: test_resample_divisions_same_length(
    params=([Timestamp('2000-01-01 00:00:00'),
      Timestamp('2000-01-02 00:00:00')],
     'W',
     'right',
     'right'),
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

# Create the specific test case that fails
index = pd.date_range('2000-01-01 00:00:00', periods=3, freq='6h')
series = pd.Series(range(len(index)), index=index)
dask_series = dd.from_pandas(series, npartitions=2)

# Apply resample with the problematic parameters
result = dask_series.resample('12h', closed='right', label='right').count()

# Try to compute - this should raise an AssertionError
try:
    computed = result.compute()
    print("No error - result:", computed)
except AssertionError as e:
    print(f"AssertionError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
AssertionError in dask/dataframe/dask_expr/_repartition.py:192
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/repo.py", line 14, in <module>
    computed = result.compute()
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
AssertionError:
```
</details>

## Why This Is A Bug

The bug violates an implicit contract in the dask resample implementation that `newdivs` and `outdivs` must have the same length. This contract is required because:

1. **Division Alignment**: In `ResampleReduction._lower()` (lines 162-164 of resample.py), the code creates `BlockwiseDep` objects from slices of `outdivs` that assume both division tuples have matching lengths.

2. **Missing Equality Case**: The bug occurs in the adjustment logic (lines 89-101) where the code handles strict inequalities (`>` and `<`) but fails to handle the equality case (`==`). When `outdivs[-1] == divisions[-1]`, neither condition is satisfied, leaving `outdivs` unmodified while `newdivs` gets an additional element appended.

3. **Downstream Failure**: The length mismatch causes the `Repartition` operation to receive inconsistent division boundaries. The assertion `assert npartitions_input > npartitions` at line 192 of `_repartition.py` fails because the incorrect division lengths lead to invalid partition calculations.

4. **Valid Use Case**: The parameters that trigger this bug (closed='right', label='right') are legitimate and documented options for pandas/dask resample operations, commonly used in time series analysis for period-end aggregations.

## Relevant Context

The `_resample_bin_and_out_divs` function is an internal utility that computes two sets of divisions:
- `newdivs`: The new divisions for repartitioning the data
- `outdivs`: The output divisions for the resampled result

The function is called from the `ResampleReduction._resample_divisions` property and its outputs are used to construct the execution graph for the resample operation. While the function lacks explicit documentation, the code usage pattern clearly shows both returned tuples must maintain the same length for the operation to succeed.

This bug affects all resample aggregation operations (count, sum, mean, etc.) when the specific parameter combination is used, making it impossible to perform legitimate time series resampling operations in certain scenarios.

## Proposed Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -95,9 +95,11 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         else:
             setter = lambda a, val: a.__setitem__(-1, val)
         setter(newdivs, divisions[-1] + res)
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
             setter(outdivs, temp.index[-1])
+        else:  # outdivs[-1] == divisions[-1]
+            setter(outdivs, outdivs[-1])

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```