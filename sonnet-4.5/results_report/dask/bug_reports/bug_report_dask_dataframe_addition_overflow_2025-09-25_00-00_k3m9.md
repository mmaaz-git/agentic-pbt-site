# Bug Report: dask.dataframe Addition with Mismatched Indices and Integer Overflow

**Target**: `dask.dataframe` (DataFrame addition operator)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When adding two Dask DataFrames with mismatched lengths and values near the int64 overflow boundary, Dask produces incorrect results that differ from pandas. Specifically, the sign of the result can be flipped in some partitions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, column, range_indexes
import dask.dataframe as dd
import pandas as pd


@settings(max_examples=100)
@given(
    df1=data_frames(
        columns=[
            column('x', dtype=int),
            column('y', dtype=int),
        ],
        index=range_indexes(min_size=1, max_size=30),
    ),
    df2=data_frames(
        columns=[
            column('x', dtype=int),
            column('y', dtype=int),
        ],
        index=range_indexes(min_size=1, max_size=30),
    ),
)
def test_add_dataframe_matches_pandas(df1, df2):
    ddf1 = dd.from_pandas(df1, npartitions=2)
    ddf2 = dd.from_pandas(df2, npartitions=2)

    dask_result = (ddf1 + ddf2).compute()
    pandas_result = df1 + df2

    pd.testing.assert_frame_equal(dask_result, pandas_result)
```

**Failing input**:
```python
df1 = pd.DataFrame({'x': [-155, -155], 'y': [0, 0]})
df2 = pd.DataFrame({'x': [-9223372036854775654, -9223372036854775654, -9223372036854775654], 'y': [0, 0, 0]})
```

## Reproducing the Bug

```python
import dask.dataframe as dd
import pandas as pd

df1 = pd.DataFrame({'x': [-155, -155], 'y': [0, 0]})
df2 = pd.DataFrame({'x': [-9223372036854775654, -9223372036854775654, -9223372036854775654], 'y': [0, 0, 0]})

pandas_result = df1 + df2
print("Pandas result:")
print(pandas_result)

ddf1 = dd.from_pandas(df1, npartitions=2)
ddf2 = dd.from_pandas(df2, npartitions=2)
dask_result = (ddf1 + ddf2).compute()
print("\nDask result:")
print(dask_result)

print(f"\nExpected x[0]: {pandas_result['x'].iloc[0]}")
print(f"Actual x[0]:   {dask_result['x'].iloc[0]}")
```

**Output:**
```
Pandas result:
              x    y
0 -9.223372e+18  0.0
1 -9.223372e+18  0.0
2           NaN  NaN

Dask result:
              x    y
0  9.223372e+18  0.0
1 -9.223372e+18  0.0
2           NaN  NaN

Expected x[0]: -9.223372036854776e+18
Actual x[0]:   9.223372036854776e+18
```

## Why This Is A Bug

Dask's documentation and API contract state that it should produce the same results as pandas for DataFrame operations. When adding DataFrames with mismatched indices (a standard pandas operation that uses index alignment), Dask produces a different result in the first partition. The sign is incorrectly flipped from negative to positive, violating the fundamental expectation that `dask_df.compute()` should match the equivalent pandas operation.

This bug occurs specifically when:
1. DataFrames have different lengths (requiring index alignment)
2. Values are near integer overflow boundaries
3. DataFrames are split into multiple partitions (npartitions > 1)

## Fix

The issue appears to be in how Dask handles partition-wise addition when DataFrames have mismatched divisions. The overflow handling differs between how Dask processes individual partitions versus how pandas handles the entire operation. A proper fix would require ensuring that:

1. Integer overflow handling in partition-wise operations matches pandas behavior
2. Index alignment across partitions is correctly handled before arithmetic operations
3. Data type conversions (int64 to float64 on overflow) are consistent with pandas

The bug is likely in the `dask/dataframe/dask_expr/_expr.py` module's binary operation handling or in the partition alignment logic.