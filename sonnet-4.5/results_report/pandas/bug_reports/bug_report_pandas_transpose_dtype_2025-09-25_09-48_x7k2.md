# Bug Report: pandas DataFrame.T.T Loses Integer Dtype

**Target**: `pandas.DataFrame.T` (transpose operation)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Double transposing a DataFrame with integer columns converts int64 dtypes to float64, violating the mathematical property that transpose is an involution.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra import pandas as pdst


@given(pdst.data_frames(columns=[
    pdst.column('A', dtype=int),
    pdst.column('B', dtype=int)
]))
@settings(max_examples=500)
def test_transpose_involution_preserves_dtype(df):
    result = df.T.T
    for col in df.columns:
        assert df[col].dtype == result[col].dtype, f"Column {col} dtype changed from {df[col].dtype} to {result[col].dtype}"
```

**Failing input**: Empty DataFrame with int64 columns

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'A': pd.Series([], dtype='int64'), 'B': pd.Series([], dtype='int64')})

print(f"Original dtypes:\n{df.dtypes}")

result = df.T.T

print(f"\nAfter df.T.T dtypes:\n{result.dtypes}")
```

Output:
```
Original dtypes:
A    int64
B    int64
dtype: object

After df.T.T dtypes:
A    float64
B    float64
dtype: object
```

## Why This Is A Bug

The transpose operation should be an involution, meaning `T(T(df)) = df` for all DataFrames. This includes preserving data types. The current implementation loses integer dtype information when transposing, converting int64 to float64. This violates user expectations and can lead to unexpected behavior in data pipelines where dtype consistency is important.

## Fix

The issue likely occurs because the transpose operation uses a common dtype for the intermediate representation. The fix would need to preserve per-column dtype information across both transpose operations.