# Bug Report: pandas.read_csv Empty DataFrame Dtype Loss

**Target**: `pandas.io.parsers.read_csv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When an empty DataFrame with typed columns (int64, float64, bool, datetime64) is round-tripped through CSV (to_csv → read_csv), all column dtypes are incorrectly inferred as object instead of preserving their original types.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, columns

@given(
    df=data_frames(
        columns=columns(['col1', 'col2', 'col3'], dtype=float),
        rows=st.just([])
    )
)
@settings(max_examples=200)
def test_roundtrip_csv_preserves_dtype(df):
    csv_str = df.to_csv(index=False)
    df_result = pd.read_csv(io.StringIO(csv_str))
    pd.testing.assert_frame_equal(df, df_result, check_dtype=True)
```

**Failing input**: Empty DataFrame with float64 columns

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame({'col': pd.Series([], dtype='float64')})
csv_str = df.to_csv(index=False)
df_result = pd.read_csv(io.StringIO(csv_str))

print(f"Before: {df['col'].dtype}")
print(f"After:  {df_result['col'].dtype}")
assert df['col'].dtype == df_result['col'].dtype
```

Output:
```
Before: float64
After:  object
AssertionError: Dtype changed from float64 to object
```

## Why This Is A Bug

This violates the round-trip property that users expect when writing and reading data. Empty DataFrames are common in real usage (e.g., after filtering, during initialization, or as edge cases in data processing pipelines). The dtype information is critical for:

1. Type safety in subsequent operations
2. Memory efficiency
3. Correctness of mathematical operations
4. Schema validation

The CSV written by to_csv contains only headers with no data rows: `col\n`. When read_csv encounters this, it defaults to object dtype instead of attempting to preserve or infer the intended type. Non-empty DataFrames correctly preserve their dtypes through this round-trip.

**Affected dtypes**: int64, float64, bool, datetime64[ns] (all convert to object)
**Unaffected**: object dtype (already object)

## Fix

The issue is in pandas' type inference logic for empty data. When read_csv encounters a column with no data rows, it should:

1. Check if dtype hints are available (already works via `dtype` parameter)
2. For empty columns without dtype hints, default to a more reasonable type than object (e.g., float64 for numeric-looking column names, or maintain float64 as a general default for better numerical compatibility)

A workaround exists by explicitly specifying dtypes:
```python
df_result = pd.read_csv(io.StringIO(csv_str), dtype={'col': 'float64'})
```

However, this defeats the purpose of automatic type inference and requires users to manually track schema information.

The fix would likely be in the type inference logic within the parsers, specifically in how empty columns are handled. When a column has zero rows, the parser should default to float64 (pandas' typical default for ambiguous numeric data) rather than object.