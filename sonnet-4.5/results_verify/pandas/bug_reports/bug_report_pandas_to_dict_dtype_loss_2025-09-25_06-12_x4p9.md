# Bug Report: pandas DataFrame.to_dict dtype loss on empty DataFrames

**Target**: `pandas.DataFrame.to_dict` / `pandas.DataFrame.from_dict`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Empty DataFrames with integer columns lose their dtype when round-tripped through `to_dict(orient='list')` and `from_dict()`, converting int64 columns to float64.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames
import pandas as pd

@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float),
]))
@settings(max_examples=100)
def test_dataframe_to_dict_from_dict_roundtrip_list_orient(df):
    d = df.to_dict(orient='list')
    result = pd.DataFrame.from_dict(d)
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: Empty DataFrame with columns [a, b] where a is int64 and b is float64

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': [], 'b': []})
df['a'] = df['a'].astype(int)
df['b'] = df['b'].astype(float)

print("Original dtypes:", df.dtypes.to_dict())

d = df.to_dict(orient='list')
result = pd.DataFrame.from_dict(d)

print("Result dtypes:", result.dtypes.to_dict())
print("Bug: int64 became", result['a'].dtype)
```

Output:
```
Original dtypes: {'a': dtype('int64'), 'b': dtype('float64')}
Result dtypes: {'a': dtype('float64'), 'b': dtype('float64')}
Bug: int64 became float64
```

## Why This Is A Bug

The `to_dict`/`from_dict` API is designed for serialization and round-tripping DataFrames. Users expect that `from_dict(df.to_dict())` preserves the DataFrame's structure and types. Empty DataFrames are common in data pipelines (e.g., filtering operations that return no results), and losing dtype information silently can cause downstream type errors or incorrect computations.

The issue arises because `from_dict` with an empty list cannot infer the original dtype and defaults to float64. Non-empty DataFrames preserve dtypes correctly.

## Fix

The fix requires `to_dict` to preserve dtype metadata for empty DataFrames, or `from_dict` to accept dtype hints. One approach:

When orient='list' and DataFrame is empty, include dtype information in the serialization format, or use orient='split' or 'tight' which preserve metadata better.