# Bug Report: pandas DataFrame transpose dtype loss on empty DataFrames

**Target**: `pandas.DataFrame.T` (transpose)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Empty DataFrames with integer columns lose their dtype when transposed twice (df.T.T), converting int64 columns to float64.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames
import pandas as pd

@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=int),
]))
@settings(max_examples=200)
def test_transpose_transpose_identity(df):
    result = df.T.T
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: Empty DataFrame with int64 columns

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': [], 'b': []})
df['a'] = df['a'].astype(int)
df['b'] = df['b'].astype(int)

print("Original dtypes:", df.dtypes.to_dict())
print("Shape:", df.shape)

result = df.T.T

print("After df.T.T dtypes:", result.dtypes.to_dict())
print("Bug: int64 columns became", result['a'].dtype)
```

Output:
```
Original dtypes: {'a': dtype('int64'), 'b': dtype('int64')}
Shape: (0, 2)
After df.T.T dtypes: {'a': dtype('float64'), 'b': dtype('float64')}
Bug: int64 columns became float64
```

## Why This Is A Bug

The transpose operation should be its own inverse - `df.T.T` should equal `df` for all DataFrames. This is a fundamental mathematical property of matrix transposition. The bug violates this property for empty DataFrames, causing silent dtype changes that can propagate through data processing pipelines.

After the first transpose of an empty (0, 2) DataFrame, we get a (2, 0) DataFrame with no columns (hence no dtypes). When transposing back, pandas cannot infer the original dtypes and defaults to float64.

## Fix

The transpose operation should preserve dtype metadata even when the resulting DataFrame has no data. One approach is to store dtype information in the transposed DataFrame's metadata or block manager, allowing the second transpose to reconstruct the original dtypes.