# Bug Report: pandas.api.interchange Converts Int64 with NA to float64

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `from_dataframe` function converts nullable Int64 columns containing NA values to float64, losing both type information and converting integers to floating-point numbers.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings


@given(st.lists(st.integers(min_value=-(2**63), max_value=2**63-1), min_size=1, max_size=20))
@settings(max_examples=50)
def test_nullable_int_dtype(values):
    df = pd.DataFrame({'col': pd.array(values, dtype='Int64')})

    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: Any DataFrame with nullable Int64 dtype containing NA values

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({'col': pd.array([1, 2, None], dtype='Int64')})
print("Original:")
print(df)
print("Dtype:", df['col'].dtype)

result = from_dataframe(df.__dataframe__())
print("\nAfter round-trip:")
print(result)
print("Dtype:", result['col'].dtype)
```

Output:
```
Original:
    col
0     1
1     2
2  <NA>
Dtype: Int64

After round-trip:
   col
0  1.0
1  2.0
2  NaN
Dtype: float64
```

## Why This Is A Bug

This violates the round-trip property and causes two issues:

1. **Type loss**: Int64 is converted to float64, losing the integer type information
2. **Precision issues**: Integers are converted to floating-point, which can cause precision issues for large integers

Note that Int64 columns **without** NA values are correctly preserved as int64:

```python
df_no_na = pd.DataFrame({'col': pd.array([1, 2, 3], dtype='Int64')})
result_no_na = from_dataframe(df_no_na.__dataframe__())
print(result_no_na['col'].dtype)  # int64 - correct!
```

The bug only occurs when NA values are present, suggesting the interchange protocol is falling back to float64 to represent missing integer values instead of using nullable Int64.

## Fix

The fix should preserve the Int64 dtype when converting back from the interchange protocol. The interchange protocol should detect when a column was originally Int64 and restore it to Int64 rather than float64.