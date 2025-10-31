# Bug Report: dask.dataframe.dask_expr.from_pandas Large Integer Type Conversion

**Target**: `dask.dataframe.dask_expr.from_pandas`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `from_pandas()` processes a DataFrame containing integers that exceed int64 range (stored in object dtype columns), it incorrectly converts them to strings, breaking arithmetic operations and changing data types.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
import dask.dataframe.dask_expr as dd


@given(st.integers(min_value=np.iinfo(np.int64).min - 1000, max_value=np.iinfo(np.int64).max + 1000))
@settings(max_examples=500)
def test_from_pandas_preserves_large_integers(value):
    df = pd.DataFrame({'x': [value]})
    ddf = dd.from_pandas(df, npartitions=1)
    result = ddf.compute()

    original_value = df['x'].iloc[0]
    result_value = result['x'].iloc[0]

    assert type(original_value) == type(result_value), \
        f"Type changed: {type(original_value)} -> {type(result_value)}"

    if isinstance(original_value, int):
        assert isinstance(result_value, int), \
            f"Integer converted to {type(result_value)}: {value}"
```

**Failing input**: `-9_223_372_036_854_775_809` (int64.min - 1)

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe.dask_expr as dd

large_int = -9_223_372_036_854_775_809

df = pd.DataFrame({'x': [large_int]})
print(f"Original dtype: {df['x'].dtype}")
print(f"Original value type: {type(df['x'].iloc[0])}")
print(f"Arithmetic works: df['x'] + 1 = {(df['x'] + 1).iloc[0]}")

ddf = dd.from_pandas(df, npartitions=1)
result = ddf.compute()

print(f"Result dtype: {result['x'].dtype}")
print(f"Result value type: {type(result['x'].iloc[0])}")
try:
    print(f"Arithmetic: result['x'] + 1 = {(result['x'] + 1).iloc[0]}")
except Exception as e:
    print(f"Arithmetic fails: {e}")
```

Output:
```
Original dtype: object
Original value type: <class 'int'>
Arithmetic works: df['x'] + 1 = -9223372036854775808
Result dtype: string
Result value type: <class 'str'>
Arithmetic fails: operation 'add' not supported for dtype 'string' with object of type <class 'int'>
```

## Why This Is A Bug

1. **Data corruption**: Integers are silently converted to strings, changing their type from `int` to `str`
2. **Breaks arithmetic**: Any mathematical operation that worked on the original DataFrame now raises an exception
3. **Type mismatch**: The result dtype is `string` instead of `object`, violating the principle that `from_pandas().compute()` should return data equivalent to the original
4. **Silent failure**: No warning or error is raised, making this a dangerous silent data corruption

This violates the documented contract of `from_pandas()`, which should preserve data types and enable parallel operations on the same data structure.

## Fix

The bug appears to be related to pyarrow string handling (based on the traceback showing `_pyarrow.py`). The issue is that when dask encounters object dtype columns, it's using pyarrow's string inference which incorrectly treats large Python integers as strings rather than preserving them as integers in an object dtype column.

The fix should:
1. Detect when object dtype columns contain integers (even if they exceed int64 range)
2. Preserve the object dtype with integer values instead of converting to string dtype
3. Alternatively, raise an error if the integers cannot be represented in the target format

A potential patch location would be in the `from_pandas` function or the `FromPandas` expression class, where pyarrow string handling is enabled. The code should check if object columns contain non-string types before applying string dtype conversion.