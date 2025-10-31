# Bug Report: pandas.io.json Empty DataFrame Roundtrip Loses Rows

**Target**: `pandas.io.json.to_json` / `pandas.io.json.read_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame with rows but no columns is serialized to JSON using `orient='records'` (and other orientations), the roundtrip through JSON loses all rows, resulting in an empty DataFrame instead of preserving the original row count.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import io

@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
def test_json_roundtrip_preserves_length(data):
    df = pd.DataFrame([data])
    json_str = df.to_json(orient='records')
    df_back = pd.read_json(io.StringIO(json_str), orient='records')
    assert len(df_back) == len(df)
```

**Failing input**: `data={}`

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame([{}])
print(f'Original DataFrame: {df.shape}')

json_str = df.to_json(orient='records')
print(f'JSON: {json_str}')

df_back = pd.read_json(io.StringIO(json_str), orient='records')
print(f'Restored DataFrame: {df_back.shape}')

print(f'Original length: {len(df)}')
print(f'Restored length: {len(df_back)}')
```

## Why This Is A Bug

The roundtrip property `read_json(df.to_json()) should preserve df` is violated. A DataFrame with 1 row and 0 columns becomes a DataFrame with 0 rows and 0 columns after the roundtrip. This silently loses data (the row exists, even if it has no columns).

This affects orient='records', 'index', 'columns', and 'values'. Only orient='split' and orient='table' correctly preserve the row count.

Expected behavior: The row count should be preserved even when there are no columns.

## Fix

The issue is that when a DataFrame has no columns, `to_json(orient='records')` produces `'[]'` (empty array), which `read_json` interprets as zero rows rather than one row with zero columns.

The fix should ensure that empty records still preserve row information. Consider:
- For orient='records': Include row metadata when columns are empty
- Document this limitation if it's by design
- Use orient='split' or 'table' for DataFrames with no columns

Alternatively, the documentation should warn users that DataFrames with no columns cannot reliably roundtrip through JSON with certain orientations.