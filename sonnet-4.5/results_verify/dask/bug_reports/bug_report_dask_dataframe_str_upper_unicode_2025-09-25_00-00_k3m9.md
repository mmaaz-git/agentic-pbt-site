# Bug Report: dask.dataframe Series.str.upper() Incorrect Unicode Handling

**Target**: `dask.dataframe.Series.str.upper()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`dask.dataframe.Series.str.upper()` does not match pandas behavior for German ß character. Dask converts `'ß'` to capital ß (`'ẞ'`, U+1E9E), while pandas follows the traditional German convention of converting to `'SS'`.

## Property-Based Test

```python
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings


@given(
    st.lists(
        st.text(min_size=1, max_size=10),
        min_size=5,
        max_size=30
    )
)
@settings(max_examples=50)
def test_str_upper_matches_pandas(strings):
    pdf = pd.DataFrame({'text': strings})
    ddf = dd.from_pandas(pdf, npartitions=2)

    pandas_result = pdf['text'].str.upper()
    dask_result = ddf['text'].str.upper().compute()

    for i in range(len(strings)):
        assert pandas_result.iloc[i] == dask_result.iloc[i], \
            f"str.upper() mismatch for '{strings[i]}': pandas='{pandas_result.iloc[i]}', dask='{dask_result.iloc[i]}'"
```

**Failing input**: Any string containing the German ß character, e.g., `['ß']`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

pdf = pd.DataFrame({'text': ['ß']})
ddf = dd.from_pandas(pdf, npartitions=1)

pandas_result = pdf['text'].str.upper().iloc[0]
dask_result = ddf['text'].str.upper().compute().iloc[0]

print(f"Input: 'ß'")
print(f"Pandas: '{pandas_result}'")
print(f"Dask:   '{dask_result}'")

assert pandas_result == 'SS'
assert dask_result == 'ẞ'
```

## Why This Is A Bug

Dask DataFrame is designed to be a distributed alternative to pandas with the same API. The docstring for dask doesn't specify different Unicode handling, so users expect `Series.str.upper()` to behave identically to pandas.

Pandas follows the traditional German typographic convention where ß (lowercase) converts to SS (uppercase). This is the standard behavior in German text processing. Dask instead uses the Unicode capital ß (ẞ, U+1E9E), which exists in Unicode but is less commonly used.

This violates the API contract with pandas and will cause incorrect results when processing German text. For example:
- Dask: `'Straße'.upper()` → `'STRAẞE'`
- Pandas: `'Straße'.upper()` → `'STRASSE'`

## Fix

The issue appears to be in how dask's string dtype handles Unicode case conversion. The fix would need to ensure dask uses the same Unicode case mapping as pandas, which follows Python's standard `str.upper()` method that converts ß → SS.

A potential fix would be to ensure dask's PyArrow-backed string operations use locale-aware or Python-compatible case conversion rather than strict Unicode uppercase mapping.