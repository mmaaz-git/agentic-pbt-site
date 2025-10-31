# Bug Report: dask.dataframe.from_pandas Silently Corrupts Mixed-Type Object Columns

**Target**: `dask.dataframe.from_pandas`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`dask.dataframe.from_pandas` silently corrupts data when converting pandas DataFrames with mixed-type object columns. Non-string values in object columns are converted to strings, causing the round-trip property `from_pandas(df).compute() == df` to fail with silent data corruption.

## Property-Based Test

```python
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings


@given(
    st.data(),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100)
def test_from_pandas_compute_roundtrip(data, n_rows, n_cols):
    if n_rows == 0:
        return

    df_dict = {}
    for i in range(n_cols):
        col_name = f'col_{i}'
        df_dict[col_name] = data.draw(
            st.lists(
                st.one_of(
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.text(max_size=10)
                ),
                min_size=n_rows,
                max_size=n_rows
            )
        )

    pdf = pd.DataFrame(df_dict)
    ddf = dd.from_pandas(pdf, npartitions=2)
    result = ddf.compute()

    pd.testing.assert_frame_equal(pdf, result)
```

**Failing input**: A DataFrame with mixed types in an object column, e.g., `pd.DataFrame({'col': ['hello', 1, None]})`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

pdf = pd.DataFrame({'col': ['hello', 1, None]})
print("Original:", pdf['col'].tolist())
print("Original types:", [type(x).__name__ for x in pdf['col']])

ddf = dd.from_pandas(pdf, npartitions=2)
result = ddf.compute()
print("Result:", result['col'].tolist())
print("Result types:", [type(x).__name__ for x in result['col']])

assert pdf['col'].iloc[1] == 1
assert isinstance(pdf['col'].iloc[1], int)

assert result['col'].iloc[1] == '1'
assert isinstance(result['col'].iloc[1], str)
```

## Why This Is A Bug

The `from_pandas` function is documented to "Construct a Dask DataFrame from a Pandas DataFrame". Users expect that `from_pandas(df).compute()` should return a DataFrame equivalent to the original. However, when the input contains object columns with mixed types (strings, integers, None, etc.), dask silently converts all values to strings:

- The integer `1` becomes the string `'1'`
- `None` becomes `<NA>` (pandas NA)
- The dtype changes from `object` to `string`

This violates the fundamental round-trip property and causes silent data corruption. Users may not notice this until the corrupted data causes downstream errors or incorrect results.

## Fix

The issue appears to be in dask's automatic dtype inference for object columns. Dask should either:

1. Preserve the original dtype (including `object`) when it contains mixed types
2. Raise an error if automatic conversion would change the data
3. Provide an explicit parameter to control this behavior

The conversion happens in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/_pyarrow.py` based on the error traceback. A fix would need to detect mixed-type object columns and preserve them as-is rather than attempting to convert them to string dtype.