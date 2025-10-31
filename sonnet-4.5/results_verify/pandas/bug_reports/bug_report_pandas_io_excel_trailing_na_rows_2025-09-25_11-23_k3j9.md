# Bug Report: pandas.io.excel read_excel Drops Trailing All-NA Rows

**Target**: `pandas.io.excel.read_excel`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pandas.read_excel()` silently drops trailing rows where all cells contain NA/None values, causing data loss during round-trip operations. This occurs even when `na_filter=False` is specified.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import io


@st.composite
def simple_dataframes(draw):
    n_rows = draw(st.integers(min_value=0, max_value=50))
    n_cols = draw(st.integers(min_value=1, max_value=10))
    col_names = [f'col_{i}' for i in range(n_cols)]
    data = {}
    for col in col_names:
        data[col] = draw(st.lists(
            st.text(min_size=0, max_size=20),
            min_size=n_rows,
            max_size=n_rows
        ))
    return pd.DataFrame(data)


@given(df=simple_dataframes())
@settings(max_examples=100)
def test_roundtrip_excel_bytesio(df):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    df_read = pd.read_excel(buffer, engine='openpyxl')
    pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
```

**Failing input**: `pd.DataFrame({'col_0': ['']})`

## Reproducing the Bug

```python
import io
import pandas as pd

df = pd.DataFrame({'col1': ['a', ''], 'col2': ['b', '']})
print(f"Original: {df.shape}")
print(df)

buffer = io.BytesIO()
df.to_excel(buffer, index=False, engine='openpyxl')
buffer.seek(0)

df_read = pd.read_excel(buffer, engine='openpyxl')
print(f"\nRead back: {df_read.shape}")
print(df_read)

buffer.seek(0)
df_no_filter = pd.read_excel(buffer, engine='openpyxl', na_filter=False)
print(f"\nWith na_filter=False: {df_no_filter.shape}")
print(df_no_filter)
```

**Output**:
```
Original: (2, 2)
  col1 col2
0    a    b
1

Read back: (1, 2)
  col1 col2
0    a    b

With na_filter=False: (1, 2)
  col1 col2
0    a    b
```

The trailing row is lost even with `na_filter=False`.

## Why This Is A Bug

1. **Silent data loss**: Rows are dropped without warning
2. **Violates round-trip property**: `read_excel(to_excel(df))` ≠ `df`
3. **Ignores na_filter parameter**: `na_filter=False` should prevent NA filtering
4. **Inconsistent behavior**: Leading/middle all-NA rows are preserved, trailing ones are not
5. **Semantic meaning lost**: Empty rows may represent placeholders, separators, or alignment markers

## Fix

The bug is in pandas' Excel reading logic, which trims trailing all-NA rows. The fix should:

1. Respect the actual data written to Excel by pandas
2. Honor `na_filter=False` by preserving all rows
3. Only trim trailing rows if they exceed Excel's written data range

Without source access, a conceptual fix would be to track the actual row count when writing and preserve that count when reading, or to add a parameter like `skip_blank_rows` (defaulting to False for pandas-written files).