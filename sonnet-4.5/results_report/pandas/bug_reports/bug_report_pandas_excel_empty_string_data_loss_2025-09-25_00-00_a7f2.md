# Bug Report: pandas.io.excel Empty String Row Data Loss

**Target**: `pandas.io.excel` (read_excel/to_excel)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Round-tripping a DataFrame through Excel causes complete row loss when all cells in a row contain None or empty strings. These rows disappear entirely when written to Excel and read back, even with `na_filter=False`. Rows with mixed empty/non-empty values are preserved correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import tempfile


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=-1000, max_value=1000),
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=50)
        ),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=200)
def test_round_trip_write_read(data):
    df = pd.DataFrame(data, columns=['int_col', 'float_col', 'str_col'])

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df.to_excel(tmp_path, index=False, engine='openpyxl')
        df_read = pd.read_excel(tmp_path, engine='openpyxl')

        assert len(df_read) == len(df)
        assert list(df_read.columns) == list(df.columns)

        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
    finally:
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Failing input**: `data=[(0, 0.0, '')]`

## Reproducing the Bug

```python
import tempfile
import pandas as pd

df = pd.DataFrame([{'text': ''}])
print("Original:", df)
print("Length:", len(df))

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

df.to_excel(tmp_path, index=False, engine='openpyxl')
df_read = pd.read_excel(tmp_path, engine='openpyxl', na_filter=False)

print("Read back:", df_read)
print("Length:", len(df_read))
```

Output:
```
Original:   text
0
Length: 1
Read back: Empty DataFrame
Columns: [text]
Index: []
Length: 0
```

Additional test case with all-None row:
```python
import tempfile
import pandas as pd

df = pd.DataFrame([{'col': None}])
with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name
df.to_excel(tmp_path, index=False, engine='openpyxl')
df_read = pd.read_excel(tmp_path, engine='openpyxl')
print(f"Original: {len(df)} rows, Read back: {len(df_read)} rows")
```
Output: `Original: 1 rows, Read back: 0 rows`

## Why This Is A Bug

This violates the fundamental expectation that data written to Excel can be read back. Empty strings are valid data and should be preserved during round-trip operations. The row exists in the Excel file (verified with openpyxl inspection showing 2 rows), but pandas skips it during reading because all cells contain `None` values.

The bug has two components:
1. **Writing**: Empty strings are written as `None` instead of empty strings
2. **Reading**: Rows where all cells are `None` are skipped entirely

## Fix

The root cause is in how empty strings are handled during the write-read cycle. One approach is to modify `read_excel` to not skip rows that exist in the file, even if they contain all NA values. This requires tracking the actual row count in the source file.

A higher-level fix would be to preserve empty strings during the write operation rather than converting them to `None`.