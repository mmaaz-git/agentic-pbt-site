# Bug Report: pandas.io.excel Empty String Data Loss

**Target**: `pandas.io.excel.read_excel` / `pandas.DataFrame.to_excel`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame containing rows with only empty strings is written to Excel and read back, the rows are silently lost. This is a data corruption bug.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import tempfile
import os

@given(st.data())
@settings(max_examples=30)
def test_string_roundtrip(data):
    """String DataFrames should round-trip through Excel"""
    n_rows = data.draw(st.integers(min_value=1, max_value=50))
    n_cols = data.draw(st.integers(min_value=1, max_value=5))

    df_data = {}
    for i in range(n_cols):
        col_name = f'col_{i}'
        df_data[col_name] = data.draw(st.lists(
            st.text(min_size=0, max_size=100),
            min_size=n_rows,
            max_size=n_rows
        ))

    df_original = pd.DataFrame(df_data)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df_original.to_excel(tmp_path, index=False)
        df_read = pd.read_excel(tmp_path)

        assert df_read.shape == df_original.shape
        pd.testing.assert_frame_equal(df_read, df_original, check_dtype=False)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Failing input**: `pd.DataFrame({'col_0': ['']})`

## Reproducing the Bug

```python
import pandas as pd
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df_original = pd.DataFrame({'col': ['']})
    print(f"Original: {df_original.shape}")
    print(df_original)

    df_original.to_excel(tmp_path, index=False)
    df_read = pd.read_excel(tmp_path)

    print(f"\nRead back: {df_read.shape}")
    print(df_read)
    print(f"\nData lost: {df_original.shape != df_read.shape}")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

Output:
```
Original: (1, 1)
  col
0

Read back: (0, 1)
Empty DataFrame
Columns: [col]
Index: []

Data lost: True
```

## Why This Is A Bug

This is a **silent data corruption bug** - one of the most serious types of bugs because:

1. Data is lost without any warning or error
2. Users expect round-trip preservation: `read_excel(df.to_excel(path))` should recover the original data
3. Empty strings are valid data that should be preserved
4. This violates user expectations and can lead to incorrect analysis results

The root cause appears to be that `read_excel` treats rows containing only empty strings as empty rows and skips them, likely due to Excel's default behavior of treating such rows as blank.

## Fix

The fix should ensure that rows containing empty strings are preserved during the write/read cycle. This likely requires:

1. Either writing empty strings in a way that Excel/openpyxl recognizes as non-empty cells
2. Or modifying `read_excel` to not skip rows that contain empty string values

A high-level approach:

```diff
--- a/pandas/io/excel/_base.py
+++ b/pandas/io/excel/_base.py

The fix would need to ensure that when writing to Excel:
- Empty strings are written as explicit string values, not as empty cells
- Or add a space/special marker to distinguish empty strings from truly empty cells

And/or when reading from Excel:
- Don't skip rows that contain cells with empty string values
- Distinguish between truly blank rows and rows with empty strings
```

Note: The exact fix depends on whether the issue is in the write path (openpyxl treating empty strings as blank cells) or the read path (pandas skipping rows it considers blank). Further investigation of the Excel file internals would clarify the best approach.