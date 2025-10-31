# Bug Report: pandas.io.excel Control Character Column Name Transformation

**Target**: `pandas.io.excel.to_excel` and `pandas.io.excel.read_excel`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Column names containing certain control characters are silently transformed during Excel round-trip operations (e.g., `\x08` becomes `_x0008_`), violating the round-trip property and causing data integrity issues.

## Property-Based Test

```python
import tempfile
import os
import pandas as pd
from hypothesis import given, strategies as st, settings


@settings(max_examples=100, deadline=None)
@given(
    col_names=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=1, max_size=10),
        min_size=1,
        max_size=10,
        unique=True
    )
)
def test_column_names_preserved(col_names):
    data = {col: [1, 2, 3] for col in col_names}
    df_original = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df_original.to_excel(tmp_path, index=False)
        df_read = pd.read_excel(tmp_path)

        assert list(df_original.columns) == list(df_read.columns)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Failing input**: `col_names=['\x08']`

## Reproducing the Bug

```python
import tempfile
import pandas as pd

df = pd.DataFrame({'\x08': [1, 2, 3]})

print(f"Original column: {repr(list(df.columns)[0])}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
    filepath = f.name

df.to_excel(filepath, index=False)
df_read = pd.read_excel(filepath)

print(f"After round-trip: {repr(list(df_read.columns)[0])}")
```

Output:
```
Original column: '\x08'
After round-trip: '_x0008_'
```

## Why This Is A Bug

Column names are being silently transformed during the round-trip operation. While the transformation is performed by the underlying Excel library (xlsxwriter) for XML safety, pandas should either:

1. **Reject invalid column names upfront** with a clear error message, OR
2. **Preserve the round-trip** by reversing the escaping when reading

Currently, the silent transformation violates user expectations and can break code that relies on specific column names. This is particularly problematic because:

- The transformation is undocumented
- Users get no warning
- Column references in downstream code will fail
- Different engines handle this differently (openpyxl raises an error, xlsxwriter silently escapes)

## Fix

Option 1: Add validation in `to_excel()` to reject column names with invalid characters:

```python
def _validate_column_names(columns):
    invalid_chars = ['\x00', '\x01', '\x08', ...]
    for col in columns:
        if isinstance(col, str):
            for char in invalid_chars:
                if char in col:
                    raise ValueError(
                        f"Column name {repr(col)} contains invalid character "
                        f"{repr(char)} that cannot be represented in Excel"
                    )
```

Option 2: Add reverse escaping in `read_excel()` to undo the transformation:

```python
def _unescape_column_name(name):
    import re
    return re.sub(r'_x([0-9a-fA-F]{4})_', lambda m: chr(int(m.group(1), 16)), name)
```

Option 1 is preferred as it provides clearer feedback to users.