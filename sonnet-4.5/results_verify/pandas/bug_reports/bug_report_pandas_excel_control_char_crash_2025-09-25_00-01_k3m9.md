# Bug Report: pandas.io.excel Control Character Crash

**Target**: `pandas.io.excel.to_excel`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

DataFrames containing control characters (e.g., `\x1f`) in string columns cause an unhandled `IllegalCharacterError` when writing to Excel via `to_excel()` with the openpyxl engine.

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
        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
    finally:
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Failing input**: `data=[(0, 0.0, '\x1f')]`

## Reproducing the Bug

```python
import tempfile
import pandas as pd

df = pd.DataFrame([{'text': '\x1f'}])

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

df.to_excel(tmp_path, index=False, engine='openpyxl')
```

Error:
```
openpyxl.utils.exceptions.IllegalCharacterError:  cannot be used in worksheets.
```

## Why This Is A Bug

While Excel files have character restrictions, pandas should handle this gracefully rather than crashing. Users may have data containing control characters from various sources (logs, user input, etc.), and a valid pandas DataFrame should not cause an unhandled exception during export.

Expected behavior:
1. Automatically sanitize/escape illegal characters with a warning, OR
2. Raise a clear pandas-specific exception with actionable guidance, OR
3. Provide a parameter to control this behavior (e.g., `sanitize_chars=True`)

The current behavior exposes an openpyxl implementation detail without proper error handling.

## Fix

One approach is to wrap the write operation and catch `IllegalCharacterError`, then either:
1. Sanitize the characters and emit a warning
2. Re-raise as a more informative pandas exception

Example patch location would be in `/pandas/io/excel/_openpyxl.py` where cell values are written, adding character validation/sanitization before the write.