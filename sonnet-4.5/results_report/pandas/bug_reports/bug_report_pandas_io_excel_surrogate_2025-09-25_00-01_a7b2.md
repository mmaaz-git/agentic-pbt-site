# Bug Report: pandas.io.excel Surrogate Character Crash

**Target**: `pandas.io.excel` (DataFrame.to_excel)
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Writing a DataFrame containing Unicode surrogate characters (U+D800 to U+DFFF) to Excel causes a `UnicodeEncodeError` crash with a cryptic error message deep in the library stack.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.pandas import data_frames, column


@given(
    df=data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
            column("C", dtype=str),
        ],
        index=st.just(pd.RangeIndex(0, 10)),
    )
)
@settings(max_examples=50)
def test_roundtrip_basic(df):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    result = pd.read_excel(buffer)
    pd.testing.assert_frame_equal(result, df, check_dtype=False)
```

**Failing input**: DataFrame with string column containing surrogate character `\ud800`

## Reproducing the Bug

```python
import io
import pandas as pd

df = pd.DataFrame({"A": [0], "B": [0.0], "C": ["\ud800"]})
buffer = io.BytesIO()
df.to_excel(buffer, index=False)
```

Output:
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 7: surrogates not allowed
```

## Why This Is A Bug

While surrogate characters are technically invalid in UTF-8, Python strings can contain them (they're valid in UTF-16). Users may encounter surrogates through:
1. Incorrectly decoded UTF-16 data
2. Data corruption
3. Manual string construction

The current behavior provides a poor user experience:
- No validation at the pandas level
- Cryptic error message deep in xlsxwriter/openpyxl stack
- Affects both xlsxwriter and openpyxl engines

pandas should either:
1. Detect and reject surrogates early with a helpful error message, or
2. Sanitize them before writing (replace with replacement character U+FFFD)

## Fix

Option 1: Early validation with clear error message

```diff
--- a/pandas/io/formats/excel.py
+++ b/pandas/io/formats/excel.py
@@ -958,6 +958,15 @@ class ExcelFormatter:
             for col_loc, col_val in enumerate(row):
                 val = self._format_value(col_val)
+                # Check for surrogate characters
+                if isinstance(val, str):
+                    try:
+                        val.encode('utf-8')
+                    except UnicodeEncodeError as e:
+                        if 'surrogates not allowed' in str(e):
+                            raise ValueError(
+                                f"Column {col_loc} contains invalid surrogate characters. "
+                                "Please clean your data before writing to Excel."
+                            ) from e
                 style = None
                 if styles:
                     style = styles[col_loc]
```