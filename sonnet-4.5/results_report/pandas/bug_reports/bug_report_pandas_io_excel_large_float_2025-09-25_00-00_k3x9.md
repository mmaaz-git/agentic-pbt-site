# Bug Report: pandas.io.excel Large Float Overflow

**Target**: `pandas.io.excel.read_excel`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Reading an Excel file containing very large float values (close to `sys.float_info.max`) causes an `OverflowError` crash in pandas, even though writing those values succeeds without error.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, column


@given(
    df=data_frames(
        columns=[
            column("X", dtype=int),
            column("Y", dtype=float),
        ],
        index=st.just(pd.RangeIndex(0, 15)),
    ),
    skipfooter=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=50)
def test_skipfooter_invariant(df, skipfooter):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    result = pd.read_excel(buffer, skipfooter=skipfooter)
    expected_len = max(0, len(df) - skipfooter)
    assert len(result) == expected_len
```

**Failing input**: DataFrame with float column containing `1.797693e+308` (sys.float_info.max)

## Reproducing the Bug

```python
import io
import pandas as pd
import sys

df = pd.DataFrame({"X": [0], "Y": [sys.float_info.max]})
buffer = io.BytesIO()
df.to_excel(buffer, index=False)
buffer.seek(0)
result = pd.read_excel(buffer)
```

Output:
```
OverflowError: cannot convert float infinity to integer
```

## Why This Is A Bug

The roundtrip property is violated: a DataFrame can be written to Excel successfully, but cannot be read back. When extremely large floats are written to Excel, they may round to infinity when stored. The `_convert_cell` method in `pandas/io/excel/_openpyxl.py` unconditionally calls `int(cell.value)` for numeric cells, which raises OverflowError when the value is infinity.

Users working with scientific or financial data containing large numbers can encounter this crash. The error occurs silently during reading, making it difficult to diagnose.

## Fix

```diff
--- a/pandas/io/excel/_openpyxl.py
+++ b/pandas/io/excel/_openpyxl.py
@@ -597,7 +597,12 @@ class OpenpyxlReader(BaseExcelReader):
         elif cell.data_type == TYPE_ERROR:
             return np.nan
         elif cell.data_type == TYPE_NUMERIC:
-            val = int(cell.value)
+            try:
+                val = int(cell.value)
+            except (OverflowError, ValueError):
+                return float(cell.value)
+
             if val == cell.value:
                 return val
             return float(cell.value)
```