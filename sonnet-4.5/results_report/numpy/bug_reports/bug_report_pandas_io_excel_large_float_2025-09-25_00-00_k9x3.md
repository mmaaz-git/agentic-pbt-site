# Bug Report: pandas.io.excel OverflowError Reading Large Floats

**Target**: `pandas.io.excel` (specifically `_openpyxl.OpenpyxlReader._convert_cell`)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Reading Excel files containing very large float values (near `sys.float_info.max`) causes an `OverflowError` in the openpyxl reader when attempting to convert infinity values to integers.

## Property-Based Test

```python
import pandas as pd
import tempfile
import os
from hypothesis import given, strategies as st, settings
from pandas.testing import assert_frame_equal

@settings(max_examples=100)
@given(
    data=st.lists(
        st.lists(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text()), min_size=1, max_size=5),
        min_size=1,
        max_size=10
    ),
    sheet_name=st.one_of(st.just(0), st.just("Sheet1"))
)
def test_read_excel_excelfile_equivalence(data, sheet_name):
    if not data or not data[0]:
        return

    num_cols = len(data[0])
    if not all(len(row) == num_cols for row in data):
        return

    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df.to_excel(tmp_path, sheet_name="Sheet1", index=False)

        result_direct = pd.read_excel(tmp_path, sheet_name=sheet_name)

        excel_file = pd.ExcelFile(tmp_path)
        result_via_excelfile = excel_file.parse(sheet_name=sheet_name)
        excel_file.close()

        assert_frame_equal(result_direct, result_via_excelfile)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Failing input**: `data=[[1.7976931348623155e+308]]`

## Reproducing the Bug

```python
import pandas as pd
import tempfile
import os

df = pd.DataFrame([[1.7976931348623155e+308]])

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False)
    result = pd.read_excel(tmp_path)
except OverflowError as e:
    print(f"OverflowError: {e}")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

## Why This Is A Bug

When pandas writes very large floats to Excel, they can be stored as infinity by the Excel format. When reading back, the `OpenpyxlReader._convert_cell` method attempts to convert all numeric cells to int first with `int(cell.value)`, which raises `OverflowError` when `cell.value` is infinity.

This breaks the fundamental round-trip property: valid DataFrames should be readable after writing to Excel. Large floats near `sys.float_info.max` are legitimate values in scientific computing.

## Fix

```diff
--- a/pandas/io/excel/_openpyxl.py
+++ b/pandas/io/excel/_openpyxl.py
@@ -597,7 +597,10 @@ class OpenpyxlReader(BaseExcelReader["Workbook"]):
         elif cell.data_type == TYPE_ERROR:
             return np.nan
         elif cell.data_type == TYPE_NUMERIC:
-            val = int(cell.value)
+            if not math.isfinite(cell.value):
+                return float(cell.value)
+
+            val = int(cell.value)
             if val == cell.value:
                 return val
             return float(cell.value)
```