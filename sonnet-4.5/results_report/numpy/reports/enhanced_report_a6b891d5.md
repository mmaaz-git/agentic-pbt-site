# Bug Report: pandas.io.excel._openpyxl OverflowError Reading Large Floats from Excel

**Target**: `pandas.io.excel._openpyxl.OpenpyxlReader._convert_cell`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The openpyxl Excel reader crashes with an OverflowError when reading Excel files containing very large float values (near `sys.float_info.max`) that were converted to infinity during the write operation.

## Property-Based Test

```python
import pandas as pd
import tempfile
import os
from hypothesis import given, strategies as st, settings, example
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
@example(data=[[1.7976931348623155e+308]], sheet_name="Sheet1")  # The failing case
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

# Run the test
if __name__ == "__main__":
    test_read_excel_excelfile_equivalence()
```

<details>

<summary>
**Failing input**: `data=[[1.7976931348623155e+308]], sheet_name='Sheet1'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/40
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_read_excel_excelfile_equivalence FAILED                    [100%]

=================================== FAILURES ===================================
____________________ test_read_excel_excelfile_equivalence _____________________
hypo.py:8: in test_read_excel_excelfile_equivalence
    @given(

/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
hypo.py:33: in test_read_excel_excelfile_equivalence
    result_direct = pd.read_excel(tmp_path, sheet_name=sheet_name)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py:508: in read_excel
    data = io.parse(
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py:1616: in parse
    return self._reader.parse(
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py:778: in parse
    data = self.get_sheet_data(sheet, file_rows_needed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_openpyxl.py:616: in get_sheet_data
    converted_row = [self._convert_cell(cell) for cell in row]
                     ^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_openpyxl.py:600: in _convert_cell
    val = int(cell.value)
          ^^^^^^^^^^^^^^^
E   OverflowError: cannot convert float infinity to integer
E   Falsifying explicit example: test_read_excel_excelfile_equivalence(
E       data=[[1.7976931348623155e+308]],
E       sheet_name='Sheet1',
E   )
=========================== short test summary info ============================
FAILED hypo.py::test_read_excel_excelfile_equivalence - OverflowError: cannot...
============================== 1 failed in 0.45s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import tempfile
import os

# Create a DataFrame with a very large float value near sys.float_info.max
df = pd.DataFrame([[1.7976931348623155e+308]])
print(f"Original DataFrame:\n{df}")
print(f"Value type: {type(df.iloc[0, 0])}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    # Write the DataFrame to Excel
    df.to_excel(tmp_path, index=False)
    print(f"\nSuccessfully wrote DataFrame to Excel file: {tmp_path}")

    # Try to read it back
    print("\nAttempting to read Excel file back...")
    result = pd.read_excel(tmp_path)
    print(f"Successfully read back:\n{result}")
except OverflowError as e:
    print(f"\nOverflowError occurred: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"\nUnexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
        print(f"\nCleaned up temporary file: {tmp_path}")
```

<details>

<summary>
OverflowError: cannot convert float infinity to integer
</summary>
```
Original DataFrame:
               0
0  1.797693e+308
Value type: <class 'numpy.float64'>

Successfully wrote DataFrame to Excel file: /tmp/tmpoey6cd08.xlsx

Attempting to read Excel file back...

OverflowError occurred: cannot convert float infinity to integer
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/repo.py", line 20, in <module>
    result = pd.read_excel(tmp_path)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py", line 508, in read_excel
    data = io.parse(
        sheet_name=sheet_name,
    ...<21 lines>...
        dtype_backend=dtype_backend,
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py", line 1616, in parse
    return self._reader.parse(
           ~~~~~~~~~~~~~~~~~~^
        sheet_name=sheet_name,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<17 lines>...
        **kwds,
        ^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py", line 778, in parse
    data = self.get_sheet_data(sheet, file_rows_needed)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_openpyxl.py", line 616, in get_sheet_data
    converted_row = [self._convert_cell(cell) for cell in row]
                     ~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_openpyxl.py", line 600, in _convert_cell
    val = int(cell.value)
OverflowError: cannot convert float infinity to integer

Cleaned up temporary file: /tmp/tmpoey6cd08.xlsx
```
</details>

## Why This Is A Bug

This violates the fundamental round-trip property expected in data persistence: a DataFrame that pandas successfully writes to Excel should be readable back without crashing. The issue occurs because:

1. **Valid input becomes unreadable**: The value `1.7976931348623155e+308` is a legitimate Python float (less than `sys.float_info.max = 1.7976931348623157e+308`) that's commonly used in scientific computing
2. **Inconsistent implementation**: The xlrd Excel reader in pandas already handles this case correctly with a `math.isfinite()` check before attempting integer conversion (see `_xlrd.py` line 124)
3. **Silent data transformation leads to crash**: When writing to Excel, openpyxl converts very large floats to infinity, but the reader doesn't handle infinity properly when reading back
4. **Unhandled exception**: The code attempts `int(infinity)` at line 600 of `_openpyxl.py`, which raises an OverflowError that propagates to the user

## Relevant Context

The bug manifests in the `OpenpyxlReader._convert_cell` method at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_openpyxl.py:589-605`. The method attempts to convert all numeric cells to integers first to preserve integer types, but doesn't check whether the value is finite before conversion.

For comparison, the xlrd reader implementation at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_xlrd.py:121-129` includes the proper check:
```python
elif cell_typ == XL_CELL_NUMBER:
    if math.isfinite(cell_contents):
        # GH54564 - don't attempt to convert NaN/Inf
        val = int(cell_contents)
        if val == cell_contents:
            cell_contents = val
```

This bug affects users working with:
- Scientific computing applications with extreme values
- Data analysis involving measurements near physical limits
- Financial calculations with very large numbers
- Any workflow that round-trips DataFrames through Excel

## Proposed Fix

```diff
--- a/pandas/io/excel/_openpyxl.py
+++ b/pandas/io/excel/_openpyxl.py
@@ -1,6 +1,7 @@
 from __future__ import annotations

 import mmap
+import math
 from typing import (
     TYPE_CHECKING,
     Any,
@@ -597,7 +598,9 @@ class OpenpyxlReader(BaseExcelReader["Workbook"]):
         elif cell.data_type == TYPE_ERROR:
             return np.nan
         elif cell.data_type == TYPE_NUMERIC:
-            val = int(cell.value)
+            if not math.isfinite(cell.value):
+                return float(cell.value)
+            val = int(cell.value)
             if val == cell.value:
                 return val
             return float(cell.value)
```