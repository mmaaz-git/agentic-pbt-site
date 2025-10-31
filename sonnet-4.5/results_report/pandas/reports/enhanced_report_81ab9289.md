# Bug Report: pandas.io.excel._openpyxl Large Float Overflow on Read

**Target**: `pandas.io.excel._openpyxl._convert_cell`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Reading Excel files containing extremely large float values (sys.float_info.max = 1.7976931348623157e+308) causes an OverflowError crash in pandas, even though writing those same values succeeds without error.

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


if __name__ == "__main__":
    test_skipfooter_invariant()
```

<details>

<summary>
**Failing input**: DataFrame with 15 rows containing float column with value 1.797693e+308 (sys.float_info.max)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 28, in <module>
    test_skipfooter_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 8, in test_skipfooter_invariant
    df=data_frames(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 22, in test_skipfooter_invariant
    result = pd.read_excel(buffer, skipfooter=skipfooter)
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
Falsifying example: test_skipfooter_invariant(
    df=
            X              Y
        0   0  1.797693e+308
        1   0  1.797693e+308
        2   0  1.797693e+308
        3   0  1.797693e+308
        4   0  1.797693e+308
        5   0  1.797693e+308
        6   0  1.797693e+308
        7   0  1.797693e+308
        8   0  1.797693e+308
        9   0  1.797693e+308
        10  0  1.797693e+308
        11  0  1.797693e+308
        12  0  1.797693e+308
        13  0  1.797693e+308
        14  0  1.797693e+308
    ,
    skipfooter=0,
)
```
</details>

## Reproducing the Bug

```python
import io
import pandas as pd
import sys
import traceback

# Test with sys.float_info.max
print(f"Testing with sys.float_info.max = {sys.float_info.max}")
print()

df = pd.DataFrame({"X": [0], "Y": [sys.float_info.max]})
print(f"Original DataFrame:")
print(df)
print()

buffer = io.BytesIO()
print("Writing to Excel...")
df.to_excel(buffer, index=False)
print("Write successful!")
print()

buffer.seek(0)
print("Reading from Excel...")
try:
    result = pd.read_excel(buffer)
    print("Read successful!")
    print(f"Result DataFrame:")
    print(result)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    print()
    print("Full traceback:")
    traceback.print_exc()
```

<details>

<summary>
OverflowError: cannot convert float infinity to integer
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/repo.py", line 24, in <module>
    result = pd.read_excel(buffer)
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
Testing with sys.float_info.max = 1.7976931348623157e+308

Original DataFrame:
   X              Y
0  0  1.797693e+308

Writing to Excel...
Write successful!

Reading from Excel...
Error occurred: OverflowError: cannot convert float infinity to integer

Full traceback:
```
</details>

## Why This Is A Bug

This violates the fundamental roundtrip property that data written to Excel should be readable back from Excel. The bug occurs because:

1. **Writing succeeds silently**: When extremely large float values (like `sys.float_info.max`) are written to Excel format, they get rounded to infinity due to Excel's floating-point representation limitations. The write operation completes without any warning or error.

2. **Reading fails with crash**: When reading back the same Excel file, the `_convert_cell` method in `/pandas/io/excel/_openpyxl.py` (line 600) unconditionally attempts to convert numeric cell values to integers first with `int(cell.value)`. When the cell contains infinity (from the rounded large float), this raises an `OverflowError`.

3. **Inconsistent behavior**: The code already handles the case where integer conversion would lose precision (line 601-603), but it doesn't handle the case where conversion is impossible due to overflow.

This contradicts pandas' general principle of data preservation and the expectation that Excel I/O operations should handle edge cases gracefully. Users working with scientific computing, astronomical data, or financial modeling may legitimately have very large numbers that approach float limits.

## Relevant Context

The bug is located in the `OpenpyxlReader._convert_cell` method at line 589-605 of `/pandas/io/excel/_openpyxl.py`. The problematic code path:

```python
elif cell.data_type == TYPE_NUMERIC:
    val = int(cell.value)  # Line 600 - crashes on infinity
    if val == cell.value:
        return val
    return float(cell.value)
```

The code tries to preserve integer types by checking if the value can be represented exactly as an integer. However, it doesn't protect against overflow when the numeric value is infinity or too large.

Related pandas documentation:
- Excel I/O: https://pandas.pydata.org/docs/user_guide/io.html#excel-files
- The openpyxl engine is the default for reading/writing `.xlsx` files

## Proposed Fix

```diff
--- a/pandas/io/excel/_openpyxl.py
+++ b/pandas/io/excel/_openpyxl.py
@@ -597,7 +597,11 @@ class OpenpyxlReader(BaseExcelReader):
         elif cell.data_type == TYPE_ERROR:
             return np.nan
         elif cell.data_type == TYPE_NUMERIC:
-            val = int(cell.value)
+            try:
+                val = int(cell.value)
+            except (OverflowError, ValueError):
+                # Value is infinity, -infinity, or NaN
+                return float(cell.value)
             if val == cell.value:
                 return val
             return float(cell.value)
```