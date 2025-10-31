# Bug Report: pandas.io.excel Empty String Data Loss and Row Disappearance

**Target**: `pandas.io.excel._openpyxl` (read_excel/to_excel)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping DataFrames through Excel, rows containing only empty strings or None values completely disappear when they are the only rows in the DataFrame. Additionally, empty strings are converted to NaN values during the round-trip process.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import tempfile
import os


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

    # Filter out illegal characters that Excel doesn't support
    for idx, row in enumerate(data):
        text = row[2]
        # Skip data with illegal Excel characters (control characters)
        if any(ord(ch) < 32 and ch not in '\t\n\r' for ch in text):
            assume(False)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df.to_excel(tmp_path, index=False, engine='openpyxl')
        df_read = pd.read_excel(tmp_path, engine='openpyxl')

        assert len(df_read) == len(df), f"Row count mismatch: expected {len(df)}, got {len(df_read)}"
        assert list(df_read.columns) == list(df.columns)

        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    # Run the property-based test
    test_round_trip_write_read()
```

<details>

<summary>
**Failing input**: `data=[(0, 0.0, '')]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 47, in <module>
    test_round_trip_write_read()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 8, in test_round_trip_write_read
    st.lists(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 39, in test_round_trip_write_read
    pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1303, in assert_frame_equal
    assert_series_equal(
    ~~~~~~~~~~~~~~~~~~~^
        lcol,
        ^^^^^
    ...<12 lines>...
        check_flags=False,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1091, in assert_series_equal
    _testing.assert_almost_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        left._values,
        ^^^^^^^^^^^^^
    ...<5 lines>...
        index_values=left.index,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "pandas/_libs/testing.pyx", line 55, in pandas._libs.testing.assert_almost_equal
  File "pandas/_libs/testing.pyx", line 173, in pandas._libs.testing.assert_almost_equal
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: DataFrame.iloc[:, 2] (column name="str_col") are different

DataFrame.iloc[:, 2] (column name="str_col") values are different (100.0 %)
[index]: [0]
[left]:  []
[right]: [nan]
At positional index 0, first diff:  != nan
Falsifying example: test_round_trip_write_read(
    data=[(0, 0.0, '')],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:52
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:3614
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:138
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:628
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:659
        (and 5 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import tempfile
import pandas as pd
import os

print("=" * 60)
print("Test Case 1: Single column with empty string")
print("=" * 60)
df = pd.DataFrame([{'text': ''}])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")
print(f"Original dtypes: {df.dtypes.to_dict()}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl', na_filter=False)

    print("\nRead back DataFrame (with na_filter=False):")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")
    print(f"Read back dtypes: {df_read.dtypes.to_dict()}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test Case 2: Single column with None value")
print("=" * 60)
df = pd.DataFrame([{'col': None}])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl')

    print("\nRead back DataFrame:")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test Case 3: Multiple columns, all empty/None")
print("=" * 60)
df = pd.DataFrame([{'col1': '', 'col2': None, 'col3': ''}])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl', na_filter=False)

    print("\nRead back DataFrame (with na_filter=False):")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test Case 4: Mixed case - one row all empty, one row with data")
print("=" * 60)
df = pd.DataFrame([{'col1': '', 'col2': None}, {'col1': 'data', 'col2': 42}])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl', na_filter=False)

    print("\nRead back DataFrame (with na_filter=False):")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test Case 5: From bug report - data=[(0, 0.0, '')]")
print("=" * 60)
data = [(0, 0.0, '')]
df = pd.DataFrame(data, columns=['int_col', 'float_col', 'str_col'])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")
print(f"Original values: {df.values.tolist()}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl')

    print("\nRead back DataFrame:")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")
    print(f"Read back values: {df_read.values.tolist()}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")

    # Check value preservation
    if len(df_read) > 0:
        if df_read['str_col'].isna().iloc[0]:
            print("❌ ERROR: Empty string converted to NaN")
        else:
            print("✓ Empty string preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

<details>

<summary>
Multiple test cases demonstrating row loss and data conversion issues
</summary>
```
============================================================
Test Case 1: Single column with empty string
============================================================
Original DataFrame:
  text
0
Original shape: (1, 1)
Original dtypes: {'text': dtype('O')}

Read back DataFrame (with na_filter=False):
Empty DataFrame
Columns: [text]
Index: []
Read back shape: (0, 1)
Read back dtypes: {'text': dtype('O')}

❌ ERROR: Row lost! Original had 1 rows, read back has 0 rows

============================================================
Test Case 2: Single column with None value
============================================================
Original DataFrame:
    col
0  None
Original shape: (1, 1)

Read back DataFrame:
Empty DataFrame
Columns: [col]
Index: []
Read back shape: (0, 1)

❌ ERROR: Row lost! Original had 1 rows, read back has 0 rows

============================================================
Test Case 3: Multiple columns, all empty/None
============================================================
Original DataFrame:
  col1  col2 col3
0       None
Original shape: (1, 3)

Read back DataFrame (with na_filter=False):
Empty DataFrame
Columns: [col1, col2, col3]
Index: []
Read back shape: (0, 3)

❌ ERROR: Row lost! Original had 1 rows, read back has 0 rows

============================================================
Test Case 4: Mixed case - one row all empty, one row with data
============================================================
Original DataFrame:
   col1  col2
0         NaN
1  data  42.0
Original shape: (2, 2)

Read back DataFrame (with na_filter=False):
   col1 col2
0
1  data   42
Read back shape: (2, 2)

✓ Row count preserved

============================================================
Test Case 5: From bug report - data=[(0, 0.0, '')]
============================================================
Original DataFrame:
   int_col  float_col str_col
0        0        0.0
Original shape: (1, 3)
Original values: [[0, 0.0, '']]

Read back DataFrame:
   int_col  float_col  str_col
0        0          0      NaN
Read back shape: (1, 3)
Read back values: [[0.0, 0.0, nan]]

✓ Row count preserved
❌ ERROR: Empty string converted to NaN
```
</details>

## Why This Is A Bug

This violates fundamental data preservation expectations in multiple ways:

1. **Data Loss**: Rows containing only empty strings or None values completely disappear when they are the only rows in the DataFrame. This is unacceptable data loss during what should be a simple I/O operation.

2. **Type Loss**: Empty strings ('') are converted to NaN values, losing the distinction between an empty string and a missing value. This changes the data type and meaning of the original data.

3. **Inconsistent Behavior**: The bug exhibits inconsistent behavior - empty rows are preserved when mixed with non-empty rows but disappear when they're the only rows. This suggests unintentional behavior rather than a design choice.

4. **No Documentation**: The pandas documentation for `read_excel` and `to_excel` does not mention that rows with all empty/NA values will be dropped. Users have no warning about this potential data loss.

5. **na_filter Ineffective**: Setting `na_filter=False` during reading, which should prevent NA value detection, does not prevent the row loss, indicating the issue is deeper than just NA handling.

## Relevant Context

The root cause appears to be in the OpenpyxlReader implementation at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/excel/_openpyxl.py`:

1. Line 595-596: `_convert_cell` returns an empty string `""` for None values
2. Lines 616-621: The code trims trailing empty cells from each row
3. Lines 626-627: **Critical Issue**: The code trims all trailing empty rows using `data = data[: last_row_with_data + 1]`

When all cells in a row are empty/None, the row is considered to have no data (`last_row_with_data` is not updated), causing the entire row to be trimmed. This explains why:
- Rows with all empty values disappear when alone
- But are preserved when followed by rows with data

Documentation references:
- pandas.DataFrame.to_excel: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html
- pandas.read_excel: https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

## Proposed Fix

The issue can be fixed by modifying the row trimming logic in `_openpyxl.py` to preserve rows that exist in the source file, even if all cells are empty:

```diff
--- a/pandas/io/excel/_openpyxl.py
+++ b/pandas/io/excel/_openpyxl.py
@@ -613,11 +613,14 @@ class OpenpyxlReader(BaseExcelReader["Workbook"]):
         data: list[list[Scalar]] = []
         last_row_with_data = -1
         for row_number, row in enumerate(sheet.rows):
             converted_row = [self._convert_cell(cell) for cell in row]
+            original_length = len(converted_row)
             while converted_row and converted_row[-1] == "":
                 # trim trailing empty elements
                 converted_row.pop()
-            if converted_row:
+            # Consider a row to have data if it had any cells originally,
+            # even if all were empty
+            if converted_row or original_length > 0:
                 last_row_with_data = row_number
             data.append(converted_row)
             if file_rows_needed is not None and len(data) >= file_rows_needed:
```