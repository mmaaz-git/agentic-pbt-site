# Bug Report: pandas.io.clipboard Index Column Detection

**Target**: `pandas.io.clipboards.read_clipboard`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_clipboard()` function incorrectly detects the number of index columns when clipboard data contains leading spaces before tabs. It counts the total number of leading whitespace **characters** instead of counting leading **tab characters**, causing the wrong number of columns to be treated as index columns.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from unittest.mock import patch
import pandas as pd

@given(st.integers(min_value=1, max_value=5))
def test_index_column_detection_counts_tabs_not_characters(num_spaces):
    assume(num_spaces > 0)

    data_rows = [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]

    leading_ws = ' ' * num_spaces + '\t'
    lines = [leading_ws + '\t'.join(row) for row in data_rows]
    text = '\n'.join(lines) + '\n'

    with patch('pandas.io.clipboard.clipboard_get', return_value=text):
        df = pd.read_clipboard()

        expected_tab_count = 1
        actual_column_count = df.shape[1]

        expected_columns = 3
        assert actual_column_count == expected_columns, \
            f"With {num_spaces} spaces + 1 tab, expected {expected_columns} columns but got {actual_column_count}"
```

**Failing input**: `num_spaces=1` (any value ≥ 1)

## Reproducing the Bug

```python
from unittest.mock import patch
import pandas as pd

text = " \ta\tb\tc\n \t1\t2\t3\n \t4\t5\t6\n"

with patch('pandas.io.clipboard.clipboard_get', return_value=text):
    df = pd.read_clipboard()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df)
```

**Output:**
```
Shape: (2, 2)
Columns: ['b', 'c']
   b  c
a  2  3
1  2  3
4  5  6
```

**Expected output:**
```
Shape: (2, 3)
Columns: ['a', 'b', 'c']
   a  b  c
0  1  2  3
1  4  5  6
```

The column 'a' is incorrectly treated as an index column because the code counts 2 characters (`" \t"`) instead of counting 1 tab.

## Why This Is A Bug

Looking at `pandas/io/clipboards.py`, lines 111-113:

```python
# check the number of leading tabs in the first line
# to account for index columns
index_length = len(lines[0]) - len(lines[0].lstrip(" \t"))
if index_length != 0:
    kwargs.setdefault("index_col", list(range(index_length)))
```

The comment says "check the number of leading **tabs**" but the code calculates `index_length` as the total **length** of leading whitespace (spaces + tabs), not the **count** of tab characters.

**Example breakdown:**
- Input: `" \ta\tb\tc"` (1 space, then 1 tab, then data)
- `index_length = len(" \ta\tb\tc") - len("a\tb\tc") = 2`
- Sets `index_col = [0, 1]` → treats first 2 columns as index
- **Should be**: `index_col = [0]` → only 1 tab = 1 index column

This violates the documented intent: index columns should correspond to leading **tabs**, not total whitespace length.

## Fix

```diff
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -108,7 +108,8 @@ def read_clipboard(
     if len(lines) > 1 and len(counts) == 1 and counts.pop() != 0:
         sep = "\t"
         # check the number of leading tabs in the first line
         # to account for index columns
-        index_length = len(lines[0]) - len(lines[0].lstrip(" \t"))
+        leading_ws_len = len(lines[0]) - len(lines[0].lstrip(" \t"))
+        index_length = lines[0][:leading_ws_len].count("\t")
         if index_length != 0:
             kwargs.setdefault("index_col", list(range(index_length)))
```

This fix correctly counts only the tab characters in the leading whitespace, not the total length of whitespace characters.