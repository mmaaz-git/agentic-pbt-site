# Bug Report: dask.utils.asciitable Crashes on Empty Rows

**Target**: `dask.utils.asciitable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`asciitable` raises `TypeError: not enough arguments for format string` when given column names but no data rows, instead of displaying just the header.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.utils

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
def test_asciitable_empty_rows(columns):
    result = dask.utils.asciitable(columns, [])
    assert isinstance(result, str)
    for col in columns:
        assert col in result
```

**Failing input**: Any list of columns with empty rows list, e.g., `columns=['A', 'B', 'C'], rows=[]`

## Reproducing the Bug

```python
import dask.utils

dask.utils.asciitable(['A', 'B', 'C'], [])
```

Output:
```
TypeError: not enough arguments for format string
```

Additional examples:
```python
dask.utils.asciitable(['X'], [])         # Crashes
dask.utils.asciitable(['X', 'Y'], [])    # Crashes
dask.utils.asciitable(['A', 'B', 'C'], [(1, 2, 3)])  # Works fine
```

## Why This Is A Bug

1. Empty data is a valid and common scenario (e.g., before data is loaded, after filtering that removes all rows)
2. The function should handle edge cases gracefully
3. Displaying just the header row would be reasonable behavior
4. The crash prevents users from using this function in dynamic contexts where the number of rows might be zero

## Fix

The function needs to handle the case where `rows` is empty. A reasonable behavior would be to display just the header:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -xxx,x +xxx,x @@ def asciitable(columns, rows):
     ...
     # Build the format string
     fmt = "|" + "|".join(f" {{:<{w}}} " for w in widths) + "|"

+    # Handle empty rows case
+    if not rows:
+        header = fmt.format(*columns)
+        return "\n".join([sep, header, sep])
+
     # Format rows
     lines = [sep]
     lines.append(fmt.format(*columns))
```