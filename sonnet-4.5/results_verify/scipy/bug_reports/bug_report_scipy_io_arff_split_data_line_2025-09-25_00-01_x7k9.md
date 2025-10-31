# Bug Report: scipy.io.arff split_data_line IndexError on Empty String

**Target**: `scipy.io.arff._arffread.split_data_line()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line()` function raises an `IndexError` when called with an empty string because it accesses `line[-1]` without checking if the string is non-empty. This can occur in practice when ARFF files contain empty lines that are not filtered by the `r_empty` regex.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line


@given(st.text())
def test_split_data_line_handles_all_strings(line):
    result, dialect = split_data_line(line)
    assert isinstance(result, list)
```

**Failing input**: `""` (empty string)

## Reproducing the Bug

```python
from scipy.io.arff._arffread import split_data_line

result, dialect = split_data_line("")
```

**Error:**
```
IndexError: string index out of range
```

This also fails when loading ARFF files with empty lines:

```python
from scipy.io.arff import loadarff
from io import StringIO

arff_content = """@relation test
@attribute x numeric
@data
1.0

2.0
"""

data, meta = loadarff(StringIO(arff_content))
```

## Why This Is A Bug

1. **Line 476** attempts to access `line[-1]` without checking if `line` is non-empty
2. **Empty strings can reach this code** because:
   - The `r_empty` regex is `r'^\s+$'` which requires at least one whitespace character
   - A truly empty string `""` does NOT match this regex
   - Therefore empty lines from file iteration are not filtered out (line 862)
3. **The bug causes a crash** with `IndexError: string index out of range`

The problematic code flow:
```python
# Line 862: Check for empty lines
if r_comment.match(raw) or r_empty.match(raw):
    continue  # Skip

# Line 865: Process line
row, dialect = split_data_line(raw, dialect)

# Inside split_data_line, line 476:
if line[-1] == '\n':  # IndexError if line is ""
    line = line[:-1]
```

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -473,7 +473,7 @@ def split_data_line(line, dialect=None):
     csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

     # Remove the line end if any
-    if line[-1] == '\n':
+    if line and line[-1] == '\n':
         line = line[:-1]

     # Remove potential trailing whitespace
```