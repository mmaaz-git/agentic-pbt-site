# Bug Report: scipy.io.arff split_data_line IndexError on Empty String

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function crashes with an IndexError when called with an empty string, due to accessing `line[-1]` without first checking if the string is non-empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line

@given(st.text())
def test_split_data_line_handles_all_strings(line):
    try:
        result, dialect = split_data_line(line)
        assert isinstance(result, list)
    except ValueError:
        pass
```

**Failing input**: `''` (empty string)

## Reproducing the Bug

```python
from scipy.io.arff._arffread import split_data_line

result, dialect = split_data_line('')
```

**Error**:
```
IndexError: string index out of range
```

## Why This Is A Bug

The function `split_data_line` is designed to parse CSV data lines in ARFF files. However, it attempts to check if the last character of the input line is a newline (`line[-1] == '\n'`) without first verifying that the line is non-empty. This causes an IndexError when the line is an empty string.

This bug can occur in practice when:
1. Processing relational attributes where splitting by newlines may produce empty strings
2. Parsing malformed or edge-case ARFF files
3. Any code path that calls `split_data_line` with user-controlled or file-based input

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