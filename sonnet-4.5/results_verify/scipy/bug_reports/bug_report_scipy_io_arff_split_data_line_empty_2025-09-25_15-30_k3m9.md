# Bug Report: scipy.io.arff split_data_line IndexError on Empty String

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function crashes with an `IndexError` when given an empty string, due to attempting to index `line[-1]` without first checking if the string is empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line


@given(st.just(''))
def test_split_data_line_empty_string(line):
    result, dialect = split_data_line(line)
```

**Failing input**: `''` (empty string)

## Reproducing the Bug

```python
from scipy.io.arff._arffread import split_data_line

result, dialect = split_data_line('')
```

Output:
```
IndexError: string index out of range
```

## Why This Is A Bug

The function is used internally when parsing ARFF data files. While empty lines in data sections might be uncommon, they could occur in edge cases or malformed files. The function should handle empty strings gracefully instead of crashing with an IndexError. The code attempts to check if the last character is a newline without first verifying the string is non-empty.

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