# Bug Report: scipy.io.arff split_data_line IndexError on Empty String

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function crashes with an `IndexError` when called with an empty string, due to unchecked array access on line 476.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line

@given(st.text(max_size=100))
def test_split_data_line_handles_all_strings(line):
    try:
        result, dialect = split_data_line(line)
        assert isinstance(result, list)
    except (IndexError, ValueError) as e:
        raise AssertionError(f"split_data_line crashed on input {line!r}: {e}")
```

**Failing input**: `''` (empty string)

## Reproducing the Bug

```python
from scipy.io.arff._arffread import split_data_line

result, dialect = split_data_line('')
```

Output:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "scipy/io/arff/_arffread.py", line 476, in split_data_line
    if line[-1] == '\n':
IndexError: string index out of range
```

## Why This Is A Bug

On line 476, the code attempts to check if the last character of `line` is a newline without first verifying that `line` is non-empty:

```python
if line[-1] == '\n':
    line = line[:-1]
```

When `line` is an empty string, `line[-1]` raises `IndexError: string index out of range`.

While empty lines in ARFF data sections are typically skipped by the `generator` function in `_loadarff` (line 862: `if r_comment.match(raw) or r_empty.match(raw): continue`), the `split_data_line` function is a public utility that should handle all string inputs gracefully, including edge cases like empty strings.

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

Alternatively, handle it after stripping:

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -473,11 +473,11 @@ def split_data_line(line, dialect=None):
     csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

-    # Remove the line end if any
-    if line[-1] == '\n':
-        line = line[:-1]
-
     # Remove potential trailing whitespace
     line = line.strip()

+    if not line:
+        return [], None
+
     sniff_line = line
```