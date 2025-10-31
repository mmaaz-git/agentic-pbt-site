# Bug Report: scipy.io.arff split_data_line IndexError on Empty String

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function crashes with an `IndexError` when called with an empty string, because it attempts to index `line[-1]` without first checking if the line is non-empty. This can be triggered via `RelationalAttribute.parse_data` which splits data by newlines, producing empty strings for trailing newlines or empty data.

## Property-Based Test

```python
from scipy.io.arff._arffread import split_data_line, RelationalAttribute
from hypothesis import given, strategies as st


@given(st.text())
def test_split_data_line_handles_any_string(line):
    """
    Property: split_data_line should handle any string input without crashing.

    This fails on empty strings due to unchecked indexing.
    """
    try:
        row, dialect = split_data_line(line)
        assert isinstance(row, (list, tuple))
    except ValueError:
        pass


@given(st.text())
def test_relational_parse_data_no_crash(data_str):
    """
    Property: RelationalAttribute.parse_data should not crash.

    This can trigger the bug when data_str ends with newline or is empty,
    because split('\\n') produces empty strings.
    """
    attr = RelationalAttribute("test")
    attr.attributes = []

    try:
        attr.parse_data(data_str)
    except IndexError as e:
        if "string index out of range" in str(e):
            raise AssertionError(f"IndexError on split_data_line with data: {repr(data_str)}") from e
    except Exception:
        pass
```

**Failing input**: `""` (empty string)

## Reproducing the Bug

```python
from scipy.io.arff._arffread import split_data_line

split_data_line("")
```

**Output:**
```
IndexError: string index out of range
```

**Stack trace location:** Line 476 in `_arffread.py`

## Why This Is A Bug

Line 476 attempts to check if the line ends with a newline:

```python
if line[-1] == '\n':
    line = line[:-1]
```

However, if `line` is an empty string, `line[-1]` raises `IndexError`.

**How it can be triggered:**

1. `RelationalAttribute.parse_data` (line 373) does:
   ```python
   for raw in escaped_string.split("\n"):
       row, self.dialect = split_data_line(raw, self.dialect)
   ```

2. The `str.split("\n")` method produces empty strings in several cases:
   - `"foo\n".split("\n")` → `['foo', '']` (trailing newline)
   - `"".split("\n")` → `['']` (empty string)
   - `"\n".split("\n")` → `['', '']` (just newline)

3. These empty strings are passed to `split_data_line`, causing the crash.

**Impact:** Valid ARFF relational attributes with trailing newlines or empty data will cause the parser to crash with `IndexError` instead of either parsing correctly or raising a meaningful error.

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