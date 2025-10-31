# Bug Report: pandas.io.excel._util._excel2num Empty String Returns Invalid Index

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_excel2num` function returns -1 when given an empty string or whitespace-only string, instead of raising a `ValueError` as documented. This invalid column index can cause downstream errors or silent failures.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from pandas.io.excel._util import _excel2num

@given(st.text())
@example('')
@example('   ')
def test_excel2num_empty_string_validation(col):
    stripped = col.strip()
    if not stripped:
        try:
            result = _excel2num(col)
            assert False, f"Empty/whitespace string returned {result}, should raise ValueError"
        except ValueError:
            pass
```

**Failing input**: `""` or `"   "`

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num

result = _excel2num('')
print(f"Result: {result}")

assert result == -1
```

Output:
```
Result: -1
```

## Why This Is A Bug

The function's docstring states it should raise `ValueError` for invalid input. An empty string is not a valid Excel column name. Instead of raising an error, the function silently returns -1, which is an invalid 0-based column index.

Looking at the code (lines 117-127 in `_util.py`):

```python
def _excel2num(x: str) -> int:
    index = 0

    for c in x.upper().strip():
        cp = ord(c)
        if cp < ord("A") or cp > ord("Z"):
            raise ValueError(f"Invalid column name: {x}")
        index = index * 26 + cp - ord("A") + 1

    return index - 1
```

When `x.upper().strip()` is empty, the for loop never executes, leaving `index = 0`, and the function returns `-1`.

This can cause:
1. Incorrect behavior if -1 accesses the last column (Python's negative indexing)
2. Confusing error messages in downstream code
3. Silent failures where invalid ranges are processed

## Fix

```diff
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -115,6 +115,9 @@ def _excel2num(x: str) -> int:
         Part of the Excel column name was invalid.
     """
     index = 0
+
+    if not x or not x.strip():
+        raise ValueError(f"Invalid column name: {x!r}")

     for c in x.upper().strip():
         cp = ord(c)
```