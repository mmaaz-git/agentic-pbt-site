# Bug Report: pandas.io.excel._util._excel2num Empty String

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_excel2num` function returns -1 for empty strings and whitespace-only strings instead of raising a `ValueError` as documented in its contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.excel._util import _excel2num

@given(st.text(max_size=10))
def test_excel2num_validates_input(text):
    # Only valid Excel column names should succeed
    stripped = text.strip().upper()
    if not stripped or not all('A' <= c <= 'Z' for c in stripped):
        # Invalid input should raise ValueError
        try:
            _excel2num(text)
            assert False, f"Should have raised ValueError for {repr(text)}"
        except ValueError:
            pass  # Expected
    else:
        # Valid input should work
        result = _excel2num(text)
        assert result >= 0
```

**Failing input**: `''` (empty string) and `'   '` (whitespace-only)

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num

result = _excel2num('')
print(f"_excel2num('') = {result}")

result = _excel2num('   ')
print(f"_excel2num('   ') = {result}")
```

**Output:**
```
_excel2num('') = -1
_excel2num('   ') = -1
```

## Why This Is A Bug

The function's docstring explicitly states:

```
Raises
------
ValueError
    Part of the Excel column name was invalid.
```

An empty string is clearly an invalid Excel column name, so the function should raise `ValueError`. Instead, it returns -1, which:

1. Violates the documented API contract
2. Returns a "valid" (though nonsensical) column index that could be used incorrectly
3. Makes error detection harder - callers expecting ValueError won't catch the error

The bug occurs because the function doesn't validate that the input contains at least one character after calling `strip()`. When the `for` loop doesn't execute (empty string after strip), `index` remains 0, and the function returns `index - 1 = -1`.

## Fix

```diff
diff --git a/pandas/io/excel/_util.py b/pandas/io/excel/_util.py
index 1234567..abcdefg 100644
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -116,7 +116,10 @@ def _excel2num(x: str) -> int:
         Part of the Excel column name was invalid.
     """
     index = 0
+    x_stripped = x.upper().strip()

-    for c in x.upper().strip():
+    if not x_stripped:
+        raise ValueError(f"Invalid column name: {x}")
+
+    for c in x_stripped:
         cp = ord(c)

         if cp < ord("A") or cp > ord("Z"):
```