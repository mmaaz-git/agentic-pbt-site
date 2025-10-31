# Bug Report: pandas.io.excel._util._excel2num Empty String Handling

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_excel2num` function returns -1 for empty or whitespace-only strings instead of raising a `ValueError`. This invalid column index can cause unexpected behavior when passed to Excel parsers.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.excel._util import _excel2num

@given(st.text(min_size=1))
def test_excel2num_valid_or_error(col_name):
    result = _excel2num(col_name)
    assert result >= 0, f"Column index must be non-negative, got {result}"
```

**Failing input**: `''` (empty string) and `'   '` (whitespace-only string)

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num

result1 = _excel2num('')
print(f"_excel2num('') = {result1}")

result2 = _excel2num('   ')
print(f"_excel2num('   ') = {result2}")

result3 = _excel2num('A')
print(f"_excel2num('A') = {result3}")
```

**Output:**
```
_excel2num('') = -1
_excel2num('   ') = -1
_excel2num('A') = 0
```

## Why This Is A Bug

The function validates that characters are within A-Z range and raises `ValueError` for invalid characters, but it doesn't validate that the input is non-empty after stripping whitespace. A column index of -1 is invalid in Excel's 0-indexed column system. This could cause:

1. When used with Python's negative indexing, `-1` selects the last column instead of an error
2. When used with explicit bounds checking, it may cause index errors
3. It violates the documented behavior that the function converts valid Excel column names

The function already validates other invalid inputs:
```python
if cp < ord("A") or cp > ord("Z"):
    raise ValueError(f"Invalid column name: {x}")
```

Empty strings should be similarly invalid.

## Fix

```diff
def _excel2num(x: str) -> int:
+   x = x.upper().strip()
+   if not x:
+       raise ValueError(f"Invalid column name: {x}")
+
    index = 0
-   for c in x.upper().strip():
+   for c in x:
        cp = ord(c)
        if cp < ord("A") or cp > ord("Z"):
            raise ValueError(f"Invalid column name: {x}")
        index = index * 26 + cp - ord("A") + 1
    return index - 1
```