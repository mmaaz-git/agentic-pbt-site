# Bug Report: pandas.io.excel._util._excel2num Whitespace Input

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_excel2num` function returns `-1` for whitespace-only or empty strings instead of raising `ValueError` as documented. This violates the function's contract and could cause silent failures in downstream code.

## Property-Based Test

```python
import string
from hypothesis import given, strategies as st
from pandas.io.excel._util import _excel2num
import pytest

@given(st.text(min_size=1, max_size=5))
def test_excel2num_invalid_chars_raise_error(text):
    """_excel2num should raise ValueError for invalid characters"""
    if not all(c.isalpha() for c in text):
        with pytest.raises(ValueError, match="Invalid column name"):
            _excel2num(text)
```

**Failing input**: `' '` (single space), `''` (empty string), `'\t'` (tab), `'\n'` (newline), and any whitespace-only string

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num

result = _excel2num(' ')
print(f"Result: {result}")

result = _excel2num('')
print(f"Result: {result}")
```

Expected: `ValueError` with message "Invalid column name: ..."
Actual: Returns `-1` (no exception raised)

## Why This Is A Bug

1. **Contract violation**: The docstring explicitly states "Raises ValueError: Part of the Excel column name was invalid"
2. **Invalid input**: Whitespace-only or empty strings are not valid Excel column names
3. **Silent failure**: Returning `-1` (an invalid column index) instead of raising an error could cause bugs in downstream code that uses this value
4. **Inconsistent behavior**: The function correctly raises `ValueError` for other invalid inputs like `'A B'` (contains space) but not for whitespace-only strings

## Fix

The issue is that the function calls `.strip()` on the input before validating, which converts whitespace-only strings to empty strings. When the string is empty, the for loop never executes, leaving `index = 0`, and the function returns `0 - 1 = -1`.

Fix: Add a check for empty strings after stripping:

```diff
def _excel2num(x: str) -> int:
    """
    Convert Excel column name like 'AB' to 0-based column index.

    Parameters
    ----------
    x : str
        The Excel column name to convert to a 0-based column index.

    Returns
    -------
    num : int
        The column index corresponding to the name.

    Raises
    ------
    ValueError
        Part of the Excel column name was invalid.
    """
    index = 0

+   stripped = x.upper().strip()
+   if not stripped:
+       raise ValueError(f"Invalid column name: {x}")
+
-   for c in x.upper().strip():
+   for c in stripped:
        cp = ord(c)

        if cp < ord("A") or cp > ord("Z"):
            raise ValueError(f"Invalid column name: {x}")

        index = index * 26 + cp - ord("A") + 1

    return index - 1
```
