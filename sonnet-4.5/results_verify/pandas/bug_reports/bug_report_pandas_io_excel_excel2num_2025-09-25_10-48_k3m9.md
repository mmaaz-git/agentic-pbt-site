# Bug Report: pandas.io.excel._excel2num Empty String Handling

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_excel2num` function returns `-1` for empty or whitespace-only strings instead of raising a `ValueError` as documented. This violates the function's contract and can lead to negative column indices being passed to downstream code.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
from pandas.io.excel._util import _excel2num


@given(st.text())
def test_excel2num_never_returns_negative(col_name):
    try:
        result = _excel2num(col_name)
        assert result >= 0, (
            f"_excel2num('{col_name}') returned {result}, "
            "but column indices should never be negative"
        )
    except ValueError:
        pass


def test_excel2num_empty_string_should_raise():
    with pytest.raises(ValueError):
        _excel2num("")
```

**Failing input**: `""` (empty string) and `"   "` (whitespace-only string)

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num

print(_excel2num(""))
print(_excel2num("   "))
```

**Output:**
```
-1
-1
```

**Expected:** Both calls should raise `ValueError` with message like "Invalid column name: "

## Why This Is A Bug

1. **Contract violation**: The docstring states:
   ```
   Raises
   ------
   ValueError
       Part of the Excel column name was invalid.
   ```
   Empty strings are invalid Excel column names, so should raise `ValueError`.

2. **Negative indices don't make sense**: Excel columns are labeled A, B, C, etc. There is no column that maps to index `-1`.

3. **Downstream impact**: The returned `-1` could cause:
   - `IndexError` when used to access arrays/lists
   - Silent bugs if negative indexing wraps around
   - Unexpected behavior in `_range2cols` when parsing user input

4. **Reachable in practice**: Can be triggered by:
   - User input like `usecols="A,,C"` (empty column name between commas)
   - Malformed Excel files
   - Whitespace-only column names

## Fix

```diff
diff --git a/pandas/io/excel/_util.py b/pandas/io/excel/_util.py
index abc123..def456 100644
--- a/pandas/io/excel/_util.py
+++ b/pandas/io/excel/_util.py
@@ -115,6 +115,9 @@ def _excel2num(x: str) -> int:
         Part of the Excel column name was invalid.
     """
     index = 0
+    x_stripped = x.upper().strip()
+    if not x_stripped:
+        raise ValueError(f"Invalid column name: {x}")

-    for c in x.upper().strip():
+    for c in x_stripped:
         cp = ord(c)