# Bug Report: pandas.io.excel._excel2num Returns -1 for Empty String

**Target**: `pandas.io.excel._util._excel2num`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_excel2num` function returns `-1` when given an empty string or whitespace-only string instead of raising a `ValueError`. This invalid return value propagates to other functions like `_range2cols`, potentially causing incorrect behavior when processing Excel column specifications.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from pandas.io.excel._util import _excel2num


@example("")
@example(" ")
@example("  ")
@given(st.text(max_size=10, alphabet=st.characters(whitelist_categories=('Zs',))))
def test_excel2num_empty_or_whitespace_only(text):
    text = text.strip()
    if not text:
        with pytest.raises(ValueError):
            _excel2num(text)
```

**Failing input**: `""` (empty string), `" "` (space), `"  "` (multiple spaces)

## Reproducing the Bug

```python
from pandas.io.excel._util import _excel2num, _range2cols

result_empty = _excel2num("")
print(f"_excel2num(''): {result_empty}")

result_space = _excel2num(" ")
print(f"_excel2num(' '): {result_space}")

result_range = _range2cols(",A")
print(f"_range2cols(',A'): {result_range}")
```

Output:
```
_excel2num(''): -1
_excel2num(' '): -1
_range2cols(',A'): [-1, 0]
```

## Why This Is A Bug

1. **Invalid column index**: Excel columns are 0-indexed starting from 'A' = 0. Returning -1 is semantically invalid.

2. **Inconsistent error handling**: The function raises `ValueError` for invalid characters like digits (`_excel2num("A1")`) but silently returns -1 for empty strings.

3. **Violates documented behavior**: The docstring states the function converts "Excel column name" to index, implying valid column names are required.

4. **Cascading errors**: The bug propagates to `_range2cols`, which can return lists containing -1 when given inputs like `",A"` or `" ,A"`.

## Fix

```diff
def _excel2num(x: str) -> int:
-    index = 0
+    x = x.upper().strip()
+
+    if not x:
+        raise ValueError(f"Invalid column name: {x}")
+
+    index = 0
-    for c in x.upper().strip():
+    for c in x:
        cp = ord(c)

        if cp < ord("A") or cp > ord("Z"):
            raise ValueError(f"Invalid column name: {x}")

        index = index * 26 + cp - ord("A") + 1

    return index - 1
```

This fix:
1. Strips and uppercases the input once at the start
2. Validates that the result is non-empty before processing
3. Raises `ValueError` for empty/whitespace-only strings, consistent with other invalid inputs