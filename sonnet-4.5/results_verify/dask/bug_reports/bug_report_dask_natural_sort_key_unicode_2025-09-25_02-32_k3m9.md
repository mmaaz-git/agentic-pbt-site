# Bug Report: dask.utils.natural_sort_key Unicode Digit Crash

**Target**: `dask.dataframe.dask_expr.io.natural_sort_key` (also exported from `dask.utils.natural_sort_key`)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `natural_sort_key` function crashes with a `ValueError` when given strings containing certain Unicode digit characters (e.g., superscript digits like '²', '³', '¹') because it uses `str.isdigit()` to check if a part is a digit, but then calls `int()` which only accepts ASCII digits.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.dask_expr.io import natural_sort_key

@given(s=st.text(min_size=1, max_size=50))
@settings(max_examples=500)
def test_natural_sort_key_returns_list(s):
    result = natural_sort_key(s)
    assert isinstance(result, list)
```

**Failing input**: `'²'`

## Reproducing the Bug

```python
from dask.dataframe.dask_expr.io import natural_sort_key

natural_sort_key('²')
```

Output:
```
ValueError: invalid literal for int() with base 10: '²'
```

The bug occurs because:
1. The regex `\d` in `re.split(r"(\d+)", s)` only matches ASCII digits [0-9]
2. For input '²', the split returns `['²']` (no split occurs)
3. The code checks `'²'.isdigit()` which returns `True`
4. Then it tries `int('²')` which raises `ValueError`

Other Unicode digits that trigger this bug: '³', '¹', and potentially any Unicode character where `isdigit()` returns True but `int()` fails.

## Why This Is A Bug

The function is documented to work with arbitrary strings for natural sorting, and would be expected to handle any valid string input. Crashing on valid Unicode characters violates this expectation. While filenames with superscript digits are uncommon, they are valid and this crash could occur in real-world usage.

## Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1579,7 +1579,7 @@ def natural_sort_key(s: str) -> list[str | int]:
     >>> sorted(a, key=natural_sort_key)
     ['f0', 'f1', 'f2', 'f8', 'f9', 'f10', 'f11', 'f19', 'f20', 'f21']
     """
-    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s)]
+    return [int(part) if part.isascii() and part.isdigit() else part for part in re.split(r"(\d+)", s)]
```

The fix adds a check for `part.isascii()` before attempting `int()` conversion. This ensures that only ASCII digit strings are converted to integers, avoiding the crash on Unicode digit characters.