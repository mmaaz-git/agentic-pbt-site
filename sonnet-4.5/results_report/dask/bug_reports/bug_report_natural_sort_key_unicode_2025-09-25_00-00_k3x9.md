# Bug Report: natural_sort_key Unicode Digit Crash

**Target**: `dask.utils.natural_sort_key`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`natural_sort_key` crashes with a `ValueError` when passed strings containing Unicode digit characters like '²', '³', '①', etc. The function uses `str.isdigit()` to detect digits, but `int()` only accepts ASCII digits, causing a crash.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.dask_expr.io import natural_sort_key

@given(st.text())
def test_natural_sort_key_deterministic(s):
    result1 = natural_sort_key(s)
    result2 = natural_sort_key(s)
    assert result1 == result2
```

**Failing input**: `'²'`

## Reproducing the Bug

```python
from dask.utils import natural_sort_key

natural_sort_key('²')

natural_sort_key('file²name')

sorted(['file²', 'file1'], key=natural_sort_key)
```

## Why This Is A Bug

The function's signature accepts any string without restrictions, and the docstring doesn't mention Unicode limitations. Unicode digit characters like '²' (superscript 2) pass Python's `str.isdigit()` check but cannot be parsed by `int()`, causing an unhandled `ValueError`. This makes the function fragile when handling arbitrary Unicode input, which is a reasonable expectation for a string sorting utility.

## Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1579,4 +1579,4 @@ def natural_sort_key(s: str) -> list[str | int]:
     >>> sorted(a, key=natural_sort_key)
     ['f0', 'f1', 'f2', 'f8', 'f9', 'f10', 'f11', 'f19', 'f20', 'f21']
     """
-    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s)]
+    return [int(part) if part.isdecimal() else part for part in re.split(r"(\d+)", s)]
```

The fix changes `isdigit()` to `isdecimal()`. The `isdecimal()` method only returns `True` for characters that can be used to form base-10 numbers (0-9), which are exactly the characters that `int()` can parse. This makes the check consistent with what `int()` accepts.