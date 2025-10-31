# Bug Report: dask.diagnostics unquote IndexError

**Target**: `dask.diagnostics.profile_visualize.unquote`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `unquote` function crashes with an `IndexError` when given an empty dict task `(dict, [])`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.diagnostics.profile_visualize import unquote


@given(st.lists(st.tuples(st.text(), st.integers())))
def test_unquote_dict_with_tuples(pairs):
    task = (dict, pairs)
    result = unquote(task)
    expected = dict(pairs)
    assert result == expected
```

**Failing input**: `pairs=[]`

## Reproducing the Bug

```python
from dask.diagnostics.profile_visualize import unquote

task = (dict, [])
result = unquote(task)
```

This crashes with:
```
IndexError: list index out of range
```

## Why This Is A Bug

The function checks `isinstance(expr[1][0], list)` on line 33 without first verifying that `expr[1]` is non-empty. When an empty list is passed, accessing `expr[1][0]` raises an `IndexError`.

An empty dict task `(dict, [])` should be converted to an empty dict `{}`, not crash.

## Fix

```diff
--- a/dask/diagnostics/profile_visualize.py
+++ b/dask/diagnostics/profile_visualize.py
@@ -30,7 +30,7 @@ def unquote(expr):
         elif (
             expr[0] == dict
             and isinstance(expr[1], list)
-            and isinstance(expr[1][0], list)
+            and (not expr[1] or isinstance(expr[1][0], list))
         ):
             return dict(map(unquote, expr[1]))
     return expr
```