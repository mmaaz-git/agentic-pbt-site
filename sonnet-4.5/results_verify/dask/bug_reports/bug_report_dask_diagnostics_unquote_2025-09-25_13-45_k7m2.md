# Bug Report: dask.diagnostics.profile_visualize.unquote IndexError on Empty Dict

**Target**: `dask.diagnostics.profile_visualize.unquote`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `unquote` function crashes with `IndexError: list index out of range` when processing an empty dict task representation `(dict, [])`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings


@st.composite
def task_dict_strategy(draw):
    num_pairs = draw(st.integers(min_value=0, max_value=10))
    pairs = []
    for _ in range(num_pairs):
        key = draw(st.text(min_size=1, max_size=10))
        value = draw(st.one_of(
            st.integers(),
            st.text(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans()
        ))
        pairs.append([key, value])
    return (dict, pairs)


@given(task_dict_strategy())
@settings(max_examples=200)
def test_unquote_dict_no_crash(task):
    from dask.diagnostics.profile_visualize import unquote
    try:
        result = unquote(task)
        assert isinstance(result, dict)
    except IndexError:
        raise AssertionError(f"unquote crashed with IndexError on input: {task}")
```

**Failing input**: `(dict, [])`

## Reproducing the Bug

```python
from dask.diagnostics.profile_visualize import unquote

expr = (dict, [])
result = unquote(expr)
```

## Why This Is A Bug

In dask's task representation format, `(dict, [])` is the valid representation of an empty dictionary. The `unquote` function should handle this case and return an empty dict `{}`, but instead crashes with an `IndexError` because it attempts to access `expr[1][0]` without first checking if `expr[1]` is non-empty.

This violates the function's contract of converting dask task representations to their corresponding Python objects, and would cause visualization to fail for any task graph containing an empty dict.

## Fix

```diff
--- a/dask/diagnostics/profile_visualize.py
+++ b/dask/diagnostics/profile_visualize.py
@@ -30,7 +30,7 @@ def unquote(expr):
             return expr[0](map(unquote, expr[1]))
         elif (
             expr[0] == dict
             and isinstance(expr[1], list)
-            and isinstance(expr[1][0], list)
+            and (not expr[1] or isinstance(expr[1][0], list))
         ):
             return dict(map(unquote, expr[1]))
     return expr
```