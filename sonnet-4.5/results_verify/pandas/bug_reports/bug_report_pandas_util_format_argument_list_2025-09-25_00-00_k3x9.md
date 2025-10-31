# Bug Report: pandas.util._decorators._format_argument_list Mutates Input List

**Target**: `pandas.util._decorators._format_argument_list`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_format_argument_list` helper function mutates its input list by removing 'self' from it. This causes the `allowed_args` parameter passed to `deprecate_nonkeyword_arguments` decorator to be mutated when the decorated function is called, violating the principle that functions should not have unexpected side effects on their inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.util._decorators import _format_argument_list


@given(st.lists(st.text()))
def test_format_argument_list_does_not_mutate(args_list):
    original = args_list.copy()
    _format_argument_list(args_list)
    assert args_list == original
```

**Failing input**: `['self', 'arg1', 'arg2']`

## Reproducing the Bug

```python
from pandas.util._decorators import deprecate_nonkeyword_arguments
import warnings

allowed = ["self", "x", "y"]
print(f"Before decoration: {allowed}")

@deprecate_nonkeyword_arguments(version="2.0", allowed_args=allowed)
def my_func(self, x, y, z=1):
    return x + y + z

print(f"After decoration: {allowed}")

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    my_func(None, 1, 2, 3)

print(f"After first call: {allowed}")
```

**Output:**
```
Before decoration: ['self', 'x', 'y']
After decoration: ['self', 'x', 'y']
After first call: ['x', 'y']
```

## Why This Is A Bug

The user's `allowed_args` list is unexpectedly mutated when the decorated function is called. This violates basic programming principles:

1. Functions should not have undocumented side effects on their inputs
2. If a user reuses the same list for multiple decorators or stores it for later use, the mutation will cause unexpected behavior
3. The mutation happens at runtime (when calling the decorated function), not at decoration time, making it even more surprising

## Fix

The fix is simple - copy the list before mutating it in `_format_argument_list`:

```diff
--- a/pandas/util/_decorators.py
+++ b/pandas/util/_decorators.py
@@ -243,7 +243,8 @@ def _format_argument_list(allow_args: list[str]) -> str:
     `format_argument_list(['a', 'b', 'c'])` ->
         "except for the arguments 'a', 'b' and 'c'"
     """
-    if "self" in allow_args:
-        allow_args.remove("self")
+    allow_args = allow_args.copy()
+    if "self" in allow_args:
+        allow_args.remove("self")
     if not allow_args:
         return ""
```