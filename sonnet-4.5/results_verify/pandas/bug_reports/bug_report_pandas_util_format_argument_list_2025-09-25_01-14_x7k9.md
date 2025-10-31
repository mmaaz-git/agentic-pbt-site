# Bug Report: pandas.util._decorators._format_argument_list Mutates Input

**Target**: `pandas.util._decorators._format_argument_list`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_format_argument_list` function mutates its input list by removing 'self', violating the expected behavior that utility functions should not modify their inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.util._decorators import _format_argument_list


@given(st.lists(st.text(min_size=1), min_size=0, max_size=10))
@settings(max_examples=1000)
def test_format_argument_list_no_mutation(args):
    original = args.copy()
    result = _format_argument_list(args)
    assert args == original, f"Should not mutate input. Expected {original}, got {args}"
```

**Failing input**: `['self', 'a', 'b']`

## Reproducing the Bug

```python
from pandas.util._decorators import _format_argument_list

args = ['self', 'a', 'b']
print(f"Before: {args}")
result = _format_argument_list(args)
print(f"After: {args}")
print(f"Result: '{result}'")
```

Output:
```
Before: ['self', 'a', 'b']
After: ['a', 'b']
Result: " except for the arguments 'a' and 'b'"
```

## Why This Is A Bug

The function modifies its input list by calling `allow_args.remove("self")` (line 247). This violates the principle that functions should not have surprising side effects on their inputs. When this function is called by `deprecate_nonkeyword_arguments`, it mutates the `allow_args` list that was passed to the decorator, which can affect multiple calls to the decorated function.

## Fix

```diff
--- a/pandas/util/_decorators.py
+++ b/pandas/util/_decorators.py
@@ -244,8 +244,9 @@ def _format_argument_list(allow_args: list[str]) -> str:
     `format_argument_list(['a', 'b', 'c'])` ->
         "except for the arguments 'a', 'b' and 'c'"
     """
+    allow_args = allow_args.copy()
     if "self" in allow_args:
         allow_args.remove("self")
     if not allow_args:
         return ""
     elif len(allow_args) == 1:
```