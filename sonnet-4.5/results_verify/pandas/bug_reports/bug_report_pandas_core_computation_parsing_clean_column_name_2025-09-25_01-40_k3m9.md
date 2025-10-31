# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError Not Caught

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clean_column_name` function fails to catch `TokenError` exceptions, causing it to crash when given column names containing null bytes, despite its docstring claiming it handles tokenization failures gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.core.computation.parsing as parsing

@given(st.text())
def test_clean_column_name_always_works(name):
    result = parsing.clean_column_name(name)
    assert result is not None
```

**Failing input**: `'\x00'` (or any string containing null bytes)

## Reproducing the Bug

```python
import pandas.core.computation.parsing as parsing

name = '\x00'
result = parsing.clean_column_name(name)
```

Output:
```
tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
```

## Why This Is A Bug

The function's docstring states:

> For some cases, a name cannot be converted to a valid Python identifier.
> In that case :func:`tokenize_string` raises a SyntaxError.
> In that case, we just return the name unmodified.

However, `tokenize_string` can raise `TokenError` (not just `SyntaxError`) when given strings containing null bytes. The function only catches `SyntaxError`, causing it to crash instead of returning the name unmodified as documented.

This violates the function's contract - it should handle all tokenization failures gracefully by returning the name unmodified, but instead crashes on certain inputs.

## Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -127,7 +127,7 @@ def clean_column_name(name: Hashable) -> Hashable:
     try:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```