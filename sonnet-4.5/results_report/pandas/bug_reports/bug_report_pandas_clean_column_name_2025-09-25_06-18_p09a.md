# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError Not Caught

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`clean_column_name` raises `TokenError` for input containing null bytes, but the docstring states it catches `SyntaxError` and returns the name unmodified when tokenization fails. This violates the documented behavior.

## Property-Based Test

```python
import tokenize
from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name


@given(st.text())
def test_clean_column_name_no_crash(name):
    try:
        result = clean_column_name(name)
        assert isinstance(result, type(name))
    except SyntaxError:
        pass
    except tokenize.TokenError:
        raise AssertionError(f"TokenError not caught for input: {name!r}")
```

**Failing input**: `'\x00'` (and any string containing null bytes)

## Reproducing the Bug

```python
from pandas.core.computation.parsing import clean_column_name

result = clean_column_name('\x00')
```

Output:
```
tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
```

## Why This Is A Bug

The function's docstring explicitly states:

> For some cases, a name cannot be converted to a valid Python identifier.
> In that case :func:`tokenize_string` raises a SyntaxError.
> In that case, we just return the name unmodified.

However, the function only catches `SyntaxError`, not `TokenError`. When `tokenize_string` encounters a null byte, Python's tokenizer raises `TokenError` instead of `SyntaxError`. This causes the exception to propagate instead of returning the name unmodified as documented.

## Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -127,5 +127,5 @@ def clean_column_name(name: Hashable) -> Hashable:
     try:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```