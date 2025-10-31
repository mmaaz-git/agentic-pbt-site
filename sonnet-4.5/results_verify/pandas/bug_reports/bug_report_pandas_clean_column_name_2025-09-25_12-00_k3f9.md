# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clean_column_name` function crashes with a `TokenError` when passed a column name containing null bytes, violating its documented behavior of returning the name unmodified when tokenization fails.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name


@given(st.text(min_size=1, max_size=50))
def test_clean_column_name_idempotent(name):
    result1 = clean_column_name(name)
    result2 = clean_column_name(result1)
    assert result1 == result2
```

**Failing input**: `'\x00'`

## Reproducing the Bug

```python
from pandas.core.computation.parsing import clean_column_name

name_with_null = '\x00'
result = clean_column_name(name_with_null)
```

This raises:
```
tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
```

## Why This Is A Bug

The function's docstring explicitly states:

> For some cases, a name cannot be converted to a valid Python identifier.
> In that case :func:`tokenize_string` raises a SyntaxError.
> In that case, we just return the name unmodified.

However, the implementation only catches `SyntaxError`, not `TokenError`. When Python's tokenizer encounters a null byte, it raises `TokenError`, not `SyntaxError`. This violates the documented contract that the function should gracefully handle tokenization failures by returning the name unmodified.

## Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -126,7 +126,7 @@ def clean_column_name(name: Hashable) -> Hashable:
         an error will be raised by :func:`tokenize_backtick_quoted_string` instead,
         which is not caught and propagates to the user level.
     """
     try:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```