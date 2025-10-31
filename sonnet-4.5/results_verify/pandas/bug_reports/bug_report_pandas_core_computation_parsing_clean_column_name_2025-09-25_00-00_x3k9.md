# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `clean_column_name` function crashes with `TokenError` when given column names containing null bytes, violating its documented behavior of returning names unmodified when they cannot be converted.

## Property-Based Test

```python
import hypothesis.strategies as st
from hypothesis import given
import pandas.core.computation.parsing as parsing


@given(st.text())
def test_clean_column_name_idempotent(name):
    result1 = parsing.clean_column_name(name)
    result2 = parsing.clean_column_name(result1)
    assert result1 == result2
```

**Failing input**: `'\x00'`

## Reproducing the Bug

```python
import pandas.core.computation.parsing as parsing

name = '\x00'
result = parsing.clean_column_name(name)
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

However, `tokenize_string` can raise `TokenError` (not a subclass of `SyntaxError`) when the input contains null bytes. The function only catches `SyntaxError`, causing it to crash instead of returning the name unmodified as documented.

## Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -1,6 +1,7 @@
 from __future__ import annotations

 from io import StringIO
 from keyword import iskeyword
 import token
 import tokenize
@@ -128,9 +129,9 @@ def clean_column_name(name: Hashable) -> Hashable:
     try:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```