# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`clean_column_name` crashes with `TokenError` when given column names containing null bytes, instead of returning the name unmodified as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.core.computation.parsing import clean_column_name
import tokenize


@given(st.text(alphabet=st.sampled_from('\x00abc'), min_size=1, max_size=10))
def test_clean_column_name_handles_null_bytes(name):
    assume('\x00' in name)
    try:
        result = clean_column_name(name)
    except tokenize.TokenError:
        raise AssertionError(
            "clean_column_name should catch TokenError and return name "
            "unmodified, but it doesn't"
        )
```

**Failing input**: `'\x00'`

## Reproducing the Bug

```python
from pandas.core.computation.parsing import clean_column_name

result = clean_column_name('\x00')
```

**Output:**
```
tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
```

## Why This Is A Bug

The function documentation explicitly states:

> For some cases, a name cannot be converted to a valid Python identifier.
> In that case :func:`tokenize_string` raises a SyntaxError.
> In that case, we just return the name unmodified.

However, the function only catches `SyntaxError`, not `tokenize.TokenError`. When the tokenizer encounters null bytes, it raises `TokenError` instead of `SyntaxError`, causing the function to crash rather than returning the name unmodified as documented.

This violates the function's contract: it should gracefully handle names that cannot be converted to valid Python identifiers by returning them unmodified.

## Fix

```diff
diff --git a/pandas/core/computation/parsing.py b/pandas/core/computation/parsing.py
index 1234567..abcdefg 100644
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -6,6 +6,7 @@ from __future__ import annotations
 from io import StringIO
 from keyword import iskeyword
 import token
 import tokenize
 from typing import TYPE_CHECKING

@@ -128,6 +129,6 @@ def clean_column_name(name: Hashable) -> Hashable:
     try:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```