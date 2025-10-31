# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError Not Caught

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clean_column_name` function crashes with `TokenError` when given strings containing null bytes, violating its documented contract which states it should return the name unmodified when tokenization fails.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name

@given(st.text(min_size=1, max_size=20))
def test_clean_column_name_returns_hashable(name):
    result = clean_column_name(name)
    hash(result)
```

**Failing input**: `'\x00'`

## Reproducing the Bug

```python
from pandas.core.computation.parsing import clean_column_name

result = clean_column_name('\x00')
```

## Why This Is A Bug

The function's docstring states:

> For some cases, a name cannot be converted to a valid Python identifier.
> In that case :func:`tokenize_string` raises a SyntaxError.
> In that case, we just return the name unmodified.

However, the function only catches `SyntaxError`, not `TokenError`. When `tokenize_string` is called with a string containing null bytes, it raises `TokenError` from Python's tokenizer, which is not caught. This violates the documented contract that the function should return the name unmodified when tokenization fails.

## Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -127,7 +127,7 @@ def clean_column_name(name: Hashable) -> Hashable:
     """
     try:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```