# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `clean_column_name` function crashes with `TokenError` when given column names containing null bytes, instead of returning the name unmodified as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name

@given(st.text(min_size=1, max_size=50))
def test_clean_column_name_returns_hashable(name):
    """clean_column_name should return a hashable value."""
    result = clean_column_name(name)
    hash(result)
```

**Failing input**: `'\x00'` (null byte)

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

The function's docstring explicitly states:

> For some cases, a name cannot be converted to a valid Python identifier.
> In that case :func:`tokenize_string` raises a SyntaxError.
> In that case, we just return the name unmodified.

However, the function only catches `SyntaxError`, not `TokenError`. When `tokenize_string` encounters null bytes or certain other problematic characters, it raises `TokenError` instead of `SyntaxError`, which is not caught. This violates the documented contract.

## Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -127,7 +127,7 @@ def clean_column_name(name: Hashable) -> Hashable:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```