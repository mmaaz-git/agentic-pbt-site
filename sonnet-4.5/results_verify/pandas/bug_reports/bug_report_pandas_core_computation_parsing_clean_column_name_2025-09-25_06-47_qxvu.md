# Bug Report: pandas.core.computation.parsing.clean_column_name Exception Handling

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clean_column_name` function fails to handle `TokenError` exceptions, violating its documented behavior of returning the name unmodified when conversion fails.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pandas.core.computation.parsing import clean_column_name

@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_clean_column_name_idempotent(name):
    try:
        first = clean_column_name(name)
        second = clean_column_name(first)
        assert first == second
    except (SyntaxError, TypeError):
        pass
```

**Failing input**: `'\x00'` (null byte)

## Reproducing the Bug

```python
from pandas.core.computation.parsing import clean_column_name
import tokenize

name = '\x00'
result = clean_column_name(name)
```

Output:
```
tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
```

Expected: Should return `'\x00'` unmodified (as per docstring).

## Why This Is A Bug

1. The function's docstring explicitly states: "For some cases, a name cannot be converted to a valid Python identifier. In that case :func:`tokenize_string` raises a SyntaxError. In that case, we just return the name unmodified."

2. However, the implementation only catches `SyntaxError`, not `TokenError`.

3. Python's tokenizer raises `TokenError` (not a subclass of `SyntaxError`) for null bytes, causing an uncaught exception.

4. Pandas DataFrames can have column names with null bytes, making this a valid input:
```python
import pandas as pd
df = pd.DataFrame({'\x00': [1, 2, 3]})  # Works fine
```

## Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -1,6 +1,7 @@
 from __future__ import annotations

 from io import StringIO
+import tokenize
 from keyword import iskeyword
 import token
-import tokenize
@@ -127,7 +128,7 @@ def clean_column_name(name: Hashable) -> Hashable:
     try:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```