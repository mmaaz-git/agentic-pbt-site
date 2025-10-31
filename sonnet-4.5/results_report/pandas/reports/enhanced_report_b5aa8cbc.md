# Bug Report: pandas.core.computation.parsing.clean_column_name Uncaught TokenError Exception

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clean_column_name` function crashes with `TokenError` when processing null bytes, violating its documented contract to return names unmodified when conversion fails.

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

if __name__ == "__main__":
    test_clean_column_name_idempotent()
```

<details>

<summary>
**Failing input**: `'\x00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 15, in <module>
    test_clean_column_name_idempotent()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 5, in test_clean_column_name_idempotent
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 8, in test_clean_column_name_idempotent
    first = clean_column_name(name)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/parsing.py", line 130, in clean_column_name
    tokval = next(tokenized)[1]
             ~~~~^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/parsing.py", line 189, in tokenize_string
    for toknum, tokval, start, _, _ in token_generator:
                                       ^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/tokenize.py", line 588, in _generate_tokens_from_c_tokenizer
    raise TokenError(msg, (e.lineno, e.offset)) from None
tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
Falsifying example: test_clean_column_name_idempotent(
    name='\x00',
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.computation.parsing import clean_column_name

# Test with null byte
name = '\x00'
print(f"Testing clean_column_name with null byte: repr(name) = {repr(name)}")

try:
    result = clean_column_name(name)
    print(f"Result: {repr(result)}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Also verify that pandas DataFrames can have null byte column names
import pandas as pd
print("\nVerifying pandas DataFrame can have null byte column names:")
df = pd.DataFrame({'\x00': [1, 2, 3]})
print(f"DataFrame created with null byte column name: {list(df.columns)}")
print(f"Column name repr: {repr(df.columns[0])}")
```

<details>

<summary>
TokenError: 'source code cannot contain null bytes'
</summary>
```
Testing clean_column_name with null byte: repr(name) = '\x00'
Exception raised: TokenError: ('source code cannot contain null bytes', (1, 0))

Verifying pandas DataFrame can have null byte column names:
DataFrame created with null byte column name: ['\x00']
Column name repr: '\x00'
```
</details>

## Why This Is A Bug

The `clean_column_name` function violates its documented contract in multiple ways:

1. **Documentation Promise**: The function's docstring (lines 120-122 in parsing.py) explicitly states: "For some cases, a name cannot be converted to a valid Python identifier. In that case :func:`tokenize_string` raises a SyntaxError. In that case, we just return the name unmodified."

2. **Incomplete Exception Handling**: The implementation (line 132) only catches `SyntaxError`, but Python's tokenizer raises `TokenError` for null bytes. `TokenError` is NOT a subclass of `SyntaxError` - it inherits directly from `Exception`.

3. **Valid Input Rejected**: Pandas DataFrames explicitly support null bytes in column names (demonstrated above), making this a legitimate input that should be handled gracefully.

4. **Function Purpose Violated**: The function is meant to be a safe wrapper that either converts names to valid Python identifiers or returns them unmodified. Crashing defeats this purpose.

## Relevant Context

This function is used internally by pandas for DataFrame.eval() and query() operations. It's called from:
- `pandas.core.generic.NDFrame._get_index_resolvers()`
- `pandas.core.generic.NDFrame._get_axis_resolvers()`

These are used to resolve column names when evaluating Python expressions on DataFrames. While null bytes in column names are rare, they are valid in pandas and the function should handle them gracefully according to its documented behavior.

The bug occurs because Python's tokenize.generate_tokens() raises `TokenError` specifically for null bytes with the message "source code cannot contain null bytes". This is a different exception type than the `SyntaxError` that the function expects and catches.

Documentation: https://docs.python.org/3/library/tokenize.html#tokenize.TokenError
Source code: /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/parsing.py:99-133

## Proposed Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -5,7 +5,7 @@ from __future__ import annotations

 from io import StringIO
 from keyword import iskeyword
 import token
 import tokenize
 from typing import TYPE_CHECKING
@@ -126,11 +126,11 @@ def clean_column_name(name: Hashable) -> Hashable:
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