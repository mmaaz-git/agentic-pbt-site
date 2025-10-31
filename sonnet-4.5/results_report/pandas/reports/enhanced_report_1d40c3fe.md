# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError on Null Bytes

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `clean_column_name` function crashes with `TokenError` when processing column names containing null bytes, violating its documented behavior of returning names unmodified when conversion fails.

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


if __name__ == "__main__":
    test_clean_column_name_idempotent()
```

<details>

<summary>
**Failing input**: `'\x00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 14, in <module>
    test_clean_column_name_idempotent()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 7, in test_clean_column_name_idempotent
    def test_clean_column_name_idempotent(name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 8, in test_clean_column_name_idempotent
    result1 = parsing.clean_column_name(name)
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
import pandas.core.computation.parsing as parsing

# Test case with null byte
name = '\x00'
result = parsing.clean_column_name(name)
```

<details>

<summary>
tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/repo.py", line 5, in <module>
    result = parsing.clean_column_name(name)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/parsing.py", line 130, in clean_column_name
    tokval = next(tokenized)[1]
             ~~~~^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/parsing.py", line 189, in tokenize_string
    for toknum, tokval, start, _, _ in token_generator:
                                       ^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/tokenize.py", line 588, in _generate_tokens_from_c_tokenizer
    raise TokenError(msg, (e.lineno, e.offset)) from None
tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
```
</details>

## Why This Is A Bug

The function's docstring explicitly promises graceful error handling (lines 120-122 of parsing.py):

> "For some cases, a name cannot be converted to a valid Python identifier. In that case :func:`tokenize_string` raises a SyntaxError. In that case, we just return the name unmodified."

However, the implementation only catches `SyntaxError` (line 132), while Python's tokenizer raises `tokenize.TokenError` when encountering null bytes. Since `TokenError` is not a subclass of `SyntaxError` (verified: `issubclass(tokenize.TokenError, SyntaxError)` returns `False`), the exception propagates uncaught, causing a crash instead of returning the name unmodified as documented.

This violates the function's contract and breaks the idempotence property that users might rely on. The function should handle all tokenization failures gracefully, not just `SyntaxError`.

## Relevant Context

- The `clean_column_name` function is used to process column names that might appear in pandas eval/query expressions
- Column names containing null bytes can occur when:
  - Processing data from binary sources or corrupted files
  - Working with improperly encoded/decoded data
  - Handling data from systems with different character encoding standards
- The Python tokenizer's `generate_tokens` function specifically checks for null bytes and raises `TokenError` before any tokenization occurs
- Source code location: `/pandas/core/computation/parsing.py` lines 99-133
- Python documentation on tokenize module: https://docs.python.org/3/library/tokenize.html

## Proposed Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -6,6 +6,7 @@ from __future__ import annotations
 from io import StringIO
 from keyword import iskeyword
 import token
 import tokenize
 from typing import TYPE_CHECKING

@@ -128,7 +129,7 @@ def clean_column_name(name: Hashable) -> Hashable:
     try:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```