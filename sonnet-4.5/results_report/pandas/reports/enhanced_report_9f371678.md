# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError Not Caught

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `clean_column_name` function crashes with `TokenError` when given column names containing null bytes, violating its documented contract to handle tokenization failures gracefully by returning the name unmodified.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.core.computation.parsing as parsing

@given(st.text())
def test_clean_column_name_always_works(name):
    result = parsing.clean_column_name(name)
    assert result is not None

# Run the test
test_clean_column_name_always_works()
```

<details>

<summary>
**Failing input**: `'\x00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 10, in <module>
    test_clean_column_name_always_works()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 5, in test_clean_column_name_always_works
    def test_clean_column_name_always_works(name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_clean_column_name_always_works
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
Falsifying example: test_clean_column_name_always_works(
    name='\x00',
)
```
</details>

## Reproducing the Bug

```python
import pandas.core.computation.parsing as parsing

name = '\x00'
result = parsing.clean_column_name(name)
print(f"Result: {result}")
```

<details>

<summary>
TokenError: source code cannot contain null bytes
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/repo.py", line 4, in <module>
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

The function's docstring explicitly states in lines 119-123:

> For some cases, a name cannot be converted to a valid Python identifier.
> In that case :func:`tokenize_string` raises a SyntaxError.
> In that case, we just return the name unmodified.

The documentation establishes a clear contract: when tokenization fails (indicated by `tokenize_string` raising an exception), the function should return the original name unmodified rather than propagating the error.

However, the implementation only catches `SyntaxError` (line 132), while Python's tokenizer can also raise `TokenError` when encountering null bytes. This occurs because:

1. The function calls `tokenize_string(f"`{name}`")` which uses Python's `tokenize.generate_tokens()`
2. When `generate_tokens()` encounters a null byte (`\x00`), it raises `TokenError` with the message "source code cannot contain null bytes"
3. This `TokenError` is not caught by the `except SyntaxError:` clause, causing the function to crash

This violates the documented behavior. The Notes section makes it clear that tokenization failures should result in returning the unmodified name, not crashing. The distinction between `SyntaxError` and `TokenError` appears to be an oversight in the implementation.

## Relevant Context

- This is an internal pandas function used by the eval/query functionality for handling column names that may not be valid Python identifiers
- The function is located in `/pandas/core/computation/parsing.py` at lines 99-133
- Null bytes in column names can occur from corrupted data files, encoding issues, or data transferred from systems that don't properly sanitize strings
- The Python tokenizer module raises different exceptions for different error conditions:
  - `SyntaxError`: for syntax issues like unclosed strings or invalid syntax
  - `TokenError`: for tokenization issues like null bytes or encoding problems
- Documentation: https://pandas.pydata.org/docs/reference/api/pandas.eval.html

## Proposed Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -7,6 +7,7 @@ from io import StringIO
 from keyword import iskeyword
 import token
 import tokenize
+from tokenize import TokenError
 from typing import TYPE_CHECKING

 if TYPE_CHECKING:
@@ -129,7 +130,7 @@ def clean_column_name(name: Hashable) -> Hashable:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, TokenError):
         return name
```