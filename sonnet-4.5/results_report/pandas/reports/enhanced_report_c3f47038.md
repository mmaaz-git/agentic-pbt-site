# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError on Null Byte Input

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `clean_column_name` function crashes with `TokenError` when processing column names containing null bytes (`\x00`), violating its documented contract that promises to return the name unmodified when tokenization fails.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name


@given(st.text(min_size=1, max_size=50))
def test_clean_column_name_idempotent(name):
    result1 = clean_column_name(name)
    result2 = clean_column_name(result1)
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
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 13, in <module>
    test_clean_column_name_idempotent()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 6, in test_clean_column_name_idempotent
    def test_clean_column_name_idempotent(name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 7, in test_clean_column_name_idempotent
    result1 = clean_column_name(name)
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

name_with_null = '\x00'
result = clean_column_name(name_with_null)
print(f"Result: {result}")
```

<details>

<summary>
TokenError: source code cannot contain null bytes
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/repo.py", line 4, in <module>
    result = clean_column_name(name_with_null)
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

The `clean_column_name` function's docstring explicitly promises graceful handling of tokenization failures:

> "For some cases, a name cannot be converted to a valid Python identifier.
> In that case :func:`tokenize_string` raises a SyntaxError.
> In that case, we just return the name unmodified."

However, the implementation has a critical oversight:

1. **The function only catches `SyntaxError`** (line 132 in pandas/core/computation/parsing.py)
2. **Python's tokenizer raises `TokenError` for null bytes**, not `SyntaxError`
3. **`TokenError` is NOT a subclass of `SyntaxError`** - they are separate exception types that both inherit directly from `Exception`

This means when `tokenize.generate_tokens()` encounters a null byte character, it raises a `TokenError` with the message "source code cannot contain null bytes". Since the function only catches `SyntaxError`, this `TokenError` propagates up to the caller, causing a crash instead of returning the name unmodified as documented.

The bug violates the principle of idempotency - applying the function twice should produce the same result, but instead it crashes on the first application when given a null byte.

## Relevant Context

The `clean_column_name` function is used internally by pandas in `DataFrame._get_index_resolvers()` and `DataFrame._get_cleaned_column_resolvers()` methods (see pandas/core/generic.py). These methods prepare column names for use in DataFrame.eval() expressions by cleaning special characters and making them valid Python identifiers where possible.

While null bytes in column names are rare in practice, pandas explicitly supports arbitrary hashable objects as column names. The function should handle all possible string inputs gracefully, as promised by its documentation.

Python's tokenize module documentation: https://docs.python.org/3/library/tokenize.html
- `TokenError` is raised when source code contains null bytes (among other cases)
- `TokenError` and `SyntaxError` are distinct exception types

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