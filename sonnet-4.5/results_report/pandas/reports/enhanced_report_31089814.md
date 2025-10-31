# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError on Null Bytes

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `clean_column_name` function crashes with an uncaught `TokenError` when given column names containing null bytes (`\x00`), instead of returning the name unmodified as documented.

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

<details>

<summary>
**Failing input**: `'\x00'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/14
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_clean_column_name_returns_hashable FAILED                  [100%]

=================================== FAILURES ===================================
___________________ test_clean_column_name_returns_hashable ____________________

    @given(st.text(min_size=1, max_size=50))
>   def test_clean_column_name_returns_hashable(name):
                   ^^^

hypo.py:5:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
hypo.py:7: in test_clean_column_name_returns_hashable
    result = clean_column_name(name)
             ^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/parsing.py:130: in clean_column_name
    tokval = next(tokenized)[1]
             ^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/parsing.py:189: in tokenize_string
    for toknum, tokval, start, _, _ in token_generator:
                                       ^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

source = <built-in method readline of _io.StringIO object at 0x751394d96b00>
encoding = None, extra_tokens = True

    def _generate_tokens_from_c_tokenizer(source, encoding=None, extra_tokens=False):
        """Tokenize a source reading Python code as unicode strings using the internal C tokenizer"""
        if encoding is None:
            it = _tokenize.TokenizerIter(source, extra_tokens=extra_tokens)
        else:
            it = _tokenize.TokenizerIter(source, encoding=encoding, extra_tokens=extra_tokens)
        try:
            for info in it:
                yield TokenInfo._make(info)
        except SyntaxError as e:
            if type(e) != SyntaxError:
                raise e from None
            msg = _transform_msg(e.msg)
>           raise TokenError(msg, (e.lineno, e.offset)) from None
E           tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
E           Falsifying example: test_clean_column_name_returns_hashable(
E               name='\x00',
E           )

/home/npc/miniconda/lib/python3.13/tokenize.py:588: TokenError
=========================== short test summary info ============================
FAILED hypo.py::test_clean_column_name_returns_hashable - tokenize.TokenError...
============================== 1 failed in 0.45s ===============================
```
</details>

## Reproducing the Bug

```python
from pandas.core.computation.parsing import clean_column_name

# Test with null byte character
result = clean_column_name('\x00')
print(f"Result: {result}")
```

<details>

<summary>
TokenError: source code cannot contain null bytes
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/repo.py", line 4, in <module>
    result = clean_column_name('\x00')
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

The function's docstring explicitly states in lines 119-122 of `pandas/core/computation/parsing.py`:

> For some cases, a name cannot be converted to a valid Python identifier.
> In that case :func:`tokenize_string` raises a SyntaxError.
> In that case, we just return the name unmodified.

However, the function implementation only catches `SyntaxError` (line 132), not `TokenError`. When Python's `tokenize.generate_tokens` encounters null bytes or certain other problematic characters, it raises `TokenError` instead of `SyntaxError`. This causes an uncaught exception that crashes the function, directly violating the documented contract that promises to "return the name unmodified" when conversion fails.

The error occurs specifically at line 189 in `tokenize_string` when iterating over the token generator. Python's tokenizer raises `TokenError` with the message "source code cannot contain null bytes" when it encounters `\x00` characters.

## Relevant Context

- **Function Location**: `/pandas/core/computation/parsing.py`, lines 99-133
- **Error Origin**: The `tokenize.generate_tokens` function in Python's standard library raises `TokenError` (not `SyntaxError`) for null bytes
- **API Status**: This is an internal function (`pandas.core.computation.parsing`), not part of the public pandas API
- **Python Version**: Confirmed on Python 3.13.2 with pandas installed
- **Impact**: While this is an edge case (null bytes in column names are rare), it represents a contract violation where the function promises graceful error handling but delivers a crash

The function is used internally by pandas for processing column names in query expressions. While null bytes in column names are uncommon, they can occur when reading data from corrupted files or certain binary formats.

## Proposed Fix

```diff
--- a/pandas/core/computation/parsing.py
+++ b/pandas/core/computation/parsing.py
@@ -126,11 +126,12 @@ def clean_column_name(name: Hashable) -> Hashable:
         an error will be raised by :func:`tokenize_backtick_quoted_string` instead,
         which is not caught and propagates to the user level.
     """
+    import tokenize
     try:
         tokenized = tokenize_string(f"`{name}`")
         tokval = next(tokenized)[1]
         return create_valid_python_identifier(tokval)
-    except SyntaxError:
+    except (SyntaxError, tokenize.TokenError):
         return name
```