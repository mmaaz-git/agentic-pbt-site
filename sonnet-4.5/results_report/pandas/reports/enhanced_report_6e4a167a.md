# Bug Report: pandas.core.computation.parsing.clean_column_name TokenError Exception Not Caught

**Target**: `pandas.core.computation.parsing.clean_column_name`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clean_column_name` function violates its documented contract by crashing with an uncaught `TokenError` when processing strings containing null bytes, instead of returning the name unmodified as promised when tokenization fails.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name

@given(st.text(min_size=1, max_size=20))
def test_clean_column_name_returns_hashable(name):
    result = clean_column_name(name)
    hash(result)

# Run the test
if __name__ == "__main__":
    test_clean_column_name_returns_hashable()
```

<details>

<summary>
**Failing input**: `'\x00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 11, in <module>
    test_clean_column_name_returns_hashable()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 5, in test_clean_column_name_returns_hashable
    def test_clean_column_name_returns_hashable(name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 6, in test_clean_column_name_returns_hashable
    result = clean_column_name(name)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/parsing.py", line 130, in clean_column_name
    tokval = next(tokenized)[1]
             ~~~~^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/parsing.py", line 189, in tokenize_string
    for toknum, tokval, start, _, _ in token_generator:
                                       ^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/tokenize.py", line 588, in _generate_tokens_from_c_tokenizer
    raise TokenError(msg, (e.lineno, e.offset)) from None
tokenize.TokenError: ('source code cannot contain null bytes', (1, 0))
Falsifying example: test_clean_column_name_returns_hashable(
    name='\x00',
)
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
  File "/home/npc/pbt/agentic-pbt/worker_/44/repo.py", line 4, in <module>
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

The `clean_column_name` function's docstring explicitly states in lines 120-122:

> "For some cases, a name cannot be converted to a valid Python identifier. In that case :func:`tokenize_string` raises a SyntaxError. In that case, we just return the name unmodified."

This establishes a clear contract: when tokenization fails, the function should return the original name unmodified. However, the current implementation only catches `SyntaxError` (line 132), not `TokenError`. When Python's tokenizer encounters null bytes (\x00), it raises `tokenize.TokenError` with the message "source code cannot contain null bytes", which propagates uncaught, violating the documented behavior.

The bug occurs because:
1. The function calls `tokenize_string(f"`{name}`")` with a null byte
2. `tokenize_string` uses Python's `tokenize.generate_tokens()` which immediately raises `TokenError` when it encounters the null byte
3. The `except SyntaxError` block doesn't catch `TokenError`, which is a different exception type
4. The function crashes instead of gracefully returning the name unmodified as documented

## Relevant Context

This function is part of pandas' public API and is used internally by `DataFrame.query()` and `DataFrame.eval()` methods to process column names. While null bytes in column names are uncommon, they can appear in:

- Data imported from certain file formats or databases
- Programmatically generated column names with encoding issues
- Data from external APIs or web scraping with malformed strings

The Python tokenize module documentation shows that `TokenError` and `SyntaxError` are distinct exception types:
- `SyntaxError`: Raised for general Python syntax errors
- `TokenError`: Raised specifically by the tokenizer for issues like incomplete strings or null bytes

Source code location: `/pandas/core/computation/parsing.py`, lines 99-133

## Proposed Fix

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