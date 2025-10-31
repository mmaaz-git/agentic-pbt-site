# Bug Report: pandas.api.types.is_re_compilable Raises PatternError Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable()` function raises `re.PatternError` for invalid regex patterns instead of returning `False` as documented, violating its contract as a boolean predicate function.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.api.types as pat

@given(st.text())
def test_is_re_compilable(text):
    result = pat.is_re_compilable(text)
    if result:
        import re
        try:
            re.compile(text)
        except Exception as e:
            raise AssertionError(f"is_re_compilable returned True but re.compile failed: {e}")

# Run the test
if __name__ == "__main__":
    test_is_re_compilable()
```

<details>

<summary>
**Failing input**: `'['` and `'\'`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 16, in <module>
  |     test_is_re_compilable()
  |     ~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 5, in test_is_re_compilable
  |     def test_is_re_compilable(text):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 6, in test_is_re_compilable
    |     result = pat.is_re_compilable(text)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py", line 188, in is_re_compilable
    |     re.compile(obj)
    |     ~~~~~~~~~~^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 289, in compile
    |     return _compile(pattern, flags)
    |   File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 350, in _compile
    |     p = _compiler.compile(pattern, flags)
    |   File "/home/npc/miniconda/lib/python3.13/re/_compiler.py", line 748, in compile
    |     p = _parser.parse(p, flags)
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 973, in parse
    |     source = Tokenizer(str)
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 238, in __init__
    |     self.__next()
    |     ~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 251, in __next
    |     raise error("bad escape (end of pattern)",
    |                 self.string, len(self.string) - 1) from None
    | re.PatternError: bad escape (end of pattern) at position 0
    | Falsifying example: test_is_re_compilable(
    |     text='\\',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:250
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 6, in test_is_re_compilable
    |     result = pat.is_re_compilable(text)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py", line 188, in is_re_compilable
    |     re.compile(obj)
    |     ~~~~~~~~~~^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 289, in compile
    |     return _compile(pattern, flags)
    |   File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 350, in _compile
    |     p = _compiler.compile(pattern, flags)
    |   File "/home/npc/miniconda/lib/python3.13/re/_compiler.py", line 748, in compile
    |     p = _parser.parse(p, flags)
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 980, in parse
    |     p = _parse_sub(source, state, flags & SRE_FLAG_VERBOSE, 0)
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 459, in _parse_sub
    |     itemsappend(_parse(source, state, verbose, nested + 1,
    |                 ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                        not nested and not items))
    |                        ^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 567, in _parse
    |     raise source.error("unterminated character set",
    |                        source.tell() - here)
    | re.PatternError: unterminated character set at position 0
    | Falsifying example: test_is_re_compilable(
    |     text='[',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:550
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:552
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:566
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:567
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas.api.types as pat

# Test with invalid regex pattern
result = pat.is_re_compilable('[')
print(f"Result: {result}")
```

<details>

<summary>
PatternError: unterminated character set
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/repo.py", line 4, in <module>
    result = pat.is_re_compilable('[')
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py", line 188, in is_re_compilable
    re.compile(obj)
    ~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 289, in compile
    return _compile(pattern, flags)
  File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 350, in _compile
    p = _compiler.compile(pattern, flags)
  File "/home/npc/miniconda/lib/python3.13/re/_compiler.py", line 748, in compile
    p = _parser.parse(p, flags)
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 980, in parse
    p = _parse_sub(source, state, flags & SRE_FLAG_VERBOSE, 0)
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 459, in _parse_sub
    itemsappend(_parse(source, state, verbose, nested + 1,
                ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       not nested and not items))
                       ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 567, in _parse
    raise source.error("unterminated character set",
                       source.tell() - here)
re.PatternError: unterminated character set at position 0
```
</details>

## Why This Is A Bug

This violates the documented behavior in multiple ways:

1. **Documentation Contract Violation**: The function's docstring explicitly states it returns `bool` - "Whether `obj` can be compiled as a regex pattern." There is no mention of exceptions being raised for invalid patterns.

2. **Semantic Inconsistency**: The function name `is_re_compilable` follows the Python convention of `is_*` predicate functions, which are expected to safely return True/False without raising exceptions. The entire purpose of such a function is to provide a safe way to check validity.

3. **Incomplete Exception Handling**: The current implementation (lines 187-192 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py`) only catches `TypeError` but misses `re.error` (also known as `re.PatternError` in Python 3.13), which `re.compile()` raises for syntactically invalid patterns.

4. **API Expectation Mismatch**: Users of this public API function (`pandas.api.types`) expect it to return False for any non-compilable input, including invalid regex syntax. The current behavior forces users to wrap the function in their own exception handling, defeating its purpose.

## Relevant Context

The issue affects common invalid regex patterns that users might encounter:
- `'['` - unterminated character set
- `'\'` - bad escape at end of pattern
- `'('` - missing closing parenthesis
- `'*'`, `'+'`, `'?'` - nothing to repeat
- `'[abc'` - unterminated character set
- And many other syntactically invalid patterns

The function is located in pandas' public API (`pandas.api.types`) and is likely used by applications that need to validate whether user input or data can be used as regex patterns. The current behavior can cause unexpected crashes in production code.

Documentation references:
- Function source: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py:165-192`
- Public API import: `pandas.api.types.is_re_compilable`

## Proposed Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
     False
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```