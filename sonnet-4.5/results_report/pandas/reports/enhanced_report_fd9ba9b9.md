# Bug Report: pandas.api.types.is_re_compilable Raises Exception on Invalid Regex Patterns Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function violates its documented contract by raising `re.PatternError` exceptions when given syntactically invalid regex patterns, instead of returning `False` as its type signature promises.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.api import types
import re


@given(st.text(min_size=1, max_size=100))
def test_is_re_compilable_should_not_raise(pattern_str):
    result = types.is_re_compilable(pattern_str)

    assert isinstance(result, bool)

    if result:
        re.compile(pattern_str)


# Run the test
if __name__ == "__main__":
    test_is_re_compilable_should_not_raise()
```

<details>

<summary>
**Failing input**: `'['`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 18, in <module>
    test_is_re_compilable_should_not_raise()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 7, in test_is_re_compilable_should_not_raise
    def test_is_re_compilable_should_not_raise(pattern_str):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 8, in test_is_re_compilable_should_not_raise
    result = types.is_re_compilable(pattern_str)
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
Falsifying example: test_is_re_compilable_should_not_raise(
    pattern_str='[',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/re/_constants.py:38
        /home/npc/miniconda/lib/python3.13/re/_parser.py:549
        /home/npc/miniconda/lib/python3.13/re/_parser.py:552
        /home/npc/miniconda/lib/python3.13/re/_parser.py:566
        /home/npc/miniconda/lib/python3.13/re/_parser.py:567
```
</details>

## Reproducing the Bug

```python
from pandas.api.types import is_re_compilable

# Test invalid regex pattern that should crash
pattern = '?'
result = is_re_compilable(pattern)
print(f"Result for pattern '{pattern}': {result}")
```

<details>

<summary>
re.PatternError: nothing to repeat at position 0
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/repo.py", line 5, in <module>
    result = is_re_compilable(pattern)
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
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 686, in _parse
    raise source.error("nothing to repeat",
                       source.tell() - here + len(this))
re.PatternError: nothing to repeat at position 0
```
</details>

## Why This Is A Bug

This function violates its documented contract in multiple ways:

1. **Type Signature Violation**: The function has an explicit return type annotation of `-> bool`, promising to always return a boolean value. However, it raises `re.PatternError` (a subclass of `re.error`) for syntactically invalid regex patterns.

2. **Documentation Contract**: The docstring states the function "Check[s] if the object can be compiled into a regex pattern instance" and returns a `bool` indicating "Whether `obj` can be compiled as a regex pattern." There is no mention of exceptions in the documentation - no "Raises" section exists.

3. **Function Naming Convention**: The function follows the `is_*` predicate naming pattern, which in Python conventionally indicates a safe boolean check that returns True/False without raising exceptions. Other functions in `pandas.api.types` like `is_float()`, `is_integer()`, and `is_bool()` follow this convention and handle invalid inputs gracefully.

4. **Incomplete Error Handling**: The current implementation only catches `TypeError` (for non-string objects like integers) but fails to catch `re.error` and its subclasses like `re.PatternError` that occur when regex compilation fails due to syntax errors. This represents incomplete error handling rather than intentional design.

5. **Purpose Defeat**: The function's purpose is to safely check if an object can be compiled as regex before attempting compilation. By crashing on invalid patterns, it defeats its own purpose - developers cannot use it to validate potentially invalid regex patterns from user input or external sources.

## Relevant Context

- **Pandas Version**: 2.3.2
- **Python Version**: 3.13
- **Source Location**: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py` (lines 187-192)
- **Documentation**: https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_re_compilable.html

Common invalid regex patterns that trigger this bug:
- `'?'`, `'*'`, `'+'` - quantifiers with nothing to repeat
- `'['` - unterminated character set
- `'('` - unterminated subpattern
- `')'` - unbalanced parenthesis
- `'\'` - incomplete escape sequence

The bug affects any code that uses `is_re_compilable` to validate user-provided or dynamically generated regex patterns before compilation, forcing developers to wrap the function in try-except blocks, which defeats its purpose as a validation utility.

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