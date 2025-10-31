# Bug Report: pandas.api.types.is_re_compilable Raises Exceptions Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function violates its documented contract by raising `re.PatternError` exceptions for invalid regex patterns instead of returning `False` as promised in the documentation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.api.types as pt
import re


@given(st.text())
def test_is_re_compilable_consistency_with_re_compile(pattern):
    """Test that is_re_compilable returns False for invalid patterns, not raises exceptions."""
    is_compilable = pt.is_re_compilable(pattern)
    try:
        re.compile(pattern)
        can_compile = True
    except re.error:
        can_compile = False
    assert is_compilable == can_compile

# Run the test
if __name__ == "__main__":
    test_is_re_compilable_consistency_with_re_compile()
```

<details>

<summary>
**Failing input**: `'\'` (and others like `'['`, `')'`, `'?'`, `'*'`)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 19, in <module>
  |     test_is_re_compilable_consistency_with_re_compile()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 7, in test_is_re_compilable_consistency_with_re_compile
  |     def test_is_re_compilable_consistency_with_re_compile(pattern):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 5 distinct failures. (5 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 9, in test_is_re_compilable_consistency_with_re_compile
    |     is_compilable = pt.is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_consistency_with_re_compile(
    |     pattern='\\',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 9, in test_is_re_compilable_consistency_with_re_compile
    |     is_compilable = pt.is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_consistency_with_re_compile(
    |     pattern='[',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 9, in test_is_re_compilable_consistency_with_re_compile
    |     is_compilable = pt.is_re_compilable(pattern)
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
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 865, in _parse
    |     raise source.error("missing ), unterminated subpattern",
    |                        source.tell() - start)
    | re.PatternError: missing ), unterminated subpattern at position 0
    | Falsifying example: test_is_re_compilable_consistency_with_re_compile(
    |     pattern='(',
    | )
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 9, in test_is_re_compilable_consistency_with_re_compile
    |     is_compilable = pt.is_re_compilable(pattern)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py", line 188, in is_re_compilable
    |     re.compile(obj)
    |     ~~~~~~~~~~^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 289, in compile
    |     return _compile(pattern, flags)
    |   File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 350, in _compile
    |     p = _compiler.compile(pattern, flags)
    |   File "/home/npc/miniconda/lib/python3.13/re/_compiler.py", line 748, in compile
    |     p = _parser.parse(p, flags)
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 985, in parse
    |     raise source.error("unbalanced parenthesis")
    | re.PatternError: unbalanced parenthesis at position 0
    | Falsifying example: test_is_re_compilable_consistency_with_re_compile(
    |     pattern=')',
    | )
    +---------------- 5 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 9, in test_is_re_compilable_consistency_with_re_compile
    |     is_compilable = pt.is_re_compilable(pattern)
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
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 686, in _parse
    |     raise source.error("nothing to repeat",
    |                        source.tell() - here + len(this))
    | re.PatternError: nothing to repeat at position 0
    | Falsifying example: test_is_re_compilable_consistency_with_re_compile(
    |     pattern='?',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas.api.types as pt

# Test various invalid regex patterns that should return False
# but actually raise re.PatternError exceptions

test_cases = [
    "[",      # unterminated character set
    ")",      # unbalanced parenthesis
    "?",      # nothing to repeat
    "\\",     # bad escape (end of pattern)
    "*",      # nothing to repeat
    "(?",     # incomplete group
    "[a-",    # incomplete character range
]

for pattern in test_cases:
    print(f"Testing pattern: {repr(pattern)}")
    try:
        result = pt.is_re_compilable(pattern)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Exception raised: {type(e).__name__}: {e}")
    print()
```

<details>

<summary>
Exception raised for all invalid regex patterns instead of returning False
</summary>
```
Testing pattern: '['
  Exception raised: PatternError: unterminated character set at position 0

Testing pattern: ')'
  Exception raised: PatternError: unbalanced parenthesis at position 0

Testing pattern: '?'
  Exception raised: PatternError: nothing to repeat at position 0

Testing pattern: '\\'
  Exception raised: PatternError: bad escape (end of pattern) at position 0

Testing pattern: '*'
  Exception raised: PatternError: nothing to repeat at position 0

Testing pattern: '(?'
  Exception raised: PatternError: unexpected end of pattern at position 2

Testing pattern: '[a-'
  Exception raised: PatternError: unterminated character set at position 0

```
</details>

## Why This Is A Bug

The function's docstring at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/inference.py:166-192` explicitly states:

1. **Return type**: "bool - Whether `obj` can be compiled as a regex pattern"
2. **Purpose**: "Check if the object can be compiled into a regex pattern instance"
3. **Expected behavior**: Should return `False` for any object that cannot be compiled as a regex

The current implementation only catches `TypeError` (line 189) for non-string inputs like integers, but fails to catch `re.error` exceptions that occur when string patterns have invalid regex syntax. This violates the documented contract that the function returns a boolean value indicating compilability.

The function name `is_re_compilable` is a predicate that semantically promises a True/False answer without side effects. Users call this function specifically to avoid exceptions from `re.compile()`, making the current behavior counterproductive.

## Relevant Context

- **Location**: `pandas/core/dtypes/inference.py:166-192`
- **Public API**: Exposed via `pandas.api.types.is_re_compilable`
- **Python regex exceptions**: `re.error` is the base class for regex compilation errors, with `re.PatternError` as an alias
- **Current exception handling**: Only catches `TypeError` for non-string types
- **Documentation**: Both inline docstring and official pandas API docs promise bool return type

This bug affects code that uses `is_re_compilable` to safely validate user input or programmatically generated patterns before attempting compilation. The function loses its primary utility as a safe validation check when it can raise the same exceptions it's meant to prevent.

## Proposed Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -186,7 +186,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```