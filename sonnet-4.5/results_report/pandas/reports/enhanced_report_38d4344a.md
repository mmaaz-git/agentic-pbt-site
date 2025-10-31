# Bug Report: pandas.api.types.is_re_compilable Raises PatternError on Invalid Regex Patterns Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function violates its documented contract by raising `re.PatternError` exceptions when given syntactically invalid regex patterns, instead of returning `False` as expected from a boolean predicate function.

## Property-Based Test

```python
import pandas.api.types as types
import re
from hypothesis import given, strategies as st


@given(st.text(min_size=1))
def test_is_re_compilable_for_valid_patterns(pattern):
    try:
        re.compile(pattern)
        can_compile = True
    except re.error:
        can_compile = False

    result = types.is_re_compilable(pattern)
    assert result == can_compile

if __name__ == "__main__":
    test_is_re_compilable_for_valid_patterns()
```

<details>

<summary>
**Failing input**: `'['` and `'?'`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 18, in <module>
  |     test_is_re_compilable_for_valid_patterns()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 7, in test_is_re_compilable_for_valid_patterns
  |     def test_is_re_compilable_for_valid_patterns(pattern):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 14, in test_is_re_compilable_for_valid_patterns
    |     result = types.is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_for_valid_patterns(
    |     pattern='[',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:550
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:552
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:566
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:567
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 14, in test_is_re_compilable_for_valid_patterns
    |     result = types.is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_for_valid_patterns(
    |     pattern='?',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:642
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:686
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas.api.types as types

# Test with invalid regex patterns that should cause crashes
test_patterns = ['(', ')', '[', '?', '*', '+']

for pattern in test_patterns:
    print(f"Testing pattern: '{pattern}'")
    try:
        result = types.is_re_compilable(pattern)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    print()
```

<details>

<summary>
All test patterns raise PatternError exceptions
</summary>
```
Testing pattern: '('
  Error: PatternError: missing ), unterminated subpattern at position 0

Testing pattern: ')'
  Error: PatternError: unbalanced parenthesis at position 0

Testing pattern: '['
  Error: PatternError: unterminated character set at position 0

Testing pattern: '?'
  Error: PatternError: nothing to repeat at position 0

Testing pattern: '*'
  Error: PatternError: nothing to repeat at position 0

Testing pattern: '+'
  Error: PatternError: nothing to repeat at position 0

```
</details>

## Why This Is A Bug

This violates the expected behavior of `is_re_compilable` in multiple ways:

1. **Function Contract Violation**: The function's docstring explicitly states it returns `bool` to indicate "Whether `obj` can be compiled as a regex pattern." A function that returns bool for some inputs but raises exceptions for others violates the principle of consistent return types.

2. **Python Convention Violation**: Functions beginning with `is_` in Python's standard library and pandas itself consistently return boolean values without raising exceptions for invalid inputs. This is a well-established pattern that users rely on.

3. **Inconsistent Error Handling**: The function correctly catches `TypeError` for non-string types (e.g., `is_re_compilable(1)` returns `False`), demonstrating that the intent is to handle failure cases gracefully. However, it fails to catch `re.error` (or `re.PatternError` in Python 3.11+) for syntactically invalid regex patterns.

4. **Documentation Example Misleading**: The documented example `is_re_compilable(1) → False` demonstrates graceful failure handling, setting the expectation that the function should handle all non-compilable inputs similarly.

## Relevant Context

The current implementation in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/inference.py:187-192` only catches `TypeError`:

```python
def is_re_compilable(obj) -> bool:
    try:
        re.compile(obj)
    except TypeError:
        return False
    else:
        return True
```

This implementation oversight means that while the function handles type errors (non-string inputs), it doesn't handle regex syntax errors (invalid pattern strings). The Python `re` module raises `re.error` (aliased as `re.PatternError` in Python 3.11+) for invalid regex patterns, which is not being caught.

Documentation link: [pandas.api.types.is_re_compilable](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.is_re_compilable.html)

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