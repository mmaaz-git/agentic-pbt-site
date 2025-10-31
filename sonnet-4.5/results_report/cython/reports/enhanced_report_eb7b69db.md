# Bug Report: pandas.core.dtypes.inference.is_re_compilable raises exceptions on invalid regex patterns

**Target**: `pandas.core.dtypes.inference.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function raises `PatternError` exceptions when given invalid regex patterns, violating its documented contract to return a boolean value.

## Property-Based Test

```python
from pandas.core.dtypes.inference import is_re_compilable
from hypothesis import given, strategies as st
import re
import pytest


@given(st.text(min_size=1, max_size=50, alphabet='abcdefghijklmnopqrstuvwxyz0123456789.*+?[]()'))
def test_is_re_compilable_never_crashes(pattern):
    try:
        re.compile(pattern)
        expected = True
    except re.error:
        expected = False

    try:
        result = is_re_compilable(pattern)
        assert result == expected, f"is_re_compilable should match re.compile for pattern '{pattern}'"
    except Exception as e:
        print(f"Exception raised for pattern '{pattern}': {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    # Run the test
    test_is_re_compilable_never_crashes()
```

<details>

<summary>
**Failing input**: `'('` (and other invalid patterns like `')'`, `'['`, `'?'`)
</summary>
```
Exception raised for pattern 'xq[l': PatternError: unterminated character set at position 2
Exception raised for pattern 'tmc[mj7y[bs.iq5': PatternError: unterminated character set at position 3
Exception raised for pattern 'h(x6zpetz': PatternError: missing ), unterminated subpattern at position 1
Exception raised for pattern '?': PatternError: nothing to repeat at position 0
Exception raised for pattern '(y[7[i0+l': PatternError: unterminated character set at position 2
Exception raised for pattern '3u((w': PatternError: missing ), unterminated subpattern at position 3
Exception raised for pattern '?': PatternError: nothing to repeat at position 0
Exception raised for pattern 'xq[l': PatternError: unterminated character set at position 2
Exception raised for pattern '3u((w': PatternError: missing ), unterminated subpattern at position 3
Exception raised for pattern 'xq[': PatternError: unterminated character set at position 2
Exception raised for pattern 'x[': PatternError: unterminated character set at position 1
Exception raised for pattern '[': PatternError: unterminated character set at position 0
Exception raised for pattern '3u((': PatternError: missing ), unterminated subpattern at position 3
Exception raised for pattern '3u(': PatternError: missing ), unterminated subpattern at position 2
Exception raised for pattern '3(': PatternError: missing ), unterminated subpattern at position 1
Exception raised for pattern '(': PatternError: missing ), unterminated subpattern at position 0
Exception raised for pattern '*': PatternError: nothing to repeat at position 0
Exception raised for pattern ')': PatternError: unbalanced parenthesis at position 0
Exception raised for pattern '+': PatternError: nothing to repeat at position 0
Exception raised for pattern '(': PatternError: missing ), unterminated subpattern at position 0
Exception raised for pattern ')': PatternError: unbalanced parenthesis at position 0
Exception raised for pattern '[': PatternError: unterminated character set at position 0
Exception raised for pattern '?': PatternError: nothing to repeat at position 0
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 25, in <module>
  |     test_is_re_compilable_never_crashes()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 8, in test_is_re_compilable_never_crashes
  |     def test_is_re_compilable_never_crashes(pattern):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 4 distinct failures. (4 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 16, in test_is_re_compilable_never_crashes
    |     result = is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_never_crashes(
    |     pattern='(',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:705
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:710
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:854
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:864
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:865
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 16, in test_is_re_compilable_never_crashes
    |     result = is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_never_crashes(
    |     pattern=')',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:528
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 16, in test_is_re_compilable_never_crashes
    |     result = is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_never_crashes(
    |     pattern='[',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:550
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:552
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:564
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:566
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:567
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 16, in test_is_re_compilable_never_crashes
    |     result = is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_never_crashes(
    |     pattern='?',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:684
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:686
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.core.dtypes.inference import is_re_compilable

# Test with an invalid regex pattern - unmatched opening parenthesis
print("Testing is_re_compilable('(')...")
try:
    result = is_re_compilable('(')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
```

<details>

<summary>
Exception raised instead of returning False
</summary>
```
Testing is_re_compilable('(')...
Exception raised: PatternError: missing ), unterminated subpattern at position 0
```
</details>

## Why This Is A Bug

The function violates its documented contract in multiple ways:

1. **Type Annotation Promise**: The function signature explicitly declares `-> bool`, indicating it should always return a boolean value. Raising exceptions violates this type contract.

2. **Docstring Contract**: The docstring states the function "Returns: bool - Whether `obj` can be compiled as a regex pattern." Invalid regex patterns cannot be compiled, so the function should return `False` for them rather than crashing.

3. **Function Naming Convention**: The function follows Python's `is_*` predicate naming convention, which by convention should return boolean values and not raise exceptions for invalid inputs. Compare with `str.isdigit()`, `str.isalpha()`, etc., which return `False` for non-matching inputs.

4. **Purpose Defeated**: The apparent purpose of this function is to safely check if a string is a valid regex pattern. If it raises exceptions on invalid patterns, users must wrap it in try/except anyway, defeating its utility as a validator function.

5. **API Exposure**: This function is part of pandas' public API (`pandas.api.types.is_re_compilable`), making it important that it behaves as documented.

## Relevant Context

The current implementation at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py:187-192` shows:

```python
try:
    re.compile(obj)
except TypeError:
    return False
else:
    return True
```

The function only catches `TypeError` (for non-string inputs like integers) but fails to catch `re.error` and its subclasses (`re.PatternError`, etc.) that are raised when a string is not a valid regex pattern.

The pandas documentation for this function can be found at: https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_re_compilable.html

## Proposed Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -187,7 +187,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```