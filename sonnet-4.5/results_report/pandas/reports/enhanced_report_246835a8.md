# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Regex Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function crashes with `re.PatternError` when given invalid regex patterns instead of returning `False` as its documentation promises.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.api.types as pt
import re

@given(st.text())
@settings(max_examples=300)
def test_is_re_compilable_on_strings(s):
    result = pt.is_re_compilable(s)

    try:
        re.compile(s)
        can_compile = True
    except re.error:
        can_compile = False

    assert result == can_compile

if __name__ == "__main__":
    test_is_re_compilable_on_strings()
```

<details>

<summary>
**Failing input**: `'?'`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 19, in <module>
  |     test_is_re_compilable_on_strings()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 6, in test_is_re_compilable_on_strings
  |     @settings(max_examples=300)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 8 distinct failures. (8 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_is_re_compilable_on_strings
    |     result = pt.is_re_compilable(s)
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
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 543, in _parse
    |     code = _escape(source, this, state)
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 391, in _escape
    |     raise source.error("incomplete escape %s" % escape, len(escape))
    | re.PatternError: incomplete escape \u at position 0
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='\\u',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:389
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_is_re_compilable_on_strings
    |     result = pt.is_re_compilable(s)
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
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 543, in _parse
    |     code = _escape(source, this, state)
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 442, in _escape
    |     raise source.error("bad escape %s" % escape, len(escape))
    | re.PatternError: bad escape \C at position 0
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='\\C',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:442
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_is_re_compilable_on_strings
    |     result = pt.is_re_compilable(s)
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
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 543, in _parse
    |     code = _escape(source, this, state)
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 439, in _escape
    |     raise source.error("invalid group reference %d" % group, len(escape) - 1)
    | re.PatternError: invalid group reference 1 at position 1
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='\\1',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:419
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_is_re_compilable_on_strings
    |     result = pt.is_re_compilable(s)
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
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='\\',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:250
    +---------------- 5 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_is_re_compilable_on_strings
    |     result = pt.is_re_compilable(s)
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
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='[',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:550
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:552
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:566
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:567
    +---------------- 6 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_is_re_compilable_on_strings
    |     result = pt.is_re_compilable(s)
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
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='(',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:709
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:710
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:854
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:864
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:865
    +---------------- 7 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_is_re_compilable_on_strings
    |     result = pt.is_re_compilable(s)
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
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s=')',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:984
    +---------------- 8 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_is_re_compilable_on_strings
    |     result = pt.is_re_compilable(s)
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
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='?',
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
import pandas.api.types as pt

# This should return False but instead crashes
result = pt.is_re_compilable('?')
print(f"Result: {result}")
```

<details>

<summary>
Crashes with PatternError: nothing to repeat at position 0
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/repo.py", line 4, in <module>
    result = pt.is_re_compilable('?')
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

The function `is_re_compilable` violates its documented contract. According to its docstring and type signature, it should "Check if the object can be compiled into a regex pattern instance" and return a `bool`. The function explicitly promises:

1. **Return type**: The function signature declares `-> bool`, promising a boolean return value
2. **Documentation**: States it should "check" compilability and return "Whether `obj` can be compiled as a regex pattern"
3. **Function naming convention**: Functions starting with "is_" are universally understood to be predicates that return True/False for all inputs

The current implementation only catches `TypeError` (for non-string inputs) but fails to catch `re.error` exceptions that occur when a string contains an invalid regex pattern. This means:
- Valid regex strings like `".*"` correctly return `True`
- Non-string objects like `1` correctly return `False`
- Invalid regex strings like `"?"`, `"*"`, `"["`, `"("` crash with `re.PatternError` instead of returning `False`

This violates the principle of least surprise for a predicate function. Users calling a function designed to check if something can be compiled expect it to handle both valid and invalid cases gracefully, not raise exceptions.

## Relevant Context

The function is located in `/pandas/core/dtypes/inference.py` at lines 166-192. The current implementation:

```python
def is_re_compilable(obj) -> bool:
    try:
        re.compile(obj)
    except TypeError:
        return False
    else:
        return True
```

The issue affects pandas version 2.3.2 (and likely earlier versions). The function is part of the public API under `pandas.api.types.is_re_compilable`.

Documentation link: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.is_re_compilable.html

This bug is particularly problematic when:
- Validating user-provided regex patterns
- Processing dynamic regex patterns from data sources
- Building tools that need to determine if a string is a valid regex

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