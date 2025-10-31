# Bug Report: pandas.api.types.is_re_compilable Raises re.PatternError on Invalid Regex Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable()` function crashes with `re.PatternError` when given invalid regex patterns instead of returning `False` as its documentation promises. The function only catches `TypeError` but fails to handle `re.PatternError` that `re.compile()` raises for malformed patterns.

## Property-Based Test

```python
import re
from hypothesis import given, strategies as st, settings
from pandas.api.types import is_re_compilable


@given(st.one_of(
    st.text(),
    st.binary(),
    st.integers(),
    st.floats(),
    st.none(),
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers())
))
@settings(max_examples=1000)
def test_is_re_compilable_correctness(obj):
    result = is_re_compilable(obj)

    if result:
        try:
            re.compile(obj)
        except TypeError:
            assert False, f"is_re_compilable({obj!r}) returned True but re.compile() raised TypeError"
    else:
        try:
            re.compile(obj)
            assert False, f"is_re_compilable({obj!r}) returned False but re.compile() succeeded"
        except TypeError:
            pass

if __name__ == "__main__":
    test_is_re_compilable_correctness()
```

<details>

<summary>
**Failing input**: `')'` (unbalanced parenthesis)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 32, in <module>
  |     test_is_re_compilable_correctness()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 7, in test_is_re_compilable_correctness
  |     st.text(),
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 17, in test_is_re_compilable_correctness
    |     result = is_re_compilable(obj)
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
    | Falsifying example: test_is_re_compilable_correctness(
    |     obj=')',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:984
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 17, in test_is_re_compilable_correctness
    |     result = is_re_compilable(obj)
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
    | Falsifying example: test_is_re_compilable_correctness(
    |     obj='?',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:549
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:641
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:686
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.api.types import is_re_compilable

# Test with single backslash - invalid regex pattern
print("Testing with single backslash '\\':")
result = is_re_compilable('\\')
print(f"Result: {result}")
```

<details>

<summary>
re.PatternError: bad escape (end of pattern) at position 0
</summary>
```
Testing with single backslash '\':
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/repo.py", line 5, in <module>
    result = is_re_compilable('\\')
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py", line 188, in is_re_compilable
    re.compile(obj)
    ~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 289, in compile
    return _compile(pattern, flags)
  File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 350, in _compile
    p = _compiler.compile(pattern, flags)
  File "/home/npc/miniconda/lib/python3.13/re/_compiler.py", line 748, in compile
    p = _parser.parse(p, flags)
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 973, in parse
    source = Tokenizer(str)
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 238, in __init__
    self.__next()
    ~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 251, in __next
    raise error("bad escape (end of pattern)",
                self.string, len(self.string) - 1) from None
re.PatternError: bad escape (end of pattern) at position 0
```
</details>

## Why This Is A Bug

This violates the function's documented contract in multiple ways:

1. **Type Contract Violation**: The function signature declares `-> bool` but raises exceptions instead of returning a boolean value. Python type checkers expect this function to always return True or False.

2. **Docstring Contract Violation**: The docstring states "Check if the object can be compiled into a regex pattern instance" and promises to return "bool - Whether `obj` can be compiled as a regex pattern". A predicate function should safely determine compilability without crashing.

3. **Incomplete Error Handling**: The function only catches `TypeError` (for non-string inputs) but ignores `re.error`/`re.PatternError` that `re.compile()` raises for syntactically invalid patterns. Per Python's documentation, `re.compile()` raises both exception types.

4. **Principle of Least Surprise**: Functions with "is_" prefix are conventionally safe predicates that return boolean values. Users expect `is_re_compilable('\\')` to return `False`, not crash.

5. **API Usability**: The function exists to safely check regex compilability before attempting compilation. If it crashes on invalid patterns, users might as well call `re.compile()` directly.

## Relevant Context

The function is located in `/pandas/core/dtypes/inference.py` at lines 166-192. It's part of pandas' public API under `pandas.api.types`.

Common invalid regex patterns that trigger this bug:
- `'\\'` - incomplete escape sequence
- `'['` - unclosed character class
- `')'` - unbalanced parenthesis
- `'?'` - nothing to repeat
- `'*'` - nothing to repeat
- `'(?P<'` - incomplete named group

The Python `re` module documentation states that `re.compile()` raises:
- `TypeError` - if the pattern is not a string or bytes
- `re.error` (alias for `re.PatternError`) - if the pattern contains invalid regex syntax

Link to pandas source: https://github.com/pandas-dev/pandas/blob/main/pandas/core/dtypes/inference.py#L166-L192

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