# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Regex Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The function `is_re_compilable()` crashes with `re.PatternError` when given invalid regex pattern strings, instead of returning `False` as documented.

## Property-Based Test

```python
import re
from hypothesis import given, strategies as st, settings
import pandas.api.types as pat


@given(st.text())
@settings(max_examples=500)
def test_is_re_compilable_should_not_crash(s):
    try:
        result = pat.is_re_compilable(s)
        assert isinstance(result, bool), f"Should return bool, got {type(result)}"
    except Exception as e:
        raise AssertionError(
            f"is_re_compilable crashed on input {s!r} with {type(e).__name__}: {e}"
        )


if __name__ == "__main__":
    # Run the test
    test_is_re_compilable_should_not_crash()
```

<details>

<summary>
**Failing input**: `'('`, `')'`, `'?'` (and many other invalid regex patterns)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 20, in <module>
  |     test_is_re_compilable_should_not_crash()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 7, in test_is_re_compilable_should_not_crash
  |     @settings(max_examples=500)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 10, in test_is_re_compilable_should_not_crash
    |     result = pat.is_re_compilable(s)
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
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 13, in test_is_re_compilable_should_not_crash
    |     raise AssertionError(
    |         f"is_re_compilable crashed on input {s!r} with {type(e).__name__}: {e}"
    |     )
    | AssertionError: is_re_compilable crashed on input '(' with PatternError: missing ), unterminated subpattern at position 0
    | Falsifying example: test_is_re_compilable_should_not_crash(
    |     s='(',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:549
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:710
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:854
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:864
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:865
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 10, in test_is_re_compilable_should_not_crash
    |     result = pat.is_re_compilable(s)
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
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 13, in test_is_re_compilable_should_not_crash
    |     raise AssertionError(
    |         f"is_re_compilable crashed on input {s!r} with {type(e).__name__}: {e}"
    |     )
    | AssertionError: is_re_compilable crashed on input ')' with PatternError: unbalanced parenthesis at position 0
    | Falsifying example: test_is_re_compilable_should_not_crash(
    |     s=')',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:984
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 10, in test_is_re_compilable_should_not_crash
    |     result = pat.is_re_compilable(s)
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
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 13, in test_is_re_compilable_should_not_crash
    |     raise AssertionError(
    |         f"is_re_compilable crashed on input {s!r} with {type(e).__name__}: {e}"
    |     )
    | AssertionError: is_re_compilable crashed on input '?' with PatternError: nothing to repeat at position 0
    | Falsifying example: test_is_re_compilable_should_not_crash(
    |     s='?',
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
import pandas.api.types as pat

# Test case that should return False but instead crashes
result = pat.is_re_compilable(')')
print(f"Result: {result}")
```

<details>

<summary>
Crashes with re.PatternError: unbalanced parenthesis at position 0
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/repo.py", line 4, in <module>
    result = pat.is_re_compilable(')')
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py", line 188, in is_re_compilable
    re.compile(obj)
    ~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 289, in compile
    return _compile(pattern, flags)
  File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 350, in _compile
    p = _compiler.compile(pattern, flags)
  File "/home/npc/miniconda/lib/python3.13/re/_compiler.py", line 748, in compile
    p = _parser.parse(p, flags)
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 985, in parse
    raise source.error("unbalanced parenthesis")
re.PatternError: unbalanced parenthesis at position 0
```
</details>

## Why This Is A Bug

The function `is_re_compilable` violates its documented contract in multiple ways:

1. **Function Name Semantics**: The function name follows the `is_*` predicate pattern used throughout pandas (e.g., `is_integer`, `is_float`), which by convention should return a boolean value without side effects or exceptions.

2. **Documented Return Type**: The docstring explicitly states:
   - "Returns: bool - Whether `obj` can be compiled as a regex pattern."
   - The function signature specifies `-> bool` return type

3. **Purpose Statement Violation**: The pandas official documentation states the function is for "helping prevent runtime errors when working with regular expressions in pandas." However, the function itself causes runtime errors for invalid regex patterns.

4. **Inconsistent Error Handling**: The function correctly returns `False` for non-string inputs (e.g., integers) by catching `TypeError`, but fails to handle the most common error case when `re.compile()` receives an invalid pattern string, which raises `re.PatternError` (or `re.error` in older Python versions).

5. **Example Behavior**: The docstring example shows `is_re_compilable(1)` returning `False`, establishing that non-compilable inputs should return `False` rather than raise exceptions. Invalid regex patterns like `')'` are equally "not compilable" and should follow the same pattern.

## Relevant Context

The issue occurs in pandas version 2.3.2 and affects Python 3.13 (though the bug exists across Python versions). The function is located in `/pandas/core/dtypes/inference.py:188`.

The current implementation only catches `TypeError` which handles non-string inputs, but misses `re.PatternError` (Python 3.13+) or `re.error` (older versions) which are raised for syntactically invalid regex patterns.

Common invalid patterns that trigger this bug include:
- Unmatched parentheses: `'('`, `')'`
- Invalid repetition operators: `'*'`, `'+'`, `'?'`
- Unterminated character classes: `'['`
- Invalid escape sequences: `'\\'`

This function is likely used in data validation scenarios where user input needs to be checked before being used as a regex pattern, making proper error handling critical.

## Proposed Fix

```diff
def is_re_compilable(obj) -> bool:
    """
    Check if the object can be compiled into a regex pattern instance.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` can be compiled as a regex pattern.

    Examples
    --------
    >>> from pandas.api.types import is_re_compilable
    >>> is_re_compilable(".*")
    True
    >>> is_re_compilable(1)
    False
    """
    try:
        re.compile(obj)
-   except TypeError:
+   except (TypeError, re.error):
        return False
    else:
        return True
```