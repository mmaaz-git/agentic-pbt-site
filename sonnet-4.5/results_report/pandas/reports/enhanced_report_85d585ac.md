# Bug Report: pandas.api.types.is_re_compilable Raises PatternError Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The function `is_re_compilable()` raises `re.PatternError` for invalid regex patterns instead of returning `False`, violating its documented contract that it should return a boolean indicating whether an object can be compiled as a regex.

## Property-Based Test

```python
import pandas.api.types as pat
import re
from hypothesis import given, strategies as st


@given(st.text())
def test_is_re_compilable_should_not_raise(s):
    try:
        re.compile(s)
        can_compile = True
    except Exception:
        can_compile = False

    result = pat.is_re_compilable(s)

    assert result == can_compile, (
        f"is_re_compilable should match re.compile behavior without raising, "
        f"but got different behavior for {s!r}"
    )


if __name__ == "__main__":
    # Run the property-based test
    test_is_re_compilable_should_not_raise()
```

<details>

<summary>
**Failing input**: `s='['`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 24, in <module>
  |     test_is_re_compilable_should_not_raise()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 7, in test_is_re_compilable_should_not_raise
  |     def test_is_re_compilable_should_not_raise(s):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 14, in test_is_re_compilable_should_not_raise
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
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 567, in _parse
    |     raise source.error("unterminated character set",
    |                        source.tell() - here)
    | re.PatternError: unterminated character set at position 0
    | Falsifying example: test_is_re_compilable_should_not_raise(
    |     s='[',
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 14, in test_is_re_compilable_should_not_raise
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
    | Falsifying example: test_is_re_compilable_should_not_raise(
    |     s='?',
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
import pandas.api.types as pat

# Test with valid regex
print("Testing with valid regex '.*':")
result = pat.is_re_compilable('.*')
print(f"Result: {result}")

# Test with invalid regex that should return False
print("\nTesting with invalid regex '(':")
result = pat.is_re_compilable('(')
print(f"Result: {result}")
```

<details>

<summary>
PatternError: missing ), unterminated subpattern at position 0
</summary>
```
Testing with valid regex '.*':
Result: True

Testing with invalid regex '(':
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/repo.py", line 10, in <module>
    result = pat.is_re_compilable('(')
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
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 865, in _parse
    raise source.error("missing ), unterminated subpattern",
                       source.tell() - start)
re.PatternError: missing ), unterminated subpattern at position 0
```
</details>

## Why This Is A Bug

The function `is_re_compilable` violates its documented contract in multiple ways:

1. **API Contract Violation**: The docstring explicitly states the function returns a `bool` indicating whether the object can be compiled as a regex pattern. It should never raise an exception for string inputs.

2. **Python Naming Convention Violation**: Functions prefixed with "is_" in Python are predicates that should return boolean values without side effects or exceptions. This is a well-established convention in the pandas API and Python in general.

3. **Incomplete Error Handling**: The implementation only catches `TypeError` (for non-string inputs like integers) but fails to catch `re.error` and its subclasses like `re.PatternError` that occur when strings contain invalid regex syntax.

4. **Defeats Function Purpose**: Users cannot safely use this function to check if a string is a valid regex without wrapping it in try-except, which completely defeats the purpose of having a checking function. The function exists specifically to provide a safe way to test regex compilability.

5. **Inconsistent with Similar Functions**: Other "is_*" functions in `pandas.api.types` (like `is_number`, `is_hashable`) handle all exceptions and return boolean values consistently.

## Relevant Context

The function is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/inference.py` at lines 166-192.

The current implementation shows clear intent to handle exceptions (it already catches `TypeError`), but the error handling is incomplete. The function's examples in the docstring demonstrate returning `False` for invalid inputs (like integers), establishing the pattern that should apply to all invalid inputs including malformed regex strings.

Common invalid regex patterns that trigger this bug include:
- `'('` - unterminated parenthesis group
- `'['` - unterminated character class
- `'*'`, `'?'`, `'+'` - quantifiers with nothing to repeat
- `'(?'` - incomplete special group
- `'\'` - incomplete escape sequence

This is particularly problematic when processing user input or data that may contain these characters not intended as regex patterns.

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