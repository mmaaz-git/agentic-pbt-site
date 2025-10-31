# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Regex Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The function `is_re_compilable` raises `re.PatternError` exceptions for invalid regex patterns instead of returning `False`, violating its documented contract as a predicate function that checks whether an object can be compiled as a regex.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.api.types as pat


@given(st.text())
@settings(max_examples=500)
def test_is_re_compilable_consistency(pattern):
    is_compilable = pat.is_re_compilable(pattern)

    if is_compilable:
        import re
        try:
            re.compile(pattern)
        except:
            assert False, f"is_re_compilable returned True but re.compile failed"


if __name__ == "__main__":
    test_is_re_compilable_consistency()
```

<details>

<summary>
**Failing input**: `'('` (also `')'`, `'?'`, `'*'`, `'['`, `'\\'`)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 19, in <module>
  |     test_is_re_compilable_consistency()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 6, in test_is_re_compilable_consistency
  |     @settings(max_examples=500)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 8, in test_is_re_compilable_consistency
    |     is_compilable = pat.is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_consistency(
    |     pattern='(',
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 8, in test_is_re_compilable_consistency
    |     is_compilable = pat.is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_consistency(
    |     pattern=')',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/re/_constants.py:38
    |         /home/npc/miniconda/lib/python3.13/re/_parser.py:528
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 8, in test_is_re_compilable_consistency
    |     is_compilable = pat.is_re_compilable(pattern)
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
    | Falsifying example: test_is_re_compilable_consistency(
    |     pattern='?',
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

# This should return False but instead crashes
result = pat.is_re_compilable(')')
print(f"Result: {result}")
```

<details>

<summary>
PatternError: unbalanced parenthesis at position 0
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/repo.py", line 4, in <module>
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

This function violates its documented contract in multiple ways:

1. **Naming convention violation**: Functions following the `is_*` predicate pattern in Python universally return boolean values and should never raise exceptions for invalid input. The function name `is_re_compilable` explicitly promises to check compilability, not attempt compilation.

2. **Documentation mismatch**: The docstring explicitly states "Check if the object can be compiled into a regex pattern instance" and promises to return "bool - Whether `obj` can be compiled as a regex pattern." A function that crashes on non-compilable input directly contradicts this promise.

3. **Defeats its purpose**: The entire reason to have `is_re_compilable` is to provide a safe way to check if something can be compiled before attempting compilation. If it crashes just like `re.compile()`, there's no reason for it to exist. Users would use this function specifically to validate untrusted input (user data, external files, etc.) where invalid regex patterns are common.

4. **Incomplete error handling**: The current implementation only catches `TypeError` (for non-string inputs like integers) but ignores `re.PatternError` which is raised for syntactically invalid regex patterns. This is an oversight since both types of errors indicate the object cannot be compiled as a regex.

5. **API consistency**: Other pandas `is_*` functions (like `is_integer`, `is_float`, etc.) safely return False for invalid input without raising exceptions. This function breaks that consistent pattern.

## Relevant Context

The function is located in `/pandas/core/dtypes/inference.py` at lines 166-192. It's part of the public pandas API (`pandas.api.types`) and is intended for users to check regex compilability before performing regex operations.

Common invalid patterns that trigger the bug include:
- `')'` - unbalanced parenthesis
- `'('` - unterminated subpattern
- `'?'`, `'*'` - nothing to repeat
- `'['` - unterminated character set
- `'\\'` - bad escape at end of pattern

The pandas documentation can be found at: https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_re_compilable.html

## Proposed Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.PatternError):
         return False
     else:
         return True
```