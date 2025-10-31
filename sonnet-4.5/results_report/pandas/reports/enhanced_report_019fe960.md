# Bug Report: pandas.api.types.is_re_compilable Raises Exception Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The function `is_re_compilable()` raises `re.PatternError` exceptions for invalid regex patterns instead of returning `False` as its documentation promises, violating its contract to always return a bool value.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.api.types as pat

@given(st.text(min_size=1, max_size=10))
def test_is_re_compilable_returns_bool(s):
    """is_re_compilable should always return a bool, never raise exceptions"""
    result = pat.is_re_compilable(s)
    assert isinstance(result, bool), f"is_re_compilable should return bool"

if __name__ == "__main__":
    # Run the property-based test
    test_is_re_compilable_returns_bool()
```

<details>

<summary>
**Failing input**: `'\\'`, `'['`, `'?'` (and other invalid regex patterns)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 12, in <module>
  |     test_is_re_compilable_returns_bool()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 5, in test_is_re_compilable_returns_bool
  |     def test_is_re_compilable_returns_bool(s):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 7, in test_is_re_compilable_returns_bool
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
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 973, in parse
    |     source = Tokenizer(str)
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 238, in __init__
    |     self.__next()
    |     ~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 251, in __next
    |     raise error("bad escape (end of pattern)",
    |                 self.string, len(self.string) - 1) from None
    | re.PatternError: bad escape (end of pattern) at position 0
    | Falsifying example: test_is_re_compilable_returns_bool(
    |     s='\\',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 7, in test_is_re_compilable_returns_bool
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
    | Falsifying example: test_is_re_compilable_returns_bool(
    |     s='[',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 7, in test_is_re_compilable_returns_bool
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
    | Falsifying example: test_is_re_compilable_returns_bool(
    |     s='?',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas.api.types as pat

# Test various invalid regex patterns that should return False
# but instead raise re.PatternError

test_cases = [
    ')',    # unbalanced parenthesis
    '?',    # nothing to repeat
    '*',    # nothing to repeat
    '+',    # nothing to repeat
    '(',    # missing closing parenthesis
    '[',    # unterminated character set
    '\\',   # bad escape at end
]

print("Testing pandas.api.types.is_re_compilable with invalid regex patterns:")
print("-" * 60)

for pattern in test_cases:
    print(f"\nInput: {repr(pattern)}")
    print("Expected: False")
    print("Actual: ", end="")
    try:
        result = pat.is_re_compilable(pattern)
        print(f"{result}")
    except Exception as e:
        print(f"Raised {e.__class__.__name__}: {e}")
```

<details>

<summary>
PatternError raised for all invalid regex patterns
</summary>
```
Testing pandas.api.types.is_re_compilable with invalid regex patterns:
------------------------------------------------------------

Input: ')'
Expected: False
Actual: Raised PatternError: unbalanced parenthesis at position 0

Input: '?'
Expected: False
Actual: Raised PatternError: nothing to repeat at position 0

Input: '*'
Expected: False
Actual: Raised PatternError: nothing to repeat at position 0

Input: '+'
Expected: False
Actual: Raised PatternError: nothing to repeat at position 0

Input: '('
Expected: False
Actual: Raised PatternError: missing ), unterminated subpattern at position 0

Input: '['
Expected: False
Actual: Raised PatternError: unterminated character set at position 0

Input: '\\'
Expected: False
Actual: Raised PatternError: bad escape (end of pattern) at position 0
```
</details>

## Why This Is A Bug

The function violates its documented contract in multiple ways:

1. **Type Annotation Contract**: The function signature explicitly declares `-> bool`, promising to always return a boolean value. However, it raises `re.PatternError` exceptions for invalid regex patterns, preventing the return of any value.

2. **Documentation Contract**: The docstring states "Returns: bool - Whether `obj` can be compiled as a regex pattern" with no mention of possible exceptions. There is no "Raises" section documenting that exceptions may occur.

3. **Semantic Contract**: The function follows the `is_*` predicate naming pattern, which by convention in the pandas API (and Python generally) implies a safe boolean check that doesn't raise exceptions. Other `is_*` functions in the same module (like `is_hashable`, `is_sequence`) catch exceptions and return False.

4. **Incomplete Exception Handling**: The current implementation only catches `TypeError` (for non-string inputs like integers) but fails to catch `re.error` (or its subclass `re.PatternError` in Python 3.13+) that `re.compile()` raises for syntactically invalid regex patterns.

5. **Purpose Violation**: The function's entire purpose is to safely check if something can be compiled as regex. Raising exceptions defeats this purpose, forcing users to wrap calls in try-except blocks, essentially duplicating the function's intended functionality.

## Relevant Context

The function is part of the public pandas API (`pandas.api.types`) and is meant for external use. Invalid regex patterns are common in real-world data processing scenarios where users need to validate user input or data before attempting regex operations.

The current implementation at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/inference.py:166-192` shows the function already attempts to handle exceptions by catching `TypeError`, indicating the intent was to return False for non-compilable inputs rather than propagating exceptions.

Similar functions in the same module demonstrate the expected pattern. For example, `is_hashable()` (lines 334-370) catches `TypeError` when `hash()` fails and returns False, providing a safe way to check hashability without exceptions.

## Proposed Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```