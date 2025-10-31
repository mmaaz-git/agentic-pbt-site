# Bug Report: pandas.api.types.is_re_compilable Raises PatternError Instead of Returning False for Invalid Regex Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function violates its documented API contract by raising `re.PatternError` exceptions for syntactically invalid regex patterns instead of returning `False` as promised in its documentation.

## Property-Based Test

```python
import pandas.api.types as pat
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=100))
@settings(max_examples=200)
def test_is_re_compilable_on_strings(s):
    """Test that is_re_compilable always returns a boolean for string inputs."""
    result = pat.is_re_compilable(s)
    assert isinstance(result, bool), f"is_re_compilable should return bool, got {type(result)}"

# Run the test
if __name__ == "__main__":
    test_is_re_compilable_on_strings()
```

<details>

<summary>
**Failing input**: `'['`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 13, in <module>
  |     test_is_re_compilable_on_strings()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 5, in test_is_re_compilable_on_strings
  |     @settings(max_examples=200)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 4 distinct failures. (4 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 8, in test_is_re_compilable_on_strings
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
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='[',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 8, in test_is_re_compilable_on_strings
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
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='(',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 8, in test_is_re_compilable_on_strings
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
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s=')',
    | )
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 8, in test_is_re_compilable_on_strings
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
    | Falsifying example: test_is_re_compilable_on_strings(
    |     s='?',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas.api.types as pat

# Test cases that should return False but instead raise exceptions
invalid_regex_patterns = ['[', '?', '*', '(unclosed', ')', '(', '[]', '(*)', '+', '++']

print("Testing pandas.api.types.is_re_compilable with invalid regex patterns:")
print("=" * 60)

for pattern in invalid_regex_patterns:
    print(f"\nTesting pattern: {repr(pattern)}")
    try:
        result = pat.is_re_compilable(pattern)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
```

<details>

<summary>
Running this code raises multiple PatternError exceptions
</summary>
```
Testing pandas.api.types.is_re_compilable with invalid regex patterns:
============================================================

Testing pattern: '['
  ERROR: PatternError: unterminated character set at position 0

Testing pattern: '?'
  ERROR: PatternError: nothing to repeat at position 0

Testing pattern: '*'
  ERROR: PatternError: nothing to repeat at position 0

Testing pattern: '(unclosed'
  ERROR: PatternError: missing ), unterminated subpattern at position 0

Testing pattern: ')'
  ERROR: PatternError: unbalanced parenthesis at position 0

Testing pattern: '('
  ERROR: PatternError: missing ), unterminated subpattern at position 0

Testing pattern: '[]'
  ERROR: PatternError: unterminated character set at position 0

Testing pattern: '(*)'
  ERROR: PatternError: nothing to repeat at position 1

Testing pattern: '+'
  ERROR: PatternError: nothing to repeat at position 0

Testing pattern: '++'
  ERROR: PatternError: nothing to repeat at position 0
```
</details>

## Why This Is A Bug

The function `is_re_compilable` explicitly violates its documented API contract. According to the docstring at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/inference.py:166-192`:

1. **Function Purpose**: "Check if the object can be compiled into a regex pattern instance"
2. **Return Type**: "Returns: bool - Whether `obj` can be compiled as a regex pattern"
3. **Function Name Convention**: The "is_" prefix follows Python's predicate naming convention, which universally indicates a function that returns a boolean value without side effects or exceptions

The current implementation (lines 187-192) only catches `TypeError` exceptions, which handles non-string inputs like integers, lists, or None. However, when `re.compile()` receives a syntactically invalid regex string, it raises `re.PatternError` (an alias for `re.error`), which is not caught. This causes the exception to propagate to the caller instead of returning `False`.

This breaks the fundamental contract that predicate functions should safely return True/False without raising exceptions for any valid input type. Users calling this function to validate user-provided regex patterns will encounter unexpected exceptions requiring additional error handling, defeating the purpose of having a predicate function.

## Relevant Context

- **Source Location**: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/inference.py:166-192`
- **Python Documentation**: According to Python's `re` module documentation, `re.compile()` raises `re.error` (also accessible as `re.PatternError` in Python 3.13) for invalid regex patterns
- **Similar Functions**: The same file contains other predicate functions like `is_hashable()` (lines 334-370) which correctly catches all relevant exceptions and returns False
- **pandas API Documentation**: https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_re_compilable.html confirms the function should return a boolean

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