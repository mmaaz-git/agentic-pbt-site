# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Unclosed Quotes

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function in Cython's build dependencies module crashes with a `KeyError` when processing strings containing unclosed quotes, due to a key mismatch between the string literal replacement and lookup operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@given(st.text())
@settings(max_examples=1000)
def test_parse_list_should_not_crash(s):
    result = parse_list(s)


if __name__ == "__main__":
    test_parse_list_should_not_crash()
```

<details>

<summary>
**Failing input**: `"'"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 12, in <module>
    test_parse_list_should_not_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 6, in test_parse_list_should_not_crash
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 8, in test_parse_list_should_not_crash
    result = parse_list(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: '__Pyx_L1'
Falsifying example: test_parse_list_should_not_crash(
    s="'",
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:132
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

# Test case: single unclosed quote
result = parse_list('"')
print("Result:", result)
```

<details>

<summary>
KeyError: '__Pyx_L1' when parsing unclosed quote
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/repo.py", line 4, in <module>
    result = parse_list('"')
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: '__Pyx_L1'
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Key Mismatch Bug**: The `strip_string_literals` function generates keys with a trailing underscore (e.g., `'__Pyx_L1_'`), but the `unquote` function attempts to look up keys without the underscore (e.g., `'__Pyx_L1'`). This is clearly an implementation error, not intentional behavior.

2. **Documentation Violation**: The function's docstring shows examples of handling quoted strings but provides no specification that quotes must be properly closed. The examples only demonstrate well-formed input, but there's no explicit requirement or warning about malformed quotes.

3. **Unhelpful Error Message**: When the function encounters unclosed quotes, it crashes with `KeyError: '__Pyx_L1'` which provides no indication that the problem is malformed input. A parsing function should either handle malformed input gracefully or raise a descriptive error.

4. **Inconsistent Behavior**: The function successfully handles other edge cases like empty strings, spaces, and complex delimiters, but fails on a simple unclosed quote character.

## Relevant Context

The bug occurs in the interaction between two functions in `/Cython/Build/Dependencies.py`:

1. **`strip_string_literals` (line 282)**: This function replaces string literals with placeholder labels. When given an unclosed quote like `'"'`, it:
   - Creates a label with underscore: `'__Pyx_L1_'`
   - Returns: `('"__Pyx_L1_', {'__Pyx_L1_': ''})`
   - Note the key in the dictionary has a trailing underscore

2. **`parse_list` inner function `unquote` (lines 129-134)**: This function attempts to restore the original literals:
   - Checks if the literal starts with a quote character
   - If yes, extracts `literal[1:-1]` which becomes `'__Pyx_L1'` (no underscore)
   - Tries to look this up in the literals dictionary
   - Fails because the actual key is `'__Pyx_L1_'` (with underscore)

This function is part of Cython's build system and is used for parsing dependency lists in build configurations. While it may be considered an internal utility, it's still accessible through the public module namespace and should handle edge cases appropriately.

Documentation: [Cython Build System](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html)
Source: [GitHub - Cython/Build/Dependencies.py](https://github.com/cython/cython/blob/master/Cython/Build/Dependencies.py)

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,10 @@ def parse_list(s):
     def unquote(literal):
         literal = literal.strip()
         if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+            # Handle potential key mismatch for unclosed quotes
+            key = literal[1:-1]
+            # Try with underscore suffix first (as generated by strip_string_literals)
+            return literals.get(key + '_', literals.get(key, literal))
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```