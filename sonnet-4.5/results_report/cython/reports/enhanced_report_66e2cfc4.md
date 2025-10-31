# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Unclosed Quotes

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError: '__Pyx_L1'` when processing input containing unclosed or lone quote characters, failing to provide meaningful error messages for malformed user input in distutils directives.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list

@given(st.text())
@settings(max_examples=1000)
def test_parse_list_no_empty_strings(s):
    result = parse_list(s)
    assert all(item != '' for item in result), f"parse_list returned empty string in result: {result}"

if __name__ == "__main__":
    test_parse_list_no_empty_strings()
```

<details>

<summary>
**Failing input**: `"'"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 11, in <module>
    test_parse_list_no_empty_strings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 5, in test_parse_list_no_empty_strings
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 7, in test_parse_list_no_empty_strings
    result = parse_list(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: '__Pyx_L1'
Falsifying example: test_parse_list_no_empty_strings(
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

# Test case with unclosed single quote
try:
    result = parse_list("'")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")

# Test case with unclosed double quote
print("\nTest with unclosed double quote:")
try:
    result = parse_list('"')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")

# Test case with unclosed quote and text
print("\nTest with unclosed quote and text:")
try:
    result = parse_list("'hello")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
```

<details>

<summary>
KeyError: '__Pyx_L1' for all test cases
</summary>
```
Exception type: KeyError
Exception message: '__Pyx_L1'

Test with unclosed double quote:
Exception type: KeyError
Exception message: '__Pyx_L1'

Test with unclosed quote and text:
Exception type: KeyError
Exception message: '__Pyx_L1'
```
</details>

## Why This Is A Bug

The `parse_list` function is used throughout Cython's build system to parse user-written distutils directive values from comments in `.pyx` and `.py` files. These directives are manually typed by developers and are prone to typos such as unclosed quotes. For example, a user might accidentally write:

```python
# distutils: libraries = ['ssl
```

When `parse_list` encounters such malformed input, it crashes with a cryptic `KeyError: '__Pyx_L1'` that provides no indication of the actual problem. The error message exposes internal implementation details (the `__Pyx_L1` label used in string literal normalization) rather than informing the user about the unclosed quote.

This violates basic error handling principles for user-facing functions. A function that processes human-written configuration should either:
1. Handle common errors gracefully
2. Provide clear error messages that help users identify and fix the problem

## Relevant Context

The bug occurs due to an interaction between two functions:

1. **`strip_string_literals`** (line 282): When it encounters an unclosed quote, it normalizes the entire remaining string to a label like `'__Pyx_L1_'` (with a trailing underscore) and stores the mapping in the `literals` dictionary.

2. **`parse_list`'s `unquote` function** (lines 129-134): This tries to look up `literal[1:-1]` which removes both the leading quote character and the trailing character. For `'__Pyx_L1_'`, this becomes `__Pyx_L1` (without the underscore), which doesn't exist in the dictionary.

The function is called from `DistutilsInfo.__init__` (line 198) when parsing directives like:
- `# distutils: libraries = ['ssl', 'crypto']`
- `# distutils: include_dirs = ['/usr/local/include']`
- `# distutils: define_macros = [('DEBUG', None)]`

## Proposed Fix

Add proper error handling in the `unquote` function to catch the `KeyError` and raise a more descriptive error message:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,14 @@ def parse_list(s):
     def unquote(literal):
         literal = literal.strip()
         if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+            key = literal[1:-1]
+            try:
+                return literals[key]
+            except KeyError:
+                # Check if this looks like an unclosed quote case
+                if key.startswith('__Pyx_L') and literal.endswith('_'):
+                    raise ValueError(f"Unclosed quote detected in input: {s!r}")
+                else:
+                    raise ValueError(f"Invalid quoted string in input: {literal!r}")
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```