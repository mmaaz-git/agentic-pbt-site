# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Unclosed Quotes

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a KeyError when given a string containing an unclosed quote character, exposing internal implementation details instead of providing a meaningful error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list

@given(st.text())
@settings(max_examples=1000)
def test_parse_list_returns_list(s):
    result = parse_list(s)
    assert isinstance(result, list)

# Run the test
test_parse_list_returns_list()
```

<details>

<summary>
**Failing input**: `"'"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 11, in <module>
    test_parse_list_returns_list()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 5, in test_parse_list_returns_list
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 7, in test_parse_list_returns_list
    result = parse_list(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: '__Pyx_L1'
Falsifying example: test_parse_list_returns_list(
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

# Test case that crashes
parse_list('"')
```

<details>

<summary>
KeyError: '__Pyx_L1'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/repo.py", line 4, in <module>
    parse_list('"')
    ~~~~~~~~~~^^^^^
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

This violates expected behavior in several ways:

1. **Unhelpful Error Message**: The KeyError exposes internal implementation details (`__Pyx_L1`) that are meaningless to users. Users seeing this error have no indication that the problem is an unclosed quote in their input.

2. **Function Purpose**: The `parse_list` function is used to parse distutils/cython directive values from source file comments (called at line 198 of Dependencies.py). These directives appear as comments like `# distutils: libraries = spam eggs` in .pyx files. Since these are user-written comments, typos and malformed input are expected.

3. **Inconsistent Error Handling**: The `strip_string_literals` function explicitly handles unclosed quotes (see line 308-310: "This probably indicates an unclosed string literal, i.e. a broken file") by creating a label with a trailing underscore. However, `parse_list`'s `unquote` function doesn't properly handle these special labels.

4. **Documentation Gap**: While the function's docstring shows examples of valid input, it doesn't specify behavior for malformed input. However, crashing with an internal KeyError is clearly not intentional design.

## Relevant Context

The root cause is a mismatch between how `strip_string_literals` and `parse_list` handle unclosed quotes:

- `strip_string_literals` (line 296) creates labels with trailing underscores for unclosed quotes: `__Pyx_L1_`
- The `unquote` function in `parse_list` (line 132) uses `literal[1:-1]` to remove quotes
- This slicing accidentally removes both the closing character AND the trailing underscore, looking for `__Pyx_L1` instead of `__Pyx_L1_`

The function is commonly used when processing Cython source files, where users might have:
- Typos in their directive comments
- Incomplete edits leaving unclosed quotes
- Copy-paste errors

Documentation: The function's docstring (lines 110-122) only shows valid input examples. There's no specification for error handling of malformed input.

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py`

## Proposed Fix

```diff
--- a/Dependencies.py
+++ b/Dependencies.py
@@ -129,7 +129,11 @@ def parse_list(s):
     def unquote(literal):
         literal = literal.strip()
         if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+            key = literal[1:-1]
+            # Handle unclosed quotes which have trailing underscore in their label
+            if key not in literals and key + '_' in literals:
+                key = key + '_'
+            return literals[key]
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```