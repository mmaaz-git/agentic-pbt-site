# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Unclosed Quotes

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a KeyError when processing strings containing unclosed quote characters, which can occur when parsing malformed distutils/cython directives from source file comments.

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
if __name__ == "__main__":
    test_parse_list_returns_list()
```

<details>

<summary>
**Failing input**: `"'"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 12, in <module>
    test_parse_list_returns_list()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 5, in test_parse_list_returns_list
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 7, in test_parse_list_returns_list
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

# Test with single double quote
try:
    result = parse_list('"')
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError with double quote: {e}")

# Test with single single quote
try:
    result = parse_list("'")
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError with single quote: {e}")
```

<details>

<summary>
KeyError: '__Pyx_L1' for both quote types
</summary>
```
KeyError with double quote: '__Pyx_L1'
KeyError with single quote: '__Pyx_L1'
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Cryptic error message**: The function crashes with an internal KeyError about `__Pyx_L1` instead of handling malformed input gracefully or providing a clear error message about unclosed quotes.

2. **Real-world usage context**: The `parse_list` function is called at line 198 of Dependencies.py when parsing distutils/cython directives from source file comments like `# distutils: libraries = foo bar`. Users may have malformed or incomplete comments in their .pyx files during development.

3. **Build system expectation**: Build tools should provide helpful error messages when encountering malformed input, not crash with internal implementation details exposed.

4. **Label mismatch bug**: The root cause is a clear logic error - `strip_string_literals()` creates placeholder labels with trailing underscores (e.g., `__Pyx_L1_` at line 296), but the `unquote()` function strips both the closing quote AND the trailing underscore using `literal[1:-1]` (line 132), then tries to look up the wrong key.

## Relevant Context

The bug occurs in the Cython build dependency parsing system. When Cython processes .pyx files, it parses special comment directives that control compilation settings. The `parse_list()` function is responsible for parsing list-valued directives.

The function uses `strip_string_literals()` to normalize string literals in the input, replacing them with placeholder labels. When an unclosed quote is encountered, `strip_string_literals()` returns the quote character followed by the label (e.g., `'__Pyx_L1_`). The `unquote()` function then incorrectly strips both ends of this string, removing both the quote and the trailing underscore, leading to a KeyError when looking up `__Pyx_L1` instead of `__Pyx_L1_`.

Documentation: https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py`

## Proposed Fix

```diff
--- a/Dependencies.py
+++ b/Dependencies.py
@@ -128,8 +128,13 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+        if len(literal) >= 1 and literal[0] in "'\"":
+            # Handle unclosed quotes - check if the key exists with trailing underscore
+            key = literal[1:]  # Remove leading quote
+            if key.endswith('_') and key in literals:
+                return literals[key]
+            elif len(literal) >= 2:
+                return literals.get(literal[1:-1], literal)
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```