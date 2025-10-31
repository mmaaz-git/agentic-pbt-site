# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Unclosed Quotes

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError: '__Pyx_L1'` when parsing strings containing unclosed quote characters (single or double quotes). This occurs due to incorrect string slicing when looking up normalized string literals in the internal dictionary.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Build.Dependencies import parse_list

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=100))
def test_parse_list_no_crash(s):
    result = parse_list(s)
    assert isinstance(result, list)
```

<details>

<summary>
**Failing input**: `"'"`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/1
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_parse_list_no_crash FAILED

=================================== FAILURES ===================================
___________________________ test_parse_list_no_crash ___________________________

    @given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=100))
>   def test_parse_list_no_crash(s):
                   ^^^

hypo.py:5:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
hypo.py:6: in test_parse_list_no_crash
    result = parse_list(s)
             ^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:135: in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

literal = '"__Pyx_L1_'

    def unquote(literal):
        literal = literal.strip()
        if literal[0] in "'\"":
>           return literals[literal[1:-1]]
                   ^^^^^^^^^^^^^^^^^^^^^^^
E           KeyError: '__Pyx_L1'
E           Falsifying example: test_parse_list_no_crash(
E               s='"',
E           )
E           Explanation:
E               These lines were always and only run by failing examples:
E                   /home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:132

/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:132: KeyError
=========================== short test summary info ============================
FAILED hypo.py::test_parse_list_no_crash - KeyError: '__Pyx_L1'
!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
============================== 1 failed in 0.30s ===============================
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

# Test the failing case
try:
    result = parse_list("'")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
```

<details>

<summary>
KeyError: '__Pyx_L1'
</summary>
```
Error type: KeyError
Error message: '__Pyx_L1'
```
</details>

## Why This Is A Bug

The `parse_list` function is documented to parse list-like strings and handle quoted elements, as shown in its doctest examples (lines 110-122 of Dependencies.py). The function is used to parse user-provided configuration values from build files when processing directives like `# distutils: sources = ['file.c']` or `# cython: include_dirs = ['/path']`.

When encountering an unclosed quote, the function crashes with a `KeyError` that exposes internal implementation details (`__Pyx_L1`). This error occurs because:

1. The `strip_string_literals` function normalizes the input string `"'"` to `"'__Pyx_L1_"` and creates a dictionary `{"__Pyx_L1_": ""}`
2. The `unquote` helper function (lines 129-134) performs `literal[1:-1]` on `"'__Pyx_L1_"`, resulting in `"__Pyx_L1"` (without the trailing underscore)
3. It then tries to look up `"__Pyx_L1"` in the dictionary, but the actual key is `"__Pyx_L1_"` (with underscore), causing the KeyError

While unclosed quotes indicate malformed input, a parsing function that processes user configuration should either handle such input gracefully or raise a meaningful error message, not crash with an internal KeyError.

## Relevant Context

The `parse_list` function is called from `DistutilsInfo.__init__` (line 198) when parsing configuration values that are expected to be lists. This occurs when processing source files with Cython directives. The function's doctest examples show it should handle quoted strings with spaces and special characters, but the implementation fails on edge cases with unclosed quotes.

Additional failing inputs include:
- `parse_list('"')` - unclosed double quote
- `parse_list("[']")` - unclosed quote in brackets
- `parse_list("#")` - returns `['#__Pyx_L1_']` (doesn't crash but returns malformed output)

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,12 @@ def parse_list(s):
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+        if literal and literal[0] in "'\"":
+            # Handle case where strip_string_literals produced a normalized label
+            if len(literal) > 2:
+                key = literal[1:-1]
+                if key in literals:
+                    return literals[key]
+            return literal  # Return as-is if key not found or malformed
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```