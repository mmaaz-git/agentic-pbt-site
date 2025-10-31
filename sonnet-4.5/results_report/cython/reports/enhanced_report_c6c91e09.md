# Bug Report: Cython.Build.Dependencies.parse_list Quote KeyError

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError` when given unbalanced quotes (single `'` or `"`) as input, producing an unhelpful error message that references internal implementation details.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Build.Dependencies import parse_list


@given(st.text(alphabet='"\'  ', max_size=20))
def test_parse_list_quotes_and_spaces(s):
    result = parse_list(s)
    assert isinstance(result, list)

if __name__ == "__main__":
    test_parse_list_quotes_and_spaces()
```

<details>

<summary>
**Failing input**: `"'"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 11, in <module>
    test_parse_list_quotes_and_spaces()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 6, in test_parse_list_quotes_and_spaces
    def test_parse_list_quotes_and_spaces(s):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 7, in test_parse_list_quotes_and_spaces
    result = parse_list(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: '__Pyx_L1'
Falsifying example: test_parse_list_quotes_and_spaces(
    s="'",
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:309
        /home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:310
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

# Test the failing case
parse_list("'")
```

<details>

<summary>
KeyError: '__Pyx_L1'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/repo.py", line 4, in <module>
    parse_list("'")
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

This violates expected behavior for several reasons:

1. **User-facing functionality**: The `parse_list` function is called when parsing distutils directives from special comments in Cython source files (e.g., `# distutils: extra_compile_args = -O2 -Wall`). Users write these directives manually in their .pyx files.

2. **Unhelpful error message**: The `KeyError: '__Pyx_L1'` error message exposes internal implementation details (placeholder names used by `strip_string_literals`) and provides no actionable information to help users understand what went wrong.

3. **Common user mistake**: An unbalanced quote is a plausible typo that users might make when editing these directive comments. The function should handle this gracefully with a clear error message or by treating it as a literal value.

4. **No documentation about restrictions**: The function's docstring shows examples with properly balanced quotes but doesn't specify that unbalanced quotes are forbidden or will cause crashes.

## Relevant Context

The root cause of the bug is a mismatch between how `strip_string_literals` generates placeholder keys and how the `unquote` function extracts them:

1. When `strip_string_literals("'")` is called, it detects an unclosed string literal and returns:
   - Transformed string: `"'__Pyx_L1_"` (note the trailing underscore in the placeholder)
   - Literals dictionary: `{'__Pyx_L1_': ''}` (key has trailing underscore)

2. The `unquote` function detects the leading quote and tries to extract the placeholder:
   - It uses `literal[1:-1]` which gives `"__Pyx_L1"` (missing the trailing underscore)
   - The actual key in the dictionary is `"__Pyx_L1_"` (with trailing underscore)
   - This mismatch causes the KeyError

Additional failing cases include:
- `'"'` → KeyError: '__Pyx_L1'
- `"''"` → KeyError: ''
- `'""'` → KeyError: ''
- `' "'` → KeyError: '__Pyx_L1'

The function is defined at `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:108-135`

Documentation: https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,11 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
+        if not literal:
+            return literal
+        if len(literal) >= 2 and literal[0] in "'\"" and literal[-1] == '_':
+            # Handle unclosed quotes that got a placeholder ending with underscore
+            return literals.get(literal[1:], '')
+        elif len(literal) >= 2 and literal[0] in "'\"":
             return literals[literal[1:-1]]
         else:
             return literal
```