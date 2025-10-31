# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Empty and Malformed Quoted Strings

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list()` function crashes with `KeyError` when parsing empty quoted strings or unclosed quotes due to implementation issues in string literal handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@settings(max_examples=1000)
@given(st.lists(st.text(min_size=0, max_size=20), min_size=0, max_size=10))
def test_parse_list_quoted_bracket_format_no_crash(items):
    s = '[' + ', '.join(f'"{item}"' for item in items) + ']'
    result = parse_list(s)


if __name__ == "__main__":
    # Run the property-based test
    test_parse_list_quoted_bracket_format_no_crash()
```

<details>

<summary>
**Failing input**: `items=['']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 14, in <module>
    test_parse_list_quoted_bracket_format_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 6, in test_parse_list_quoted_bracket_format_no_crash
    @given(st.lists(st.text(min_size=0, max_size=20), min_size=0, max_size=10))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 9, in test_parse_list_quoted_bracket_format_no_crash
    result = parse_list(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: ''
Falsifying example: test_parse_list_quoted_bracket_format_no_crash(
    items=[''],
)
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

# Test all the failing cases mentioned in the report
test_cases = [
    '[""]',           # Empty string in list
    "['']",           # Empty single-quoted string in list
    '""',             # Empty string without brackets
    "''",             # Empty single-quoted string without brackets
    '"',              # Unclosed double quote
    "'",              # Unclosed single quote
    '["\\"]',         # Escaped quote in list
    '[a, "", b]',     # Empty string in mixed list
]

for test in test_cases:
    try:
        result = parse_list(test)
        print(f"parse_list('{test}') = {result}")
    except KeyError as e:
        print(f"parse_list('{test}') -> KeyError: {e}")
    except Exception as e:
        print(f"parse_list('{test}') -> {type(e).__name__}: {e}")
```

<details>

<summary>
KeyError crashes for all test cases
</summary>
```
parse_list('[""]') -> KeyError: ''
parse_list('['']') -> KeyError: ''
parse_list('""') -> KeyError: ''
parse_list('''') -> KeyError: ''
parse_list('"') -> KeyError: '__Pyx_L1'
parse_list(''') -> KeyError: '__Pyx_L1'
parse_list('["\"]') -> KeyError: '__Pyx_L1'
parse_list('[a, "", b]') -> KeyError: ''
```
</details>

## Why This Is A Bug

The `parse_list()` function is a public API in Cython's build system used to parse configuration values from distutils directives in source code comments (e.g., `# distutils: libraries = foo bar`). The function should handle any valid Python string literal syntax without crashing.

The crashes occur due to two distinct implementation issues:

1. **Empty String Literals Not Captured**: The `strip_string_literals()` function fails to capture empty string literals (`""` or `''`) in its mapping dictionary. When `parse_list()` encounters `'[""]'`, the `strip_string_literals()` returns `('[""]', {})` with an empty dictionary. The nested `unquote()` function then tries to look up an empty string key in this empty dictionary, causing `KeyError: ''`.

2. **Label Format Mismatch**: For non-empty strings, `strip_string_literals()` creates placeholder labels with format `__Pyx_L{N}_` (note the trailing underscore). However, when the `unquote()` function processes a quoted placeholder like `"__Pyx_L1_"`, it uses `literal[1:-1]` to extract the key, which removes both the opening quote AND the trailing character, producing `__Pyx_L1` without the underscore. This causes `KeyError: '__Pyx_L1'` when looking up in the literals dictionary.

The function's docstring shows it should support various quoted string formats:
```python
>>> parse_list('a " " b')
['a', ' ', 'b']
>>> parse_list('[a, ",a", "a,", ",", ]')
['a', ',a', 'a,', ',']
```

Since the function already handles quoted strings with spaces and special characters, it should logically also handle empty strings, which are valid Python string literals. The current behavior of crashing with an unhandled KeyError is clearly unintended.

## Relevant Context

- Location: `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py`
- Function definition starts at line 108
- The problematic `unquote` nested function is at lines 129-134
- The `strip_string_literals` function that should capture empty strings starts at line 282
- This function is used internally by Cython's build system to parse configuration from source comments

The issue affects users who might have empty strings in their distutils directives, either intentionally or accidentally. While empty library names or source files don't make practical sense, the parser should handle this gracefully rather than crashing.

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -128,8 +128,13 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
+        if not literal:
+            return literal
         if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+            if literal == '""' or literal == "''":
+                return ''
+            key = literal[1:-1]
+            return literals.get(key, key)
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```