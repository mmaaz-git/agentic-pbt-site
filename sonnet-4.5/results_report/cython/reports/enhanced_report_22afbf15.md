# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Empty Quoted Strings

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function in Cython's build dependencies module crashes with a `KeyError` when parsing bracketed lists containing empty quoted strings like `[""]`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@given(st.lists(st.text(min_size=0, max_size=10), min_size=0, max_size=20))
@settings(max_examples=1000)
def test_parse_list_no_crash_on_empty_strings(items):
    quoted_items = [f'"{item}"' for item in items]
    input_str = '[' + ', '.join(quoted_items) + ']'
    result = parse_list(input_str)
    assert isinstance(result, list)

if __name__ == "__main__":
    test_parse_list_no_crash_on_empty_strings()
```

<details>

<summary>
**Failing input**: `items=['']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 14, in <module>
    test_parse_list_no_crash_on_empty_strings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 6, in test_parse_list_no_crash_on_empty_strings
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 10, in test_parse_list_no_crash_on_empty_strings
    result = parse_list(input_str)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: ''
Falsifying example: test_parse_list_no_crash_on_empty_strings(
    items=[''],
)
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

# Test case that should crash with KeyError
result = parse_list('[""]')
print(f"Result: {result}")
```

<details>

<summary>
KeyError crash when parsing empty quoted string
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/repo.py", line 4, in <module>
    result = parse_list('[""]')
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: ''
```
</details>

## Why This Is A Bug

The `parse_list` function is documented to handle various forms of quoted strings within bracketed lists. The function's docstring explicitly shows it should handle strings like `" "` (a single space), `","` (a comma), and other quoted content:

```python
>>> parse_list('[a, ",a", "a,", ",", ]')
['a', ',a', 'a,', ',']
>>> parse_list('a " " b')
['a', ' ', 'b']
```

Empty quoted strings (`""`) are valid string literals in Python and should be handled consistently with other quoted strings. However, the function crashes with a `KeyError` when encountering them.

The root cause is in the interaction between two functions:
1. `strip_string_literals()` - This function normalizes string literals in the input but doesn't create an entry in the `literals` dictionary for empty strings. When it encounters `""`, it leaves it as-is rather than adding an empty string to the literals map.
2. `unquote()` helper in `parse_list` - This function assumes all quoted strings have been extracted into the `literals` dictionary and tries to look them up using `literals[literal[1:-1]]`. For `""`, this becomes `literals['']`, causing a `KeyError` since empty strings weren't added to the dictionary.

This violates the principle of least surprise - if the function can handle `" "` (space) and `","` (comma), it should also handle `""` (empty string) without crashing.

## Relevant Context

The `parse_list` function is used in Cython's build system to parse configuration values that can be specified as either space-separated strings or comma-separated bracketed lists. It's particularly used for parsing build options and dependencies.

The function is located in `/Cython/Build/Dependencies.py` (lines 108-135) and relies on `strip_string_literals` from the same module to handle quoted strings that might contain delimiters.

Documentation link: https://cython.readthedocs.io/en/latest/
Source code: https://github.com/cython/cython/blob/master/Cython/Build/Dependencies.py

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,10 @@ def parse_list(s):
     def unquote(literal):
         literal = literal.strip()
         if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+            key = literal[1:-1]
+            if key == '':
+                return ''
+            return literals[key]
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```