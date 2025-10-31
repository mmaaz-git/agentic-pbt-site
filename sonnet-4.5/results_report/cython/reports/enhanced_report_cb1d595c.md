# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Empty String

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a KeyError when parsing bracket-delimited lists containing empty quoted strings (e.g., `[""]` or `['']`).

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text()))
def test_parse_list_bracket_delimited_round_trip(items):
    assume(all('"' not in item for item in items))
    quoted_items = [f'"{item}"' for item in items]
    input_str = '[' + ', '.join(quoted_items) + ']'
    result = parse_list(input_str)
    assert result == items

# Run the test
if __name__ == "__main__":
    test_parse_list_bracket_delimited_round_trip()
```

<details>

<summary>
**Failing input**: `items=['']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 14, in <module>
    test_parse_list_bracket_delimited_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 5, in test_parse_list_bracket_delimited_round_trip
    def test_parse_list_bracket_delimited_round_trip(items):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 9, in test_parse_list_bracket_delimited_round_trip
    result = parse_list(input_str)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: ''
Falsifying example: test_parse_list_bracket_delimited_round_trip(
    items=[''],
)
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

# Test case that should work but crashes with KeyError
result = parse_list('[""]')
print(f"Result: {result}")
```

<details>

<summary>
KeyError: '' when parsing empty string in quotes
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/repo.py", line 4, in <module>
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

The `parse_list` function is designed to parse distutils-style configuration lists. According to its docstring, it should handle quoted strings in bracket-delimited lists. The function includes examples showing successful parsing of various quoted strings like `" "` (a space) and `","` (a comma). An empty string `""` is a valid quoted string literal that follows the same pattern.

The root cause is an implementation mismatch between two internal functions:

1. **`strip_string_literals` function (line 359)**: This function explicitly skips empty string literals. When it encounters a quote token where `len(quote) == 2` (like `""` or `''`), it does NOT replace it with a label. This check appears designed to ignore empty quoted strings.

2. **`parse_list` function (lines 129-135)**: The `unquote` helper function assumes ALL quoted strings have been replaced with labels by `strip_string_literals`. When it encounters `""`, it tries to look up the empty string (`literal[1:-1]` = `""[1:-1]` = `""`) in the `literals` dictionary, causing a KeyError.

This violates the principle of least surprise. Users would reasonably expect empty strings to be valid list elements since:
- The function handles other quoted strings correctly
- Empty strings are valid in Python lists
- Empty values might be used as placeholders or to indicate current directory in path lists

## Relevant Context

The `parse_list` function is used throughout Cython's build system to parse configuration values from distutils-style comments in `.pyx` and `.pxd` files. For example, it parses directives like:
```python
# distutils: include_dirs = ["", "/usr/include", ""]
# distutils: libraries = ["mylib", ""]
```

The bug affects line 198 of Dependencies.py where `parse_list` is called to parse these configuration values:
```python
if type in (list, transitive_list):
    value = parse_list(value)  # <-- Crashes here if value contains [""]
```

Links to relevant code:
- Function definition: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py:108`
- Where crash occurs: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py:132`
- Root cause in strip_string_literals: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py:359`

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -128,8 +128,12 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+        if literal and literal[0] in "'\"" and literal[0] == literal[-1]:
+            # Handle empty strings that weren't replaced by strip_string_literals
+            if len(literal) == 2:
+                return ''
+            else:
+                return literals[literal[1:-1]]
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```