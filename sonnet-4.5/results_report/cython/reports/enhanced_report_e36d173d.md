# Bug Report: Cython.Build.Dependencies.parse_list Empty String KeyError

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a KeyError when parsing lists containing empty string literals (`[""]` or `['']`), due to a mismatch between how `strip_string_literals` and `parse_list` handle empty quoted strings.

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
test_parse_list_bracket_delimited_round_trip()
```

<details>

<summary>
**Failing input**: `items=['']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 13, in <module>
    test_parse_list_bracket_delimited_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 5, in test_parse_list_bracket_delimited_round_trip
    def test_parse_list_bracket_delimited_round_trip(items):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 9, in test_parse_list_bracket_delimited_round_trip
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

# Test case from the bug report - empty string in bracket-delimited list
result = parse_list('[""]')
print(f"Result: {result}")
```

<details>

<summary>
KeyError: '' when parsing empty string literal
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/repo.py", line 4, in <module>
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

The `parse_list` function is designed to parse distutils-style list configurations, which are used extensively in Cython's build system for processing directives like `include_dirs`, `libraries`, and other build settings. The function's docstring shows it should handle quoted strings within bracket-delimited lists, as evidenced by examples like `'a " " b'` (single space) and `'[a, ",a", "a,", ",", ]'` (strings with commas).

Empty strings are valid Python string literals and legitimate configuration values. Users might need empty strings in configurations for various reasons:
- As placeholders in configuration lists
- To represent the current directory (e.g., `include_dirs=[""]`)
- As default values that get populated later

The crash occurs due to an implementation inconsistency between two internal functions:

1. `strip_string_literals` (line 359 in Dependencies.py) explicitly checks `if quote and len(quote) != 2:` which causes it to skip processing empty string literals (`""` or `''`), leaving them unchanged in the text rather than replacing them with placeholder labels.

2. `parse_list`'s `unquote` function (lines 131-132) assumes ALL quoted strings have been replaced with labels by `strip_string_literals` and tries to look them up in the `literals` dictionary. When it encounters `""`, it tries to access `literals['']` which doesn't exist, causing the KeyError.

## Relevant Context

The issue affects all versions of Cython that use this parsing mechanism for distutils configuration. The bug is triggered when:
- Using bracket-delimited lists with commas: `[""]`, `['']`
- Using space-delimited lists with empty strings: `parse_list('"" "hello"')`
- Mixed lists containing empty strings: `["hello", "", "world"]`

The function correctly handles non-empty quoted strings, including single spaces (`" "`), which shows the special case for empty strings was likely an oversight rather than intentional.

Documentation: The function's docstring at lines 109-122 in Dependencies.py shows various examples but doesn't explicitly demonstrate empty string handling, though it implies all valid quoted strings should work.

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,10 @@ def parse_list(s):
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
+        if len(literal) >= 2 and literal[0] in "'\"" and literal[-1] == literal[0]:
+            # Handle empty strings that weren't processed by strip_string_literals
+            if len(literal) == 2:
+                return ''
             return literals[literal[1:-1]]
         else:
             return literal
```