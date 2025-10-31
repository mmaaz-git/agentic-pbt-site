# Bug Report: Cython.Build.Dependencies parse_list Incorrect Comment Handling

**Target**: `Cython.Build.Dependencies.parse_list` and `DistutilsInfo.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_list` function incorrectly processes strings containing `#` characters by transforming them into label placeholders (`#__Pyx_L1_`), causing distutils directives with inline comments to include invalid library names that lead to linker errors.

## Property-Based Test

```python
#!/usr/bin/env python3

from hypothesis import given, strategies as st, assume, settings, example
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1)))
@settings(max_examples=1000)
@example(['#', '0'])  # Known failing case
def test_parse_list_space_separated_count(items):
    assume(all(item.strip() for item in items))
    assume(all(' ' not in item and ',' not in item and '"' not in item and "'" not in item for item in items))

    list_str = ' '.join(items)
    result = parse_list(list_str)
    assert len(result) == len(items), f"Expected {len(items)} items from input {items!r} (string: {list_str!r}), got {len(result)} items: {result!r}"

# Run the test
if __name__ == "__main__":
    from hypothesis import reproduce_failure
    import sys
    try:
        test_parse_list_space_separated_count()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)
```

<details>

<summary>
**Failing input**: `items=['#', '0']`
</summary>
```
Test failed with error: Expected 2 items from input ['#', '0'] (string: '# 0'), got 1 items: ['#__Pyx_L1_']
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3

from Cython.Build.Dependencies import DistutilsInfo

source_with_comment = """
# distutils: libraries = foo # this is a comment explaining foo
"""

info = DistutilsInfo(source_with_comment)
print("Parsed libraries:", info.values.get('libraries'))

# Show what was actually parsed
if info.values.get('libraries'):
    print("Number of libraries:", len(info.values.get('libraries')))
    for i, lib in enumerate(info.values.get('libraries')):
        print(f"  Library {i}: '{lib}'")

# Test the assertion
expected = ['foo']
actual = info.values.get('libraries')
print(f"\nExpected: {expected}")
print(f"Actual: {actual}")

if actual == expected:
    print("✓ Test passed: Libraries parsed correctly")
else:
    print("✗ Test failed: Libraries include comment text")
    if actual and len(actual) > 1 and actual[1].startswith('#'):
        print(f"  Bug confirmed: Comment transformed into label '{actual[1]}'")
```

<details>

<summary>
Libraries incorrectly include comment placeholder
</summary>
```
Parsed libraries: ['foo', '#__Pyx_L1_']
Number of libraries: 2
  Library 0: 'foo'
  Library 1: '#__Pyx_L1_'

Expected: ['foo']
Actual: ['foo', '#__Pyx_L1_']
✗ Test failed: Libraries include comment text
  Bug confirmed: Comment transformed into label '#__Pyx_L1_'
```
</details>

## Why This Is A Bug

This bug violates the fundamental Python convention that `#` introduces a comment. When users write distutils directives with inline comments for documentation purposes (a standard practice in Python development), they expect the comment to be ignored, not transformed into configuration values.

The issue occurs because:
1. The `strip_string_literals` function (called by `parse_list`) treats `#` as the start of a comment and replaces everything after it with a label like `#__Pyx_L1_`
2. This label is then parsed as if it were an actual library name
3. The build system attempts to link against this non-existent library, causing compilation failures

This breaks real-world use cases where developers document their build configuration:
```python
# distutils: libraries = m pthread  # math and threading libraries
# distutils: include_dirs = /usr/local/include  # custom headers location
```

## Relevant Context

The bug stems from the interaction between two functions:
- `strip_string_literals()` at Dependencies.py:282 - replaces comments with labels
- `parse_list()` at Dependencies.py:108 - parses space-separated values but receives pre-processed strings with comment labels
- `DistutilsInfo.__init__()` at Dependencies.py:179 - extracts directive values but doesn't strip inline comments before parsing

The root cause is in `DistutilsInfo.__init__()` which passes the full directive value (including inline comments) to `parse_list`. The `strip_string_literals` function then transforms the comment into a label placeholder that gets incorrectly interpreted as a value.

Documentation: While Cython's documentation doesn't explicitly state that inline comments are supported in distutils directives, Python's universal convention is that `#` introduces a comment. The current behavior creates invalid configuration values, which is objectively incorrect regardless of documentation.

## Proposed Fix

Strip inline comments from directive values before parsing them in `DistutilsInfo.__init__`:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -191,7 +191,10 @@ class DistutilsInfo:
                 line = line[1:].lstrip()
                 kind = next((k for k in ("distutils:","cython:") if line.startswith(k)), None)
                 if kind is not None:
-                    key, _, value = [s.strip() for s in line[len(kind):].partition('=')]
+                    directive_line = line[len(kind):]
+                    # Strip inline comments before parsing the value
+                    directive_line = directive_line.split('#')[0]
+                    key, _, value = [s.strip() for s in directive_line.partition('=')]
                     type = distutils_settings.get(key, None)
                     if line.startswith("cython:") and type is None: continue
                     if type in (list, transitive_list):
```