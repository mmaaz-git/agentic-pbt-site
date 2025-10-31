# Bug Report: Cython.Build parse_list Comment Handling

**Target**: `Cython.Build.Dependencies.parse_list` and `DistutilsInfo.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When parsing distutils directive values that contain inline comments, `parse_list` incorrectly includes the comment text as a label placeholder (`#__Pyx_L1_`) instead of stripping it. This causes invalid entries in configuration lists like `libraries`, `include_dirs`, etc.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1)))
@settings(max_examples=1000)
def test_parse_list_space_separated_count(items):
    assume(all(item.strip() for item in items))
    assume(all(' ' not in item and ',' not in item and '"' not in item and "'" not in item for item in items))

    list_str = ' '.join(items)
    result = parse_list(list_str)
    assert len(result) == len(items), f"Expected {len(items)} items, got {len(result)}"
```

**Failing input**: `items=['#', '0']` produces `list_str='# 0'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, "/path/to/cython/installation")

from Cython.Build.Dependencies import DistutilsInfo

source_with_comment = """
# distutils: libraries = foo # this is a comment explaining foo
"""

info = DistutilsInfo(source_with_comment)
print("Parsed libraries:", info.values.get('libraries'))

assert info.values.get('libraries') == ['foo', '#__Pyx_L1_']
```

Expected: `['foo']`
Actual: `['foo', '#__Pyx_L1_']`

## Why This Is A Bug

When users write distutils directives with inline comments for documentation:
```python
# distutils: libraries = m pthread  # math and threading libraries
```

They expect only `['m', 'pthread']` to be added to the libraries list. Instead, the bogus entry `'#__Pyx_L1_'` is included, which will cause linker errors when trying to find a library with that name.

This violates the principle that `#` introduces a comment in Python-like syntax, and the documented behavior that distutils directives follow Python commenting conventions.

## Fix

The bug occurs in `DistutilsInfo.__init__` (Dependencies.py:179-212). The value extracted from the directive line includes any trailing comment. Before passing to `parse_list`, comments should be stripped:

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