# Bug Report: Cython.Build DistutilsInfo Comment Handling

**Target**: `Cython.Build.Dependencies.DistutilsInfo.__init__` and `parse_list`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When parsing distutils directive values containing inline comments, `parse_list` incorrectly includes comment text as a placeholder label (`#__Pyx_L1_`) instead of stripping it, causing invalid entries in configuration lists.

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
    assert len(result) == len(items)
```

**Failing input**: `items=['#', '0']`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, "/path/to/cython")

from Cython.Build.Dependencies import DistutilsInfo

source = """
# distutils: libraries = foo # this is a comment
"""

info = DistutilsInfo(source)
print("Libraries:", info.values.get('libraries'))
```

Expected: `['foo']`
Actual: `['foo', '#__Pyx_L1_']`

## Why This Is A Bug

Users writing distutils directives with inline comments expect standard Python comment semantics. The directive `# distutils: libraries = foo # comment` should parse only `['foo']`, not include a placeholder label. This causes build failures when the linker searches for a library named `#__Pyx_L1_`.

## Fix

Strip comments from directive values before parsing:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -191,7 +191,10 @@ class DistutilsInfo:
                 line = line[1:].lstrip()
                 kind = next((k for k in ("distutils:","cython:") if line.startswith(k)), None)
                 if kind is not None:
-                    key, _, value = [s.strip() for s in line[len(kind):].partition('=')]
+                    directive_line = line[len(kind):]
+                    comment_idx = directive_line.find('#')
+                    if comment_idx != -1:
+                        directive_line = directive_line[:comment_idx]
+                    key, _, value = [s.strip() for s in directive_line.partition('=')]
                     type = distutils_settings.get(key, None)
                     if line.startswith("cython:") and type is None: continue
                     if type in (list, transitive_list):
```