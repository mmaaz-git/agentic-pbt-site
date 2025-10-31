# Bug Report: Cython.Distutils Unvalidated Directives Cause Crash

**Target**: `Cython.Distutils.build_ext.finalize_options`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `cython_directives` is set as a string (e.g., from command-line), `finalize_options` doesn't parse or validate it, causing a crash when `build_extension` attempts to use it.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(st.text(min_size=1, max_size=100))
def test_directives_type_validation(directive_value):
    """
    Property: finalize_options should ensure cython_directives is a dict
    """
    dist = Distribution()
    cmd = build_ext(dist)
    cmd.initialize_options()

    cmd.cython_directives = directive_value
    cmd.finalize_options()

    assert isinstance(cmd.cython_directives, dict), \
        f"Expected dict, got {type(cmd.cython_directives)}"
```

**Failing input**: Any string, e.g., `"boundscheck=True"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import build_ext

dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()

cmd.cython_directives = "boundscheck=True,wraparound=False"
cmd.finalize_options()

directives = dict(cmd.cython_directives)
```

**Output**:
```
ValueError: dictionary update sequence element #0 has length 1; 2 is required
```

## Why This Is A Bug

The `cython-directives` option is defined in `user_options` (line 44-45) as a string option that can be set from the command-line. When set from command-line, it will be a string. However, `finalize_options` (lines 77-78) only handles the None case:

```python
if self.cython_directives is None:
    self.cython_directives = {}
```

Later, in `build_extension` (line 107), the code assumes it's already a dict:
```python
directives = dict(self.cython_directives)
```

When `cython_directives` is a string, this crashes because `dict()` tries to interpret the string as an iterable of key-value pairs.

This is a critical flaw because:
1. Users can set this option from command-line: `python setup.py build_ext --cython-directives="..."`
2. The failure happens during build, not during option parsing
3. The error message is confusing and doesn't indicate the real problem

## Fix

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -76,6 +76,17 @@ class build_ext(_build_ext):
                 self.cython_include_dirs.split(os.pathsep)
         if self.cython_directives is None:
             self.cython_directives = {}
+        elif isinstance(self.cython_directives, str):
+            directives = {}
+            if self.cython_directives.strip():
+                for directive in self.cython_directives.split(','):
+                    directive = directive.strip()
+                    if '=' in directive:
+                        key, value = directive.split('=', 1)
+                        directives[key.strip()] = value.strip()
+                    elif directive:
+                        directives[directive] = True
+            self.cython_directives = directives
```