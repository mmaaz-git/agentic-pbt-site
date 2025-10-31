# Bug Report: Cython.Distutils Directives Type Validation Missing

**Target**: `Cython.Distutils.build_ext.finalize_options`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `finalize_options` method does not validate or parse the `cython_directives` option when it's a string (as it would be when set from command-line), causing a crash in `build_extension`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(st.text(min_size=1, max_size=100))
def test_directives_type_validation(directive_value):
    """
    Property: cython_directives should be validated or converted to dict
    """
    dist = Distribution()
    cmd = build_ext(dist)
    cmd.initialize_options()

    cmd.cython_directives = directive_value
    cmd.finalize_options()

    result = cmd.cython_directives
    assert isinstance(result, dict), \
        f"cython_directives should be dict after finalize, got {type(result)}"
```

**Failing input**: Any string value, e.g., `"boundscheck=True"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import build_ext

dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()

cmd.cython_directives = "boundscheck=True"
cmd.finalize_options()

directives = dict(cmd.cython_directives)
```

**Output**:
```
ValueError: dictionary update sequence element #0 has length 1; 2 is required
```

## Why This Is A Bug

The `cython_directives` option can be set from the command-line as a string (per the user_options definition at line 44-45). However, `finalize_options` only checks if it's None and converts it to `{}`, but doesn't handle the string case:

```python
if self.cython_directives is None:
    self.cython_directives = {}
```

Later, at line 107 in `build_extension`, the code assumes it's a dict:
```python
directives = dict(self.cython_directives)
```

When `cython_directives` is a string, `dict()` tries to interpret it as a sequence of key-value pairs and fails.

This violates the contract that command-line options should be properly processed by `finalize_options`.

## Fix

The fix should parse the string format (likely key=value pairs) into a dictionary:

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -75,6 +75,16 @@ class build_ext(_build_ext):
                 self.cython_include_dirs.split(os.pathsep)
         if self.cython_directives is None:
             self.cython_directives = {}
+        elif isinstance(self.cython_directives, str):
+            directives = {}
+            for directive in self.cython_directives.split(','):
+                directive = directive.strip()
+                if '=' in directive:
+                    key, value = directive.split('=', 1)
+                    directives[key.strip()] = value.strip()
+                elif directive:
+                    directives[directive] = True
+            self.cython_directives = directives
```