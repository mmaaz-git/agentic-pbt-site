# Bug Report: Cython.Distutils.old_build_ext Option Precedence Bug

**Target**: `Cython.Distutils.old_build_ext.cython_sources`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cython_sources` method incorrectly handles falsy values when merging command-line and extension options, causing explicitly set command-line values of `0` or `False` to be overridden by extension attribute values.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.booleans())
def test_create_listing_or_operator_bug(ext_value):
    """Command value should take precedence over extension value, even when falsy."""
    dist = Distribution()
    cmd = old_build_ext(dist)
    cmd.initialize_options()
    cmd.finalize_options()

    cmd.cython_create_listing = 0
    ext = Extension("test", ["test.pyx"], cython_create_listing=ext_value)

    create_listing = cmd.cython_create_listing or getattr(ext, 'cython_create_listing', 0)

    assert create_listing == 0, f"Bug: expected 0 (cmd value), got {create_listing}"
```

**Failing input**: `ext_value=True`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import Extension
from Cython.Distutils.old_build_ext import old_build_ext

dist = Distribution()
cmd = old_build_ext(dist)
cmd.initialize_options()
cmd.finalize_options()

cmd.cython_create_listing = 0
ext = Extension("test", ["test.pyx"], cython_create_listing=True)

create_listing = cmd.cython_create_listing or getattr(ext, 'cython_create_listing', 0)

print(f"Command value: {cmd.cython_create_listing}")
print(f"Extension value: {ext.cython_create_listing}")
print(f"Result: {create_listing}")
```

Output:
```
Command value: 0
Extension value: True
Result: True
```

## Why This Is A Bug

The `cython_sources` method uses the `or` operator to merge command-line and extension options (lines 223-234 in old_build_ext.py). This treats falsy values (`0`, `False`) as "not set", causing the extension value to be used instead of the explicitly set command value.

This affects all options merged this way:
- `cython_create_listing` (lines 223-224)
- `cython_line_directives` (lines 225-226)
- `no_c_in_traceback` (lines 227-228)
- `cython_cplus` (lines 229-230)
- `cython_gen_pxi` (line 231)
- `cython_gdb` (line 232)
- `cython_compile_time_env` (lines 233-234)

This violates the documented precedence where command-line options should override extension settings.

Note: This module is marked as deprecated in the docstring, but the bug still affects users who haven't migrated to `cythonize()`.

## Fix

Replace the `or` operators with proper None-aware checks:

```diff
--- a/Cython/Distutils/old_build_ext.py
+++ b/Cython/Distutils/old_build_ext.py
@@ -220,17 +220,27 @@ class old_build_ext(_build_ext.build_ext):
         #                (extension.language != None and \
         #                    extension.language.lower() == 'c++')

-        create_listing = self.cython_create_listing or \
-            getattr(extension, 'cython_create_listing', 0)
-        line_directives = self.cython_line_directives or \
-            getattr(extension, 'cython_line_directives', 0)
-        no_c_in_traceback = self.no_c_in_traceback or \
-            getattr(extension, 'no_c_in_traceback', 0)
-        cplus = self.cython_cplus or getattr(extension, 'cython_cplus', 0) or \
+        create_listing = self.cython_create_listing if self.cython_create_listing else \
+            getattr(extension, 'cython_create_listing', 0)
+        line_directives = self.cython_line_directives if self.cython_line_directives else \
+            getattr(extension, 'cython_line_directives', 0)
+        no_c_in_traceback = self.no_c_in_traceback if self.no_c_in_traceback else \
+            getattr(extension, 'no_c_in_traceback', 0)
+        cplus = (self.cython_cplus if self.cython_cplus else \
+            getattr(extension, 'cython_cplus', 0)) or \
                 (extension.language and extension.language.lower() == 'c++')
-        cython_gen_pxi = self.cython_gen_pxi or getattr(extension, 'cython_gen_pxi', 0)
-        cython_gdb = self.cython_gdb or getattr(extension, 'cython_gdb', False)
-        cython_compile_time_env = self.cython_compile_time_env or \
+        cython_gen_pxi = self.cython_gen_pxi if self.cython_gen_pxi else \
+            getattr(extension, 'cython_gen_pxi', 0)
+        cython_gdb = self.cython_gdb if self.cython_gdb is not False else \
+            getattr(extension, 'cython_gdb', False)
+        cython_compile_time_env = self.cython_compile_time_env if self.cython_compile_time_env is not None else \
             getattr(extension, 'cython_compile_time_env', None)
+
+Alternative fix (better): Initialize these values to None instead of 0/False, then check for None:
+
+```python
+create_listing = (self.cython_create_listing
+                  if self.cython_create_listing is not None
+                  else getattr(extension, 'cython_create_listing', 0))
+```