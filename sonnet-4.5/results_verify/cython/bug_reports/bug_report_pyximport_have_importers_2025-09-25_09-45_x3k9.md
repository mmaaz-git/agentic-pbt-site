# Bug Report: pyximport._have_importers() Fails to Detect PyImportMetaFinder

**Target**: `pyximport.pyximport._have_importers()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_have_importers()` function fails to detect `PyImportMetaFinder` instances in `sys.meta_path` due to incorrect `isinstance` check ordering. Since `PyImportMetaFinder` and `PyxImportMetaFinder` are sibling classes (not parent-child), the outer `isinstance` check excludes `PyImportMetaFinder` from detection. This causes `install(pyimport=True)` to install duplicate importers when called multiple times.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from pyximport.pyximport import _have_importers, PyxImportMetaFinder, PyImportMetaFinder


@settings(max_examples=500)
@given(st.booleans(), st.booleans())
def test_have_importers_detects_both_types(add_pyx, add_py):
    original_meta_path = sys.meta_path.copy()

    try:
        sys.meta_path = []

        if add_pyx:
            sys.meta_path.append(PyxImportMetaFinder())

        if add_py:
            sys.meta_path.append(PyImportMetaFinder())

        has_py, has_pyx = _have_importers()

        assert has_pyx == add_pyx, \
            f"Expected has_pyx={add_pyx}, got {has_pyx}. meta_path types: {[type(x).__name__ for x in sys.meta_path]}"

        assert has_py == add_py, \
            f"Expected has_py={add_py}, got {has_py}. meta_path types: {[type(x).__name__ for x in sys.meta_path]}"

    finally:
        sys.meta_path[:] = original_meta_path
```

**Failing input**: `add_pyx=False, add_py=True`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import _have_importers, PyImportMetaFinder

original_meta_path = sys.meta_path.copy()

sys.meta_path = [PyImportMetaFinder()]
has_py, has_pyx = _have_importers()

print(f"meta_path contains: PyImportMetaFinder")
print(f"_have_importers() returned: has_py={has_py}, has_pyx={has_pyx}")
print(f"Expected: has_py=True, has_pyx=False")
print(f"Bug: has_py={has_py} (should be True)")

sys.meta_path[:] = original_meta_path
```

## Why This Is A Bug

The `_have_importers()` function (lines 355-365 in pyximport.py) has the following logic:

```python
for importer in sys.meta_path:
    if isinstance(importer, PyxImportMetaFinder):
        if isinstance(importer, PyImportMetaFinder):
            has_py_importer = True
        else:
            has_pyx_importer = True
```

This logic assumes `PyImportMetaFinder` is a subclass of `PyxImportMetaFinder`. However, both classes independently inherit from `MetaPathFinder` (lines 222 and 256), making them sibling classes. Therefore:

1. A `PyImportMetaFinder` instance fails the outer `isinstance(importer, PyxImportMetaFinder)` check
2. The function returns `has_py_importer=False` even when `PyImportMetaFinder` is present
3. When `install(pyimport=True)` is called multiple times, it installs duplicate `PyImportMetaFinder` instances
4. This violates the expected idempotence of the `install()` function

Real-world impact: Performance degradation and unexpected behavior when import hooks are reinstalled.

## Fix

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -355,11 +355,10 @@ class PyxArgs(object):
 def _have_importers():
     has_py_importer = False
     has_pyx_importer = False
     for importer in sys.meta_path:
-        if isinstance(importer, PyxImportMetaFinder):
-            if isinstance(importer, PyImportMetaFinder):
-                has_py_importer = True
-            else:
-                has_pyx_importer = True
+        if isinstance(importer, PyImportMetaFinder):
+            has_py_importer = True
+        elif isinstance(importer, PyxImportMetaFinder):
+            has_pyx_importer = True

     return has_py_importer, has_pyx_importer
```