# Bug Report: pyximport._have_importers() Fails to Detect PyImportMetaFinder

**Target**: `pyximport.pyximport._have_importers()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_have_importers()` function fails to detect when `PyImportMetaFinder` is installed in `sys.meta_path`, incorrectly returning `has_py_importer=False` when it should return `True`.

## Property-Based Test

```python
import sys
from pyximport.pyximport import _have_importers, PyImportMetaFinder, PyxImportMetaFinder
from hypothesis import given, strategies as st, settings


@settings(max_examples=50)
@given(
    add_py=st.booleans(),
    add_pyx=st.booleans()
)
def test_have_importers_detects_all_finders(add_py, add_pyx):
    original_meta_path = sys.meta_path.copy()
    try:
        sys.meta_path = []

        if add_py:
            sys.meta_path.append(PyImportMetaFinder())
        if add_pyx:
            sys.meta_path.append(PyxImportMetaFinder())

        has_py, has_pyx = _have_importers()

        assert has_py == add_py, f"Expected has_py={add_py}, got {has_py}"
        assert has_pyx == add_pyx, f"Expected has_pyx={add_pyx}, got {has_pyx}"

    finally:
        sys.meta_path[:] = original_meta_path
```

**Failing input**: `add_py=True, add_pyx=False`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import _have_importers, PyImportMetaFinder

original_meta_path = sys.meta_path.copy()
try:
    sys.meta_path = []
    sys.meta_path.append(PyImportMetaFinder())

    has_py, has_pyx = _have_importers()
    print(f"has_py={has_py}, has_pyx={has_pyx}")
    assert has_py == True, f"Bug: has_py is {has_py}, expected True"

finally:
    sys.meta_path[:] = original_meta_path
```

## Why This Is A Bug

The function `_have_importers()` (lines 355-365) uses incorrect nested `isinstance` checks that assume `PyImportMetaFinder` is a subclass of `PyxImportMetaFinder`. However, both classes independently inherit from `MetaPathFinder` (lines 222 and 256), making them siblings, not parent-child.

The nested check at lines 359-363:
```python
if isinstance(importer, PyxImportMetaFinder):
    if isinstance(importer, PyImportMetaFinder):
        has_py_importer = True
    else:
        has_pyx_importer = True
```

When a `PyImportMetaFinder` is in `sys.meta_path`, the outer `isinstance(importer, PyxImportMetaFinder)` check fails, so `has_py_importer` is never set to `True`. This causes `install()` (line 434) to incorrectly re-install an already-installed `PyImportMetaFinder`.

## Fix

```diff
--- a/pyximport.py
+++ b/pyximport.py
@@ -356,10 +356,10 @@ def _have_importers():
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