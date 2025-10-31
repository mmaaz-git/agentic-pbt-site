# Bug Report: pyximport.install Duplicate Importers

**Target**: `pyximport.install()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Calling `pyximport.install(pyimport=True)` multiple times adds duplicate `PyImportMetaFinder` instances to `sys.meta_path` due to a logic error in the `_have_importers()` helper function.

## Property-Based Test

```python
import sys
import copy
from hypothesis import given, settings, strategies as st
import pyximport

@settings(max_examples=50)
@given(
    pyximport_flag=st.booleans(),
    pyimport_flag=st.booleans()
)
def test_install_twice_no_duplicates(pyximport_flag, pyimport_flag):
    original_meta_path = copy.copy(sys.meta_path)

    try:
        py1, pyx1 = pyximport.install(pyximport=pyximport_flag, pyimport=pyimport_flag)
        meta_path_after_first = copy.copy(sys.meta_path)

        py2, pyx2 = pyximport.install(pyximport=pyximport_flag, pyimport=pyimport_flag)
        meta_path_after_second = copy.copy(sys.meta_path)

        pyx_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyxImportMetaFinder))
        py_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyImportMetaFinder))

        if pyximport_flag:
            assert pyx_count <= 1
        if pyimport_flag:
            assert py_count <= 1
    finally:
        pyximport.uninstall(py1, pyx1)
        try:
            pyximport.uninstall(py2, pyx2)
        except:
            pass
        sys.meta_path[:] = original_meta_path
```

**Failing input**: `pyximport_flag=False, pyimport_flag=True`

## Reproducing the Bug

```python
import sys
import pyximport

py1, pyx1 = pyximport.install(pyximport=False, pyimport=True)
py2, pyx2 = pyximport.install(pyximport=False, pyimport=True)

py_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyImportMetaFinder))
print(f"PyImportMetaFinder instances in sys.meta_path: {py_count}")

pyximport.uninstall(py1, pyx1)
pyximport.uninstall(py2, pyx2)
```

## Why This Is A Bug

The `_have_importers()` function uses incorrect nested `isinstance` checks. Since `PyImportMetaFinder` and `PyxImportMetaFinder` are independent classes (neither inherits from the other), the code:

```python
if isinstance(importer, PyxImportMetaFinder):
    if isinstance(importer, PyImportMetaFinder):
        has_py_importer = True
```

Never detects existing `PyImportMetaFinder` instances because they don't satisfy the outer `isinstance(importer, PyxImportMetaFinder)` check. This causes `install()` to repeatedly add new importers to `sys.meta_path`, violating the function's intent to prevent duplicates.

## Fix

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -1,9 +1,10 @@
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
+        if isinstance(importer, PyxImportMetaFinder):
+            has_pyx_importer = True

     return has_py_importer, has_pyx_importer
```