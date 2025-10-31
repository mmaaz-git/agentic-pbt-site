# Bug Report: pyximport install() Not Idempotent for pyimport=True

**Target**: `pyximport.install()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `install()` function is not idempotent when called with `pyimport=True`. Each call adds a new `PyImportMetaFinder` instance to `sys.meta_path`, causing the import system to check the same hook multiple times and degrading performance.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import sys
import pyximport

@given(
    pyximport_flag=st.booleans(),
    pyimport_flag=st.booleans(),
    num_calls=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_install_idempotence(pyximport_flag, pyimport_flag, num_calls):
    assume(pyximport_flag or pyimport_flag)

    original_meta_path = sys.meta_path.copy()

    try:
        for _ in range(num_calls):
            pyximport.install(
                pyximport=pyximport_flag,
                pyimport=pyimport_flag
            )

        py_importers = [imp for imp in sys.meta_path
                       if isinstance(imp, pyximport.PyImportMetaFinder)]

        if pyimport_flag:
            assert len(py_importers) == 1, \
                f"Expected 1 py importer, got {len(py_importers)}"
    finally:
        sys.meta_path = original_meta_path
```

**Failing input**: `test_install_idempotence(pyximport_flag=False, pyimport_flag=True, num_calls=2)`

## Reproducing the Bug

```python
import sys
import pyximport

original_meta_path = sys.meta_path.copy()

print("Initial count:", len([i for i in sys.meta_path if isinstance(i, pyximport.PyImportMetaFinder)]))

pyximport.install(pyimport=True, pyximport=False)
print("After 1st install:", len([i for i in sys.meta_path if isinstance(i, pyximport.PyImportMetaFinder)]))

pyximport.install(pyimport=True, pyximport=False)
print("After 2nd install:", len([i for i in sys.meta_path if isinstance(i, pyximport.PyImportMetaFinder)]))

pyximport.install(pyimport=True, pyximport=False)
print("After 3rd install:", len([i for i in sys.meta_path if isinstance(i, pyximport.PyImportMetaFinder)]))

sys.meta_path = original_meta_path
```

Expected output: `0, 1, 1, 1` (idempotent)
Actual output: `0, 1, 2, 3` (not idempotent)

## Why This Is A Bug

The root cause is in the `_have_importers()` function at lines 355-365. The function uses flawed logic that assumes `PyImportMetaFinder` is a subclass of `PyxImportMetaFinder`, but they are sibling classes:

```python
def _have_importers():
    has_py_importer = False
    has_pyx_importer = False
    for importer in sys.meta_path:
        if isinstance(importer, PyxImportMetaFinder):  # PyImportMetaFinder fails this check!
            if isinstance(importer, PyImportMetaFinder):
                has_py_importer = True
            else:
                has_pyx_importer = True
    return has_py_importer, has_pyx_importer
```

Since `PyImportMetaFinder` is NOT a subclass of `PyxImportMetaFinder`, instances of `PyImportMetaFinder` never pass the outer `isinstance` check, so `has_py_importer` always returns `False`. This causes `install()` to add a new importer every time.

The function violates its documented intent (checking if importers are already installed) and breaks idempotence - a reasonable expectation for an `install()` function.

## Fix

```diff
--- a/pyximport.py
+++ b/pyximport.py
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