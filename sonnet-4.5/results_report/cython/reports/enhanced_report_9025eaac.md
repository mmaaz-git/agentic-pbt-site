# Bug Report: pyximport.install Duplicate PyImportMetaFinder Instances

**Target**: `pyximport.install()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Calling `pyximport.install(pyimport=True)` multiple times adds duplicate `PyImportMetaFinder` instances to `sys.meta_path` due to incorrect nested `isinstance` checks in the `_have_importers()` helper function.

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
            assert pyx_count <= 1, f"Found {pyx_count} PyxImportMetaFinder instances, expected <= 1"
        if pyimport_flag:
            assert py_count <= 1, f"Found {py_count} PyImportMetaFinder instances, expected <= 1"
    finally:
        pyximport.uninstall(py1, pyx1)
        try:
            pyximport.uninstall(py2, pyx2)
        except:
            pass
        sys.meta_path[:] = original_meta_path

if __name__ == "__main__":
    test_install_twice_no_duplicates()
```

<details>

<summary>
**Failing input**: `pyximport_flag=False, pyimport_flag=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 37, in <module>
    test_install_twice_no_duplicates()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 7, in test_install_twice_no_duplicates
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 27, in test_install_twice_no_duplicates
    assert py_count <= 1, f"Found {py_count} PyImportMetaFinder instances, expected <= 1"
           ^^^^^^^^^^^^^
AssertionError: Found 2 PyImportMetaFinder instances, expected <= 1
Falsifying example: test_install_twice_no_duplicates(
    pyximport_flag=False,
    pyimport_flag=True,
)
```
</details>

## Reproducing the Bug

```python
import sys
import pyximport

# First install with pyimport=True
print("Initial sys.meta_path length:", len(sys.meta_path))
py1, pyx1 = pyximport.install(pyximport=False, pyimport=True)

# Count PyImportMetaFinder instances after first install
py_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyImportMetaFinder))
print(f"PyImportMetaFinder instances after first install: {py_count}")

# Second install with same parameters
py2, pyx2 = pyximport.install(pyximport=False, pyimport=True)

# Count PyImportMetaFinder instances after second install
py_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyImportMetaFinder))
print(f"PyImportMetaFinder instances after second install: {py_count}")

# This demonstrates the bug - we should have at most 1, but we have 2
if py_count > 1:
    print(f"BUG: Found {py_count} PyImportMetaFinder instances, expected <= 1")
else:
    print("No bug detected")

# Cleanup
pyximport.uninstall(py1, pyx1)
pyximport.uninstall(py2, pyx2)
```

<details>

<summary>
Output showing duplicate importers
</summary>
```
Initial sys.meta_path length: 4
PyImportMetaFinder instances after first install: 1
PyImportMetaFinder instances after second install: 2
BUG: Found 2 PyImportMetaFinder instances, expected <= 1
```
</details>

## Why This Is A Bug

The `_have_importers()` function at lines 355-365 in `/home/npc/miniconda/lib/python3.13/site-packages/pyximport/pyximport.py` uses incorrect nested `isinstance` checks:

```python
def _have_importers():
    has_py_importer = False
    has_pyx_importer = False
    for importer in sys.meta_path:
        if isinstance(importer, PyxImportMetaFinder):
            if isinstance(importer, PyImportMetaFinder):
                has_py_importer = True
            else:
                has_pyx_importer = True
    return has_py_importer, has_pyx_importer
```

This violates expected behavior because:

1. **The classes are siblings, not parent-child**: Both `PyImportMetaFinder` and `PyxImportMetaFinder` inherit directly from `importlib.abc.MetaPathFinder`. Neither class inherits from the other.

2. **The nested check never succeeds**: A `PyImportMetaFinder` instance will never pass the outer `isinstance(importer, PyxImportMetaFinder)` check, so existing `PyImportMetaFinder` instances are never detected.

3. **The install() function relies on accurate detection**: Lines 434-449 in `install()` use the return values from `_have_importers()` to conditionally add importers only when they don't already exist:
   ```python
   has_py_importer, has_pyx_importer = _have_importers()
   if pyimport and not has_py_importer:
       py_importer = PyImportMetaFinder(...)
       sys.meta_path.insert(0, py_importer)
   ```

4. **Performance impact**: Each duplicate importer in `sys.meta_path` is consulted for every Python import operation, causing unnecessary overhead that accumulates with each duplicate.

5. **Documentation implies idempotency**: While not explicitly stated, the presence of the `_have_importers()` check and the conditional logic demonstrates clear intent to prevent duplicate installations.

## Relevant Context

- The `pyimport` feature is documented as "experimental" and "use at your own risk" in the docstring (line 381-386), warning it will "heavily slow down your imports".
- The bug only affects `PyImportMetaFinder` instances. `PyxImportMetaFinder` instances are correctly detected because they pass the outer isinstance check and fail the inner one, reaching the `else` branch.
- The issue has existed since the nested isinstance structure was introduced and affects any code that calls `install()` multiple times, which could happen in complex applications or during testing.
- Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pyximport/pyximport.py:355-365`

## Proposed Fix

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -355,13 +355,11 @@ class PyxArgs(object):
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