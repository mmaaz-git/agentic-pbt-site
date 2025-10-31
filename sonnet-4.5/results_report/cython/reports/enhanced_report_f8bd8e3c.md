# Bug Report: pyximport._have_importers() Fails to Detect PyImportMetaFinder Due to Incorrect isinstance Logic

**Target**: `pyximport.pyximport._have_importers()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_have_importers()` function contains a logic error in its nested `isinstance` checks that prevents it from detecting `PyImportMetaFinder` in `sys.meta_path`, causing the `install()` function to incorrectly re-install already-installed importers.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

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


if __name__ == "__main__":
    test_have_importers_detects_all_finders()
```

<details>

<summary>
**Failing input**: `add_py=True, add_pyx=False`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 33, in <module>
    test_have_importers_detects_all_finders()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 9, in test_have_importers_detects_all_finders
    @given(

  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 25, in test_have_importers_detects_all_finders
    assert has_py == add_py, f"Expected has_py={add_py}, got {has_py}"
           ^^^^^^^^^^^^^^^^
AssertionError: Expected has_py=True, got False
Falsifying example: test_have_importers_detects_all_finders(
    add_py=True,
    add_pyx=False,
)

```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import _have_importers, PyImportMetaFinder

# Save original state
original_meta_path = sys.meta_path.copy()

try:
    # Clear sys.meta_path and add only PyImportMetaFinder
    sys.meta_path = []
    sys.meta_path.append(PyImportMetaFinder())

    # Call _have_importers() to check detection
    has_py, has_pyx = _have_importers()

    print(f"Result from _have_importers(): has_py={has_py}, has_pyx={has_pyx}")
    print(f"Expected: has_py=True, has_pyx=False")

    # Verify this is actually a bug
    if has_py != True:
        print(f"\n✗ BUG DETECTED: PyImportMetaFinder is in sys.meta_path but has_py={has_py}")
        print(f"  The function failed to detect PyImportMetaFinder")
    else:
        print("\n✓ No bug detected")

finally:
    # Restore original state
    sys.meta_path[:] = original_meta_path
```

<details>

<summary>
Output shows PyImportMetaFinder not detected
</summary>
```
Result from _have_importers(): has_py=False, has_pyx=False
Expected: has_py=True, has_pyx=False

✗ BUG DETECTED: PyImportMetaFinder is in sys.meta_path but has_py=False
  The function failed to detect PyImportMetaFinder
```
</details>

## Why This Is A Bug

The `_have_importers()` function at lines 355-365 uses incorrect nested `isinstance` checks that assume `PyImportMetaFinder` would be a subclass of `PyxImportMetaFinder`. However, examining the class definitions reveals:

- `PyxImportMetaFinder` (line 222): `class PyxImportMetaFinder(MetaPathFinder)`
- `PyImportMetaFinder` (line 256): `class PyImportMetaFinder(MetaPathFinder)`

Both classes independently inherit from `MetaPathFinder`, making them sibling classes, not a parent-child relationship. The nested check structure:

```python
if isinstance(importer, PyxImportMetaFinder):
    if isinstance(importer, PyImportMetaFinder):
        has_py_importer = True
    else:
        has_pyx_importer = True
```

When `PyImportMetaFinder` is in `sys.meta_path`, the outer `isinstance(importer, PyxImportMetaFinder)` returns `False` (since they are siblings), so the inner check is never reached and `has_py_importer` is never set to `True`.

This violates the expected behavior because:
1. The function name `_have_importers()` (plural) indicates it should detect multiple importer types independently
2. The return tuple `(has_py_importer, has_pyx_importer)` structure clearly shows two independent boolean flags
3. The `install()` function at line 434 uses this to avoid re-installing importers, but due to this bug, it incorrectly re-installs `PyImportMetaFinder` when it's already present

## Relevant Context

- The `pyimport` feature is documented as "experimental" (lines 33-34 of module docstring)
- The bug affects users who call `pyximport.install(pyimport=True, pyximport=False)` to enable experimental Python compilation
- The `install()` function relies on `_have_importers()` at line 434 to check if importers are already installed
- Both `PyImportMetaFinder` and `PyxImportMetaFinder` are MetaPathFinder implementations that handle different file types (.py and .pyx respectively)
- Code location: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/pyximport/pyximport.py:355-365`

## Proposed Fix

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -356,11 +356,10 @@ def _have_importers():
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