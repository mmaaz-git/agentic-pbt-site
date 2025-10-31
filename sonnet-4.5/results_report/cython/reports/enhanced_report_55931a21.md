# Bug Report: pyximport._have_importers() Fails to Detect PyImportMetaFinder Instances

**Target**: `pyximport.pyximport._have_importers()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_have_importers()` function fails to detect `PyImportMetaFinder` instances in `sys.meta_path` due to incorrect class hierarchy assumptions, causing duplicate importers to be installed when `install(pyimport=True)` is called multiple times.

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

if __name__ == "__main__":
    test_have_importers_detects_both_types()
```

<details>

<summary>
**Failing input**: `add_pyx=False, add_py=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 34, in <module>
    test_have_importers_detects_both_types()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 9, in test_have_importers_detects_both_types
    @given(st.booleans(), st.booleans())
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 27, in test_have_importers_detects_both_types
    assert has_py == add_py, \
           ^^^^^^^^^^^^^^^^
AssertionError: Expected has_py=True, got False. meta_path types: ['PyImportMetaFinder']
Falsifying example: test_have_importers_detects_both_types(
    add_pyx=False,
    add_py=True,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import _have_importers, PyImportMetaFinder

# Save the original meta_path
original_meta_path = sys.meta_path.copy()

# Set up a meta_path with only PyImportMetaFinder
sys.meta_path = [PyImportMetaFinder()]

# Call _have_importers() to check detection
has_py, has_pyx = _have_importers()

# Display results
print(f"meta_path contains: PyImportMetaFinder")
print(f"_have_importers() returned: has_py={has_py}, has_pyx={has_pyx}")
print(f"Expected: has_py=True, has_pyx=False")
print(f"Bug: has_py={has_py} (should be True)")

# Restore original meta_path
sys.meta_path[:] = original_meta_path
```

<details>

<summary>
PyImportMetaFinder not detected despite being present in sys.meta_path
</summary>
```
meta_path contains: PyImportMetaFinder
_have_importers() returned: has_py=False, has_pyx=False
Expected: has_py=True, has_pyx=False
Bug: has_py=False (should be True)
```
</details>

## Why This Is A Bug

The `_have_importers()` function contains flawed logic that assumes `PyImportMetaFinder` is a subclass of `PyxImportMetaFinder`. However, examination of the source code reveals both classes independently inherit from `MetaPathFinder`:

- Line 222: `class PyxImportMetaFinder(MetaPathFinder):`
- Line 256: `class PyImportMetaFinder(MetaPathFinder):`

The buggy implementation (lines 355-365):
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

Since `PyImportMetaFinder` instances are not subclasses of `PyxImportMetaFinder`, they fail the outer `isinstance` check entirely, resulting in `has_py_importer` remaining `False` even when `PyImportMetaFinder` is present.

This violates the expected behavior of the `install()` function which relies on `_have_importers()` to avoid installing duplicate importers (lines 437-447). The function explicitly checks `not has_py_importer` before installing a new `PyImportMetaFinder`, but since detection fails, duplicate importers get installed on subsequent calls.

## Relevant Context

The `install()` function is the main entry point for pyximport and is designed to be idempotent. The code structure at lines 434-447 clearly shows the intent to prevent duplicate importers:

```python
has_py_importer, has_pyx_importer = _have_importers()
# ...
if pyimport and not has_py_importer:
    py_importer = PyImportMetaFinder(...)
    sys.meta_path.insert(0, py_importer)

if pyximport and not has_pyx_importer:
    pyx_importer = PyxImportMetaFinder(...)
    sys.meta_path.append(pyx_importer)
```

This pattern is common in import hook implementations to ensure idempotent behavior. The bug breaks this contract, potentially causing:
- Performance degradation from duplicate import hooks
- Unexpected behavior in development environments where `install()` might be called multiple times
- Issues in Jupyter notebooks or interactive Python sessions

Source code location: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/pyximport/pyximport.py`

## Proposed Fix

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