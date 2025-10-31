# Bug Report: pyximport install() Not Idempotent with pyimport=True

**Target**: `pyximport.install()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `install()` function fails to be idempotent when called with `pyimport=True`. Each call incorrectly adds a new `PyImportMetaFinder` instance to `sys.meta_path`, degrading Python's import performance as the system checks duplicate hooks.

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

if __name__ == "__main__":
    test_install_idempotence()
```

<details>

<summary>
**Failing input**: `test_install_idempotence(pyximport_flag=False, pyimport_flag=True, num_calls=2)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 33, in <module>
    test_install_idempotence()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 6, in test_install_idempotence
    pyximport_flag=st.booleans(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 27, in test_install_idempotence
    assert len(py_importers) == 1, \
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 1 py importer, got 2
Falsifying example: test_install_idempotence(
    # The test sometimes passed when commented parts were varied together.
    pyximport_flag=False,  # or any other generated value
    pyimport_flag=True,  # or any other generated value
    num_calls=2,  # or any other generated value
)
```
</details>

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

<details>

<summary>
Duplicate importers accumulate with each call
</summary>
```
Initial count: 0
After 1st install: 1
After 2nd install: 2
After 3rd install: 3
```
</details>

## Why This Is A Bug

The bug stems from incorrect class hierarchy checking in the `_have_importers()` function at lines 355-365. The function's logic assumes `PyImportMetaFinder` is a subclass of `PyxImportMetaFinder`, but they are actually independent sibling classes that both inherit from `MetaPathFinder`.

The flawed logic:
```python
def _have_importers():
    has_py_importer = False
    has_pyx_importer = False
    for importer in sys.meta_path:
        if isinstance(importer, PyxImportMetaFinder):  # PyImportMetaFinder instances fail this check!
            if isinstance(importer, PyImportMetaFinder):  # This nested check is never reached
                has_py_importer = True
            else:
                has_pyx_importer = True
    return has_py_importer, has_pyx_importer
```

Since `PyImportMetaFinder` is not a subclass of `PyxImportMetaFinder`, instances of `PyImportMetaFinder` never pass the outer `isinstance` check. As a result, `has_py_importer` always returns `False`, causing `install()` to add duplicate importers on every call.

This violates the function's documented intent to check for existing importers and breaks idempotence - a fundamental expectation for installation functions. While the documentation doesn't explicitly promise idempotence, the code structure with `_have_importers()` clearly shows this was the intended behavior. The presence of conditional checks `if pyimport and not has_py_importer` further confirms the intent to prevent duplicate installations.

## Relevant Context

The class hierarchy in pyximport.py shows:
- `PyxImportMetaFinder(MetaPathFinder)` - defined at line 222
- `PyImportMetaFinder(MetaPathFinder)` - defined at line 256

Both classes inherit directly from `MetaPathFinder`, making them sibling classes rather than having a parent-child relationship. The nested isinstance checks in `_have_importers()` incorrectly assume one is a subclass of the other.

This bug impacts:
1. **Performance**: Each duplicate importer adds overhead to Python's import mechanism
2. **Memory**: Unnecessary importer objects accumulate in memory
3. **Module reloading**: Complex applications that might reinstall hooks suffer degraded performance
4. **sitecustomize.py usage**: The documentation suggests putting `pyximport.install()` in sitecustomize.py, which could be sourced multiple times

The `pyimport` feature is marked as experimental in the documentation (line 383-386), but the bug in `_have_importers()` also affects the interaction between `pyximport` and `pyimport` flags when both are used together.

## Proposed Fix

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