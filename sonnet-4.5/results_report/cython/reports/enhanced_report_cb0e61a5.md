# Bug Report: pyximport PyImportMetaFinder found flag prevents multiple module compilation

**Target**: `pyximport.PyImportMetaFinder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `found` flag in `PyImportMetaFinder` class is set to `True` when the first .py module is imported but never reset, causing all subsequent .py modules to be skipped from Cython compilation when using `pyimport=True`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import sys
import os
import tempfile
import importlib
import pyximport

@given(st.integers(min_value=2, max_value=5))
def test_pyimport_compiles_all_modules(num_modules):
    """Test that pyximport with pyimport=True compiles all .py modules, not just the first one"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create num_modules simple Python modules
        for i in range(num_modules):
            with open(os.path.join(tmpdir, f'mod{i}.py'), 'w') as f:
                f.write(f'VALUE = {i}')

        # Add tmpdir to path
        sys.path.insert(0, tmpdir)
        try:
            # Install pyximport with pyimport=True
            py_imp, pyx_imp = pyximport.install(pyimport=True)

            # Import all modules and check if they were compiled
            compiled_count = 0
            for i in range(num_modules):
                mod = importlib.import_module(f'mod{i}')
                # Check if module was compiled (has .so or .pyd extension)
                if '.so' in mod.__file__ or '.pyd' in mod.__file__:
                    compiled_count += 1

            # All modules should be compiled
            assert compiled_count == num_modules, f"Expected all {num_modules} modules to be compiled, only {compiled_count} were"
        finally:
            # Cleanup
            sys.path.remove(tmpdir)
            pyximport.uninstall(py_imp, pyx_imp)

if __name__ == "__main__":
    # Run the test
    test_pyimport_compiles_all_modules()
```

<details>

<summary>
**Failing input**: `num_modules=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 40, in <module>
    test_pyimport_compiles_all_modules()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 9, in test_pyimport_compiles_all_modules
    def test_pyimport_compiles_all_modules(num_modules):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 32, in test_pyimport_compiles_all_modules
    assert compiled_count == num_modules, f"Expected all {num_modules} modules to be compiled, only {compiled_count} were"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected all 2 modules to be compiled, only 1 were
Falsifying example: test_pyimport_compiles_all_modules(
    num_modules=2,
)
```
</details>

## Reproducing the Bug

```python
import os
import sys
import tempfile
import importlib
import pyximport

with tempfile.TemporaryDirectory() as tmpdir:
    # Create two simple Python modules
    with open(os.path.join(tmpdir, 'first.py'), 'w') as f:
        f.write('X = 1')
    with open(os.path.join(tmpdir, 'second.py'), 'w') as f:
        f.write('Y = 2')

    # Add tmpdir to path so modules can be imported
    sys.path.insert(0, tmpdir)

    # Install pyximport with pyimport=True to compile .py files
    py_imp, pyx_imp = pyximport.install(pyimport=True)

    # Import first module
    mod1 = importlib.import_module('first')
    print(f"first module file: {mod1.__file__}")
    print(f"first module X value: {mod1.X}")

    # Import second module
    mod2 = importlib.import_module('second')
    print(f"second module file: {mod2.__file__}")
    print(f"second module Y value: {mod2.Y}")

    # Check if modules were compiled (should have .so or .pyd extension)
    compiled_first = '.so' in mod1.__file__ or '.pyd' in mod1.__file__
    compiled_second = '.so' in mod2.__file__ or '.pyd' in mod2.__file__

    print(f"\nfirst module compiled: {compiled_first}")
    print(f"second module compiled: {compiled_second}")

    if compiled_first and not compiled_second:
        print("\nBUG CONFIRMED: Only first module was compiled!")

    # Cleanup
    sys.path.remove(tmpdir)
    pyximport.uninstall(py_imp, pyx_imp)
```

<details>

<summary>
Output showing only first module is Cython-compiled
</summary>
```
first module file: /home/npc/.pyxbld/lib.linux-x86_64-cpython-313/first.cpython-313-x86_64-linux-gnu.so
first module X value: 1
second module file: /tmp/tmpgp9cs9l5/second.py
second module Y value: 2

first module compiled: True
second module compiled: False

BUG CONFIRMED: Only first module was compiled!
```
</details>

## Why This Is A Bug

The pyximport documentation explicitly states that with `pyimport=True`, the module will "automatically run Cython on every .pyx and .py module that Python imports" (pyximport.py lines 33-36). However, due to a logic error in the `PyImportMetaFinder` class, only the first .py module gets compiled.

The bug occurs because:
1. When `PyImportMetaFinder.find_spec()` is called for the first .py module, it sets `self.found = True` at line 296
2. For all subsequent imports, the check at lines 270-271 immediately returns `None` because `self.found` is `True`
3. The `self.found` flag is never reset back to `False`, permanently disabling the importer after the first module

This violates the documented contract that "every" .py module should be compiled, not just the first one. While the feature is marked as experimental, the current behavior makes it completely unusable for any real-world scenario involving multiple Python modules.

## Relevant Context

The `self.found` flag appears to be an attempt to prevent some form of recursion or duplicate processing, but its implementation is incorrect. The flag is initialized in `__init__` at line 267, checked at lines 270-271, and set at line 296, but crucially it is never reset.

The `PyxImportMetaFinder` class (for .pyx files) does not have this `self.found` flag pattern and works correctly for multiple .pyx files, suggesting this is a bug specific to the experimental .py compilation feature.

Code locations:
- Bug location: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/pyximport/pyximport.py:267-296`
- Documentation: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/pyximport/pyximport.py:33-42`

## Proposed Fix

The `self.found` flag logic should be removed entirely as it serves no useful purpose and breaks the feature. The `self.blocked_modules` list already prevents infinite recursion.

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -264,11 +264,8 @@ class PyImportMetaFinder(MetaPathFinder):
         self.blocked_modules = ['Cython', 'pyxbuild', 'pyximport.pyxbuild',
                                 'distutils', 'cython']
         self.blocked_packages = ['Cython.', 'distutils.']
-        self.found = False

     def find_spec(self, fullname, path, target=None):
-        if self.found:
-            return None
         if fullname in sys.modules:
             return None
         if any([fullname.startswith(pkg) for pkg in self.blocked_packages]):
@@ -293,7 +290,6 @@ class PyImportMetaFinder(MetaPathFinder):
                 if not os.path.exists(filename):
                     continue

-                self.found = True
                 return spec_from_file_location(
                     fullname, filename,
                     loader=PyxImportLoader(filename, self.pyxbuild_dir, self.inplace, self.language_level),
```