# Bug Report: pyximport PyImportMetaFinder found flag prevents multiple module compilation

**Target**: `pyximport.PyImportMetaFinder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `found` flag in `PyImportMetaFinder` is set to `True` when the first .py module is found but never reset, preventing subsequent .py modules from being Cython-compiled when using `pyimport=True`.

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
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_modules):
            with open(os.path.join(tmpdir, f'mod{i}.py'), 'w') as f:
                f.write(f'VALUE = {i}')

        sys.path.insert(0, tmpdir)
        try:
            py_imp, pyx_imp = pyximport.install(pyimport=True)

            compiled_count = 0
            for i in range(num_modules):
                mod = importlib.import_module(f'mod{i}')
                if '.so' in mod.__file__ or '.pyd' in mod.__file__:
                    compiled_count += 1

            assert compiled_count == num_modules, f"Expected all {num_modules} to be compiled, only {compiled_count} were"
        finally:
            sys.path.remove(tmpdir)
            pyximport.uninstall(py_imp, pyx_imp)
```

**Failing input**: Any scenario where 2 or more .py modules need to be imported

## Reproducing the Bug

```python
import os
import sys
import tempfile
import importlib
import pyximport

with tempfile.TemporaryDirectory() as tmpdir:
    with open(os.path.join(tmpdir, 'first.py'), 'w') as f:
        f.write('X = 1')
    with open(os.path.join(tmpdir, 'second.py'), 'w') as f:
        f.write('Y = 2')

    sys.path.insert(0, tmpdir)

    py_imp, pyx_imp = pyximport.install(pyimport=True)

    mod1 = importlib.import_module('first')
    print(f"first module: {mod1.__file__}")

    mod2 = importlib.import_module('second')
    print(f"second module: {mod2.__file__}")

    sys.path.remove(tmpdir)
    pyximport.uninstall(py_imp, pyx_imp)
```

Output:
```
first module: /home/user/.pyxbld/lib.linux-x86_64-cpython-313/first.cpython-313-x86_64-linux-gnu.so
second module: /tmp/tmpXXX/second.py
```

The first module is compiled to `.so` (Cython-compiled), but the second is not.

## Why This Is A Bug

The documentation states (pyximport.py lines 33-36):
> "the :mod:`pyximport` module also has experimental compilation support for normal Python modules. This allows you to automatically run Cython on every .pyx and .py module that Python imports"

The `pyimport=True` feature should compile all .py modules, but due to the `found` flag bug, only the first module is compiled. The flag is set at line 296 but never reset, causing line 270-271 to return None for all subsequent find_spec calls.

## Fix

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -267,8 +267,6 @@ class PyImportMetaFinder(MetaPathFinder):
         self.found = False

     def find_spec(self, fullname, path, target=None):
-        if self.found:
-            return None
         if fullname in sys.modules:
             return None
         if any([fullname.startswith(pkg) for pkg in self.blocked_packages]):
@@ -293,7 +291,6 @@ class PyImportMetaFinder(MetaPathFinder):
                 if not os.path.exists(filename):
                     continue

-                self.found = True
                 return spec_from_file_location(
                     fullname, filename,
                     loader=PyxImportLoader(filename, self.pyxbuild_dir, self.inplace, self.language_level),
```