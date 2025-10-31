# Bug Report: pyximport.handle_special_build AttributeError

**Target**: `pyximport.handle_special_build`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `handle_special_build` function crashes with `AttributeError: 'NoneType' object has no attribute 'sources'` when a `.pyxbld` file defines `make_setup_args()` but not `make_ext()`.

## Property-Based Test

```python
import os
import tempfile
from hypothesis import given, strategies as st, settings
import pyximport


@given(st.booleans())
@settings(max_examples=50)
def test_handle_special_build_with_setup_args_only(include_make_ext):
    with tempfile.TemporaryDirectory() as tmpdir:
        pyxfile = os.path.join(tmpdir, 'test.pyx')
        pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

        with open(pyxfile, 'w') as f:
            f.write('# cython code')

        if include_make_ext:
            pyxbld_content = '''
from distutils.extension import Extension

def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])

def make_setup_args():
    return {'extra_compile_args': ['-O3']}
'''
        else:
            pyxbld_content = '''
def make_setup_args():
    return {'extra_compile_args': ['-O3']}
'''

        with open(pyxbld_file, 'w') as f:
            f.write(pyxbld_content)

        ext, setup_args = pyximport.handle_special_build('test', pyxfile)
```

**Failing input**: `include_make_ext=False`

## Reproducing the Bug

```python
import os
import tempfile
import pyximport

with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, 'test.pyx')
    pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

    with open(pyxfile, 'w') as f:
        f.write('# cython code')

    with open(pyxbld_file, 'w') as f:
        f.write('''def make_setup_args():
    return {'extra_compile_args': ['-O3']}
''')

    ext, setup_args = pyximport.handle_special_build('test', pyxfile)
```

**Error**: `AttributeError: 'NoneType' object has no attribute 'sources'` at line 132 in `pyximport/pyximport.py`

## Why This Is A Bug

The function has an assertion `assert ext or setup_args` that explicitly allows either `ext` OR `setup_args` to be present. However, the code unconditionally tries to access `ext.sources` at line 132, which crashes when `ext` is None. This violates the function's own contract that setup_args alone should be sufficient.

## Fix

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -129,8 +129,9 @@ def handle_special_build(modname, pyxfilename):
                                          % special_build)
         assert ext or setup_args, ("neither make_ext nor make_setup_args %s"
                                          % special_build)
-        ext.sources = [os.path.join(os.path.dirname(special_build), source)
-                       for source in ext.sources]
+        if ext:
+            ext.sources = [os.path.join(os.path.dirname(special_build), source)
+                           for source in ext.sources]
     return ext, setup_args
```