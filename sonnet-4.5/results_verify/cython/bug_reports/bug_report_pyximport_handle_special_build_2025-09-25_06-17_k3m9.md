# Bug Report: pyximport.handle_special_build AttributeError with make_setup_args

**Target**: `pyximport.handle_special_build`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`handle_special_build()` crashes with AttributeError when a `.pyxbld` file defines only `make_setup_args()` without `make_ext()`, despite documentation explicitly allowing this configuration.

## Property-Based Test

```python
import os
import tempfile
from hypothesis import given, strategies as st
import pyximport

@given(st.just(None))
def test_handle_special_build_with_only_setup_args(x):
    with tempfile.TemporaryDirectory() as tmpdir:
        pyx_file = os.path.join(tmpdir, 'test.pyx')
        pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

        with open(pyx_file, 'w') as f:
            f.write('def hello(): return "world"')

        with open(pyxbld_file, 'w') as f:
            f.write('def make_setup_args():\n    return {"script_args": ["--verbose"]}')

        ext, setup_args = pyximport.handle_special_build('test', pyx_file)
        assert isinstance(setup_args, dict)
```

**Failing input**: Any `.pyxbld` file with only `make_setup_args()` defined

## Reproducing the Bug

```python
import os
import tempfile
import pyximport

with tempfile.TemporaryDirectory() as tmpdir:
    pyx_file = os.path.join(tmpdir, 'example.pyx')
    pyxbld_file = os.path.join(tmpdir, 'example.pyxbld')

    with open(pyx_file, 'w') as f:
        f.write('def hello(): return "world"')

    with open(pyxbld_file, 'w') as f:
        f.write('def make_setup_args():\n    return {"script_args": ["--verbose"]}')

    ext, setup_args = pyximport.handle_special_build('example', pyx_file)
```

Output:
```
AttributeError: 'NoneType' object has no attribute 'sources'
```

## Why This Is A Bug

The module docstring (lines 17-28 in pyximport.py) explicitly shows that `.pyxbld` files can define `make_setup_args()` without `make_ext()`:

```python
def make_setup_args():
    return dict(script_args=["--compiler=mingw32"])
```

The code asserts `ext or setup_args` (line 129) allowing `ext` to be `None`, but then unconditionally accesses `ext.sources` (lines 131-132), causing a crash.

## Fix

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -128,8 +128,9 @@ def handle_special_build(modname, pyxfilename):
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