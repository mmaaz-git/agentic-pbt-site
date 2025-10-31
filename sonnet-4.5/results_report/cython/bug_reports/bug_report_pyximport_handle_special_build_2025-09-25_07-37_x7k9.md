# Bug Report: pyximport handle_special_build() AttributeError on None Extension

**Target**: `pyximport.pyximport.handle_special_build()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`handle_special_build()` crashes with `AttributeError` when a `.pyxbld` file defines only `make_setup_args()` without `make_ext()`, because it attempts to access `ext.sources` when `ext` is `None`.

## Property-Based Test

```python
import sys
import os
import tempfile
from pyximport.pyximport import handle_special_build
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(
    has_make_ext=st.booleans(),
    has_make_setup_args=st.booleans()
)
def test_handle_special_build_combinations(has_make_ext, has_make_setup_args):
    with tempfile.TemporaryDirectory() as tmpdir:
        pyxfile = os.path.join(tmpdir, "test.pyx")
        pyxbld = os.path.join(tmpdir, "test.pyxbld")

        open(pyxfile, 'w').close()

        pyxbld_content = ""
        if has_make_ext:
            pyxbld_content += """
from distutils.extension import Extension
def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])
"""
        if has_make_setup_args:
            pyxbld_content += """
def make_setup_args():
    return {'script_args': ['--verbose']}
"""

        if pyxbld_content:
            with open(pyxbld, 'w') as f:
                f.write(pyxbld_content)

            ext, setup_args = handle_special_build("test", pyxfile)
```

**Failing input**: `has_make_ext=False, has_make_setup_args=True`

## Reproducing the Bug

```python
import sys
import os
import tempfile
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import handle_special_build

with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, "module.pyx")
    pyxbld = os.path.join(tmpdir, "module.pyxbld")

    open(pyxfile, 'w').close()

    with open(pyxbld, 'w') as f:
        f.write("""
def make_setup_args():
    return {'script_args': ['--verbose']}
""")

    ext, setup_args = handle_special_build("module", pyxfile)
```

## Why This Is A Bug

According to the docstring at lines 17-28, a `.pyxbld` file can define either `make_ext()` or `make_setup_args()` or both. The assertion at line 129 allows `ext` to be `None` as long as `setup_args` is truthy. However, line 131 unconditionally attempts to access `ext.sources`, causing an `AttributeError` when `ext` is `None`.

## Fix

```diff
--- a/pyximport.py
+++ b/pyximport.py
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