# Bug Report: pyximport.get_distutils_extension crashes with non-str paths

**Target**: `pyximport.get_distutils_extension`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `get_distutils_extension()` receives non-string path arguments (bytes or pathlib.Path objects), it crashes due to broken Python 2→3 migration code. Bytes paths cause `TypeError: can't concat str to bytes`, while Path objects cause `AttributeError: 'PosixPath' object has no attribute 'encode'`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env')

import os
import tempfile
import pyximport
from hypothesis import given, strategies as st, settings
import string
from pathlib import Path

@given(st.text(alphabet=string.ascii_letters + string.digits + '_', min_size=1, max_size=50))
@settings(max_examples=100)
def test_get_distutils_extension_bytes_vs_str(modname):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pyx', delete=False) as f:
        f.write("# dummy\n")
        str_path = f.name
        bytes_path = str_path.encode('utf-8')

    try:
        ext_str, args_str = pyximport.get_distutils_extension(modname, str_path)
        ext_bytes, args_bytes = pyximport.get_distutils_extension(modname, bytes_path)
        assert ext_str.name == ext_bytes.name
    finally:
        os.unlink(str_path)

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1, max_size=30))
@settings(max_examples=100)
def test_get_distutils_extension_with_pathlib(modname):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pyx', delete=False) as f:
        f.write("# dummy\n")
        path_str = f.name
        path_obj = Path(path_str)

    try:
        ext_str, args_str = pyximport.get_distutils_extension(modname, path_str)
        ext_path, args_path = pyximport.get_distutils_extension(modname, path_obj)
        assert ext_str.name == ext_path.name
    finally:
        os.unlink(path_str)
```

**Failing inputs**:
- Bytes path: `b'/tmp/test.pyx'`
- Path object: `Path('/tmp/test.pyx')`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env')

import os
import tempfile
import pyximport
from pathlib import Path

with tempfile.NamedTemporaryFile(mode='w', suffix='.pyx', delete=False) as f:
    f.write("# dummy\n")
    path = f.name

try:
    print("Testing bytes path:")
    pyximport.get_distutils_extension('mymod', path.encode('utf-8'))
except TypeError as e:
    print(f"  Bytes bug: {e}")

try:
    print("\nTesting Path object:")
    pyximport.get_distutils_extension('mymod', Path(path))
except AttributeError as e:
    print(f"  Path bug: {e}")
finally:
    os.unlink(path)
```

## Why This Is A Bug

The function `get_distutils_extension()` has explicit logic to handle non-string paths:

```python
if not isinstance(pyxfilename, str):
    # distutils is stupid in Py2 and requires exactly 'str'
    # => encode accidentally coerced unicode strings back to str
    pyxfilename = pyxfilename.encode(sys.getfilesystemencoding())
```

This Python 2 legacy code is fundamentally broken in Python 3:

1. **Bytes path**: In Python 3, bytes pass the `not isinstance(pyxfilename, str)` check but don't have an `encode()` method (they have `decode()`). The bytes are passed unchanged to `handle_special_build()` which tries `os.path.splitext(pyxfilename)[0] + PYXBLD_EXT`, failing with `TypeError: can't concat str to bytes`.

2. **Path objects**: pathlib.Path objects also fail the isinstance check but don't have an `encode()` method, causing `AttributeError: 'PosixPath' object has no attribute 'encode'`.

Both failures indicate the code doesn't properly support modern Python 3 path types despite having logic that suggests it should.

## Fix

```diff
--- a/pyximport.py
+++ b/pyximport.py
@@ -96,9 +96,11 @@ def get_distutils_extension(modname, pyxfilename, language_level=None):
 #    modname = modname + extra
     extension_mod,setup_args = handle_special_build(modname, pyxfilename)
     if not extension_mod:
+        if isinstance(pyxfilename, bytes):
+            pyxfilename = pyxfilename.decode(sys.getfilesystemencoding())
-        if not isinstance(pyxfilename, str):
-            # distutils is stupid in Py2 and requires exactly 'str'
-            # => encode accidentally coerced unicode strings back to str
-            pyxfilename = pyxfilename.encode(sys.getfilesystemencoding())
+        elif not isinstance(pyxfilename, str):
+            pyxfilename = str(pyxfilename)
         from distutils.extension import Extension
         extension_mod = Extension(name = modname, sources=[pyxfilename])
         if language_level is not None:
```