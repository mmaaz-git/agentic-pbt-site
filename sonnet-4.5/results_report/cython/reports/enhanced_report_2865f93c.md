# Bug Report: pyximport.get_distutils_extension crashes with non-string path arguments

**Target**: `pyximport.get_distutils_extension`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The function `pyximport.get_distutils_extension()` crashes when receiving non-string path arguments (bytes or pathlib.Path objects) due to broken Python 2→3 migration code that attempts to handle such paths but fails.

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

if __name__ == "__main__":
    test_get_distutils_extension_bytes_vs_str()
    test_get_distutils_extension_with_pathlib()
```

<details>

<summary>
**Failing input**: `modname='0'` (bytes path test)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 42, in <module>
    test_get_distutils_extension_bytes_vs_str()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 12, in test_get_distutils_extension_bytes_vs_str
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 21, in test_get_distutils_extension_bytes_vs_str
    ext_bytes, args_bytes = pyximport.get_distutils_extension(modname, bytes_path)
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pyximport/pyximport.py", line 97, in get_distutils_extension
    extension_mod,setup_args = handle_special_build(modname, pyxfilename)
                               ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pyximport/pyximport.py", line 111, in handle_special_build
    special_build = os.path.splitext(pyxfilename)[0] + PYXBLD_EXT
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~
TypeError: can't concat str to bytes
Falsifying example: test_get_distutils_extension_bytes_vs_str(
    modname='0',  # or any other generated value
)
```
</details>

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

<details>

<summary>
Output demonstrating both crashes
</summary>
```
Testing bytes path:
  Bytes bug: can't concat str to bytes

Testing Path object:
  Path bug: 'PosixPath' object has no attribute 'encode'
```
</details>

## Why This Is A Bug

The function contains explicit code at lines 99-102 that attempts to handle non-string path types:

```python
if not isinstance(pyxfilename, str):
    # distutils is stupid in Py2 and requires exactly 'str'
    # => encode accidentally coerced unicode strings back to str
    pyxfilename = pyxfilename.encode(sys.getfilesystemencoding())
```

This code was designed for Python 2 to handle unicode strings, but in Python 3 it causes two distinct failures:

1. **Bytes paths**: When a bytes object is passed, it fails the `isinstance(pyxfilename, str)` check. Since bytes objects don't have an `encode()` method in Python 3, the bytes pass through unchanged to `handle_special_build()`. There, at line 111, the code attempts `os.path.splitext(pyxfilename)[0] + PYXBLD_EXT`, which fails with `TypeError: can't concat str to bytes` because `PYXBLD_EXT` is a string (".pyxbld") and cannot be concatenated to bytes.

2. **Path objects**: When a pathlib.Path object is passed, it also fails the isinstance check. The code then tries to call `pyxfilename.encode()`, but Path objects don't have an `encode()` method, resulting in `AttributeError: 'PosixPath' object has no attribute 'encode'`.

The presence of this type-handling code indicates the function was intended to support different path types, but the implementation is fundamentally broken for Python 3's type system.

## Relevant Context

This bug represents incomplete Python 2 to Python 3 migration. The comment in the code explicitly mentions "distutils is stupid in Py2", indicating this was Python 2 compatibility code that should have been updated or removed for Python 3.

The distutils.extension.Extension class (which this function creates) accepts string paths in its sources parameter. Modern Python conventions increasingly use pathlib.Path objects for file paths, and bytes paths are still used in some contexts (e.g., when dealing with filesystem encoding issues).

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/pyximport/pyximport.py:90-107`

## Proposed Fix

```diff
--- a/pyximport.py
+++ b/pyximport.py
@@ -96,10 +96,13 @@ def get_distutils_extension(modname, pyxfilename, language_level=None):
 #    modname = modname + extra
     extension_mod,setup_args = handle_special_build(modname, pyxfilename)
     if not extension_mod:
-        if not isinstance(pyxfilename, str):
-            # distutils is stupid in Py2 and requires exactly 'str'
-            # => encode accidentally coerced unicode strings back to str
-            pyxfilename = pyxfilename.encode(sys.getfilesystemencoding())
+        # Convert non-string paths to strings for distutils compatibility
+        if isinstance(pyxfilename, bytes):
+            pyxfilename = pyxfilename.decode(sys.getfilesystemencoding())
+        elif hasattr(pyxfilename, '__fspath__'):
+            pyxfilename = os.fspath(pyxfilename)
+        elif not isinstance(pyxfilename, str):
+            pyxfilename = str(pyxfilename)
         from distutils.extension import Extension
         extension_mod = Extension(name = modname, sources=[pyxfilename])
         if language_level is not None:
```