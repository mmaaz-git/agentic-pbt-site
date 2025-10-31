# Bug Report: pyximport.build_module commonprefix bug on Windows

**Target**: `pyximport.build_module`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

On Windows, when handling long path names, `build_module()` uses `os.path.commonprefix()` which can return an invalid directory path, causing `os.chdir()` to fail with FileNotFoundError.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env')

import os
from hypothesis import given, strategies as st, settings, example

@given(
    st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10),
    st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10)
)
@settings(max_examples=300)
@example('abc', 'abd')
def test_commonprefix_not_valid_path(prefix1, prefix2):
    path1 = f"/tmp/very_long_directory_name_{prefix1}/subdir/file.pyx"
    path2 = f"/tmp/very_long_directory_name_{prefix2}/build"

    common = os.path.commonprefix([path1, path2])

    if len(common) > 30 and common != '/':
        assert os.path.isdir(common), \
            f"commonprefix returned invalid directory: {common}"
```

**Failing input**: Any pair of paths with similar prefixes but different directory names, e.g.:
- `pyxbuild_dir = "C:\\very_long_directory_name_abc\\build"`
- `pyxfilename = "C:\\very_long_directory_name_abd\\test.pyx"`

Results in `common = "C:\\very_long_directory_name_ab"` which is not a valid directory.

## Reproducing the Bug

```python
import os

path1 = "/tmp/very_long_directory_name_abc/subdir/file.pyx"
path2 = "/tmp/very_long_directory_name_abd/build"

common = os.path.commonprefix([path1, path2])
print(f"Common prefix: {common}")
print(f"Is valid directory: {os.path.isdir(common)}")
print(f"Would crash on: os.chdir('{common}')")
```

## Why This Is A Bug

In `build_module()`, when running on Windows with long paths:

```python
if pyxbuild_dir and sys.platform == 'win32':
    common = os.path.commonprefix([pyxbuild_dir, pyxfilename])
if len(common) > 30:
    pyxfilename = os.path.relpath(pyxfilename, common)
    pyxbuild_dir = os.path.relpath(pyxbuild_dir, common)
    os.chdir(common)  # <-- CRASH: common might not be a valid directory
```

The problem is that `os.path.commonprefix()` operates on strings character-by-character, not path components:

- Input: `["C:\\abc\\file.pyx", "C:\\abd\\build"]`
- Returns: `"C:\\ab"` (invalid!)
- Should use: `os.path.commonpath()` which returns `"C:\\"` (valid)

When `os.chdir(common)` is called with an invalid directory, it raises `FileNotFoundError`.

## Fix

```diff
--- a/pyximport.py
+++ b/pyximport.py
@@ -138,7 +138,7 @@ def build_module(name, pyxfilename, pyxbuild_dir=None, inplace=False, language_
     if pyxbuild_dir and sys.platform == 'win32':
         # Windows concatenates the pyxbuild_dir to the pyxfilename when
         # compiling, and then complains that the filename is too long
-        common = os.path.commonprefix([pyxbuild_dir, pyxfilename])
+        common = os.path.commonpath([pyxbuild_dir, pyxfilename])
     if len(common) > 30:
         pyxfilename = os.path.relpath(pyxfilename, common)
         pyxbuild_dir = os.path.relpath(pyxbuild_dir, common)
```