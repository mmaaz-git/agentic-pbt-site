# Bug Report: pyximport build_module() Uses Wrong Path Function

**Target**: `pyximport.pyximport.build_module`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `build_module()` function uses `os.path.commonprefix()` to find the common path between two directories, then calls `os.chdir()` on the result. However, `os.path.commonprefix()` operates on strings (not path components), which can return partial directory names that don't exist, causing `FileNotFoundError` when trying to change to that directory.

## Property-Based Test

```python
import os
from hypothesis import given, settings, strategies as st
from hypothesis import HealthCheck

@settings(max_examples=500, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    st.integers(min_value=1, max_value=99),
    st.integers(min_value=1, max_value=99)
)
def test_commonprefix_partial_directory_name(num1, num2):
    if num1 == num2:
        return

    base = "a" * 31
    pyxbuild_dir = f"/very/long/base/{base}{num1}/build"
    pyxfilename = f"/very/long/base/{base}{num2}/file.pyx"

    common_prefix = os.path.commonprefix([pyxbuild_dir, pyxfilename])
    common_path = os.path.commonpath([pyxbuild_dir, pyxfilename])

    if len(common_prefix) > 30:
        assert common_prefix == common_path, (
            f"os.path.commonprefix returns partial path {common_prefix!r}, "
            f"but should use os.path.commonpath which returns {common_path!r}"
        )
```

**Failing input**: `num1=1, num2=2` (any different numbers)

## Reproducing the Bug

```python
import os

pyxbuild_dir = "/very/long/base/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1/build"
pyxfilename = "/very/long/base/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa2/file.pyx"

common_prefix = os.path.commonprefix([pyxbuild_dir, pyxfilename])
print(f"commonprefix: {common_prefix!r}")
print(f"Length: {len(common_prefix)}")

common_path = os.path.commonpath([pyxbuild_dir, pyxfilename])
print(f"commonpath: {common_path!r}")

if len(common_prefix) > 30:
    print(f"\nThe build_module() workaround would trigger:")
    print(f"  os.chdir({common_prefix!r})")
    try:
        os.chdir(common_prefix)
        print("  ERROR: chdir succeeded but shouldn't!")
    except FileNotFoundError as e:
        print(f"  ✓ FileNotFoundError: {e}")
        print(f"\nThis directory doesn't exist - it's a partial directory name!")
```

Output:
```
commonprefix: '/very/long/base/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
Length: 47
commonpath: '/very/long/base'
The build_module() workaround would trigger:
  os.chdir('/very/long/base/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
  ✓ FileNotFoundError: [Errno 2] No such file or directory: '/very/long/base/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
This directory doesn't exist - it's a partial directory name!
```

## Why This Is A Bug

In `build_module()` at lines 187-195 of `pyximport.py`, there's a Windows-specific workaround for long path names:

```python
common = ''
if pyxbuild_dir and sys.platform == 'win32':
    # Windows concatenates the pyxbuild_dir to the pyxfilename when
    # compiling, and then complains that the filename is too long
    common = os.path.commonprefix([pyxbuild_dir, pyxfilename])
if len(common) > 30:
    pyxfilename = os.path.relpath(pyxfilename, common)
    pyxbuild_dir = os.path.relpath(pyxbuild_dir, common)
    os.chdir(common)
```

The bug is on line 191: `os.path.commonprefix()` finds the longest common **string prefix**, not the longest common **path**.

Example:
- `pyxbuild_dir = "/proj/longname123/build"`
- `pyxfilename = "/proj/longname456/file.pyx"`
- `os.path.commonprefix()` → `/proj/longname` (partial directory!)
- `os.path.commonpath()` → `/proj` (valid directory)

When `os.chdir(common)` is called on line 195 with a partial directory name, it will crash with `FileNotFoundError`.

This workaround is specifically for Windows where path lengths matter, so the bug would manifest on Windows systems with long, similar project paths.

## Fix

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -188,7 +188,7 @@ def build_module(name, pyxfilename, pyxbuild_dir=None, inplace=False, language_
     if pyxbuild_dir and sys.platform == 'win32':
         # Windows concatenates the pyxbuild_dir to the pyxfilename when
         # compiling, and then complains that the filename is too long
-        common = os.path.commonprefix([pyxbuild_dir, pyxfilename])
+        common = os.path.commonpath([pyxbuild_dir, pyxfilename])
     if len(common) > 30:
         pyxfilename = os.path.relpath(pyxfilename, common)
         pyxbuild_dir = os.path.relpath(pyxbuild_dir, common)
```