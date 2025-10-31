# Bug Report: numpy.ctypeslib.load_library TypeError when EXT_SUFFIX is None

**Target**: `numpy.ctypeslib.load_library`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`load_library` crashes with a TypeError when `sysconfig.get_config_var("EXT_SUFFIX")` returns None, instead of gracefully handling the missing configuration.

## Property-Based Test

```python
from unittest.mock import patch
from hypothesis import given, settings, strategies as st
import numpy as np
import tempfile

@given(
    libname=st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10)
)
@settings(max_examples=100)
def test_load_library_handles_none_ext_suffix(libname):
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('sysconfig.get_config_var', return_value=None):
            try:
                np.ctypeslib.load_library(libname, tmpdir)
            except OSError as e:
                if "no file with expected extension" in str(e):
                    pass
                else:
                    raise
            except TypeError as e:
                if "unsupported operand type" in str(e) or "can only concatenate" in str(e):
                    assert False, f"Bug: load_library crashes when EXT_SUFFIX is None: {e}"
                else:
                    raise
```

**Failing input**: `libname='mylib'` (or any string)

## Reproducing the Bug

```python
import tempfile
from unittest.mock import patch

import numpy as np

with tempfile.TemporaryDirectory() as tmpdir:
    with patch('sysconfig.get_config_var', return_value=None):
        np.ctypeslib.load_library('mylib', tmpdir)
```

## Why This Is A Bug

`sysconfig.get_config_var()` is documented to return None when a configuration variable doesn't exist. While EXT_SUFFIX is typically set in standard Python installations, custom builds or edge cases might not have it. The code should handle this gracefully (e.g., by only trying `base_ext`) rather than crashing with a confusing TypeError about string concatenation.

The crash occurs at line 146:
```python
libname_ext.insert(0, libname + so_ext)
```
when `so_ext` is None.

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -142,7 +142,7 @@ def load_library(libname, loader_path):
                 base_ext = ".dll"
             libname_ext = [libname + base_ext]
             so_ext = sysconfig.get_config_var("EXT_SUFFIX")
-            if not so_ext == base_ext:
+            if so_ext is not None and so_ext != base_ext:
                 libname_ext.insert(0, libname + so_ext)
         else:
             libname_ext = [libname]
```