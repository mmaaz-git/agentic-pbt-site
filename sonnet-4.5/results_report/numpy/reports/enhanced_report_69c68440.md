# Bug Report: numpy.ctypeslib.load_library TypeError with None EXT_SUFFIX

**Target**: `numpy.ctypeslib.load_library`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.ctypeslib.load_library` crashes with a TypeError when `sysconfig.get_config_var("EXT_SUFFIX")` returns None, failing to handle this documented return value gracefully.

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

# Run the test
test_load_library_handles_none_ext_suffix()
```

<details>

<summary>
**Failing input**: `libname='_'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 14, in test_load_library_handles_none_ext_suffix
    np.ctypeslib.load_library(libname, tmpdir)
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 146, in load_library
    libname_ext.insert(0, libname + so_ext)
                          ~~~~~~~~^~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 27, in <module>
    test_load_library_handles_none_ext_suffix()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 7, in test_load_library_handles_none_ext_suffix
    libname=st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 22, in test_load_library_handles_none_ext_suffix
    assert False, f"Bug: load_library crashes when EXT_SUFFIX is None: {e}"
           ^^^^^
AssertionError: Bug: load_library crashes when EXT_SUFFIX is None: can only concatenate str (not "NoneType") to str
Falsifying example: test_load_library_handles_none_ext_suffix(
    libname='_',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import tempfile
from unittest.mock import patch

import numpy as np

# Mock sysconfig.get_config_var to return None (simulating missing EXT_SUFFIX)
with tempfile.TemporaryDirectory() as tmpdir:
    with patch('sysconfig.get_config_var', return_value=None):
        try:
            # This should crash with a TypeError
            np.ctypeslib.load_library('mylib', tmpdir)
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
```

<details>

<summary>
TypeError when concatenating None with string
</summary>
```
Error type: TypeError
Error message: can only concatenate str (not "NoneType") to str
```
</details>

## Why This Is A Bug

This violates expected behavior because `sysconfig.get_config_var()` is explicitly documented in Python's standard library to return None when a configuration variable doesn't exist. The NumPy code fails to handle this documented return value properly.

The crash occurs in `/numpy/ctypeslib/_ctypeslib.py` at line 146:
```python
libname_ext.insert(0, libname + so_ext)
```

When `so_ext` is None (retrieved from line 144), Python cannot concatenate a string with None, causing a TypeError. The conditional check on line 145 (`if not so_ext == base_ext:`) evaluates to True when `so_ext` is None (since `None != base_ext`), allowing execution to reach the problematic concatenation.

The code shows intent to handle different EXT_SUFFIX values through its conditional logic, but overlooks the None case. This is a defensive programming failure - robust code should handle all documented return values without crashing.

## Relevant Context

- **Python sysconfig documentation**: States that `get_config_var()` returns None if the configuration variable is not found
- **Standard installations**: EXT_SUFFIX is typically set to values like `.cpython-313-x86_64-linux-gnu.so` on Linux systems
- **Edge case occurrence**: This bug manifests in custom Python builds or environments with incomplete configuration
- **Code location**: The bug is in numpy/ctypeslib/_ctypeslib.py, specifically in the `load_library` function
- **Impact**: While rare, the bug causes an uninformative TypeError instead of a clear error message about missing libraries

Documentation links:
- Python sysconfig: https://docs.python.org/3/library/sysconfig.html#sysconfig.get_config_var
- NumPy ctypeslib: https://numpy.org/doc/stable/reference/routines.ctypeslib.html

## Proposed Fix

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