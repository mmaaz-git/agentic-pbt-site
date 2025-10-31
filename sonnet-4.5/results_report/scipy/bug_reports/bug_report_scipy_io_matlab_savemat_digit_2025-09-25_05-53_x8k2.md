# Bug Report: scipy.io.matlab.savemat allows variable names starting with digits

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `savemat` function violates its documented contract by allowing variable names that start with digits to be saved to MATLAB files, when the documentation explicitly states they should be ignored with a warning.

## Property-Based Test

```python
@given(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122)))
@settings(max_examples=100)
def test_variable_names_starting_with_digit(name):
    varname = '0' + name
    arr = np.array([1, 2, 3])

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        mdict = {varname: arr}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            savemat(fname, mdict)
            if len(w) > 0:
                assert issubclass(w[0].category, MatWriteWarning)

        result = loadmat(fname)
        assert varname not in result or len(result[varname]) == 0
    finally:
        if os.path.exists(fname):
            os.unlink(fname)
```

**Failing input**: `name='A'` (or any other generated value)

## Reproducing the Bug

```python
import numpy as np
import tempfile
import os
import warnings
from scipy.io.matlab import loadmat, savemat, MatWriteWarning

arr = np.array([1, 2, 3])
varname = '0test'

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

try:
    mdict = {varname: arr}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(fname, mdict)
        print(f"Warnings raised: {len(w)}")

    result = loadmat(fname)

    if varname in result:
        print(f"BUG: Variable '{varname}' was saved despite starting with a digit!")
        print(f"Loaded value: {result[varname]}")
finally:
    if os.path.exists(fname):
        os.unlink(fname)
```

Output:
```
Warnings raised: 0
BUG: Variable '0test' was saved despite starting with a digit!
Loaded value: [[1 2 3]]
```

## Why This Is A Bug

The `savemat` function's docstring (in `_mio.py` lines 269-273) explicitly states:

> Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` or a digit, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued.

However, the implementation only checks for underscore prefix, not digits:

- In `_mio5.py` line 884: only checks `if name[0] == '_':`
- In `_mio4.py` line 631-632: no validation at all

This violates the documented contract in two ways:
1. Variables starting with digits ARE saved (when they shouldn't be)
2. No `MatWriteWarning` is issued (when it should be)

Note that struct/dict fields are correctly validated (in `_mio5.py` line 486), but top-level variable names are not.

## Fix

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -881,9 +881,10 @@ class MatFile5Writer:
             self.write_file_header()
         self._matrix_writer = VarWriter5(self)
         for name, var in mdict.items():
-            if name[0] == '_':
-                msg = (f"Starting field name with a "
-                       f"underscore ({name}) is ignored")
+            if name[0] in '_0123456789':
+                msg = (f"Starting field name with a underscore "
+                       f"or a digit ({name}) is ignored")
                 warnings.warn(msg, MatWriteWarning, stacklevel=2)
                 continue
             is_global = name in self.global_vars
--- a/scipy/io/matlab/_mio4.py
+++ b/scipy/io/matlab/_mio4.py
@@ -629,6 +629,11 @@ class MatFile4Writer:
         # with the matlab 5 version of this method
         self._matrix_writer = VarWriter4(self)
         for name, var in mdict.items():
+            if name[0] in '_0123456789':
+                msg = (f"Starting field name with a underscore "
+                       f"or a digit ({name}) is ignored")
+                warnings.warn(msg, MatWriteWarning, stacklevel=2)
+                continue
             self._matrix_writer.write(var, name)
```