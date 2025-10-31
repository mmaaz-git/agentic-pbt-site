# Bug Report: scipy.io.matlab.savemat Variable Name Validation

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `savemat` function documentation claims that variable names starting with a digit will not be saved and should issue a `MatWriteWarning`, but the implementation allows such names to be saved without warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import tempfile
import os
import numpy as np
from scipy.io.matlab import loadmat, savemat
import warnings

@settings(max_examples=300)
@given(
    var_name=st.text(alphabet='0123456789', min_size=1, max_size=1)
)
def test_variable_names_starting_with_digit(var_name):
    arr = np.array([1.0, 2.0, 3.0])

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.mat')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            savemat(file_path, {var_name: arr})

        loaded = loadmat(file_path)

        assert var_name not in loaded
```

**Failing input**: `var_name='0'`

## Reproducing the Bug

```python
import tempfile
import os
import numpy as np
from scipy.io.matlab import loadmat, savemat
import warnings

arr = np.array([1.0, 2.0, 3.0])

with tempfile.TemporaryDirectory() as tmpdir:
    file_path = os.path.join(tmpdir, 'test.mat')

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(file_path, {'0': arr, '9abc': arr, 'abc': arr})
        print(f"Warnings issued: {len(w)}")

    loaded = loadmat(file_path)

    print("Variables saved:")
    for key in loaded.keys():
        if not key.startswith('__'):
            print(f"  {key}")

    print(f"\nVariable '0' in file: {'0' in loaded}")
    print(f"Variable '9abc' in file: {'9abc' in loaded}")
    print(f"Variable 'abc' in file: {'abc' in loaded}")
```

Expected output:
```
Warnings issued: 2
Variables saved:
  abc
Variable '0' in file: False
Variable '9abc' in file: False
Variable 'abc' in file: True
```

Actual output:
```
Warnings issued: 0
Variables saved:
  0
  9abc
  abc
Variable '0' in file: True
Variable '9abc' in file: True
Variable 'abc' in file: True
```

## Why This Is A Bug

The documentation in `/scipy/io/matlab/_mio.py` lines 271-273 explicitly states:

> "Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` or a digit, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued."

However, the implementation in `/scipy/io/matlab/_mio5.py` lines 884-888 only checks for underscore:

```python
for name, var in mdict.items():
    if name[0] == '_':
        msg = (f"Starting field name with a "
               f"underscore ({name}) is ignored")
        warnings.warn(msg, MatWriteWarning, stacklevel=2)
        continue
```

It should also check for digits. The v4 writer in `_mio4.py` has no validation at all.

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
+                msg = (f"Starting field name with an "
+                       f"underscore or digit ({name}) is ignored")
                 warnings.warn(msg, MatWriteWarning, stacklevel=2)
                 continue
             is_global = name in self.global_vars
```

```diff
--- a/scipy/io/matlab/_mio4.py
+++ b/scipy/io/matlab/_mio4.py
@@ -629,6 +629,12 @@ class MatFile4Writer:
         # with the matlab 5 version of this method
         self._matrix_writer = VarWriter4(self)
         for name, var in mdict.items():
+            if name[0] in '_0123456789':
+                msg = (f"Starting field name with an "
+                       f"underscore or digit ({name}) is ignored")
+                warnings.warn(msg, MatWriteWarning, stacklevel=2)
+                continue
             self._matrix_writer.write(var, name)
```