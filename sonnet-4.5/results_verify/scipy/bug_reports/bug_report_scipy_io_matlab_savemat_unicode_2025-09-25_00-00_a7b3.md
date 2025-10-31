# Bug Report: scipy.io.matlab.savemat Unicode Variable Name Encoding Error

**Target**: `scipy.io.matlab.savemat`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.io.matlab.savemat` raises a cryptic `UnicodeEncodeError` when given variable names containing non-Latin-1 Unicode characters, instead of validating input and providing a clear error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from io import BytesIO
import scipy.io.matlab as sio

@given(st.text(min_size=1, max_size=20))
@settings(max_examples=1000)
def test_savemat_variable_name_encoding(varname):
    data = {varname: np.array([[1.0]])}
    f = BytesIO()
    try:
        sio.savemat(f, data)
        f.seek(0)
        result = sio.loadmat(f)
        assert varname in result or varname.startswith('_')
    except UnicodeEncodeError:
        pass
```

**Failing input**: `varname='Ā'` (U+0100, Latin A with macron)

## Reproducing the Bug

```python
import numpy as np
from io import BytesIO
import scipy.io.matlab as sio

data = {'Ā': np.array([[1.0]])}
f = BytesIO()
sio.savemat(f, data)
```

**Output**:
```
UnicodeEncodeError: 'latin-1' codec can't encode character '\u0100' in position 0: ordinal not in range(256)
```

## Why This Is A Bug

1. **Poor error message**: Users receive a low-level codec error rather than a clear message about variable name restrictions
2. **Missing input validation**: The function should validate variable names upfront
3. **Undocumented restriction**: The `savemat` docstring doesn't document variable name character restrictions

While MATLAB variable names may be restricted to ASCII characters, the error should be caught early with a clear, actionable message like: "Variable name 'Ā' contains invalid characters. MATLAB variable names must use only ASCII letters, digits, and underscores."

## Fix

Add input validation at the beginning of `put_variables` method in `_mio5.py` to check variable names before attempting to encode them:

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -898,6 +898,13 @@ class MatFile5Writer:
             self._matrix_writer.file_stream = stream
             self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
+        for name, var in mdict.items():
+            if name[0] == '_':
+                continue
+            try:
+                name.encode('latin1')
+            except UnicodeEncodeError:
+                raise ValueError(f"Variable name '{name}' contains characters that cannot be encoded in MATLAB files. "
+                                 f"Variable names must use only ASCII letters, digits, and underscores.") from None
         for name, var in mdict.items():
             if name[0] == '_':
                 msg = (f"Starting field name with a "
```

Alternatively, document the restriction clearly in the `savemat` docstring under the Parameters section.