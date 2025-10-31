# Bug Report: scipy.io.matlab savemat Does Not Filter Variable Names Starting with Digits

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `savemat` function documentation states that variable names starting with a digit will not be saved and `MatWriteWarning` will be issued, but the implementation only filters variable names starting with underscore, allowing digit-starting names through without warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat, MatWriteWarning
import numpy as np
import tempfile
import warnings

@given(
    value=st.floats(min_value=-1e10, max_value=1e10,
                   allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_digit_start_vars_not_saved(value):
    arr = np.array([[value]])
    mdict = {'1invalid': arr, 'valid': arr}

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            savemat(fname, mdict, format='5')
            warning_issued = any(issubclass(warn.category, MatWriteWarning) for warn in w)

        loaded = loadmat(fname)

        assert 'valid' in loaded
        assert '1invalid' not in loaded, "Variables starting with digit should not be saved"
        assert warning_issued, "MatWriteWarning should be issued for digit-starting variables"
    finally:
        if os.path.exists(fname):
            os.unlink(fname)
```

**Failing input**: Any valid float value (e.g., `value=0.0`)

## Reproducing the Bug

```python
import numpy as np
import tempfile
import warnings
from scipy.io.matlab import loadmat, savemat, MatWriteWarning

arr = np.array([[1.0]])
mdict = {'1invalid': arr, 'valid': arr}

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(fname, mdict, format='5')

    print(f"MatWriteWarning issued: {any(issubclass(warn.category, MatWriteWarning) for warn in w)}")

loaded = loadmat(fname)
print(f"'1invalid' in loaded file: {'1invalid' in loaded}")
```

Output:
```
MatWriteWarning issued: False
'1invalid' in loaded file: True
```

## Why This Is A Bug

The documentation for `savemat` (scipy/io/matlab/_mio.py:271-273) explicitly states:

> Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` or a digit, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued.

However, the implementation in `_mio5.py:884-888` only checks for underscore:

```python
if name[0] == '_':
    msg = (f"Starting field name with a "
           f"underscore ({name}) is ignored")
    warnings.warn(msg, MatWriteWarning, stacklevel=2)
    continue
```

There is no check for digit-starting variable names, causing a contract violation between documentation and implementation.

## Fix

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -881,7 +881,10 @@ class MatFile5Writer:
             self.write_file_header()
         self._matrix_writer = VarWriter5(self)
         for name, var in mdict.items():
-            if name[0] == '_':
+            if name[0] == '_' or name[0].isdigit():
                 msg = (f"Starting field name with a "
-                       f"underscore ({name}) is ignored")
+                       f"{'underscore' if name[0] == '_' else 'digit'} "
+                       f"({name}) is ignored")
                 warnings.warn(msg, MatWriteWarning, stacklevel=2)
                 continue
```