# Bug Report: scipy.io.matlab Allows Digit-Prefixed Top-Level Keys

**Target**: `scipy.io.matlab.savemat`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `savemat` function's documentation states that keys starting with digits should not be saved and a warning should be issued, but this validation is only enforced for struct fields, not for top-level variable names in the dictionary.

## Property-Based Test

```python
from io import BytesIO
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat
import warnings


@settings(max_examples=50)
@given(st.from_regex(r'^[0-9][a-zA-Z0-9_]*$', fullmatch=True))
def test_digit_key_not_saved(key):
    bio = BytesIO()
    data = {key: np.array([1, 2, 3])}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(bio, data)

        if len(w) > 0:
            assert any("MatWriteWarning" in str(warn.category) for warn in w)

    bio.seek(0)
    loaded = loadmat(bio)

    assert key not in loaded
```

**Failing input**: `key='0'`

## Reproducing the Bug

```python
from io import BytesIO
import numpy as np
from scipy.io.matlab import loadmat, savemat
import warnings

bio = BytesIO()
data = {'0': np.array([1, 2, 3])}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(bio, data)
    print(f"Warnings issued: {len(w)}")

bio.seek(0)
loaded = loadmat(bio)
print(f"'0' in loaded: {'0' in loaded}")
```

**Output:**
```
Warnings issued: 0
'0' in loaded: True
```

## Why This Is A Bug

The `savemat` docstring explicitly states:

> "Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` or a digit, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued."

However:
1. Top-level keys starting with digits (like `'0'`, `'1test'`) ARE saved
2. No `MatWriteWarning` is issued
3. This creates inconsistent behavior: digit-prefixed keys are rejected in struct fields but accepted at the top level

## Fix

The bug is in `_mio5.py` in the `MatFile5Writer.put_variables` method around line 884. The code only checks for underscore-prefixed keys:

```python
for name, var in mdict.items():
    if name[0] == '_':
        msg = (f"Starting field name with a "
               f"underscore ({name}) is ignored")
        warnings.warn(msg, MatWriteWarning, stacklevel=2)
        continue
```

But doesn't check for digit-prefixed keys. The fix:

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -882,8 +882,9 @@ class MatFile5Writer:
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
```

This matches the existing check for struct fields at line 486 and makes the behavior consistent with the documentation.