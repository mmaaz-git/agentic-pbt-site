# Bug Report: scipy.io.matlab.savemat Unicode Variable Names

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`savemat` crashes with an unhelpful `UnicodeEncodeError` when variable names contain Unicode characters outside the latin1 range (U+0100+), despite Python 3 allowing Unicode identifiers.

## Property-Based Test

```python
import io
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import savemat, loadmat

valid_var_name = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1, max_size=31
).filter(lambda x: x[0].isalpha() or x[0] == '_').filter(lambda x: not x.startswith('_'))

@given(
    var_name=valid_var_name,
    data=st.integers(min_value=-1e10, max_value=1e10).map(lambda x: np.array([[x]])),
)
@settings(max_examples=200, deadline=None)
def test_round_trip_savemat_loadmat(var_name, data):
    f = io.BytesIO()
    original_dict = {var_name: data}

    savemat(f, original_dict)
    f.seek(0)

    loaded_dict = loadmat(f)
    assert var_name in loaded_dict
```

**Failing input**: `var_name='Ā', data=array([[0]])`

## Reproducing the Bug

```python
import io
import numpy as np
from scipy.io.matlab import savemat

f = io.BytesIO()
data = {'Ā': np.array([[1, 2, 3]])}
savemat(f, data)
```

Output:
```
UnicodeEncodeError: 'latin-1' codec can't encode character '\u0100' in position 0: ordinal not in range(256)
```

## Why This Is A Bug

1. **Python 3 allows Unicode identifiers** (PEP 3131), so users can legitimately have variables named 'Ā', '中', etc.
2. **The documentation doesn't warn** about this limitation - users have no way to know variable names must be latin1-encodable
3. **The error message is unhelpful** - it doesn't explain that MATLAB variable names are limited to latin1 characters
4. **Similar validation exists** for underscore-prefixed names, but not for encoding issues

## Fix

The fix should add validation for variable names and provide a helpful error message:

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -23,6 +23,13 @@ class MatFile5Writer:
         self._matrix_writer = VarWriter5(self)
         for name, var in mdict.items():
             if name[0] == '_':
                 msg = (f"Starting field name with a "
                        f"underscore ({name}) is ignored")
                 warnings.warn(msg, MatWriteWarning, stacklevel=2)
                 continue
+            try:
+                name.encode('latin1')
+            except UnicodeEncodeError:
+                raise ValueError(
+                    f"Variable name '{name}' contains characters outside the latin1 "
+                    f"encoding range (U+0000 to U+00FF). MATLAB variable names must "
+                    f"use only ASCII or latin1 characters."
+                ) from None
             is_global = name in self.global_vars
             if self.do_compression:
```