# Bug Report: scipy.io.matlab.savemat Variable Name Validation

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.io.matlab.savemat` fails to validate variable names starting with digits. According to the documentation, such variables should not be saved and should trigger a `MatWriteWarning`, but they are actually saved without any warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst
import scipy.io.matlab as mio
import numpy as np
import tempfile
import os
import warnings

@given(
    var_name_with_digit=st.text(
        alphabet='0123456789',
        min_size=1,
        max_size=1
    ).flatmap(lambda digit: st.just(digit + 'var')),
    arr=npst.arrays(dtype=npst.floating_dtypes(), shape=(2, 2))
)
def test_variable_names_starting_with_digit(var_name_with_digit, arr):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.mat')
        data = {var_name_with_digit: arr}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mio.savemat(file_path, data)
            loaded = mio.loadmat(file_path)

            # According to docs, this should not be in loaded
            assert var_name_with_digit not in loaded, \
                f"Variable {var_name_with_digit} should not be saved (starts with digit)"
```

**Failing input**: `var_name_with_digit='0var'`, `arr=array([[0., 0.], [0., 0.]])`

## Reproducing the Bug

```python
import scipy.io.matlab as mio
import numpy as np
import tempfile
import warnings

arr = np.array([[1.0, 2.0], [3.0, 4.0]])

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    file_path = f.name

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    mio.savemat(file_path, {'0digit': arr, '9name': arr})
    print(f"Warnings issued: {len(w)}")
    for warning in w:
        print(f"  {warning.message}")

loaded = mio.loadmat(file_path)
user_vars = {k: v for k, v in loaded.items() if not k.startswith('__')}
print(f"Variables saved: {list(user_vars.keys())}")
```

Output:
```
Warnings issued: 0
Variables saved: ['0digit', '9name']
```

Expected behavior (according to documentation):
```
Warnings issued: 2
  MatWriteWarning: Starting field name with a digit (0digit) is ignored
  MatWriteWarning: Starting field name with a digit (9name) is ignored
Variables saved: []
```

## Why This Is A Bug

The `savemat` docstring explicitly states:

> "Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` **or a digit**, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued."

The implementation correctly handles keys starting with `_` (they are not saved and a warning is issued), but fails to handle keys starting with digits. This violates the documented API contract.

Additionally, MATLAB itself does not allow variable names starting with digits, so saving them creates .mat files that may not be readable or usable in MATLAB.

## Fix

The issue is in `/home/npc/.local/lib/python3.13/site-packages/scipy/io/matlab/_mio5.py` (or similar file). The validation logic checks for underscores but not digits. The fix should add digit validation similar to underscore validation:

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -xxx,x +xxx,x @@ def put_variables(self, mdict, write_header=None):
     for name, var in mdict.items():
         if name[0] == '_':
             warnings.warn(f'Starting field name with a underscore ({name}) is ignored',
                          MatWriteWarning, stacklevel=2)
             continue
+        if name[0].isdigit():
+            warnings.warn(f'Starting field name with a digit ({name}) is ignored',
+                         MatWriteWarning, stacklevel=2)
+            continue
         self.write_var(name, var)
```

Note: The exact location and implementation may vary depending on the actual code structure. The key is to add validation that checks if `name[0].isdigit()` and skip saving with a warning, similar to the underscore check.