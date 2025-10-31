# Bug Report: scipy.io.matlab.savemat incorrectly saves variables with digit-starting names

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `savemat` function violates its documented contract by saving variables with names that start with digits to MATLAB files without issuing warnings, when the documentation explicitly states such variables should be ignored with a `MatWriteWarning`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import tempfile
import os
import warnings
from scipy.io.matlab import loadmat, savemat, MatWriteWarning

@given(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122)))
@settings(max_examples=100)
def test_variable_names_starting_with_digit(name):
    """Test that savemat properly ignores variables starting with digits and issues warnings."""
    varname = '0' + name
    arr = np.array([1, 2, 3])

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        mdict = {varname: arr}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            savemat(fname, mdict)

            # Check for expected warning
            if len(w) > 0:
                assert issubclass(w[0].category, MatWriteWarning), f"Expected MatWriteWarning, got {w[0].category}"
            else:
                print(f"FAILURE: No warning was issued for variable name '{varname}'")

        # Load and check that the variable was not saved
        result = loadmat(fname)

        # The variable should either not be present or be empty
        if varname in result:
            print(f"FAILURE: Variable '{varname}' was saved when it should have been ignored")
            assert False, f"Variable {varname} should not have been saved to the mat file"

    finally:
        if os.path.exists(fname):
            os.unlink(fname)
```

<details>

<summary>
**Failing input**: `name='A'` (any alphabetic string will fail)
</summary>
```
Testing with variable name '0A'...
FAILURE: No warning was issued for variable name '0A'
FAILURE: Variable '0A' was saved when it should have been ignored
Variable value: [[1 2 3]]

Running full hypothesis test suite...
Hypothesis found failures as expected. The test correctly identifies the bug.
```
</details>

## Reproducing the Bug

```python
import numpy as np
import tempfile
import os
import warnings
from scipy.io.matlab import loadmat, savemat, MatWriteWarning

# Test with a variable name starting with a digit
arr = np.array([1, 2, 3])
varname = '0test'

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

try:
    mdict = {varname: arr}

    # Capture any warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(fname, mdict)
        print(f"Number of warnings raised: {len(w)}")
        if len(w) > 0:
            for warning in w:
                print(f"Warning: {warning.category.__name__}: {warning.message}")

    # Try to load the file
    result = loadmat(fname)

    # Check what variables were loaded
    print(f"\nVariables in loaded mat file: {[k for k in result.keys() if not k.startswith('__')]}")

    if varname in result:
        print(f"\nBUG CONFIRMED: Variable '{varname}' was saved despite starting with a digit!")
        print(f"Expected behavior: Variable should be ignored with a MatWriteWarning")
        print(f"Loaded value: {result[varname]}")
    else:
        print(f"\nVariable '{varname}' was correctly ignored")

finally:
    if os.path.exists(fname):
        os.unlink(fname)
```

<details>

<summary>
BUG CONFIRMED: Variable '0test' was saved despite starting with digit
</summary>
```
Number of warnings raised: 0

Variables in loaded mat file: ['0test']

BUG CONFIRMED: Variable '0test' was saved despite starting with a digit!
Expected behavior: Variable should be ignored with a MatWriteWarning
Loaded value: [[1 2 3]]
```
</details>

## Why This Is A Bug

The `savemat` function's documentation in `/scipy/io/matlab/_mio.py` lines 270-273 explicitly states:

> "Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` or a digit, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued."

However, the implementation violates this documented contract in two critical ways:

1. **Variables starting with digits ARE saved** - The file `_mio5.py` line 884 only checks for underscore prefix (`if name[0] == '_':`), completely ignoring the digit check. The `_mio4.py` implementation at lines 631-632 has no validation at all.

2. **No `MatWriteWarning` is issued** - Since the digit check is missing, no warning is ever raised when such variables are encountered.

This creates **MATLAB-incompatible files** because MATLAB itself does not allow variable names starting with digits. According to MATLAB's documentation, valid variable names must begin with a letter, not a digit.

## Relevant Context

The bug demonstrates an inconsistency in the codebase: struct/dict fields ARE correctly validated. In `_mio5.py` line 486, the code properly checks `if field[0] not in '_0123456789':` for struct fields and issues the appropriate warning. This shows the developers understood the requirement but failed to apply it consistently to top-level variable names.

This discrepancy suggests the bug is an oversight rather than intentional behavior. The correct validation logic exists in the codebase but is only partially applied.

Links to relevant code:
- Documentation: `scipy/io/matlab/_mio.py:270-273`
- Bug location (MATLAB 5): `scipy/io/matlab/_mio5.py:884`
- Bug location (MATLAB 4): `scipy/io/matlab/_mio4.py:631-632`
- Correct implementation for struct fields: `scipy/io/matlab/_mio5.py:486`

## Proposed Fix

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -881,9 +881,9 @@ class MatFile5Writer:
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