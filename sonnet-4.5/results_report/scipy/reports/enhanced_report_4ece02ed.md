# Bug Report: scipy.io.matlab.savemat Does Not Filter Variable Names Starting with Digits

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `savemat` function documentation promises to filter out and warn about variable names starting with digits, but the implementation only filters variables starting with underscores, allowing digit-prefixed names to be saved without warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat, MatWriteWarning
import numpy as np
import tempfile
import warnings
import os

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

# Run the test
test_digit_start_vars_not_saved()
```

<details>

<summary>
**Failing input**: `value=0.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 36, in <module>
    test_digit_start_vars_not_saved()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 9, in test_digit_start_vars_not_saved
    value=st.floats(min_value=-1e10, max_value=1e10,
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 29, in test_digit_start_vars_not_saved
    assert '1invalid' not in loaded, "Variables starting with digit should not be saved"
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Variables starting with digit should not be saved
Falsifying example: test_digit_start_vars_not_saved(
    value=0.0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import tempfile
import warnings
import os
from scipy.io.matlab import loadmat, savemat, MatWriteWarning

# Create test data
arr = np.array([[1.0]])
mdict = {'1invalid': arr, 'valid': arr}

# Create temporary file
with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

try:
    # Save the data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(fname, mdict, format='5')

        # Check if MatWriteWarning was issued
        warning_issued = any(issubclass(warn.category, MatWriteWarning) for warn in w)
        print(f"MatWriteWarning issued: {warning_issued}")

        # Show all warnings if any
        if w:
            for warn in w:
                print(f"Warning: {warn.category.__name__}: {warn.message}")
        else:
            print("No warnings were issued")

    # Load the data back
    loaded = loadmat(fname)

    # Check what was loaded
    print(f"\nKeys in loaded file: {list(loaded.keys())}")
    print(f"'1invalid' in loaded file: {'1invalid' in loaded}")
    print(f"'valid' in loaded file: {'valid' in loaded}")

    # If '1invalid' is in the loaded data, print its value
    if '1invalid' in loaded:
        print(f"Value of '1invalid': {loaded['1invalid']}")

finally:
    # Clean up
    if os.path.exists(fname):
        os.unlink(fname)
```

<details>

<summary>
Output shows '1invalid' was incorrectly saved without warning
</summary>
```
MatWriteWarning issued: False
No warnings were issued

Keys in loaded file: ['__header__', '__version__', '__globals__', '1invalid', 'valid']
'1invalid' in loaded file: True
'valid' in loaded file: True
Value of '1invalid': [[1.]]
```
</details>

## Why This Is A Bug

This violates the documented behavior and breaks MATLAB compatibility. The `savemat` function documentation explicitly states in `/scipy/io/matlab/_mio.py` lines 271-273:

> Note that if this dict has a key starting with `_` or a sub-dict has a key starting with `_` or a digit, these key's items will not be saved in the mat file and `MatWriteWarning` will be issued.

However, the implementation in `/scipy/io/matlab/_mio5.py` lines 884-888 only checks for underscore-starting names:

```python
if name[0] == '_':
    msg = (f"Starting field name with a "
           f"underscore ({name}) is ignored")
    warnings.warn(msg, MatWriteWarning, stacklevel=2)
    continue
```

This creates three problems:
1. **Contract violation**: The function does not behave as documented
2. **MATLAB incompatibility**: Variables starting with digits are invalid MATLAB identifiers and cannot be loaded in MATLAB
3. **Silent failure**: No warning is issued to alert users that their data may be incompatible with MATLAB

## Relevant Context

According to MATLAB's official documentation, valid MATLAB variable names must:
- Start with a letter (not a digit)
- Contain only letters, digits, and underscores
- Be no longer than `namelengthmax` characters (typically 63)

The current scipy implementation correctly filters underscore-prefixed variables (which are also invalid in MATLAB), but fails to filter digit-prefixed variables, creating .mat files that may not load correctly in MATLAB.

Documentation link: https://github.com/scipy/scipy/blob/main/scipy/io/matlab/_mio.py#L271-L273
Implementation link: https://github.com/scipy/scipy/blob/main/scipy/io/matlab/_mio5.py#L884-L888

## Proposed Fix

```diff
--- a/scipy/io/matlab/_mio5.py
+++ b/scipy/io/matlab/_mio5.py
@@ -881,10 +881,13 @@ class MatFile5Writer:
             self.write_file_header()
         self._matrix_writer = VarWriter5(self)
         for name, var in mdict.items():
-            if name[0] == '_':
-                msg = (f"Starting field name with a "
-                       f"underscore ({name}) is ignored")
-                warnings.warn(msg, MatWriteWarning, stacklevel=2)
+            if name and (name[0] == '_' or name[0].isdigit()):
+                if name[0] == '_':
+                    msg = f"Starting field name with underscore ({name}) is ignored"
+                else:
+                    msg = f"Starting field name with digit ({name}) is ignored"
+                warnings.warn(msg, MatWriteWarning, stacklevel=2)
                 continue
             is_global = name in self.global_vars
             if self.do_compression:
```