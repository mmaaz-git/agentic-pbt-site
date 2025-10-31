# Bug Report: scipy.io.matlab savemat appendmat Parameter Does Not Append Extension

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `appendmat=True` parameter in `savemat()` does not append the `.mat` extension when saving files, contrary to the documented behavior. Files are saved without the extension.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import tempfile
import os
from scipy.io.matlab import savemat

@given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
       st.booleans())
def test_appendmat_behavior(base_name, appendmat):
    assume('.' not in base_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, base_name)
        data = {'x': 1.0}

        savemat(fname, data, appendmat=appendmat)

        if appendmat:
            expected_fname = fname + '.mat'
            assert os.path.exists(expected_fname), f"Expected file {expected_fname} not found"
        else:
            assert os.path.exists(fname), f"Expected file {fname} not found"
```

**Failing input**: `base_name='test'`, `appendmat=True`

## Reproducing the Bug

```python
import tempfile
import os
from scipy.io.matlab import savemat

with tempfile.TemporaryDirectory() as tmpdir:
    fname = os.path.join(tmpdir, 'test')
    data = {'x': 1.0}

    savemat(fname, data, appendmat=True)

    print(f"Files in directory: {os.listdir(tmpdir)}")
    print(f"Expected 'test.mat': {os.path.exists(fname + '.mat')}")
    print(f"Actual 'test' exists: {os.path.exists(fname)}")
```

Output:
```
Files in directory: ['test']
Expected 'test.mat': False
Actual 'test' exists: True
```

## Why This Is A Bug

1. **Documentation violation**: The `savemat` docstring states: "True (the default) to append the .mat extension to the end of the given filename, if not already present." This clearly promises that the extension will be appended when `appendmat=True`.

2. **Inconsistent with loadmat**: The `loadmat` function with `appendmat=True` successfully appends `.mat` when reading files. This asymmetry breaks round-trip expectations.

3. **API contract violation**: The parameter name and documentation create a clear expectation that doesn't match the actual behavior.

## Fix

The bug is in `/scipy/io/matlab/_mio.py` in the `_open_file` function. The function only appends `.mat` if the initial `open()` call fails:

```python
def _open_file(file_like, appendmat, mode='rb'):
    # ...
    try:
        return open(file_like, mode), True
    except OSError as e:
        if isinstance(file_like, str):
            if appendmat and not file_like.endswith('.mat'):
                file_like += '.mat'
            return open(file_like, mode), True
```

The problem: when opening a file for writing (`mode='wb'`), `open()` succeeds even if the file doesn't exist—it creates the file. So the exception handler that appends `.mat` is never reached.

**Fix**: Append `.mat` before attempting to open the file when `appendmat=True`:

```diff
--- a/scipy/io/matlab/_mio.py
+++ b/scipy/io/matlab/_mio.py
@@ -33,6 +33,11 @@ def _open_file(file_like, appendmat, mode='rb'):
     if reqs.issubset(dir(file_like)):
         return file_like, False

+    # Append .mat extension before opening if requested
+    if isinstance(file_like, str) and appendmat and not file_like.endswith('.mat'):
+        file_like += '.mat'
+
     try:
         return open(file_like, mode), True
     except OSError as e:
-        # Probably "not found"
-        if isinstance(file_like, str):
-            if appendmat and not file_like.endswith('.mat'):
-                file_like += '.mat'
-            return open(file_like, mode), True
-        else:
-            raise OSError(
-                'Reader needs file name or open file-like object'
-            ) from e
+        raise OSError(
+            'Reader needs file name or open file-like object'
+        ) from e
```