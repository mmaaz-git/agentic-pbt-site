# Bug Report: scipy.io.matlab.savemat appendmat Parameter Not Working

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `appendmat` parameter in `scipy.io.matlab.savemat` does not work as documented. According to the documentation, when `appendmat=True`, the function should "append the .mat extension to the end of the given filename, if not already present." However, in practice, the function does NOT append the .mat extension when writing files.

## Property-Based Test

```python
@settings(max_examples=50)
@given(
    varname=st.text(
        alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')),
        min_size=1,
        max_size=10
    ),
    appendmat=st.booleans(),
)
def test_appendmat_parameter(varname, appendmat):
    with tempfile.TemporaryDirectory() as tmpdir:
        if appendmat:
            filename = os.path.join(tmpdir, 'test')
            expected_file = os.path.join(tmpdir, 'test.mat')
        else:
            filename = os.path.join(tmpdir, 'test.mat')
            expected_file = filename

        arr = np.array([1, 2, 3])
        data = {varname: arr}

        savemat(filename, data, appendmat=appendmat)

        assert os.path.exists(expected_file), f"Expected file {expected_file} does not exist"
```

**Failing input**: `varname='a', appendmat=True`

## Reproducing the Bug

```python
import tempfile
import os
import numpy as np
from scipy.io.matlab import savemat, loadmat

with tempfile.TemporaryDirectory() as tmpdir:
    filename = os.path.join(tmpdir, 'test')
    arr = np.array([1, 2, 3])
    data = {'a': arr}

    savemat(filename, data, appendmat=True)

    print("Files created:", os.listdir(tmpdir))
    print("Expected: ['test.mat']")
    print("Actual:", os.listdir(tmpdir))

    filename_with_mat = os.path.join(tmpdir, 'test.mat')
    print(f"test.mat exists: {os.path.exists(filename_with_mat)}")

    filename_without_mat = os.path.join(tmpdir, 'test')
    print(f"test exists: {os.path.exists(filename_without_mat)}")
```

Output:
```
Files created: ['test']
Expected: ['test.mat']
Actual: ['test']
test.mat exists: False
test exists: True
```

## Why This Is A Bug

The documentation for `savemat` states:

> appendmat : bool, optional
>     True (the default) to append the .mat extension to the end of the
>     given filename, if not already present.

This clearly indicates that when `appendmat=True` and the filename doesn't have a `.mat` extension, the function should append it. However, the function creates a file without the `.mat` extension.

Interestingly, the `appendmat` parameter works correctly for `loadmat`:

```python
filename_write = os.path.join(tmpdir, 'test.mat')
savemat(filename_write, data, appendmat=False)

filename_read = os.path.join(tmpdir, 'test')
loaded = loadmat(filename_read, appendmat=True)
```

This succeeds - `loadmat` correctly appends `.mat` and finds the file.

## Fix

The root cause is in the `_open_file` function in `scipy/io/matlab/_mio.py`:

```python
def _open_file(file_like, appendmat, mode='rb'):
    try:
        return open(file_like, mode), True
    except OSError as e:
        if isinstance(file_like, str):
            if appendmat and not file_like.endswith('.mat'):
                file_like += '.mat'
            return open(file_like, mode), True
        else:
            raise OSError(
                'Reader needs file name or open file-like object'
            ) from e
```

The logic appends `.mat` only when the initial `open()` call fails with an `OSError`. For read mode (`'rb'`), this happens when the file doesn't exist. However, for write mode (`'wb'`), `open()` succeeds even if the file doesn't exist - it creates the file. So the `.mat` extension is never appended.

The fix is to check `appendmat` before attempting to open the file:

```diff
 def _open_file(file_like, appendmat, mode='rb'):
     reqs = {'read'} if set(mode) & set('r+') else set()
     if set(mode) & set('wax+'):
         reqs.add('write')
     if reqs.issubset(dir(file_like)):
         return file_like, False

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