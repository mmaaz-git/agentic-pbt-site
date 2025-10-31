# Bug Report: scipy.io.matlab.savemat appendmat Parameter Fails to Append .mat Extension

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `appendmat=True` parameter in `scipy.io.matlab.savemat()` fails to append the `.mat` extension when saving files, directly contradicting its documented behavior that promises to "append the .mat extension to the end of the given filename, if not already present."

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, example
import tempfile
import os
from scipy.io.matlab import savemat

@given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
       st.booleans())
@example('test', True)  # Known failing example
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

# Run the test
if __name__ == "__main__":
    # Run hypothesis test
    print("Running Hypothesis test with known failing input base_name='test', appendmat=True:")
    try:
        test_appendmat_behavior()
    except AssertionError as e:
        print(f"Falsifying example: base_name='test', appendmat=True")
        print(f"AssertionError: {e}")
```

<details>

<summary>
**Failing input**: `base_name='test', appendmat=True`
</summary>
```
Running Hypothesis test with known failing input base_name='test', appendmat=True:
Falsifying example: base_name='test', appendmat=True
AssertionError: Expected file /tmp/tmpctnf00oe/test.mat not found
```
</details>

## Reproducing the Bug

```python
import tempfile
import os
from scipy.io.matlab import savemat

with tempfile.TemporaryDirectory() as tmpdir:
    fname = os.path.join(tmpdir, 'test')
    data = {'x': 1.0}

    # Test with appendmat=True (should append .mat extension)
    savemat(fname, data, appendmat=True)

    print(f"Files in directory: {os.listdir(tmpdir)}")
    print(f"Expected 'test.mat' exists: {os.path.exists(fname + '.mat')}")
    print(f"Actual 'test' exists: {os.path.exists(fname)}")

    # Also test with appendmat=False for comparison
    fname2 = os.path.join(tmpdir, 'test2')
    savemat(fname2, data, appendmat=False)
    print(f"\nWith appendmat=False:")
    print(f"Files in directory: {os.listdir(tmpdir)}")
    print(f"'test2' exists: {os.path.exists(fname2)}")
    print(f"'test2.mat' exists: {os.path.exists(fname2 + '.mat')}")
```

<details>

<summary>
Output showing the bug: file saved as 'test' instead of 'test.mat'
</summary>
```
Files in directory: ['test']
Expected 'test.mat' exists: False
Actual 'test' exists: True

With appendmat=False:
Files in directory: ['test2', 'test']
'test2' exists: True
'test2.mat' exists: False
```
</details>

## Why This Is A Bug

1. **Clear Documentation Violation**: The `savemat` docstring at line 275-276 in `/scipy/io/matlab/_mio.py` explicitly states: "True (the default) to append the .mat extension to the end of the given filename, if not already present." The function fails to deliver this promised behavior.

2. **Broken API Symmetry**: The companion function `loadmat` with `appendmat=True` correctly attempts to append `.mat` when loading files (it tries both with and without the extension). This asymmetry breaks the principle of least surprise and round-trip expectations between save and load operations.

3. **Parameter Name Creates Clear Expectation**: The parameter name "appendmat" unambiguously suggests "append .mat extension". When set to `True`, users reasonably expect the extension to be appended.

4. **Default Value Implications**: Since `appendmat=True` is the default, most users will encounter this bug without explicitly setting the parameter, expecting files to be saved with the standard `.mat` extension.

5. **Implementation Bug is Clear**: The bug occurs because `_open_file` (lines 25-49) only appends `.mat` in the exception handler (lines 40-45). When opening a file for writing (`mode='wb'`), Python's `open()` succeeds even if the file doesn't exist—it simply creates the file. Therefore, the exception handler that would append `.mat` is never reached.

## Relevant Context

The issue stems from the `_open_file` function in `/scipy/io/matlab/_mio.py`. The function's logic assumes that `open()` will fail if a file doesn't exist, allowing it to retry with `.mat` appended. However, this assumption only holds for read mode. In write mode (used by `savemat`), `open()` creates the file immediately, bypassing the exception handler entirely.

The function's comment at lines 29-30 states: "If that fails, and `file_like` is a string, and `appendmat` is true, append '.mat' and try again." This describes the current implementation but doesn't align with the public API documentation's promise to always append `.mat` when `appendmat=True`.

SciPy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html

## Proposed Fix

Move the `.mat` appending logic before the initial `open()` call when in write mode:

```diff
--- a/scipy/io/matlab/_mio.py
+++ b/scipy/io/matlab/_mio.py
@@ -35,13 +35,18 @@ def _open_file(file_like, appendmat, mode='rb'):
     if reqs.issubset(dir(file_like)):
         return file_like, False

+    # When writing, append .mat extension before opening if requested
+    if isinstance(file_like, str) and appendmat and not file_like.endswith('.mat'):
+        if 'w' in mode or 'a' in mode or 'x' in mode:
+            file_like += '.mat'
+
     try:
         return open(file_like, mode), True
     except OSError as e:
-        # Probably "not found"
+        # For read mode, try appending .mat if initial open fails
         if isinstance(file_like, str):
             if appendmat and not file_like.endswith('.mat'):
-                file_like += '.mat'
+                file_like += '.mat'
             return open(file_like, mode), True
         else:
             raise OSError(
```