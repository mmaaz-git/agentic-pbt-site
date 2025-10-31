# Bug Report: scipy.io.matlab.savemat appendmat Parameter Documentation Mismatch

**Target**: `scipy.io.matlab.savemat`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `appendmat` parameter in `scipy.io.matlab.savemat` does not behave as documented. The documentation claims it will append the `.mat` extension to filenames, but in reality, `savemat` always uses the exact filename provided regardless of the `appendmat` parameter value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import tempfile
import os
import numpy as np
from scipy.io.matlab import savemat

@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=20))
@settings(max_examples=100)
def test_savemat_appendmat_adds_extension(filename):
    """Contract: appendmat=True should append .mat extension per documentation"""
    data = {"x": np.array([1, 2, 3])}

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)

        savemat(filepath, data, appendmat=True)

        expected_file = filepath + ".mat"
        actual_file = filepath

        assert os.path.exists(expected_file) or not os.path.exists(actual_file), \
            f"Expected {expected_file} but found {actual_file}"
```

**Failing input**: Any filename without `.mat` extension

## Reproducing the Bug

```python
import tempfile
import os
import numpy as np
from scipy.io.matlab import savemat

data = {"x": np.array([1, 2, 3])}

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test")

    savemat(filepath, data, appendmat=True)

    print(f"Files created: {os.listdir(tmpdir)}")

    if os.path.exists(filepath + '.mat'):
        print("Expected: File saved as 'test.mat'")
    elif os.path.exists(filepath):
        print("BUG: File saved as 'test' (no .mat extension)")
```

Output:
```
Files created: ['test']
BUG: File saved as 'test' (no .mat extension)
```

## Why This Is A Bug

The documentation for `savemat` states:

```
appendmat : bool, optional
    True (the default) to append the .mat extension to the end of the
    given filename, if not already present.
```

However, the actual behavior is that `savemat` ignores the `appendmat` parameter entirely and always saves to the exact filename provided. This creates a mismatch between the API contract (documentation) and the implementation.

Note that `loadmat` correctly implements the `appendmat` behavior - it does add `.mat` when loading files. The inconsistency is specific to `savemat`.

## Fix

The fix depends on the intended behavior:

**Option 1: Fix the implementation** (make `savemat` honor `appendmat`):

```diff
--- a/scipy/io/matlab/_mio.py
+++ b/scipy/io/matlab/_mio.py
@@ -XXX,X +XXX,X @@ def savemat(file_name, mdict, appendmat=True, format='5',
+    if appendmat and not file_name.endswith('.mat'):
+        file_name = file_name + '.mat'
+
     file_stream, file_opened = _open_file_write(file_name)
```

**Option 2: Fix the documentation** (remove claim that `savemat` appends `.mat`):

```diff
--- a/scipy/io/matlab/_mio.py
+++ b/scipy/io/matlab/_mio.py
@@ -XXX,X +XXX,X @@ def savemat(file_name, mdict, appendmat=True, format='5',
 appendmat : bool, optional
-    True (the default) to append the .mat extension to the end of the
-    given filename, if not already present.
+    Not used for saving. This parameter exists for API consistency with
+    loadmat, where it controls whether to append .mat when loading.
```

The recommended fix is **Option 2** (documentation fix), as changing the implementation could break existing code that relies on the current behavior where `savemat` uses the exact filename provided.