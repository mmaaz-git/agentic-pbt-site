# Bug Report: scipy.io.matlab.savemat appendmat Parameter Fails to Append .mat Extension

**Target**: `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `appendmat` parameter in `scipy.io.matlab.savemat` fails to append the `.mat` extension to filenames during write operations, violating its documented behavior where it should "append the .mat extension to the end of the given filename, if not already present."

## Property-Based Test

```python
import tempfile
import os
import numpy as np
from scipy.io.matlab import savemat, loadmat
from hypothesis import given, settings, strategies as st

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

if __name__ == "__main__":
    test_appendmat_parameter()
```

<details>

<summary>
**Failing input**: `varname='a', appendmat=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 33, in <module>
    test_appendmat_parameter()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 8, in test_appendmat_parameter
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 30, in test_appendmat_parameter
    assert os.path.exists(expected_file), f"Expected file {expected_file} does not exist"
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
AssertionError: Expected file /tmp/tmpp70zqc4i/test.mat does not exist
Falsifying example: test_appendmat_parameter(
    varname='a',  # or any other generated value
    appendmat=True,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/8/hypo.py:19
```
</details>

## Reproducing the Bug

```python
import tempfile
import os
import numpy as np
from scipy.io.matlab import savemat, loadmat

with tempfile.TemporaryDirectory() as tmpdir:
    # Test case: using savemat with appendmat=True (default)
    filename = os.path.join(tmpdir, 'test')
    arr = np.array([1, 2, 3])
    data = {'a': arr}

    print("=== Testing savemat with appendmat=True ===")
    print(f"Filename provided: {filename}")
    print(f"appendmat parameter: True")

    savemat(filename, data, appendmat=True)

    print("\nFiles created in directory:")
    files = os.listdir(tmpdir)
    print(f"  Actual: {files}")
    print(f"  Expected: ['test.mat']")

    filename_with_mat = os.path.join(tmpdir, 'test.mat')
    filename_without_mat = os.path.join(tmpdir, 'test')

    print(f"\nFile existence check:")
    print(f"  'test.mat' exists: {os.path.exists(filename_with_mat)}")
    print(f"  'test' exists: {os.path.exists(filename_without_mat)}")

    print("\n=== Verifying the bug ===")
    if os.path.exists(filename_without_mat) and not os.path.exists(filename_with_mat):
        print("BUG CONFIRMED: savemat created 'test' instead of 'test.mat' when appendmat=True")
    else:
        print("Bug not reproduced")

    print("\n=== Testing loadmat for comparison ===")
    print("Now testing if loadmat correctly handles appendmat=True...")

    # First save with explicit .mat extension
    filename_explicit = os.path.join(tmpdir, 'test2.mat')
    savemat(filename_explicit, data, appendmat=False)
    print(f"Saved file with explicit .mat: {filename_explicit}")

    # Now try to load without .mat extension but with appendmat=True
    filename_load = os.path.join(tmpdir, 'test2')
    print(f"Loading with filename: {filename_load} and appendmat=True")
    try:
        loaded_data = loadmat(filename_load, appendmat=True)
        print(f"SUCCESS: loadmat correctly found 'test2.mat' when given 'test2' with appendmat=True")
        print(f"Loaded data: {loaded_data['a']}")
    except FileNotFoundError as e:
        print(f"FAILED: loadmat couldn't find the file: {e}")

    print("\n=== Summary ===")
    print("savemat with appendmat=True: FAILS to append .mat extension")
    print("loadmat with appendmat=True: CORRECTLY appends .mat extension")
    print("This inconsistency confirms the bug in savemat.")
```

<details>

<summary>
Output showing the bug
</summary>
```
=== Testing savemat with appendmat=True ===
Filename provided: /tmp/tmpopu8gqez/test
appendmat parameter: True

Files created in directory:
  Actual: ['test']
  Expected: ['test.mat']

File existence check:
  'test.mat' exists: False
  'test' exists: True

=== Verifying the bug ===
BUG CONFIRMED: savemat created 'test' instead of 'test.mat' when appendmat=True

=== Testing loadmat for comparison ===
Now testing if loadmat correctly handles appendmat=True...
Saved file with explicit .mat: /tmp/tmpopu8gqez/test2.mat
Loading with filename: /tmp/tmpopu8gqez/test2 and appendmat=True
SUCCESS: loadmat correctly found 'test2.mat' when given 'test2' with appendmat=True
Loaded data: [[1 2 3]]

=== Summary ===
savemat with appendmat=True: FAILS to append .mat extension
loadmat with appendmat=True: CORRECTLY appends .mat extension
This inconsistency confirms the bug in savemat.
```
</details>

## Why This Is A Bug

The `savemat` function's documentation explicitly states that when `appendmat=True` (which is the default value), it should "append the .mat extension to the end of the given filename, if not already present." This is documented in `/scipy/io/matlab/_mio.py` lines 274-276:

```python
appendmat : bool, optional
    True (the default) to append the .mat extension to the end of the
    given filename, if not already present.
```

However, the actual behavior contradicts this documentation:
1. When calling `savemat('test', data, appendmat=True)`, the function creates a file named 'test' instead of 'test.mat'
2. The default value of `appendmat` is `True`, meaning most users are affected by this bug
3. The same `appendmat` parameter works correctly in `loadmat`, creating an inconsistency within the same module
4. This breaks MATLAB interoperability since MATLAB expects files with the `.mat` extension

## Relevant Context

The root cause lies in the `_open_file` function at `/scipy/io/matlab/_mio.py:25-49`. The function only appends `.mat` when the initial `open()` call fails with an OSError:

- For read mode (`'rb'`): `open()` fails when the file doesn't exist, triggering the exception handler that appends `.mat`
- For write mode (`'wb'`): `open()` succeeds immediately (creating the file), so the exception handler never runs and `.mat` is never appended

This explains why `loadmat` (which opens files for reading) works correctly while `savemat` (which opens files for writing) fails.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html

## Proposed Fix

```diff
--- a/scipy/io/matlab/_mio.py
+++ b/scipy/io/matlab/_mio.py
@@ -35,17 +35,15 @@ def _open_file(file_like, appendmat, mode='rb'):
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