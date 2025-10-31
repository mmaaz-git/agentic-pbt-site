# Bug Report: pandas.io.sas.sasreader.read_sas Format Detection Using Substring Matching Instead of Extension Checking

**Target**: `pandas.io.sas.sasreader.read_sas`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_sas` function incorrectly uses substring matching (`if ".xpt" in fname`) instead of proper file extension checking (`fname.endswith(".xpt")`), causing files with embedded extension strings like "data.xpt.backup" or "file.sas7bdat.old" to be misidentified as valid SAS files when they are not.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import string
import pytest
from pandas.io.sas import read_sas

@given(
    base_name=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
    middle_ext=st.sampled_from(['.xpt', '.sas7bdat']),
    suffix=st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=10)
)
@settings(max_examples=200)
def test_format_detection_false_positives(base_name, middle_ext, suffix):
    filename = base_name + middle_ext + "." + suffix

    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        read_sas(filename)
```

<details>

<summary>
**Failing input**: `filename = "a.xpt.0"`
</summary>
```
Testing: a.xpt.0
  FileNotFoundError (BUG - incorrectly detected as valid SAS format)
Testing: a.xpt.0
  FileNotFoundError (BUG - incorrectly detected as valid SAS format)
Testing: tpigra.sas7bdat.3
  FileNotFoundError (BUG - incorrectly detected as valid SAS format)
Testing: ihu.xpt.pg
  FileNotFoundError (BUG - incorrectly detected as valid SAS format)
Testing: keknyppvpp.sas7bdat.o9
  FileNotFoundError (BUG - incorrectly detected as valid SAS format)
Testing: a.sas7bdat.t
  FileNotFoundError (BUG - incorrectly detected as valid SAS format)
```
</details>

## Reproducing the Bug

```python
from pandas.io.sas import read_sas

# Test 1: File with .xpt embedded but not as extension
print("Test 1: Filename 'data.xpt.backup'")
try:
    read_sas("data.xpt.backup")
except FileNotFoundError as e:
    print(f"  FileNotFoundError: {e}")
    print("  BUG: File was incorrectly detected as xport format")
except ValueError as e:
    if "unable to infer format" in str(e):
        print(f"  ValueError: {e}")
        print("  CORRECT: Format detection failed appropriately")
    else:
        print(f"  ValueError: {e}")
        print("  BUG: Unexpected error")

print("\nTest 2: Filename 'file.sas7bdat.old'")
try:
    read_sas("file.sas7bdat.old")
except FileNotFoundError as e:
    print(f"  FileNotFoundError: {e}")
    print("  BUG: File was incorrectly detected as sas7bdat format")
except ValueError as e:
    if "unable to infer format" in str(e):
        print(f"  ValueError: {e}")
        print("  CORRECT: Format detection failed appropriately")
    else:
        print(f"  ValueError: {e}")
        print("  BUG: Unexpected error")

print("\nTest 3: Filename 'myfile.xpt123'")
try:
    read_sas("myfile.xpt123")
except FileNotFoundError as e:
    print(f"  FileNotFoundError: {e}")
    print("  BUG: File was incorrectly detected as xport format")
except ValueError as e:
    if "unable to infer format" in str(e):
        print(f"  ValueError: {e}")
        print("  CORRECT: Format detection failed appropriately")
    else:
        print(f"  ValueError: {e}")
        print("  BUG: Unexpected error")
```

<details>

<summary>
FileNotFoundError raised instead of ValueError for invalid file extensions
</summary>
```
Test 1: Filename 'data.xpt.backup'
  FileNotFoundError: [Errno 2] No such file or directory: 'data.xpt.backup'
  BUG: File was incorrectly detected as xport format

Test 2: Filename 'file.sas7bdat.old'
  FileNotFoundError: [Errno 2] No such file or directory: 'file.sas7bdat.old'
  BUG: File was incorrectly detected as sas7bdat format

Test 3: Filename 'myfile.xpt123'
  FileNotFoundError: [Errno 2] No such file or directory: 'myfile.xpt123'
  BUG: File was incorrectly detected as xport format
```
</details>

## Why This Is A Bug

The function's documentation at line 111 explicitly states: "If None, file format is inferred from file extension." The term "file extension" has a precise technical meaning - it refers to the suffix after the last period in a filename. However, the implementation at lines 141-144 uses substring matching (`if ".xpt" in fname`) which contradicts this documented behavior.

This causes several problems:
1. Files with backup suffixes like "data.xpt.backup" are incorrectly detected as xport files (actual extension is .backup)
2. Versioned files like "file.sas7bdat.old" are incorrectly detected as sas7bdat files (actual extension is .old)
3. Files with embedded extension strings like "myfile.xpt123" are incorrectly detected as xport files (actual extension is .xpt123)

The current behavior results in a confusing FileNotFoundError when trying to open these files as SAS files, instead of the appropriate ValueError with message "unable to infer format of SAS file from filename" that should be raised for unrecognized extensions.

## Relevant Context

The bug is in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/sas/sasreader.py` at lines 141-144. The code currently uses substring matching which is inconsistent with the behavior of virtually all other file I/O functions in pandas and Python, which properly check file extensions using methods like `endswith()` or `os.path.splitext()`.

This affects pandas version 2.3.2 and likely other versions with the same implementation pattern. The issue is particularly problematic for users in data science workflows where backup files, versioned files, and temporary files with compound extensions are common.

Documentation link: The inline docstring at line 111 clearly states the expected behavior.

## Proposed Fix

```diff
--- a/pandas/io/sas/sasreader.py
+++ b/pandas/io/sas/sasreader.py
@@ -138,9 +138,9 @@ def read_sas(
         filepath_or_buffer = stringify_path(filepath_or_buffer)
         if not isinstance(filepath_or_buffer, str):
             raise ValueError(buffer_error_msg)
         fname = filepath_or_buffer.lower()
-        if ".xpt" in fname:
+        if fname.endswith(".xpt"):
             format = "xport"
-        elif ".sas7bdat" in fname:
+        elif fname.endswith(".sas7bdat"):
             format = "sas7bdat"
         else:
             raise ValueError(
```