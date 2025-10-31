# Bug Report: pandas.io.sas.read_sas Incorrectly Uses Substring Matching for Format Detection

**Target**: `pandas.io.sas.read_sas`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_sas` function incorrectly detects file format by checking if extension strings like ".xpt" or ".sas7bdat" appear anywhere in the filename as substrings, rather than checking if the filename actually ends with these extensions. This causes files with compound extensions like "data.xpt.backup" to be incorrectly identified as SAS files.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import string
from pandas.io.sas import read_sas


@given(
    base=st.text(
        alphabet=string.ascii_letters + string.digits,
        min_size=1, max_size=20
    ),
    extension=st.sampled_from([".txt", ".csv", ".dat", ".backup", ".old", ".tmp"])
)
@settings(max_examples=500)
def test_format_detection_should_check_extension_not_substring(base, extension):
    filename = f"{base}.xpt{extension}"

    try:
        read_sas(filename)
        format_detected = True
    except FileNotFoundError:
        format_detected = True
    except ValueError as e:
        if "unable to infer format" in str(e):
            format_detected = False
        else:
            raise

    assert not format_detected, \
        f"File '{filename}' should not be detected as xport format (extension is {extension}, not .xpt)"

if __name__ == "__main__":
    test_format_detection_should_check_extension_not_substring()
```

<details>

<summary>
**Failing input**: `base='0', extension='.txt'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 32, in <module>
    test_format_detection_should_check_extension_not_substring()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 7, in test_format_detection_should_check_extension_not_substring
    base=st.text(
               ^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 28, in test_format_detection_should_check_extension_not_substring
    assert not format_detected, \
           ^^^^^^^^^^^^^^^^^^^
AssertionError: File '0.xpt.txt' should not be detected as xport format (extension is .txt, not .xpt)
Falsifying example: test_format_detection_should_check_extension_not_substring(
    # The test always failed when commented parts were varied together.
    base='0',  # or any other generated value
    extension='.txt',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.sas import read_sas

filename = "data.xpt.backup"
try:
    read_sas(filename)
    print(f"BUG: '{filename}' was detected as xport format (FileNotFoundError raised)")
    print("Expected: ValueError('unable to infer format...')")
except FileNotFoundError as e:
    print(f"BUG: '{filename}' was detected as xport format (FileNotFoundError raised)")
    print(f"Error: {e}")
    print("Expected: ValueError('unable to infer format...')")
except ValueError as e:
    if "unable to infer format" in str(e):
        print(f"CORRECT: '{filename}' was not detected as a SAS file")
        print(f"ValueError raised: {e}")
    else:
        print(f"Different ValueError: {e}")
```

<details>

<summary>
FileNotFoundError raised for 'data.xpt.backup'
</summary>
```
BUG: 'data.xpt.backup' was detected as xport format (FileNotFoundError raised)
Error: [Errno 2] No such file or directory: 'data.xpt.backup'
Expected: ValueError('unable to infer format...')
```
</details>

## Why This Is A Bug

This behavior violates the documented contract of the `read_sas` function. The documentation at line 111-112 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/sas/sasreader.py` explicitly states:

> "If None, file format is inferred from file extension."

A file extension is the suffix after the last dot in a filename. The term "file extension" has a universally accepted meaning in computing - it refers to the final part of the filename after the last period. However, the current implementation at lines 141-144 uses substring matching:

```python
if ".xpt" in fname:
    format = "xport"
elif ".sas7bdat" in fname:
    format = "sas7bdat"
```

This causes several problems:
1. **Backup files** with names like "data.xpt.backup" or "analysis.sas7bdat.old" are incorrectly identified as SAS files
2. **Temporary files** like "data.xpt.tmp" or "report.sas7bdat.temp" are misidentified
3. **Archived files** with compound extensions are incorrectly processed
4. The error message is misleading - users get `FileNotFoundError` instead of the more informative `ValueError: unable to infer format`

The bug affects common real-world use cases where users work with backup files, archived files, or temporary files that contain ".xpt" or ".sas7bdat" as part of their filename but have different actual extensions.

## Relevant Context

The pandas `read_sas` function is designed to read SAS data files in two formats:
- **XPORT** format (files ending with `.xpt`)
- **SAS7BDAT** format (files ending with `.sas7bdat`)

The function provides a `format` parameter that can be explicitly set to bypass automatic detection. When `format=None` (the default), the function attempts to infer the format from the filename. This inference is documented to be based on "file extension" but is implemented using substring matching.

Relevant documentation:
- Official pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_sas.html
- Source code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/sas/sasreader.py`

Workaround for users encountering this bug:
- Explicitly specify the `format` parameter when calling `read_sas` for files with compound extensions
- Rename files to have the correct extension before processing

## Proposed Fix

```diff
--- a/pandas/io/sas/sasreader.py
+++ b/pandas/io/sas/sasreader.py
@@ -138,9 +138,9 @@ def read_sas(
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