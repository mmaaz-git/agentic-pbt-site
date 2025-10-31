# Bug Report: pandas.io.sas.read_sas Format Detection Uses Substring Matching

**Target**: `pandas.io.sas.read_sas`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_sas` function incorrectly detects file format by searching for extension substrings anywhere in the filename, rather than checking the actual file extension. This causes files like `"data.xpt.backup"` to be treated as xport files when they should fail format detection.

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
```

**Failing input**: `base="data", extension=".backup"` produces filename `"data.xpt.backup"`

## Reproducing the Bug

```python
from pandas.io.sas import read_sas

filename = "data.xpt.backup"
try:
    read_sas(filename)
except FileNotFoundError:
    print(f"BUG: '{filename}' was detected as xport format")
    print("Expected: ValueError('unable to infer format...')")
except ValueError as e:
    print(f"Correct: {e}")
```

**Output:**
```
BUG: 'data.xpt.backup' was detected as xport format
Expected: ValueError('unable to infer format...')
```

## Why This Is A Bug

Files with extensions like `.xpt.backup`, `.sas7bdat.old`, or even filenames containing these substrings in the middle (e.g., `"encrypted_data.txt"` won't match, but `"text.xpt_archive.dat"` would) are incorrectly identified as SAS files. This violates the documented behavior which states format is "inferred from file extension."

The function should check if the filename **ends with** the extension, not if it **contains** the extension substring anywhere.

## Fix

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