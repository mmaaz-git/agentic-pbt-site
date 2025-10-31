# Bug Report: pandas.io.sas Format Detection Using Substring Match

**Target**: `pandas.io.sas.read_sas`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_sas` function uses substring matching (`in` operator) instead of proper extension checking to infer file format, causing files with embedded extensions in their names (like `file.xpt0` or `data.sas7bdat.backup`) to be incorrectly identified as SAS files.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import tempfile
import os
from pandas.io.sas import read_sas

@given(
    extension=st.sampled_from(['.xpt', '.sas7bdat']),
    suffix=st.text(st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
                   min_size=1, max_size=10)
)
@settings(max_examples=100)
def test_format_detection_embedded_extension(extension, suffix):
    filename = f"file{extension}{suffix}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{filename}') as tmp:
        tmp_path = tmp.name

    try:
        read_sas(tmp_path)
        assert False, f"Should have failed for {tmp_path}"
    except ValueError as e:
        error_msg = str(e).lower()
        if "unable to infer format" in error_msg:
            pass
        elif "header record" in error_msg:
            assert False, f"BUG: File '{filename}' was incorrectly detected as SAS format!"
    finally:
        os.unlink(tmp_path)
```

**Failing input**: `extension='.xpt', suffix='0'` (creates filename `file.xpt0`)

## Reproducing the Bug

```python
import tempfile
import os
from pandas.io.sas import read_sas

with tempfile.NamedTemporaryFile(delete=False, suffix='.file.xpt0') as tmp:
    tmp_path = tmp.name

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "Header record" in str(e):
        print("BUG: File was incorrectly detected as xport format")
finally:
    os.unlink(tmp_path)
```

## Why This Is A Bug

The code at lines 141-144 in `sasreader.py` uses substring matching:

```python
if ".xpt" in fname:
    format = "xport"
elif ".sas7bdat" in fname:
    format = "sas7bdat"
```

This violates the documented behavior that format is "inferred from file extension". A file extension should be at the end of the filename, not embedded anywhere in it. Files like `data.xpt.backup`, `file.xpt0`, or `myxptfile.txt` should not be detected as SAS files.

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