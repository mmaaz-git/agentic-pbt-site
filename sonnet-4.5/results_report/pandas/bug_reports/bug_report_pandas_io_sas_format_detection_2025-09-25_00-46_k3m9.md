# Bug Report: pandas.io.sas Format Detection Substring Matching

**Target**: `pandas.io.sas.sasreader.read_sas`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_sas` function uses substring matching (`if ".xpt" in fname`) instead of proper extension checking (`fname.endswith(".xpt")`), causing incorrect format detection for files with embedded extension strings like `"data.xpt.backup"` or `"file.sas7bdat.old"`.

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

**Failing input**: `filename = "test.xpt.backup"`

## Reproducing the Bug

```python
from pandas.io.sas import read_sas

try:
    read_sas("data.xpt.backup")
except FileNotFoundError:
    print("BUG: File was incorrectly detected as xport format")
except ValueError as e:
    if "unable to infer format" in str(e):
        print("Correct: Format detection failed appropriately")
    else:
        print(f"BUG: Unexpected error: {e}")
```

## Why This Is A Bug

The function uses substring matching which incorrectly detects files like:
- `"file.xpt.backup"` → Detected as xport (wrong - should fail format detection)
- `"data.sas7bdat.old"` → Detected as sas7bdat (wrong)
- `"myfile.xpt123"` → Detected as xport (wrong)

Users who have backup files or version-controlled data files will experience incorrect behavior.

## Fix

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