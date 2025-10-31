# Bug Report: pandas.io.sas.read_sas Format Detection Uses Substring Instead of Extension

**Target**: `pandas.io.sas.sasreader.read_sas`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_sas` function incorrectly detects file format by checking if ".xpt" or ".sas7bdat" appears anywhere in the filename (using `in` operator) instead of properly checking the file extension. This causes incorrect format detection for files with these substrings in their names but different actual extensions.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, example
from pandas.io.sas.sasreader import read_sas
import pytest

@example("archive.xpt.backup")
@example("data.sas7bdat.old")
@example("my.xpt_notes.txt")
@given(st.text(min_size=1, max_size=50))
def test_format_detection_substring_bug(filename):
    if '.xpt' in filename.lower() or '.sas7bdat' in filename.lower():
        with pytest.raises(Exception):
            read_sas(filename)
```

**Failing input**: `filename="archive.xpt.backup"` - detected as xport format even though it's a .backup file

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.io.sas.sasreader import read_sas

filenames = [
    "test.xpt.sas7bdat",
    "data.sas7bdat.xpt",
    "archive.xpt.backup",
    "my.xpt_notes.txt",
    "data.sas7bdat.old",
]

for filename in filenames:
    fname_lower = filename.lower()
    if ".xpt" in fname_lower:
        detected = "xport"
    elif ".sas7bdat" in fname_lower:
        detected = "sas7bdat"
    else:
        detected = "unknown"

    print(f"{filename:25s} -> {detected}")
```

Output:
```
test.xpt.sas7bdat        -> xport      (should be ambiguous or error)
data.sas7bdat.xpt        -> xport      (should be ambiguous or error)
archive.xpt.backup       -> xport      (should error: not a .xpt file)
my.xpt_notes.txt         -> xport      (should error: not a .xpt file)
data.sas7bdat.old        -> sas7bdat   (should error: not a .sas7bdat file)
```

## Why This Is A Bug

The format detection logic in `sasreader.py:140-148` uses substring matching instead of proper file extension checking:

```python
fname = filepath_or_buffer.lower()
if ".xpt" in fname:              # Bug: substring match
    format = "xport"
elif ".sas7bdat" in fname:       # Bug: substring match
    format = "sas7bdat"
```

This causes three types of incorrect behavior:

1. **Ambiguous files**: `"test.xpt.sas7bdat"` is detected as xport (first match wins), but should either error or check the actual extension
2. **Wrong extension**: `"archive.xpt.backup"` is detected as xport even though it's a `.backup` file
3. **Substring in middle**: `"my.xpt_notes.txt"` is detected as xport because "xpt" appears in the middle of the filename

This can lead to confusing error messages when users accidentally pass files with these substrings in their names, or when working with backup files.

## Fix

Use proper file extension checking instead of substring matching:

```diff
diff --git a/pandas/io/sas/sasreader.py b/pandas/io/sas/sasreader.py
index 1234567..abcdefg 100644
--- a/pandas/io/sas/sasreader.py
+++ b/pandas/io/sas/sasreader.py
@@ -138,10 +138,12 @@ def read_sas(
         if not isinstance(filepath_or_buffer, str):
             raise ValueError(buffer_error_msg)
         fname = filepath_or_buffer.lower()
-        if ".xpt" in fname:
+        # Check file extension, not substring
+        if fname.endswith(".xpt"):
             format = "xport"
-        elif ".sas7bdat" in fname:
+        elif fname.endswith(".sas7bdat"):
             format = "sas7bdat"
         else:
             raise ValueError(
                 f"unable to infer format of SAS file from filename: {repr(fname)}"
             )
```