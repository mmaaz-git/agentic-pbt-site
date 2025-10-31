# Bug Report: pandas.io.sas.read_sas Incorrect Format Detection via Substring Matching

**Target**: `pandas.io.sas.read_sas`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_sas` function incorrectly uses substring matching (`".xpt" in fname`) instead of proper file extension checking (`fname.endswith(".xpt")`), causing files with extensions embedded anywhere in their names to be misidentified as SAS files, leading to cryptic parsing errors instead of clear format detection failures.

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
        elif "magic number" in error_msg:
            assert False, f"BUG: File '{filename}' was incorrectly detected as SAS format!"
    finally:
        os.unlink(tmp_path)

# Run the test
test_format_detection_embedded_extension()
```

<details>

<summary>
**Failing input**: `extension='.xpt', suffix='0'` and `extension='.sas7bdat', suffix='0'`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 33, in <module>
  |     test_format_detection_embedded_extension()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 7, in test_format_detection_embedded_extension
  |     extension=st.sampled_from(['.xpt', '.sas7bdat']),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 19, in test_format_detection_embedded_extension
    |     read_sas(tmp_path)
    |     ~~~~~~~~^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sasreader.py", line 164, in read_sas
    |     reader = SAS7BDATReader(
    |         filepath_or_buffer,
    |     ...<3 lines>...
    |         compression=compression,
    |     )
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py", line 229, in __init__
    |     self._get_properties()
    |     ~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py", line 258, in _get_properties
    |     raise ValueError("magic number mismatch (not a SAS file?)")
    | ValueError: magic number mismatch (not a SAS file?)
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 28, in test_format_detection_embedded_extension
    |     assert False, f"BUG: File '{filename}' was incorrectly detected as SAS format!"
    |            ^^^^^
    | AssertionError: BUG: File 'file.sas7bdat0' was incorrectly detected as SAS format!
    | Falsifying example: test_format_detection_embedded_extension(
    |     # The test always failed when commented parts were varied together.
    |     extension='.sas7bdat',  # or any other generated value
    |     suffix='0',  # or any other generated value
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 19, in test_format_detection_embedded_extension
    |     read_sas(tmp_path)
    |     ~~~~~~~~^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sasreader.py", line 154, in read_sas
    |     reader = XportReader(
    |         filepath_or_buffer,
    |     ...<3 lines>...
    |         compression=compression,
    |     )
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas_xport.py", line 280, in __init__
    |     self._read_header()
    |     ~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas_xport.py", line 303, in _read_header
    |     raise ValueError("Header record is not an XPORT file.")
    | ValueError: Header record is not an XPORT file.
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 26, in test_format_detection_embedded_extension
    |     assert False, f"BUG: File '{filename}' was incorrectly detected as SAS format!"
    |            ^^^^^
    | AssertionError: BUG: File 'file.xpt0' was incorrectly detected as SAS format!
    | Falsifying example: test_format_detection_embedded_extension(
    |     # The test always failed when commented parts were varied together.
    |     extension='.xpt',  # or any other generated value
    |     suffix='0',  # or any other generated value
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import tempfile
import os
from pandas.io.sas import read_sas

# Test case: file.xpt0 - should NOT be detected as xport format
print("Testing file.xpt0:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.file.xpt0') as tmp:
    tmp_path = tmp.name

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "Header record" in str(e):
        print("BUG CONFIRMED: File was incorrectly detected as xport format!")
    elif "unable to infer format" in str(e):
        print("CORRECT: File was not detected as a SAS file")
finally:
    os.unlink(tmp_path)

print("\n" + "="*50 + "\n")

# Test case: data.xpt.backup - backup file should NOT be detected as xport
print("Testing data.xpt.backup:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.data.xpt.backup') as tmp:
    tmp_path = tmp.name

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "Header record" in str(e):
        print("BUG CONFIRMED: Backup file was incorrectly detected as xport format!")
    elif "unable to infer format" in str(e):
        print("CORRECT: File was not detected as a SAS file")
finally:
    os.unlink(tmp_path)

print("\n" + "="*50 + "\n")

# Test case: file.sas7bdat0 - should NOT be detected as sas7bdat format
print("Testing file.sas7bdat0:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.file.sas7bdat0') as tmp:
    tmp_path = tmp.name

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "magic number" in str(e):
        print("BUG CONFIRMED: File was incorrectly detected as sas7bdat format!")
    elif "unable to infer format" in str(e):
        print("CORRECT: File was not detected as a SAS file")
finally:
    os.unlink(tmp_path)
```

<details>

<summary>
BUG CONFIRMED: All three test cases incorrectly detect files as SAS formats
</summary>
```
Testing file.xpt0:
Error: Header record is not an XPORT file.
BUG CONFIRMED: File was incorrectly detected as xport format!

==================================================

Testing data.xpt.backup:
Error: Header record is not an XPORT file.
BUG CONFIRMED: Backup file was incorrectly detected as xport format!

==================================================

Testing file.sas7bdat0:
Error: magic number mismatch (not a SAS file?)
BUG CONFIRMED: File was incorrectly detected as sas7bdat format!
```
</details>

## Why This Is A Bug

The pandas documentation for `read_sas` explicitly states that when `format=None`, the format is "inferred from file extension." In standard computing terminology, a file extension is the suffix that follows the last dot in a filename (e.g., `.txt`, `.pdf`, `.xpt`, `.sas7bdat`).

The current implementation at lines 141-144 in `/pandas/io/sas/sasreader.py` violates this contract:

```python
if ".xpt" in fname:         # BUG: Uses substring matching
    format = "xport"
elif ".sas7bdat" in fname:   # BUG: Uses substring matching
    format = "sas7bdat"
```

This causes several problems:
1. **False positives**: Files like `file.xpt0`, `data.xpt.backup`, or `report.sas7bdat.old` are incorrectly identified as SAS files
2. **Confusing errors**: Users see cryptic parser errors ("Header record is not an XPORT file", "magic number mismatch") instead of the clear "unable to infer format" message
3. **Real-world impact**: Backup files (`.xpt.backup`), versioned files (`.xpt.v2`), and temporary files (`.xpt.tmp`) are common patterns that trigger this bug
4. **Documentation violation**: The behavior directly contradicts the documented promise of inferring from "file extension"

## Relevant Context

- **File location**: `/pandas/io/sas/sasreader.py` lines 141-144
- **pandas documentation**: Clearly states format is "inferred from file extension" when `format=None`
- **Workaround**: Users can explicitly specify `format='xport'` or `format='sas7bdat'` to bypass the faulty detection logic
- **Impact scope**: Affects both XPORT (`.xpt`) and SAS7BDAT (`.sas7bdat`) file formats
- **Error messages differ by format**:
  - XPORT: "Header record is not an XPORT file"
  - SAS7BDAT: "magic number mismatch (not a SAS file?)"

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