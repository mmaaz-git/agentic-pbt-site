# Bug Report: pandas.io.sas.read_sas Incorrectly Uses Substring Matching Instead of File Extension Checking

**Target**: `pandas.io.sas.sasreader.read_sas`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_sas` function incorrectly detects SAS file formats by checking if ".xpt" or ".sas7bdat" appears anywhere in the filename using the `in` operator, rather than properly checking if the filename ends with these extensions. This causes files like "archive.xpt.backup" or "data.sas7bdat.old" to be incorrectly accepted as valid SAS files.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, example, assume
from pandas.io.sas.sasreader import read_sas
import pytest

@example("archive.xpt.backup")
@example("data.sas7bdat.old")
@example("my.xpt_notes.txt")
@given(st.text(min_size=1, max_size=50))
def test_format_detection_substring_bug(filename):
    # The bug: read_sas incorrectly accepts files with .xpt or .sas7bdat
    # anywhere in the filename, not just as the file extension
    fname_lower = filename.lower()

    # Files that contain the substrings but don't end with them should be rejected
    has_xpt_substring = '.xpt' in fname_lower
    has_sas7bdat_substring = '.sas7bdat' in fname_lower
    ends_with_xpt = fname_lower.endswith('.xpt')
    ends_with_sas7bdat = fname_lower.endswith('.sas7bdat')

    if (has_xpt_substring and not ends_with_xpt) or \
       (has_sas7bdat_substring and not ends_with_sas7bdat):
        # This SHOULD raise a ValueError for "unable to infer format"
        # But due to the bug, it incorrectly accepts these files
        try:
            read_sas(filename)
            # If we get here without exception, the bug is present
            # (it would be FileNotFoundError since file doesn't exist)
            assert False, f"Bug detected: {filename} was incorrectly accepted as a valid SAS file"
        except ValueError as e:
            if "unable to infer format" in str(e):
                # This is the correct behavior
                pass
            else:
                raise
        except FileNotFoundError:
            # This means the format was detected (incorrectly) and it tried to open
            assert False, f"Bug detected: {filename} was incorrectly accepted as a valid SAS file"

if __name__ == "__main__":
    test_format_detection_substring_bug()
```

<details>

<summary>
**Failing input**: `filename='archive.xpt.backup'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/61
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_format_detection_substring_bug FAILED                      [100%]

=================================== FAILURES ===================================
_____________________ test_format_detection_substring_bug ______________________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 9, in test_format_detection_substring_bug
  |     @example("data.sas7bdat.old")
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 28, in test_format_detection_substring_bug
    |     read_sas(filename)
    |     ~~~~~~~~^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sasreader.py", line 154, in read_sas
    |     reader = XportReader(
    |         filepath_or_buffer,
    |     ...<3 lines>...
    |         compression=compression,
    |     )
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas_xport.py", line 270, in __init__
    |     self.handles = get_handle(
    |                    ~~~~~~~~~~^
    |         filepath_or_buffer,
    |         ^^^^^^^^^^^^^^^^^^^
    |     ...<3 lines>...
    |         compression=compression,
    |         ^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py", line 882, in get_handle
    |     handle = open(handle, ioargs.mode)
    | FileNotFoundError: [Errno 2] No such file or directory: 'archive.xpt.backup'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 40, in test_format_detection_substring_bug
    |     assert False, f"Bug detected: {filename} was incorrectly accepted as a valid SAS file"
    | AssertionError: Bug detected: archive.xpt.backup was incorrectly accepted as a valid SAS file
    | assert False
    | Falsifying explicit example: test_format_detection_substring_bug(
    |     filename='archive.xpt.backup',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 28, in test_format_detection_substring_bug
    |     read_sas(filename)
    |     ~~~~~~~~^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sasreader.py", line 164, in read_sas
    |     reader = SAS7BDATReader(
    |         filepath_or_buffer,
    |     ...<3 lines>...
    |         compression=compression,
    |     )
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py", line 209, in __init__
    |     self.handles = get_handle(
    |                    ~~~~~~~~~~^
    |         path_or_buf, "rb", is_text=False, compression=compression
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py", line 882, in get_handle
    |     handle = open(handle, ioargs.mode)
    | FileNotFoundError: [Errno 2] No such file or directory: 'data.sas7bdat.old'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 40, in test_format_detection_substring_bug
    |     assert False, f"Bug detected: {filename} was incorrectly accepted as a valid SAS file"
    | AssertionError: Bug detected: data.sas7bdat.old was incorrectly accepted as a valid SAS file
    | assert False
    | Falsifying explicit example: test_format_detection_substring_bug(
    |     filename='data.sas7bdat.old',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 28, in test_format_detection_substring_bug
    |     read_sas(filename)
    |     ~~~~~~~~^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sasreader.py", line 154, in read_sas
    |     reader = XportReader(
    |         filepath_or_buffer,
    |     ...<3 lines>...
    |         compression=compression,
    |     )
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas_xport.py", line 270, in __init__
    |     self.handles = get_handle(
    |                    ~~~~~~~~~~^
    |         filepath_or_buffer,
    |         ^^^^^^^^^^^^^^^^^^^
    |     ...<3 lines>...
    |         compression=compression,
    |         ^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py", line 882, in get_handle
    |     handle = open(handle, ioargs.mode)
    | FileNotFoundError: [Errno 2] No such file or directory: 'my.xpt_notes.txt'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 40, in test_format_detection_substring_bug
    |     assert False, f"Bug detected: {filename} was incorrectly accepted as a valid SAS file"
    | AssertionError: Bug detected: my.xpt_notes.txt was incorrectly accepted as a valid SAS file
    | assert False
    | Falsifying explicit example: test_format_detection_substring_bug(
    |     filename='my.xpt_notes.txt',
    | )
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_format_detection_substring_bug - ExceptionGroup: Hypothe...
============================== 1 failed in 0.35s ===============================
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.io.sas.sasreader import read_sas

print("Testing SAS format detection bug with substring matching\n")
print("=" * 60)

test_cases = [
    ("archive.xpt.backup", "Should reject as .backup file"),
    ("data.sas7bdat.old", "Should reject as .old file"),
    ("my.xpt_notes.txt", "Should reject as .txt file"),
    ("test.xpt.sas7bdat", "Ambiguous: contains both substrings"),
    ("data.sas7bdat.xpt", "Ambiguous: contains both substrings"),
    (".xpt", "Edge case: no filename, just extension"),
    (".sas7bdat", "Edge case: no filename, just extension"),
    ("test.txt", "Should reject: no SAS substring"),
]

for filename, description in test_cases:
    print(f"\nTest: {filename}")
    print(f"Description: {description}")
    print("-" * 40)

    try:
        # Try to read the file (will fail due to file not existing, but we'll see format detection)
        reader = read_sas(filename)
    except ValueError as e:
        if "unable to infer format" in str(e):
            print(f"Result: ValueError - unable to infer format")
            print(f"Error: {e}")
        else:
            print(f"Result: ValueError (other)")
            print(f"Error: {e}")
    except FileNotFoundError as e:
        # This means format was detected and it tried to open the file
        fname_lower = filename.lower()
        if ".xpt" in fname_lower:
            detected = "xport"
        elif ".sas7bdat" in fname_lower:
            detected = "sas7bdat"
        else:
            detected = "unknown"
        print(f"Result: Format detected as '{detected}' (FileNotFoundError when opening)")
        print(f"Error: {e}")
    except Exception as e:
        print(f"Result: Unexpected error")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")

print("\n" + "=" * 60)
print("\nSUMMARY: The bug allows files with '.xpt' or '.sas7bdat' anywhere")
print("in their filename to be incorrectly accepted as valid SAS files,")
print("regardless of their actual file extension.")
```

<details>

<summary>
FileNotFoundError confirms format was incorrectly detected for non-SAS files
</summary>
```
Testing SAS format detection bug with substring matching

============================================================

Test: archive.xpt.backup
Description: Should reject as .backup file
----------------------------------------
Result: Format detected as 'xport' (FileNotFoundError when opening)
Error: [Errno 2] No such file or directory: 'archive.xpt.backup'

Test: data.sas7bdat.old
Description: Should reject as .old file
----------------------------------------
Result: Format detected as 'sas7bdat' (FileNotFoundError when opening)
Error: [Errno 2] No such file or directory: 'data.sas7bdat.old'

Test: my.xpt_notes.txt
Description: Should reject as .txt file
----------------------------------------
Result: Format detected as 'xport' (FileNotFoundError when opening)
Error: [Errno 2] No such file or directory: 'my.xpt_notes.txt'

Test: test.xpt.sas7bdat
Description: Ambiguous: contains both substrings
----------------------------------------
Result: Format detected as 'xport' (FileNotFoundError when opening)
Error: [Errno 2] No such file or directory: 'test.xpt.sas7bdat'

Test: data.sas7bdat.xpt
Description: Ambiguous: contains both substrings
----------------------------------------
Result: Format detected as 'xport' (FileNotFoundError when opening)
Error: [Errno 2] No such file or directory: 'data.sas7bdat.xpt'

Test: .xpt
Description: Edge case: no filename, just extension
----------------------------------------
Result: Format detected as 'xport' (FileNotFoundError when opening)
Error: [Errno 2] No such file or directory: '.xpt'

Test: .sas7bdat
Description: Edge case: no filename, just extension
----------------------------------------
Result: Format detected as 'sas7bdat' (FileNotFoundError when opening)
Error: [Errno 2] No such file or directory: '.sas7bdat'

Test: test.txt
Description: Should reject: no SAS substring
----------------------------------------
Result: ValueError - unable to infer format
Error: unable to infer format of SAS file from filename: 'test.txt'

============================================================

SUMMARY: The bug allows files with '.xpt' or '.sas7bdat' anywhere
in their filename to be incorrectly accepted as valid SAS files,
regardless of their actual file extension.
```
</details>

## Why This Is A Bug

This violates the documented behavior and user expectations in several ways:

1. **Documentation Contradiction**: The docstring at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/sas/sasreader.py:111` explicitly states: "If None, file format is inferred from file extension." A file extension is universally understood as the suffix after the last dot in a filename, not any substring within the filename.

2. **Incorrect Implementation**: The code at lines 140-148 uses `if ".xpt" in fname:` and `elif ".sas7bdat" in fname:` which performs substring matching instead of extension checking. This means:
   - `"archive.xpt.backup"` is detected as xport format even though its extension is `.backup`
   - `"data.sas7bdat.old"` is detected as sas7bdat format even though its extension is `.old`
   - `"my.xpt_notes.txt"` is detected as xport format even though its extension is `.txt`

3. **Real-World Impact**: This affects common file naming patterns:
   - Backup files: `data.xpt.backup`, `data.xpt.old`, `data.xpt.2024`
   - Version control: `data.sas7bdat.orig`, `data.sas7bdat.new`
   - Descriptive names: `my.xpt_analysis.csv`, `contains.sas7bdat.info.txt`

4. **Ambiguous Behavior**: Files containing both substrings like `"test.xpt.sas7bdat"` are handled inconsistently - the first match wins (xport), even though the actual extension is `.sas7bdat`.

## Relevant Context

The bug exists in the pandas library's SAS file reader module. The `read_sas` function is designed to automatically detect whether a file is in XPORT or SAS7BDAT format based on its file extension when the `format` parameter is not explicitly provided.

**Source Code Location**: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/sas/sasreader.py` lines 140-148

**Documentation Link**: The pandas.read_sas function is documented at https://pandas.pydata.org/docs/reference/api/pandas.read_sas.html

**Workaround**: Users can explicitly specify the format parameter to bypass auto-detection:
```python
read_sas("archive.xpt.backup", format="xport")  # if it's actually an xport file
```

However, this requires users to know the actual format, defeating the purpose of automatic detection.

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