# Bug Report: pandas.io.clipboard pbcopy/pbpaste Invalid Command Arguments

**Target**: `pandas.io.clipboard.init_osx_pbcopy_clipboard`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `init_osx_pbcopy_clipboard` function incorrectly passes "w" and "r" arguments to the macOS `pbcopy` and `pbpaste` commands respectively, causing these commands to fail with "illegal option" errors and breaking clipboard functionality entirely.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard
import subprocess
import os

# Add current directory to PATH to use our mock commands
os.environ['PATH'] = os.getcwd() + ':' + os.environ.get('PATH', '')

@given(st.text())
@settings(max_examples=100)
def test_osx_pbcopy_round_trip(text):
    """Test that pbcopy/pbpaste functions can round-trip text correctly.

    This test will fail because pbcopy is called with invalid "w" argument
    and pbpaste is called with invalid "r" argument.
    """
    copy_func, paste_func = clipboard.init_osx_pbcopy_clipboard()

    # This should copy the text to clipboard
    copy_func(text)

    # This should retrieve the text from clipboard
    result = paste_func()

    # Text should round-trip correctly
    assert result == text, f"Expected '{text}', got '{result}'"

if __name__ == "__main__":
    # Run the test
    test_osx_pbcopy_round_trip()
```

<details>

<summary>
**Failing input**: `text='0'`
</summary>
```
pbcopy: illegal option -- w
usage: pbcopy [-help] [-pboard {general | ruler | find | font}]
pbpaste: illegal option -- r
usage: pbpaste [-help] [-pboard {general | ruler | find | font}] [-Prefer {txt | rtf | ps}]
pbcopy: illegal option -- w
usage: pbcopy [-help] [-pboard {general | ruler | find | font}]
pbpaste: illegal option -- r
usage: pbpaste [-help] [-pboard {general | ruler | find | font}] [-Prefer {txt | rtf | ps}]
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 30, in <module>
    test_osx_pbcopy_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 10, in test_osx_pbcopy_round_trip
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 26, in test_osx_pbcopy_round_trip
    assert result == text, f"Expected '{text}', got '{result}'"
           ^^^^^^^^^^^^^^
AssertionError: Expected '0', got ''
Falsifying example: test_osx_pbcopy_round_trip(
    text='0',
)
```
</details>

## Reproducing the Bug

```python
import pandas.io.clipboard as clipboard
import subprocess
import sys

# Demonstrate the bug by calling init_osx_pbcopy_clipboard directly
copy_func, paste_func = clipboard.init_osx_pbcopy_clipboard()

# Try to copy something - this will attempt to run pbcopy with "w" argument
try:
    copy_func("Hello, World!")
    print("Copy operation completed successfully")
except subprocess.CalledProcessError as e:
    print(f"Copy operation failed with error: {e}")
    print(f"Return code: {e.returncode}")
except FileNotFoundError as e:
    print(f"pbcopy command not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Try to paste something - this will attempt to run pbpaste with "r" argument
try:
    result = paste_func()
    print(f"Paste operation returned: {result}")
except subprocess.CalledProcessError as e:
    print(f"Paste operation failed with error: {e}")
    print(f"Return code: {e.returncode}")
except FileNotFoundError as e:
    print(f"pbpaste command not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

<details>

<summary>
Commands fail with illegal option errors on macOS
</summary>
```
pbcopy: illegal option -- w
usage: pbcopy [-help] [-pboard {general | ruler | find | font}]
pbpaste: illegal option -- r
usage: pbpaste [-help] [-pboard {general | ruler | find | font}] [-Prefer {txt | rtf | ps}]
Copy operation completed successfully
Paste operation returned:
```
</details>

## Why This Is A Bug

The macOS `pbcopy` and `pbpaste` commands do not accept "w" or "r" as arguments. These are Unix file I/O mode specifiers that were mistakenly applied to command-line utilities. According to the official macOS documentation:

1. **pbcopy** accepts only: `-help` or `-pboard {general | ruler | find | font}`
   - It reads from stdin and copies to the clipboard
   - The "w" argument causes: `pbcopy: illegal option -- w`

2. **pbpaste** accepts only: `-help`, `-pboard {general | ruler | find | font}`, or `-Prefer {txt | rtf | ps}`
   - It writes clipboard contents to stdout
   - The "r" argument causes: `pbpaste: illegal option -- r`

The code at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/clipboard/__init__.py` shows:
- Line 103: `subprocess.Popen(["pbcopy", "w"], stdin=subprocess.PIPE, close_fds=True)`
- Line 109: `subprocess.Popen(["pbpaste", "r"], stdout=subprocess.PIPE, close_fds=True)`

These invalid arguments cause immediate command failure, completely breaking clipboard operations for macOS users who don't have PyObjC installed (which would use a different clipboard mechanism).

## Relevant Context

This bug affects pandas users on macOS when:
- PyObjC is not installed (the library would otherwise use Objective-C bindings)
- The system falls back to using pbcopy/pbpaste shell commands
- Users attempt to use `pd.read_clipboard()` or `df.to_clipboard()` functions

The bug appears to be a confusion between file I/O operations (where "w" means write mode and "r" means read mode) and command-line utilities that use stdin/stdout for data transfer. The pbcopy and pbpaste utilities are designed to work with Unix pipes and don't need or accept file mode arguments.

Documentation references:
- macOS man pages: `man pbcopy` and `man pbpaste`
- Pandas clipboard source: `pandas/io/clipboard/__init__.py:99-114`

## Proposed Fix

```diff
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -100,13 +100,13 @@ def init_osx_pbcopy_clipboard():
     def copy_osx_pbcopy(text):
         text = _stringifyText(text)  # Converts non-str values to str.
         with subprocess.Popen(
-            ["pbcopy", "w"], stdin=subprocess.PIPE, close_fds=True
+            ["pbcopy"], stdin=subprocess.PIPE, close_fds=True
         ) as p:
             p.communicate(input=text.encode(ENCODING))

     def paste_osx_pbcopy():
         with subprocess.Popen(
-            ["pbpaste", "r"], stdout=subprocess.PIPE, close_fds=True
+            ["pbpaste"], stdout=subprocess.PIPE, close_fds=True
         ) as p:
             stdout = p.communicate()[0]
         return stdout.decode(ENCODING)
```