# Bug Report: Cython.TestUtils.write_newer_file Infinite Loop When newer_than File Doesn't Exist

**Target**: `Cython.TestUtils.write_newer_file`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `write_newer_file()` function enters an infinite loop when the `newer_than` parameter points to a non-existent file, causing test suites to hang indefinitely without any error message.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Cython.TestUtils.write_newer_file using Hypothesis."""

import sys
import os
import tempfile
import signal
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck

# Add Cython to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
from Cython.TestUtils import write_newer_file

# Set a timeout to prevent infinite loop from hanging forever
def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out after 5 seconds - infinite loop detected!")

@given(st.text(min_size=0, max_size=100))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None, max_examples=5)
def test_write_newer_file_terminates(content):
    """Test that write_newer_file terminates when newer_than file doesn't exist."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout per test

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            nonexistent = os.path.join(tmpdir, "nonexistent.txt")

            # This should write the file and return, not hang
            write_newer_file(file_path, nonexistent, content)

            # Verify the file was created with correct content
            assert os.path.exists(file_path)
            with open(file_path) as f:
                assert f.read() == content

            signal.alarm(0)  # Cancel alarm on success

    except TimeoutError:
        signal.alarm(0)  # Cancel alarm
        raise AssertionError(f"write_newer_file entered infinite loop with content: {repr(content)}")
    finally:
        signal.alarm(0)  # Ensure alarm is cancelled

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for write_newer_file...")
    print("Testing with various string inputs when newer_than file doesn't exist.\n")

    try:
        test_write_newer_file_terminates()
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        print("\nThis confirms the bug: write_newer_file enters an infinite loop")
        print("when the newer_than file doesn't exist.")
    else:
        print("✅ All tests passed!")
        print("This should not happen if the bug exists.")
```

<details>

<summary>
**Failing input**: `content=''` (any string triggers the bug)
</summary>
```
Running property-based test for write_newer_file...
Testing with various string inputs when newer_than file doesn't exist.

Falsifying example: test_write_newer_file_terminates(
    content='',
)
❌ Test failed: write_newer_file entered infinite loop with content: ''

This confirms the bug: write_newer_file enters an infinite loop
when the newer_than file doesn't exist.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the Cython.TestUtils.write_newer_file infinite loop bug."""

import sys
import os
import tempfile
import signal

# Add Cython to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# Set a timeout to prevent infinite loop from hanging forever
def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out after 5 seconds - infinite loop detected!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5 second timeout

try:
    from Cython.TestUtils import write_newer_file

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")
        nonexistent = os.path.join(tmpdir, "nonexistent.txt")

        print(f"Creating file at: {file_path}")
        print(f"Comparing against non-existent file: {nonexistent}")
        print("Calling write_newer_file...")

        # This should return immediately but instead hangs forever
        write_newer_file(file_path, nonexistent, "test content")

        print("Function returned successfully!")
        print(f"File contents: {open(file_path).read()}")

except TimeoutError as e:
    print(f"\n❌ ERROR: {e}")
    print("The function entered an infinite loop as predicted by the bug report.")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
finally:
    signal.alarm(0)  # Cancel the alarm
```

<details>

<summary>
Process hangs indefinitely and must be killed with timeout
</summary>
```
Creating file at: /tmp/tmp88eutfxe/test.txt
Comparing against non-existent file: /tmp/tmp88eutfxe/nonexistent.txt
Calling write_newer_file...

❌ ERROR: Function call timed out after 5 seconds - infinite loop detected!
The function entered an infinite loop as predicted by the bug report.
```
</details>

## Why This Is A Bug

The function violates its documented behavior and the developer's explicit intent:

1. **Docstring Contract Violation**: The docstring states the function should "make sure [the file] is newer than the file `newer_than`". When `newer_than` doesn't exist, any existing file is by definition newer, so the function should succeed immediately.

2. **Comment Intent Violation**: Line 385 contains the comment "Support writing a fresh file (which is always newer than a non-existent one)", explicitly acknowledging this use case should be supported.

3. **Logic Error**: The bug occurs due to a flawed while loop condition:
   - When `newer_than` doesn't exist, `other_time` is set to `None` (line 386)
   - The while condition `other_time is None or other_time >= os.path.getmtime(file_path)` (line 388)
   - Since `other_time is None` always evaluates to `True`, the loop runs forever
   - `other_time` is never updated inside the loop body
   - Result: infinite loop

4. **Incomplete Error Handling**: The try-except block catches the OSError when `newer_than` doesn't exist but then proceeds with flawed logic instead of returning after the successful initial write on line 380.

## Relevant Context

- **Function Location**: `/Cython/TestUtils.py`, lines 373-390
- **Function Purpose**: Creates or updates a file to ensure it has a modification time newer than a reference file
- **Common Use Case**: Used in test suites to ensure test artifacts are regenerated when dependencies change
- **Impact**: Any test suite using this function with non-existent reference files will hang indefinitely
- **Existing Tests**: The current test suite only tests the case where `file_path == newer_than`, missing this critical edge case

The bug is particularly problematic because:
- It causes a silent hang with no error message
- There's no built-in timeout mechanism
- It affects testing infrastructure, potentially breaking CI/CD pipelines
- The hang makes it difficult to diagnose the root cause

## Proposed Fix

```diff
--- a/Cython/TestUtils.py
+++ b/Cython/TestUtils.py
@@ -382,10 +382,11 @@ def write_newer_file(file_path, newer_than, content, dedent=False, encoding=Non
     try:
         other_time = os.path.getmtime(newer_than)
     except OSError:
-        # Support writing a fresh file (which is always newer than a non-existent one)
-        other_time = None
-
-    while other_time is None or other_time >= os.path.getmtime(file_path):
+        # Support writing a fresh file (which is always newer than a non-existent one).
+        # The file was already written on line 380, so we're done.
+        return
+
+    # Keep rewriting until file_path is newer than newer_than
+    while other_time >= os.path.getmtime(file_path):
         write_file(file_path, content, dedent=dedent, encoding=encoding)
```