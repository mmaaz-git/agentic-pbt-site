# Bug Report: pandas.io.clipboard waitForPaste/waitForNewPaste Minimum Timeout Enforcement Failure

**Target**: `pandas.io.clipboard.waitForPaste` and `pandas.io.clipboard.waitForNewPaste`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `waitForPaste()` and `waitForNewPaste()` functions always wait at least ~0.01 seconds before timing out, even when timeout values ≤ 0.01 seconds are specified, violating the expected timeout behavior for immediate or sub-10ms timeouts.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for pandas.io.clipboard timeout precision bug.
Tests that waitForPaste() and waitForNewPaste() respect small/zero/negative timeouts.
"""

from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard
import time

# Mock the paste function to always return empty string
def mock_empty_paste():
    return ""

@given(st.floats(min_value=-100, max_value=0.005, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_waitForPaste_timeout_precision(timeout):
    # Override paste function with mock
    original_paste = clipboard.paste
    clipboard.paste = mock_empty_paste

    try:
        start = time.time()
        try:
            clipboard.waitForPaste(timeout)
            assert False, f"Should have raised PyperclipTimeoutException for timeout={timeout}"
        except clipboard.PyperclipTimeoutException:
            elapsed = time.time() - start

            if timeout <= 0:
                assert elapsed < 0.005, (
                    f"With timeout={timeout} (≤0), expected immediate timeout "
                    f"but waited {elapsed:.4f}s (>0.005s). "
                    f"Bug: timeout check happens AFTER sleep(0.01)"
                )
    finally:
        clipboard.paste = original_paste

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for waitForPaste timeout precision...")
    print("Testing with timeouts from -100 to 0.005")
    print("=" * 60)

    try:
        test_waitForPaste_timeout_precision()
        print("\nAll tests passed! No bug detected.")
    except AssertionError as e:
        print(f"\nBUG DETECTED: {e}")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `timeout=0.0`
</summary>
```
Running property-based test for waitForPaste timeout precision...
Testing with timeouts from -100 to 0.005
============================================================

BUG DETECTED: With timeout=0.0 (≤0), expected immediate timeout but waited 0.0101s (>0.005s). Bug: timeout check happens AFTER sleep(0.01)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the pandas.io.clipboard timeout bug.
Demonstrates that waitForPaste() and waitForNewPaste() do not respect
timeout values <= 0.01 seconds.
"""

import time
import pandas.io.clipboard as clipboard

# Mock the paste function to always return empty string
# This simulates an empty clipboard that never gets filled
original_paste = clipboard.paste
def mock_empty_paste():
    return ""
clipboard.paste = mock_empty_paste

print("Testing waitForPaste() with various timeout values:")
print("=" * 50)

# Test with timeout=0 (should timeout immediately)
print("\nTest 1: timeout=0 (should timeout immediately)")
start = time.time()
try:
    clipboard.waitForPaste(timeout=0)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: <0.005s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited longer than expected")

# Test with timeout=-1 (negative, should timeout immediately)
print("\nTest 2: timeout=-1 (negative, should timeout immediately)")
start = time.time()
try:
    clipboard.waitForPaste(timeout=-1)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: <0.005s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited longer than expected")

# Test with timeout=0.001 (1ms, should timeout after ~1ms)
print("\nTest 3: timeout=0.001 (should timeout after ~0.001s)")
start = time.time()
try:
    clipboard.waitForPaste(timeout=0.001)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: ~0.001s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited much longer than expected")

print("\n" + "=" * 50)
print("Testing waitForNewPaste() with various timeout values:")
print("=" * 50)

# Mock to return a constant non-empty string for waitForNewPaste
def mock_constant_paste():
    return "constant text"
clipboard.paste = mock_constant_paste

# Test waitForNewPaste with timeout=0
print("\nTest 4: waitForNewPaste with timeout=0")
start = time.time()
try:
    clipboard.waitForNewPaste(timeout=0)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: <0.005s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited longer than expected")

# Test waitForNewPaste with timeout=-1
print("\nTest 5: waitForNewPaste with timeout=-1")
start = time.time()
try:
    clipboard.waitForNewPaste(timeout=-1)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: <0.005s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited longer than expected")

print("\n" + "=" * 50)
print("Summary: The bug exists because the timeout check occurs")
print("AFTER time.sleep(0.01), causing a minimum wait of ~0.01s")
print("regardless of the timeout value specified.")
```

<details>

<summary>
All 5 tests confirm the bug - functions always wait ~0.01s minimum
</summary>
```
Testing waitForPaste() with various timeout values:
==================================================

Test 1: timeout=0 (should timeout immediately)
Timed out after 0.0101s
Expected: <0.005s, Actual: 0.0101s
BUG CONFIRMED: Waited longer than expected

Test 2: timeout=-1 (negative, should timeout immediately)
Timed out after 0.0101s
Expected: <0.005s, Actual: 0.0101s
BUG CONFIRMED: Waited longer than expected

Test 3: timeout=0.001 (should timeout after ~0.001s)
Timed out after 0.0101s
Expected: ~0.001s, Actual: 0.0101s
BUG CONFIRMED: Waited much longer than expected

==================================================
Testing waitForNewPaste() with various timeout values:
==================================================

Test 4: waitForNewPaste with timeout=0
Timed out after 0.0101s
Expected: <0.005s, Actual: 0.0101s
BUG CONFIRMED: Waited longer than expected

Test 5: waitForNewPaste with timeout=-1
Timed out after 0.0101s
Expected: <0.005s, Actual: 0.0101s
BUG CONFIRMED: Waited longer than expected

==================================================
Summary: The bug exists because the timeout check occurs
AFTER time.sleep(0.01), causing a minimum wait of ~0.01s
regardless of the timeout value specified.
```
</details>

## Why This Is A Bug

This violates expected timeout behavior in several ways:

1. **timeout=0 convention**: In standard programming practice, `timeout=0` universally means "check once and return immediately" or "don't wait at all". Functions like `select()`, `poll()`, and similar APIs follow this convention. Users expect `waitForPaste(timeout=0)` to check the clipboard once and immediately timeout if empty, not wait 10ms.

2. **Negative timeout handling**: The functions accept negative timeout values without error but treat them incorrectly. They should either raise a ValueError for invalid negative inputs, or treat them as 0 (immediate timeout). Instead, they wait the full 10ms before checking the timeout condition.

3. **Precision violation**: Users specifying `timeout=0.001` (1 millisecond) receive a 10x longer wait than requested. This breaks any code requiring fine-grained timing control or quick polling cycles.

4. **Documentation mismatch**: The docstrings state the function "raises PyperclipTimeoutException if timeout was set to a number of seconds that has elapsed" - but 0 seconds have already elapsed at function entry for `timeout=0`, yet it still waits 10ms.

The root cause is the implementation's control flow: both functions execute `time.sleep(0.01)` before checking if the timeout has expired, guaranteeing a minimum 10ms wait regardless of the specified timeout value.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/clipboard/__init__.py`:

- `waitForPaste()`: Lines 694-711
- `waitForNewPaste()`: Lines 714-733

Both functions follow an identical flawed pattern:
```python
while True:
    clipboardText = paste()  # Check clipboard
    if clipboardText != "":  # Check if non-empty (or changed)
        return clipboardText
    time.sleep(0.01)  # ALWAYS sleeps 10ms first

    # Only THEN checks if timeout expired
    if timeout is not None and time.time() > startTime + timeout:
        raise PyperclipTimeoutException(...)
```

This affects any code that:
- Uses clipboard monitoring with precise timing requirements
- Expects immediate timeout behavior with `timeout=0`
- Implements high-frequency clipboard polling
- Relies on sub-10ms timeout precision

PyPerclip documentation: https://pyperclip.readthedocs.io/

## Proposed Fix

```diff
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -700,11 +700,14 @@ def waitForPaste(timeout=None):
     the clipboard."""
     startTime = time.time()
     while True:
+        # Check timeout BEFORE sleeping
+        if timeout is not None and time.time() > startTime + timeout:
+            raise PyperclipTimeoutException(
+                "waitForPaste() timed out after " + str(timeout) + " seconds."
+            )
+
         clipboardText = paste()
         if clipboardText != "":
             return clipboardText
         time.sleep(0.01)

-        if timeout is not None and time.time() > startTime + timeout:
-            raise PyperclipTimeoutException(
-                "waitForPaste() timed out after " + str(timeout) + " seconds."
-            )

@@ -721,11 +724,14 @@ def waitForNewPaste(timeout=None):
     startTime = time.time()
     originalText = paste()
     while True:
+        # Check timeout BEFORE sleeping
+        if timeout is not None and time.time() > startTime + timeout:
+            raise PyperclipTimeoutException(
+                "waitForNewPaste() timed out after " + str(timeout) + " seconds."
+            )
+
         currentText = paste()
         if currentText != originalText:
             return currentText
         time.sleep(0.01)

-        if timeout is not None and time.time() > startTime + timeout:
-            raise PyperclipTimeoutException(
-                "waitForNewPaste() timed out after " + str(timeout) + " seconds."
-            )
```