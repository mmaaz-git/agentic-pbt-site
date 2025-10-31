# Bug Report: pandas.io.clipboard waitForPaste/waitForNewPaste Timeout Precision

**Target**: `pandas.io.clipboard.waitForPaste` and `pandas.io.clipboard.waitForNewPaste`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `waitForPaste()` and `waitForNewPaste()` functions do not respect timeout values ≤ 0.01 seconds. Even with `timeout=0` or negative values, both functions wait at least ~0.01 seconds before timing out, rather than timing out immediately.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard
import time


def mock_empty_paste():
    return ""


@given(st.floats(min_value=-100, max_value=0.005, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_waitForPaste_timeout_precision(timeout):
    def waitForPaste_mock(timeout_val):
        startTime = time.time()
        while True:
            clipboardText = mock_empty_paste()
            if clipboardText != "":
                return clipboardText
            time.sleep(0.01)

            if timeout_val is not None and time.time() > startTime + timeout_val:
                raise clipboard.PyperclipTimeoutException(
                    f"waitForPaste() timed out after {timeout_val} seconds."
                )

    start = time.time()
    try:
        waitForPaste_mock(timeout)
        assert False, f"Should have raised PyperclipTimeoutException for timeout={timeout}"
    except clipboard.PyperclipTimeoutException:
        elapsed = time.time() - start

        if timeout <= 0:
            assert elapsed < 0.005, (
                f"With timeout={timeout} (≤0), expected immediate timeout "
                f"but waited {elapsed:.4f}s (>0.005s). "
                f"Bug: timeout check happens AFTER sleep(0.01)"
            )
```

**Failing input**: `timeout=0.0` (or any value ≤ 0)

## Reproducing the Bug

```python
import pandas.io.clipboard as clipboard
import time


def mock_empty_paste():
    return ""


def waitForPaste_current(timeout=None):
    startTime = time.time()
    while True:
        clipboardText = mock_empty_paste()
        if clipboardText != "":
            return clipboardText
        time.sleep(0.01)

        if timeout is not None and time.time() > startTime + timeout:
            raise clipboard.PyperclipTimeoutException(
                f"waitForPaste() timed out after {timeout} seconds."
            )


start = time.time()
try:
    waitForPaste_current(timeout=0)
except clipboard.PyperclipTimeoutException:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")

start = time.time()
try:
    waitForPaste_current(timeout=-1)
except clipboard.PyperclipTimeoutException:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
```

Output:
```
Timed out after 0.0101s
Timed out after 0.0101s
```

Expected: Both should timeout in < 0.005s since the timeout has already expired.

## Why This Is A Bug

The timeout check occurs AFTER `time.sleep(0.01)` in the loop, meaning the function always waits at least one full iteration (~0.01s) before checking if the timeout has been exceeded. This violates the expected behavior where `timeout=0` or negative timeouts should cause an immediate timeout.

For users who want fine-grained timeout control (e.g., timeout=0.001s) or expect immediate timeout with timeout=0, this behavior is incorrect.

## Fix

Move the timeout check to occur before the sleep, or add an initial timeout check before entering the loop:

```diff
 def waitForPaste(timeout=None):
     """This function call blocks until a non-empty text string exists on the
     clipboard. It returns this text.

     This function raises PyperclipTimeoutException if timeout was set to
     a number of seconds that has elapsed without non-empty text being put on
     the clipboard."""
     startTime = time.time()
     while True:
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


 def waitForNewPaste(timeout=None):
     """This function call blocks until a new text string exists on the
     clipboard that is different from the text that was there when the function
     was first called. It returns this text.

     This function raises PyperclipTimeoutException if timeout was set to
     a number of seconds that has elapsed without non-empty text being put on
     the clipboard."""
     startTime = time.time()
     originalText = paste()
     while True:
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