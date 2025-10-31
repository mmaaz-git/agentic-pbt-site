# Bug Report: pandas.io.clipboard Timeout Functions Hang with NaN or Infinity

**Target**: `pandas.io.clipboard.waitForPaste` and `pandas.io.clipboard.waitForNewPaste`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `waitForPaste` and `waitForNewPaste` functions hang indefinitely when passed `float('nan')` or `float('inf')` as timeout values, instead of either rejecting these values or handling them appropriately.

## Property-Based Test

```python
import time
import pytest
from unittest.mock import patch
import pandas.io.clipboard as clip


@settings(max_examples=10)
def test_waitForPaste_nan_timeout_hangs():
    with patch.object(clip, 'paste', return_value=''):
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Test timed out - function is hanging!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)

        try:
            clip.waitForPaste(timeout=float('nan'))
        finally:
            signal.alarm(0)
```

**Failing input**: `timeout=float('nan')` or `timeout=float('inf')`

## Reproducing the Bug

```python
import time
import math
from unittest.mock import patch
import pandas.io.clipboard as clip

with patch.object(clip, 'paste', return_value=''):
    start = time.time()
    timeout_value = float('nan')

    print(f"Testing timeout comparison: time.time() > start + timeout_value")
    print(f"  {time.time()} > {start + timeout_value} = {time.time() > start + timeout_value}")

    print(f"Calling waitForPaste(timeout={timeout_value})")
    clip.waitForPaste(timeout=timeout_value)
```

When run, this will hang indefinitely because:
- `NaN + any_number = NaN`
- `any_number > NaN = False` (always)
- Similarly, `any_number > infinity = False` (always)

## Why This Is A Bug

The timeout comparison logic in both functions uses:
```python
if timeout is not None and time.time() > startTime + timeout:
    raise PyperclipTimeoutException(...)
```

When `timeout` is `NaN`:
- `startTime + NaN = NaN`
- `time.time() > NaN` is always `False` (NaN comparisons always return False)
- The timeout condition never triggers, causing an infinite loop

When `timeout` is `infinity`:
- `startTime + infinity = infinity`
- `time.time() > infinity` is always `False`
- The timeout condition never triggers, causing an infinite loop

This violates the documented behavior that specifies the function "raises PyperclipTimeoutException if timeout was set to a number of seconds that has elapsed". Users would reasonably expect that invalid timeout values would either be rejected with a clear error or handled gracefully.

## Fix

Add validation for the timeout parameter to reject NaN and negative values, and clarify the behavior for infinity:

```diff
 def waitForPaste(timeout=None):
     """This function call blocks until a non-empty text string exists on the
     clipboard. It returns this text.

     This function raises PyperclipTimeoutException if timeout was set to
     a number of seconds that has elapsed without non-empty text being put on
     the clipboard."""
+    if timeout is not None and (math.isnan(timeout) or timeout < 0):
+        raise ValueError(f"timeout must be a non-negative number, got {timeout}")
+
     startTime = time.time()
     while True:
         clipboardText = paste()
         if clipboardText != "":
             return clipboardText
         time.sleep(0.01)

         if timeout is not None and time.time() > startTime + timeout:
             raise PyperclipTimeoutException(
                 "waitForPaste() timed out after " + str(timeout) + " seconds."
             )


 def waitForNewPaste(timeout=None):
     """This function call blocks until a new text string exists on the
     clipboard that is different from the text that was there when the function
     was first called. It returns this text.

     This function raises PyperclipTimeoutException if timeout was set to
     a number of seconds that has elapsed without non-empty text being put on
     the clipboard."""
+    if timeout is not None and (math.isnan(timeout) or timeout < 0):
+        raise ValueError(f"timeout must be a non-negative number, got {timeout}")
+
     startTime = time.time()
     originalText = paste()
     while True:
         currentText = paste()
         if currentText != originalText:
             return currentText
         time.sleep(0.01)

         if timeout is not None and time.time() > startTime + timeout:
             raise PyperclipTimeoutException(
                 "waitForNewPaste() timed out after " + str(timeout) + " seconds."
             )
```

Note: This fix requires adding `import math` at the top of the file.