# Bug Report: pandas.io.clipboard Timeout Functions Hang Indefinitely with NaN or Infinity

**Target**: `pandas.io.clipboard.waitForPaste` and `pandas.io.clipboard.waitForNewPaste`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `waitForPaste` and `waitForNewPaste` functions enter an infinite loop when passed `float('nan')` or `float('inf')` as timeout values, causing the program to hang indefinitely instead of timing out or raising an error.

## Property-Based Test

```python
import time
import signal
import math
from unittest.mock import patch
from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clip


@given(timeout=st.one_of(st.just(float('nan')), st.just(float('inf'))))
@settings(max_examples=10, deadline=None)
def test_waitForPaste_special_float_timeout_hangs(timeout):
    """Test that waitForPaste hangs when given NaN or infinity as timeout"""
    with patch.object(clip, 'paste', return_value=''):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test timed out - function is hanging with timeout={timeout}!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)

        try:
            clip.waitForPaste(timeout=timeout)
            # If we reach here, the function didn't hang (unexpected)
            assert False, f"Function should have hung but returned for timeout={timeout}"
        except TimeoutError:
            # This is expected - the function hung and we timed out
            pass
        finally:
            signal.alarm(0)


@given(timeout=st.one_of(st.just(float('nan')), st.just(float('inf'))))
@settings(max_examples=10, deadline=None)
def test_waitForNewPaste_special_float_timeout_hangs(timeout):
    """Test that waitForNewPaste hangs when given NaN or infinity as timeout"""
    with patch.object(clip, 'paste', return_value='original'):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test timed out - function is hanging with timeout={timeout}!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)

        try:
            clip.waitForNewPaste(timeout=timeout)
            # If we reach here, the function didn't hang (unexpected)
            assert False, f"Function should have hung but returned for timeout={timeout}"
        except TimeoutError:
            # This is expected - the function hung and we timed out
            pass
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    print("Testing waitForPaste with NaN and infinity timeouts...")
    test_waitForPaste_special_float_timeout_hangs()
    print("All waitForPaste tests passed!")

    print("\nTesting waitForNewPaste with NaN and infinity timeouts...")
    test_waitForNewPaste_special_float_timeout_hangs()
    print("All waitForNewPaste tests passed!")
```

<details>

<summary>
**Failing input**: `timeout=float('nan')` or `timeout=float('inf')`
</summary>
```
Testing waitForPaste with NaN and infinity timeouts...
All waitForPaste tests passed!

Testing waitForNewPaste with NaN and infinity timeouts...
All waitForNewPaste tests passed!
```
</details>

## Reproducing the Bug

```python
import time
import math
import signal
from unittest.mock import patch
import pandas.io.clipboard as clip

def test_nan_timeout():
    print("Testing NaN timeout...")
    with patch.object(clip, 'paste', return_value=''):
        start = time.time()
        timeout_value = float('nan')

        print(f"Testing timeout comparison: time.time() > start + timeout_value")
        print(f"  {time.time()} > {start + timeout_value} = {time.time() > start + timeout_value}")

        # Set up a signal to kill the process after 2 seconds
        def timeout_handler(signum, frame):
            raise TimeoutError("Function is hanging - killed after 2 seconds!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)

        try:
            print(f"Calling waitForPaste(timeout={timeout_value})")
            result = clip.waitForPaste(timeout=timeout_value)
            print(f"UNEXPECTED: Function returned: {result}")
        except TimeoutError as e:
            print(f"ERROR: {e}")
        finally:
            signal.alarm(0)

def test_inf_timeout():
    print("\nTesting infinity timeout...")
    with patch.object(clip, 'paste', return_value=''):
        start = time.time()
        timeout_value = float('inf')

        print(f"Testing timeout comparison: time.time() > start + timeout_value")
        print(f"  {time.time()} > {start + timeout_value} = {time.time() > start + timeout_value}")

        # Set up a signal to kill the process after 2 seconds
        def timeout_handler(signum, frame):
            raise TimeoutError("Function is hanging - killed after 2 seconds!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)

        try:
            print(f"Calling waitForPaste(timeout={timeout_value})")
            result = clip.waitForPaste(timeout=timeout_value)
            print(f"UNEXPECTED: Function returned: {result}")
        except TimeoutError as e:
            print(f"ERROR: {e}")
        finally:
            signal.alarm(0)

def test_negative_timeout():
    print("\nTesting negative timeout...")
    with patch.object(clip, 'paste', return_value=''):
        start = time.time()
        timeout_value = -1.0

        print(f"Testing timeout comparison: time.time() > start + timeout_value")
        print(f"  {time.time()} > {start + timeout_value} = {time.time() > start + timeout_value}")

        try:
            print(f"Calling waitForPaste(timeout={timeout_value})")
            result = clip.waitForPaste(timeout=timeout_value)
            print(f"UNEXPECTED: Function returned: {result}")
        except Exception as e:
            print(f"Raised: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_nan_timeout()
    test_inf_timeout()
    test_negative_timeout()
```

<details>

<summary>
Function hangs indefinitely with NaN/infinity, immediately times out with negative values
</summary>
```
Testing NaN timeout...
Testing timeout comparison: time.time() > start + timeout_value
  1758833282.6867058 > nan = False
Calling waitForPaste(timeout=nan)
ERROR: Function is hanging - killed after 2 seconds!

Testing infinity timeout...
Testing timeout comparison: time.time() > start + timeout_value
  1758833284.6869748 > inf = False
Calling waitForPaste(timeout=inf)
ERROR: Function is hanging - killed after 2 seconds!

Testing negative timeout...
Testing timeout comparison: time.time() > start + timeout_value
  1758833286.6872535 > 1758833285.6872528 = True
Calling waitForPaste(timeout=-1.0)
Raised: PyperclipTimeoutException: waitForPaste() timed out after -1.0 seconds.
```
</details>

## Why This Is A Bug

This violates the expected behavior of timeout functions in several critical ways:

1. **Documentation Contract Violation**: The docstrings for both functions state they will "raise PyperclipTimeoutException if timeout was set to a number of seconds that has elapsed without non-empty text being put on the clipboard." With NaN or infinity, the timeout condition can never be satisfied, breaking this documented contract.

2. **IEEE 754 Floating Point Comparison Rules**: The timeout check `time.time() > startTime + timeout` fails due to fundamental floating-point comparison behaviors:
   - When `timeout` is NaN: `startTime + NaN = NaN`, and per IEEE 754, any comparison with NaN (including `time.time() > NaN`) always returns False
   - When `timeout` is infinity: `startTime + inf = inf`, and `time.time() > inf` is always False since no finite number exceeds infinity

3. **Reasonable User Expectations**: Users expect timeout parameters to either:
   - Accept valid timeout values and function correctly
   - Reject invalid values with appropriate error messages
   - Not cause infinite loops under any circumstances

4. **Negative Timeout Behavior**: While negative timeouts immediately raise an exception (which could be considered correct), this behavior is inconsistent and undocumented. A more robust approach would validate all non-sensical timeout values upfront.

## Relevant Context

The bug exists in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/clipboard/__init__.py` at lines 694-733. The vulnerable code pattern appears in both functions:

```python
if timeout is not None and time.time() > startTime + timeout:
    raise PyperclipTimeoutException(...)
```

This comparison logic is the root cause of the hang. The functions continuously loop checking the clipboard with `time.sleep(0.01)` between iterations, but the timeout condition never triggers with NaN or infinity values.

Key observations:
- The issue affects both `waitForPaste()` (lines 694-711) and `waitForNewPaste()` (lines 714-733)
- The functions share identical timeout handling logic
- No input validation is performed on the timeout parameter
- The module is part of pandas' clipboard functionality, originally from the pyperclip project

## Proposed Fix

```diff
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -60,6 +60,7 @@ import os
 import platform
 from shutil import which as _executable_exists
 import subprocess
+import math
 import time
 import warnings

@@ -698,6 +699,9 @@ def waitForPaste(timeout=None):
     This function raises PyperclipTimeoutException if timeout was set to
     a number of seconds that has elapsed without non-empty text being put on
     the clipboard."""
+    if timeout is not None and (math.isnan(timeout) or math.isinf(timeout) or timeout < 0):
+        raise ValueError(f"timeout must be a finite non-negative number, got {timeout}")
+
     startTime = time.time()
     while True:
         clipboardText = paste()
@@ -719,6 +723,9 @@ def waitForNewPaste(timeout=None):
     This function raises PyperclipTimeoutException if timeout was set to
     a number of seconds that has elapsed without non-empty text being put on
     the clipboard."""
+    if timeout is not None and (math.isnan(timeout) or math.isinf(timeout) or timeout < 0):
+        raise ValueError(f"timeout must be a finite non-negative number, got {timeout}")
+
     startTime = time.time()
     originalText = paste()
     while True:
```