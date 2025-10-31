# Bug Report: pandas.io.clipboard waitForPaste/waitForNewPaste Timeout Violation

**Target**: `pandas.io.clipboard.waitForPaste`, `pandas.io.clipboard.waitForNewPaste`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `waitForPaste` and `waitForNewPaste` functions violate their documented contracts by raising `PyperclipException` immediately when the clipboard is unavailable, instead of waiting for the specified timeout and raising `PyperclipTimeoutException`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard


@settings(max_examples=10)
@given(st.floats(min_value=0.001, max_value=0.1))
def test_waitForPaste_timeout_raises(timeout_seconds):
    clipboard.set_clipboard('no')
    try:
        clipboard.waitForPaste(timeout=timeout_seconds)
        assert False, "Should have raised PyperclipTimeoutException"
    except clipboard.PyperclipTimeoutException:
        pass
```

**Failing input**: Any timeout value (e.g., `0.0625`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas.io.clipboard as clipboard

clipboard.set_clipboard('no')

clipboard.waitForPaste(timeout=1.0)
```

Expected: `PyperclipTimeoutException` after 1 second
Actual: `PyperclipException` raised immediately

## Why This Is A Bug

The docstring for `waitForPaste` (line 694-700) explicitly states:

> "This function raises PyperclipTimeoutException if timeout was set to a number of seconds that has elapsed without non-empty text being put on the clipboard."

Similarly, `waitForNewPaste` (line 714-721) has the same contract.

However, when the clipboard is unavailable (e.g., `set_clipboard('no')`), both functions call `paste()` without catching the `PyperclipException` it raises. This causes the exception to propagate immediately, bypassing the timeout logic entirely.

## Fix

The functions should catch `PyperclipException` from `paste()` and treat it as an empty clipboard, allowing the timeout logic to work correctly:

```diff
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -700,7 +700,10 @@ def waitForPaste(timeout=None):
     the clipboard."""
     startTime = time.time()
     while True:
-        clipboardText = paste()
+        try:
+            clipboardText = paste()
+        except PyperclipException:
+            clipboardText = ""
         if clipboardText != "":
             return clipboardText
         time.sleep(0.01)
@@ -721,7 +724,10 @@ def waitForNewPaste(timeout=None):
     the clipboard."""
     startTime = time.time()
-    originalText = paste()
+    try:
+        originalText = paste()
+    except PyperclipException:
+        originalText = ""
     while True:
-        currentText = paste()
+        try:
+            currentText = paste()
+        except PyperclipException:
+            currentText = ""
         if currentText != originalText:
             return currentText
         time.sleep(0.01)
```