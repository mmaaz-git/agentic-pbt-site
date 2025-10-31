# Bug Report: pandas.io.clipboard.waitForPaste Type Checking Issue

**Target**: `pandas.io.clipboard.waitForPaste`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`waitForPaste()` incorrectly returns on any non-empty-string value (including None, 0, False, [], {}) instead of waiting for an actual non-empty text string as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard

@given(value=st.one_of(st.none(), st.just(0), st.just(False), st.just([]), st.just({})))
@settings(max_examples=50, deadline=None)
def test_waitForPaste_should_wait_for_string(value):
    original_paste = clipboard.paste
    clipboard.paste = lambda: value

    try:
        result = clipboard.waitForPaste(timeout=0.1)
        assert isinstance(result, str) and result != "", \
            f"waitForPaste should wait for non-empty string, not {type(value).__name__}"
    except clipboard.PyperclipTimeoutException:
        pass
    finally:
        clipboard.paste = original_paste
```

**Failing input**: `None`, `0`, `False`, `[]`, `{}`

## Reproducing the Bug

```python
import pandas.io.clipboard as clipboard

original_paste = clipboard.paste
clipboard.paste = lambda: None

result = clipboard.waitForPaste(timeout=0.05)
print(f"Result: {repr(result)}, Type: {type(result).__name__}")

clipboard.paste = original_paste
```

Output:
```
Result: None, Type: NoneType
```

Expected: Should timeout or wait for actual non-empty string.
Actual: Returns None immediately.

## Why This Is A Bug

The docstring states: "This function call blocks until a non-empty text string exists on the clipboard."

However, the implementation checks `clipboardText != ""` which is True for any value that's not an empty string, including None, 0, False, [], {}, etc.

This is a realistic bug because:
1. On macOS with pyobjc, `paste_osx_pyobjc()` calls `board.stringForType_(AppKit.NSStringPboardType)` which returns None if the clipboard doesn't contain string data
2. The function should wait for a valid text string, not return None or other non-string types

## Fix

```diff
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -700,7 +700,7 @@ def waitForPaste(timeout=None):
     startTime = time.time()
     while True:
         clipboardText = paste()
-        if clipboardText != "":
+        if isinstance(clipboardText, str) and clipboardText != "":
             return clipboardText
         time.sleep(0.01)

```