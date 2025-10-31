# Bug Report: pandas.io.clipboard.waitForNewPaste Type Checking Issue

**Target**: `pandas.io.clipboard.waitForNewPaste`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`waitForNewPaste()` incorrectly returns on any changed value (including None, 0, False, [], {}) instead of waiting for an actual text string as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard

@given(
    initial=st.text(),
    changed=st.one_of(st.none(), st.integers(), st.booleans(), st.lists(st.text()))
)
@settings(max_examples=100, deadline=None)
def test_waitForNewPaste_should_wait_for_string(initial, changed):
    if initial == changed:
        return

    original_paste = clipboard.paste
    call_count = [0]

    def mock_paste():
        call_count[0] += 1
        return initial if call_count[0] == 1 else changed

    clipboard.paste = mock_paste

    try:
        result = clipboard.waitForNewPaste(timeout=0.1)
        assert isinstance(result, str), \
            f"waitForNewPaste should return string, not {type(result).__name__}"
    finally:
        clipboard.paste = original_paste
```

**Failing input**: `initial="text"`, `changed=None` (or any non-string type)

## Reproducing the Bug

```python
import pandas.io.clipboard as clipboard

original_paste = clipboard.paste
call_count = [0]

def mock_paste():
    call_count[0] += 1
    if call_count[0] == 1:
        return "initial text"
    else:
        return None

clipboard.paste = mock_paste

result = clipboard.waitForNewPaste(timeout=0.1)
print(f"Result: {repr(result)}, Type: {type(result).__name__}")

clipboard.paste = original_paste
```

Output:
```
Result: None, Type: NoneType
```

Expected: Should wait for a new text string, not return None.
Actual: Returns None when clipboard changes to None.

## Why This Is A Bug

The docstring states: "This function call blocks until a new text string exists on the clipboard that is different from the text that was there when the function was first called."

However, the implementation only checks `currentText != originalText` without validating that `currentText` is actually a string.

This is a realistic bug because:
1. On macOS with pyobjc, `paste_osx_pyobjc()` can return None if the clipboard type changes to non-string data
2. The function should wait for a valid text string, not return None or other non-string types

## Fix

```diff
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -722,7 +722,7 @@ def waitForNewPaste(timeout=None):
     originalText = paste()
     while True:
         currentText = paste()
-        if currentText != originalText:
+        if isinstance(currentText, str) and currentText != originalText:
             return currentText
         time.sleep(0.01)

```