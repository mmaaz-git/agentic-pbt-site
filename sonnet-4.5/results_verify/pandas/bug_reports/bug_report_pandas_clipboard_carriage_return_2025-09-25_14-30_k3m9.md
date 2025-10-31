# Bug Report: pandas.io.clipboard Carriage Return Corruption

**Target**: `pandas.io.clipboard.init_dev_clipboard_clipboard`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `init_dev_clipboard_clipboard()` function silently corrupts carriage return (`\r`) characters by converting them to newlines (`\n`) when copying and pasting text. The function issues a warning that it "cannot handle \r characters" but then proceeds to write them anyway, resulting in silent data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import tempfile
import os


@given(st.text(alphabet='\r\n', min_size=1, max_size=50))
@settings(max_examples=100)
def test_carriage_return_preservation(text):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "clipboard")

        with open(test_file, "w", encoding="utf-8") as fd:
            fd.write(text)

        with open(test_file, encoding="utf-8") as fd:
            result = fd.read()

        if '\r' in text and '\n' not in text:
            assert text == result, f"Carriage returns not preserved: {repr(text)} != {repr(result)}"
```

**Failing input**: `'\r'`, `'hello\rworld'`, `'\r\r\r'`, etc.

## Reproducing the Bug

```python
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    clipboard_file = os.path.join(tmpdir, "clipboard")

    def copy_dev_clipboard(text):
        with open(clipboard_file, "w", encoding="utf-8") as fd:
            fd.write(text)

    def paste_dev_clipboard():
        with open(clipboard_file, encoding="utf-8") as fd:
            return fd.read()

    original = "hello\rworld"
    copy_dev_clipboard(original)
    result = paste_dev_clipboard()

    print(f"Original: {repr(original)}")
    print(f"Result:   {repr(result)}")

    assert result == "hello\nworld"
```

## Why This Is A Bug

The function's warning message states it "cannot handle \r characters on Cygwin", which is accurate - carriage returns ARE corrupted. However, the function continues to write the text anyway, resulting in silent data corruption where `\r` is converted to `\n`.

This violates the round-trip property fundamental to clipboard operations: `paste()` should return the exact text passed to `copy()`.

The root cause is that Python's `open()` with default `newline=None` performs newline translation in text mode, converting `\r` to `\n` on read.

## Fix

```diff
 def init_dev_clipboard_clipboard():
     def copy_dev_clipboard(text):
         text = _stringifyText(text)
         if text == "":
             warnings.warn(
                 "Pyperclip cannot copy a blank string to the clipboard on Cygwin. "
                 "This is effectively a no-op.",
                 stacklevel=find_stack_level(),
             )
-        if "\r" in text:
-            warnings.warn(
-                "Pyperclip cannot handle \\r characters on Cygwin.",
-                stacklevel=find_stack_level(),
-            )

-        with open("/dev/clipboard", "w", encoding="utf-8") as fd:
+        with open("/dev/clipboard", "w", encoding="utf-8", newline='') as fd:
             fd.write(text)

     def paste_dev_clipboard() -> str:
-        with open("/dev/clipboard", encoding="utf-8") as fd:
+        with open("/dev/clipboard", encoding="utf-8", newline='') as fd:
             content = fd.read()
         return content

     return copy_dev_clipboard, paste_dev_clipboard
```

The `newline=''` parameter disables Python's newline translation, preserving all characters exactly as written. The warning about `\r` characters can be removed since they will now be handled correctly.