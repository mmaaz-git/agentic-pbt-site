# Bug Report: pandas.io.clipboard macOS pbcopy/pbpaste Invalid Arguments

**Target**: `pandas.io.clipboard.init_osx_pbcopy_clipboard`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `init_osx_pbcopy_clipboard` function passes invalid arguments `"w"` and `"r"` to the `pbcopy` and `pbpaste` commands, causing clipboard operations to fail on macOS systems that use this clipboard mechanism.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard


@given(st.text())
@settings(max_examples=100)
def test_osx_pbcopy_round_trip(text):
    copy_func, paste_func = clipboard.init_osx_pbcopy_clipboard()
    copy_func(text)
    result = paste_func()
    assert result == text
```

**Failing input**: Any text input causes the copy operation to fail.

## Reproducing the Bug

```python
import pandas.io.clipboard as clipboard

copy_func, paste_func = clipboard.init_osx_pbcopy_clipboard()

copy_func("Hello, World!")
```

On macOS, this raises an error because `pbcopy` is called with an invalid argument `"w"`:
```
subprocess.Popen(["pbcopy", "w"], ...)
```

The `pbcopy` and `pbpaste` commands do not accept `"w"` or `"r"` as arguments. These appear to be mistakenly copied from file I/O code where these would be file modes.

## Why This Is A Bug

The macOS `pbcopy` and `pbpaste` utilities have simple interfaces:
- `pbcopy` reads from stdin and copies to clipboard (no arguments needed)
- `pbpaste` writes clipboard contents to stdout (no arguments needed)

Passing `"w"` to `pbcopy` and `"r"` to `pbpaste` causes these commands to fail, making clipboard operations non-functional for users on macOS who don't have PyObjC installed (which would use a different clipboard mechanism).

## Fix

```diff
diff --git a/pandas/io/clipboard/__init__.py b/pandas/io/clipboard/__init__.py
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -1,13 +1,13 @@
 def init_osx_pbcopy_clipboard():
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