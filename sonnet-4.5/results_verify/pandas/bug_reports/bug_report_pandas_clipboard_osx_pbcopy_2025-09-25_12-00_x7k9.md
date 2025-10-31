# Bug Report: pandas.io.clipboard macOS pbcopy/pbpaste Invalid Arguments

**Target**: `pandas.io.clipboard.init_osx_pbcopy_clipboard`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The macOS pbcopy/pbpaste clipboard backend passes invalid arguments `"w"` and `"r"` to the `pbcopy` and `pbpaste` commands, causing clipboard operations to fail on macOS systems when the pyobjc library is not installed.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard

@settings(max_examples=100)
@given(st.text())
def test_copy_paste_roundtrip(text):
    if not clipboard.is_available():
        return

    clipboard.copy(text)
    result = clipboard.paste()
    assert result == text
```

**Failing input**: Any text when using the pbcopy/pbpaste backend on macOS

## Reproducing the Bug

```python
import subprocess

p = subprocess.Popen(
    ["pbcopy", "w"],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
    close_fds=True
)
_, stderr = p.communicate(input=b"test text")
print(f"Return code: {p.returncode}")
print(f"Error: {stderr.decode()}")
```

On macOS, this produces an error like: `pbcopy: unknown option -- w`

Similarly for pbpaste:
```python
p = subprocess.Popen(["pbpaste", "r"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = p.communicate()
```

This produces: `pbpaste: unknown option -- r`

## Why This Is A Bug

According to macOS documentation:
- `pbcopy` usage: `pbcopy [-pboard {general | ruler | find | font}]`
- `pbpaste` usage: `pbpaste [-pboard {general | ruler | find | font}] [-Prefer {txt | rtf | ps}]`

Neither command accepts positional arguments `"w"` or `"r"`. These arguments cause the commands to fail with "unknown option" errors, making the entire pbcopy/pbpaste clipboard backend non-functional.

This affects macOS users who don't have pyobjc installed, as pandas falls back to the pbcopy/pbpaste backend which is completely broken.

## Fix

```diff
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -xxx,7 +xxx,7 @@ def init_osx_pbcopy_clipboard():
     def copy_osx_pbcopy(text):
         text = _stringifyText(text)  # Converts non-str values to str.
         with subprocess.Popen(
-            ["pbcopy", "w"], stdin=subprocess.PIPE, close_fds=True
+            ["pbcopy"], stdin=subprocess.PIPE, close_fds=True
         ) as p:
             p.communicate(input=text.encode(ENCODING))

@@ -xxx,7 +xxx,7 @@ def init_osx_pbcopy_clipboard():
     def paste_osx_pbcopy():
         with subprocess.Popen(
-            ["pbpaste", "r"], stdout=subprocess.PIPE, close_fds=True
+            ["pbpaste"], stdout=subprocess.PIPE, close_fds=True
         ) as p:
             stdout = p.communicate()[0]
         return stdout.decode(ENCODING)
```

The fix is simple: remove the invalid `"w"` and `"r"` arguments from the subprocess command lists.