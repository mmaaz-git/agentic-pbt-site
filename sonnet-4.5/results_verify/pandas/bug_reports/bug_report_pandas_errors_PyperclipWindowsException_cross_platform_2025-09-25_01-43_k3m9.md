# Bug Report: pandas.errors.PyperclipWindowsException Cross-Platform Instantiation Crash

**Target**: `pandas.errors.PyperclipWindowsException`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`PyperclipWindowsException` crashes with an `AttributeError` when instantiated on non-Windows platforms (Linux, macOS) because it unconditionally calls `ctypes.WinError()`, which only exists on Windows.

## Property-Based Test

```python
import platform

import hypothesis.strategies as st
import pytest
from hypothesis import given

from pandas.errors import PyperclipWindowsException


@given(st.text())
def test_pyperclip_windows_exception_crashes_with_any_message(message):
    if platform.system() != 'Windows':
        with pytest.raises(AttributeError):
            PyperclipWindowsException(message)
```

**Failing input**: Any string message on non-Windows platforms (e.g., `"test message"`)

## Reproducing the Bug

```python
import platform

from pandas.errors import PyperclipWindowsException

print(f"Platform: {platform.system()}")

try:
    exc = PyperclipWindowsException("Clipboard access denied")
    print(f"Success: {exc}")
except AttributeError as e:
    print(f"Bug confirmed! AttributeError: {e}")
```

On Linux/macOS, this outputs:
```
Platform: Linux
Bug confirmed! AttributeError: module 'ctypes' has no attribute 'WinError'
```

## Why This Is A Bug

1. `PyperclipWindowsException` is part of pandas' public API (exported from `pandas.errors`)
2. The exception is designed for cross-platform code - it may be imported and used in exception handling on any platform
3. While the exception is meant to be raised on Windows, code may need to catch or reference it on other platforms
4. The docstring and implementation suggest this should be a Windows-specific exception, but the `__init__` unconditionally calls a Windows-only function
5. The code even includes a comment acknowledging the platform-specific nature ("# attr only exists on Windows"), but doesn't handle it properly

## Fix

The fix should check for platform or the existence of `WinError` before calling it:

```diff
diff --git a/pandas/errors/__init__.py b/pandas/errors/__init__.py
index 1234567..abcdefg 100644
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -634,7 +634,11 @@ class PyperclipWindowsException(PyperclipException):
     """

     def __init__(self, message: str) -> None:
-        # attr only exists on Windows, so typing fails on other platforms
-        message += f" ({ctypes.WinError()})"  # type: ignore[attr-defined]
+        # WinError only exists on Windows
+        if hasattr(ctypes, 'WinError'):
+            message += f" ({ctypes.WinError()})"  # type: ignore[attr-defined]
+        else:
+            # On non-Windows platforms, just use the message as-is
+            pass
         super().__init__(message)
```

Alternatively, a simpler fix using platform detection:

```diff
diff --git a/pandas/errors/__init__.py b/pandas/errors/__init__.py
index 1234567..abcdefg 100644
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -1,5 +1,6 @@
 from __future__ import annotations

+import platform
 import ctypes
 from typing import TYPE_CHECKING

@@ -634,7 +635,8 @@ class PyperclipWindowsException(PyperclipException):
     """

     def __init__(self, message: str) -> None:
-        # attr only exists on Windows, so typing fails on other platforms
-        message += f" ({ctypes.WinError()})"  # type: ignore[attr-defined]
+        # WinError only exists on Windows
+        if platform.system() == 'Windows':
+            message += f" ({ctypes.WinError()})"  # type: ignore[attr-defined]
         super().__init__(message)
```