# Bug Report: pandas.errors.PyperclipWindowsException Crashes on Non-Windows Platforms

**Target**: `pandas.errors.PyperclipWindowsException`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`PyperclipWindowsException` crashes when instantiated on non-Windows platforms (Linux, macOS) because it unconditionally calls `ctypes.WinError()`, which only exists on Windows. This makes the exception unusable for error handling on non-Windows systems.

## Property-Based Test

```python
import sys
import pandas.errors


def test_pyperclipwindowsexception_non_windows():
    """
    Property: PyperclipWindowsException should be instantiable on all platforms
    without crashing, even though it's Windows-specific.
    """
    if sys.platform == 'win32':
        return

    error = pandas.errors.PyperclipWindowsException("test message")
    assert "test message" in str(error)
```

**Failing input**: Any message string on Linux/macOS

## Reproducing the Bug

```python
import sys
import pandas as pd

print(f"Platform: {sys.platform}")

error = pd.errors.PyperclipWindowsException("Clipboard access denied")
print(f"Error message: {error}")
```

**Output on Linux:**
```
Platform: linux
AttributeError: module 'ctypes' has no attribute 'WinError'
```

## Why This Is A Bug

The exception is defined in the public API and can be raised by pandas code that might run on any platform. However, simply instantiating this exception crashes on non-Windows systems, making it impossible to handle these errors properly in cross-platform code.

While the exception is Windows-specific by name, exception classes should be instantiable on all platforms to allow for proper error handling in libraries that support multiple platforms.

## Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -634,7 +634,10 @@ class PyperclipWindowsException(PyperclipException):
     """

     def __init__(self, message: str) -> None:
-        # attr only exists on Windows, so typing fails on other platforms
-        message += f" ({ctypes.WinError()})"  # type: ignore[attr-defined]
+        import sys
+        if sys.platform == 'win32':
+            # attr only exists on Windows, so typing fails on other platforms
+            message += f" ({ctypes.WinError()})"  # type: ignore[attr-defined]
         super().__init__(message)
```