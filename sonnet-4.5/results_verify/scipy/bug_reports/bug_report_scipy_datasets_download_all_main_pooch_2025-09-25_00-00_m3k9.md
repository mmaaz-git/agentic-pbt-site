# Bug Report: scipy.datasets._download_all main() Crashes When Pooch Not Installed

**Target**: `scipy.datasets._download_all.main`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `main()` function in `_download_all.py` crashes with `AttributeError` when pooch is not installed, instead of providing a clear error message. The bug occurs because the default argument to `argparse` calls `pooch.os_cache()` at parse time, before the `download_all()` function can check if pooch is installed.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.booleans())
def test_main_handles_missing_pooch(pooch_installed):
    """Property: main() should give clear error when pooch is missing, not crash."""
    import sys
    from unittest.mock import patch

    pooch_value = None if not pooch_installed else __import__('pooch')

    with patch.dict('sys.modules', {'pooch': pooch_value}):
        try:
            from scipy.datasets._download_all import main
            if not pooch_installed:
                assert False, "Should have crashed or raised ImportError"
        except (ImportError, AttributeError) as e:
            if not pooch_installed and isinstance(e, AttributeError):
                assert False, f"Wrong error type: {e}"
```

**Failing input**: `pooch_installed=False`

## Reproducing the Bug

```python
import sys

pooch_backup = sys.modules.get('pooch')
try:
    sys.modules['pooch'] = None

    from scipy.datasets._download_all import main
    import argparse

    try:
        main()
    except AttributeError as e:
        print(f"Error: {e}")
        print("Expected: ImportError with clear message about missing pooch")
        print("Got: AttributeError about NoneType")
finally:
    if pooch_backup:
        sys.modules['pooch'] = pooch_backup
```

Output:
```
Error: 'NoneType' object has no attribute 'os_cache'
Expected: ImportError with clear message about missing pooch
Got: AttributeError about NoneType
```

## Why This Is A Bug

1. **Confusing error message**: Users see `'NoneType' object has no attribute 'os_cache'` instead of a clear message about installing pooch
2. **Premature evaluation**: The default argument `pooch.os_cache('scipy-data')` is evaluated when the argument parser is created, not when it's needed
3. **Inconsistent error handling**: The `download_all()` function has proper error handling for missing pooch, but `main()` crashes before it can be called
4. **Bad UX**: When run as a script (`python _download_all.py`), the error is cryptic

The issue is on line 64:
```python
parser.add_argument("path", nargs='?', type=str,
                    default=pooch.os_cache('scipy-data'),  # Crashes here if pooch is None!
                    help="Directory path to download all the data files.")
```

## Fix

```diff
--- a/scipy/datasets/_download_all.py
+++ b/scipy/datasets/_download_all.py
@@ -61,9 +61,10 @@ def download_all(path=None):

 def main():
     parser = argparse.ArgumentParser(description='Download SciPy data files.')
     parser.add_argument("path", nargs='?', type=str,
-                        default=pooch.os_cache('scipy-data'),
+                        default=None,
                         help="Directory path to download all the data files.")
     args = parser.parse_args()
-    download_all(args.path)
+    path = args.path if args.path is not None else (
+        pooch.os_cache('scipy-data') if pooch is not None else None)
+    download_all(path)
```

This defers the `pooch.os_cache()` call until after argument parsing, allowing `download_all()` to handle the missing pooch dependency with a clear error message.