# Bug Report: scipy.datasets._download_all.main() Crashes Without Pooch

**Target**: `scipy.datasets._download_all.main()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When running `_download_all.py` as a script without the optional `pooch` dependency installed, the `main()` function crashes with a confusing `AttributeError` instead of the helpful `ImportError` that `download_all()` would provide.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import argparse


@given(st.booleans())
def test_main_gives_helpful_error_without_pooch(has_pooch):
    pooch = MockPooch() if has_pooch else None

    if not has_pooch:
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("path", nargs='?', type=str,
                                default=pooch.os_cache('scipy-data'))

            assert False, "Should have raised an error"
        except ImportError as e:
            assert "pooch" in str(e).lower()
        except AttributeError:
            assert False, "Got confusing AttributeError instead of helpful ImportError"
```

**Failing input**: `has_pooch=False`

## Reproducing the Bug

```python
import argparse

pooch = None


def download_all(path=None):
    if pooch is None:
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")


def main():
    parser = argparse.ArgumentParser(description='Download SciPy data files.')
    parser.add_argument("path", nargs='?', type=str,
                        default=pooch.os_cache('scipy-data'),
                        help="Directory path to download all the data files.")
    args = parser.parse_args()
    download_all(args.path)


try:
    main()
except AttributeError as e:
    print(f"Got confusing error: {e}")
except ImportError as e:
    print(f"Got helpful error: {e}")
```

**Output:**
```
Got confusing error: 'NoneType' object has no attribute 'os_cache'
```

**Expected:**
```
Got helpful error: Missing optional dependency 'pooch' required for scipy.datasets module. Please use pip or conda to install 'pooch'.
```

## Why This Is A Bug

The `main()` function in `_download_all.py` (line 64) evaluates `pooch.os_cache('scipy-data')` when creating the argument parser, before the helpful error check in `download_all()` can run. When `pooch` is not installed (set to `None` on line 13), this causes an unhelpful `AttributeError` instead of the intended `ImportError` with installation instructions.

This violates the error message contract - users should get clear guidance about missing dependencies, not cryptic attribute errors.

## Fix

Move the default value evaluation to after the pooch availability check, or make it conditional:

```diff
--- a/_download_all.py
+++ b/_download_all.py
@@ -61,7 +61,10 @@ def download_all(path=None):

 def main():
     parser = argparse.ArgumentParser(description='Download SciPy data files.')
+    default_path = pooch.os_cache('scipy-data') if pooch is not None else None
     parser.add_argument("path", nargs='?', type=str,
-                        default=pooch.os_cache('scipy-data'),
+                        default=default_path,
                         help="Directory path to download all the data files.")
     args = parser.parse_args()
+    if args.path is None and pooch is not None:
+        args.path = pooch.os_cache('scipy-data')
     download_all(args.path)
```