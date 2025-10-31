# Bug Report: scipy.datasets._download_all.main() Crashes with AttributeError Instead of Helpful ImportError When Pooch Not Installed

**Target**: `scipy.datasets._download_all.main()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When running `scipy/datasets/_download_all.py` as a standalone script without the optional `pooch` dependency installed, the `main()` function crashes with a confusing `AttributeError` instead of providing the intended helpful `ImportError` with installation instructions.

## Property-Based Test

```python
"""
Hypothesis test to verify scipy.datasets._download_all.main()
provides helpful error messages when pooch is not installed.
"""
from hypothesis import given, strategies as st, example
import argparse
import sys
import io
from contextlib import redirect_stdout


class MockPooch:
    """Mock pooch module with os_cache method"""
    @staticmethod
    def os_cache(name):
        return f"/mock/cache/{name}"


@given(st.booleans())
@example(False)  # Explicitly test the failing case
def test_main_gives_helpful_error_without_pooch(has_pooch):
    """Test that main() gives a helpful ImportError when pooch is missing"""
    # Set up pooch as either a mock or None
    pooch = MockPooch() if has_pooch else None

    def download_all(path=None):
        """Same logic as scipy.datasets._download_all.download_all"""
        if pooch is None:
            raise ImportError("Missing optional dependency 'pooch' required "
                              "for scipy.datasets module. Please use pip or "
                              "conda to install 'pooch'.")
        if path is None:
            path = pooch.os_cache('scipy-data')
        return f"Would download to: {path}"

    def main():
        """Same logic as scipy.datasets._download_all.main"""
        parser = argparse.ArgumentParser(description='Download SciPy data files.')
        parser.add_argument("path", nargs='?', type=str,
                            default=pooch.os_cache('scipy-data'),  # BUG: This line causes AttributeError when pooch is None
                            help="Directory path to download all the data files.")
        args = parser.parse_args([])  # Empty args for testing
        return download_all(args.path)

    if has_pooch:
        # With pooch, it should work fine
        try:
            # Capture output to avoid test noise
            result = main()
            # Test passes - main() executed without error
        except Exception as e:
            assert False, f"Should not have raised an error with pooch installed, got: {e}"
    else:
        # Without pooch, we expect a helpful ImportError
        try:
            main()
            assert False, "Should have raised an error without pooch"
        except ImportError as e:
            # Good! Got the helpful error
            assert "pooch" in str(e).lower(), f"ImportError should mention 'pooch', got: {e}"
            assert "pip" in str(e).lower() or "conda" in str(e).lower(), f"ImportError should mention installation method, got: {e}"
        except AttributeError as e:
            # Bad! Got confusing AttributeError instead
            assert False, f"Got confusing AttributeError instead of helpful ImportError: {e}"
        except Exception as e:
            assert False, f"Got unexpected error type {type(e).__name__}: {e}"


# Run the hypothesis test
if __name__ == "__main__":
    print("Running Hypothesis test for scipy.datasets._download_all.main() error handling...")
    print("=" * 70)
    test_main_gives_helpful_error_without_pooch()
```

<details>

<summary>
**Failing input**: `has_pooch=False`
</summary>
```
Running Hypothesis test for scipy.datasets._download_all.main() error handling...
======================================================================
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 56, in test_main_gives_helpful_error_without_pooch
    main()
    ~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 40, in main
    default=pooch.os_cache('scipy-data'),  # BUG: This line causes AttributeError when pooch is None
            ^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'os_cache'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 73, in <module>
    test_main_gives_helpful_error_without_pooch()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 20, in test_main_gives_helpful_error_without_pooch
    @example(False)  # Explicitly test the failing case
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 64, in test_main_gives_helpful_error_without_pooch
    assert False, f"Got confusing AttributeError instead of helpful ImportError: {e}"
           ^^^^^
AssertionError: Got confusing AttributeError instead of helpful ImportError: 'NoneType' object has no attribute 'os_cache'
Falsifying explicit example: test_main_gives_helpful_error_without_pooch(
    has_pooch=False,
)
```
</details>

## Reproducing the Bug

```python
"""
Minimal reproduction of the scipy.datasets._download_all bug
when pooch is not installed.
"""
import argparse

# Simulate pooch not being installed
pooch = None


def download_all(path=None):
    """Same logic as scipy.datasets._download_all.download_all"""
    if pooch is None:
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")
    if path is None:
        path = pooch.os_cache('scipy-data')
    print(f"Would download to: {path}")


def main():
    """Same logic as scipy.datasets._download_all.main"""
    parser = argparse.ArgumentParser(description='Download SciPy data files.')
    parser.add_argument("path", nargs='?', type=str,
                        default=pooch.os_cache('scipy-data'),  # BUG: This line causes AttributeError
                        help="Directory path to download all the data files.")
    args = parser.parse_args()
    download_all(args.path)


if __name__ == "__main__":
    try:
        main()
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except ImportError as e:
        print(f"ImportError: {e}")
```

<details>

<summary>
AttributeError: 'NoneType' object has no attribute 'os_cache'
</summary>
```
AttributeError: 'NoneType' object has no attribute 'os_cache'
```
</details>

## Why This Is A Bug

This bug violates the error handling contract that SciPy developers explicitly implemented. The code at `scipy/datasets/_download_all.py` lines 47-50 contains carefully crafted error handling specifically designed to detect when `pooch` is not installed and provide a helpful `ImportError` with installation instructions:

```python
if pooch is None:
    raise ImportError("Missing optional dependency 'pooch' required "
                      "for scipy.datasets module. Please use pip or "
                      "conda to install 'pooch'.")
```

However, this error handling never executes because Python evaluates the default argument `pooch.os_cache('scipy-data')` when creating the argument parser on line 64, before the `download_all()` function can check if `pooch` is `None`. This results in users seeing a cryptic `AttributeError: 'NoneType' object has no attribute 'os_cache'` instead of the intended helpful message explaining how to install the missing dependency.

The script's docstring explicitly states it's meant to be run directly: "Run: python _download_all.py <download_dir>", indicating this is a supported use case for packagers who need to pre-download data files in restricted build environments where external downloads may be forbidden.

## Relevant Context

The `_download_all.py` script is a utility specifically designed for SciPy packagers (e.g., Linux distributions) who need to pre-download dataset files before building SciPy packages. In many build environments, external network access is restricted or forbidden, so packagers need to download these files separately.

The script location: `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/datasets/_download_all.py`

The bug occurs at line 64 where the default value for the path argument is evaluated immediately:
```python
parser.add_argument("path", nargs='?', type=str,
                    default=pooch.os_cache('scipy-data'),  # Line 64
                    help="Directory path to download all the data files.")
```

When `pooch` is not installed, it's set to `None` on line 13, causing the AttributeError when Python tries to call `None.os_cache()`.

## Proposed Fix

```diff
--- a/scipy/datasets/_download_all.py
+++ b/scipy/datasets/_download_all.py
@@ -60,8 +60,11 @@ def download_all(path=None):

 def main():
     parser = argparse.ArgumentParser(description='Download SciPy data files.')
+    # Only evaluate default if pooch is available
+    default_path = pooch.os_cache('scipy-data') if pooch is not None else None
     parser.add_argument("path", nargs='?', type=str,
-                        default=pooch.os_cache('scipy-data'),
+                        default=default_path,
                         help="Directory path to download all the data files.")
     args = parser.parse_args()
+    # Let download_all handle the missing pooch error
     download_all(args.path)
```