# Bug Report: scipy.datasets._download_all.main() Crashes With AttributeError When Pooch Not Installed

**Target**: `scipy.datasets._download_all.main`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `main()` function in scipy.datasets._download_all crashes with `AttributeError: 'NoneType' object has no attribute 'os_cache'` when the optional dependency pooch is not installed, instead of raising the intended ImportError with installation instructions.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for scipy.datasets._download_all.main() missing pooch handling."""

from hypothesis import given, strategies as st, settings
import sys
from unittest.mock import patch

@given(st.booleans())
@settings(max_examples=5, verbosity=2, deadline=None)
def test_main_handles_missing_pooch(pooch_installed):
    """Property: main() should give clear error when pooch is missing, not crash."""

    # Set up pooch based on the test parameter
    pooch_value = None if not pooch_installed else __import__('pooch')

    with patch.dict('sys.modules', {'pooch': pooch_value}):
        try:
            # Clear any cached imports
            if 'scipy.datasets._download_all' in sys.modules:
                del sys.modules['scipy.datasets._download_all']

            # Try to import and use main()
            from scipy.datasets._download_all import main

            # Try to call main() with empty args to trigger any potential error
            with patch('sys.argv', ['test']):
                result = main()

            # If pooch is not installed, we should never get here
            if not pooch_installed:
                print(f"UNEXPECTED: Import and call succeeded when pooch_installed={pooch_installed}")
                print(f"main() returned: {result}")
                assert False, "Should have crashed or raised ImportError when pooch is not installed"

        except (ImportError, AttributeError) as e:
            # Check that we got the right error type
            if not pooch_installed and isinstance(e, AttributeError):
                # This is the bug: we get AttributeError instead of ImportError
                print(f"BUG FOUND: Got AttributeError '{e}' instead of ImportError")
                assert False, f"Wrong error type when pooch not installed: AttributeError instead of ImportError. Message: {e}"
```

<details>

<summary>
**Failing input**: `pooch_installed=False`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/__init__.py:44: UserWarning: The NumPy module was reloaded (imported a second time). This can in some cases result in small but subtle issues and is discouraged.
  from numpy import __version__ as __numpy_version__
Running Hypothesis test for scipy.datasets._download_all.main() pooch handling
----------------------------------------------------------------------
Trying example: test_main_handles_missing_pooch(
    pooch_installed=False,
)
BUG FOUND: Got AttributeError ''NoneType' object has no attribute 'os_cache'' instead of ImportError
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 27, in test_main_handles_missing_pooch
    result = main()
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/datasets/_download_all.py", line 64, in main
    default=pooch.os_cache('scipy-data'),
            ^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'os_cache'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 40, in test_main_handles_missing_pooch
    assert False, f"Wrong error type when pooch not installed: AttributeError instead of ImportError. Message: {e}"
           ^^^^^
AssertionError: Wrong error type when pooch not installed: AttributeError instead of ImportError. Message: 'NoneType' object has no attribute 'os_cache'

BUG FOUND: Got AttributeError ''NoneType' object has no attribute 'os_cache'' instead of ImportError
Test failed as expected, demonstrating the bug:
  Wrong error type when pooch not installed: AttributeError instead of ImportError. Message: 'NoneType' object has no attribute 'os_cache'

This confirms the bug: main() raises AttributeError instead of ImportError
when pooch is not installed.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the scipy.datasets._download_all main() crash when pooch is not installed."""

import sys
import os

# Ensure we're in a clean state without pooch
pooch_backup = sys.modules.get('pooch')
try:
    # Remove pooch from sys.modules to simulate it not being installed
    sys.modules['pooch'] = None

    # Now try to use the scipy.datasets._download_all.main() function
    from scipy.datasets._download_all import main
    import argparse

    print("Testing scipy.datasets._download_all.main() without pooch installed...")
    print("-" * 70)

    try:
        # This should fail gracefully with a clear ImportError message
        # about needing to install pooch, but instead it crashes with AttributeError
        main()
        print("ERROR: main() succeeded when it should have failed!")
    except AttributeError as e:
        print(f"AttributeError raised (WRONG ERROR TYPE): {e}")
        print(f"Full error type: {type(e).__name__}")
    except ImportError as e:
        print(f"ImportError raised (CORRECT): {e}")
        print(f"Full error type: {type(e).__name__}")
    except SystemExit as e:
        # argparse might exit if arguments are wrong
        print(f"SystemExit raised: {e}")
    except Exception as e:
        print(f"Unexpected error type {type(e).__name__}: {e}")

    print("-" * 70)
    print("\nEXPECTED: ImportError with message about installing pooch")
    print("ACTUAL: AttributeError: 'NoneType' object has no attribute 'os_cache'")

finally:
    # Restore original pooch module if it existed
    if pooch_backup:
        sys.modules['pooch'] = pooch_backup
    elif 'pooch' in sys.modules:
        del sys.modules['pooch']
```

<details>

<summary>
AttributeError: 'NoneType' object has no attribute 'os_cache'
</summary>
```
Testing scipy.datasets._download_all.main() without pooch installed...
----------------------------------------------------------------------
AttributeError raised (WRONG ERROR TYPE): 'NoneType' object has no attribute 'os_cache'
Full error type: AttributeError
----------------------------------------------------------------------

EXPECTED: ImportError with message about installing pooch
ACTUAL: AttributeError: 'NoneType' object has no attribute 'os_cache'
```
</details>

## Why This Is A Bug

This violates expected behavior because the code already has proper error handling for missing pooch, but it never gets executed. The module's design treats pooch as an optional dependency:

1. **Lines 10-13** catch ImportError and set `pooch = None` when the import fails
2. **Lines 47-50** in `download_all()` check if pooch is None and raise a clear ImportError: "Missing optional dependency 'pooch' required for scipy.datasets module. Please use pip or conda to install 'pooch'."
3. **Line 64** prematurely evaluates `pooch.os_cache('scipy-data')` during argparse setup, causing AttributeError before the proper error handling can execute

The documentation states this script can be run as `python _download_all.py <download_dir>`, but this crashes unhelpfully when pooch isn't installed. Users get a confusing AttributeError about NoneType instead of clear installation instructions.

## Relevant Context

The bug occurs in `/home/npc/.local/lib/python3.13/site-packages/scipy/datasets/_download_all.py:64` when creating the argument parser. The same error pattern would affect any user who:

- Runs `python -m scipy.datasets._download_all` without pooch installed
- Imports and calls `scipy.datasets._download_all.main()` directly
- Uses the script form `python _download_all.py` for dataset management

The `download_all()` function works correctly when called directly, as it properly checks for pooch before use. Only the `main()` entry point has this issue.

Documentation: https://docs.scipy.org/doc/scipy/reference/datasets.html

## Proposed Fix

```diff
--- a/scipy/datasets/_download_all.py
+++ b/scipy/datasets/_download_all.py
@@ -61,9 +61,13 @@ def download_all(path=None):
 def main():
     parser = argparse.ArgumentParser(description='Download SciPy data files.')
     parser.add_argument("path", nargs='?', type=str,
-                        default=pooch.os_cache('scipy-data'),
+                        default=None,
                         help="Directory path to download all the data files.")
     args = parser.parse_args()
-    download_all(args.path)
+    # Only use pooch.os_cache() if path not provided and pooch is available
+    if args.path is None and pooch is not None:
+        args.path = pooch.os_cache('scipy-data')
+    # download_all() will handle the case where pooch is None
+    download_all(args.path)


```