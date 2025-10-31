# Bug Report: scipy.datasets.clear_cache Bypasses Dataset Validation When Cache Directory Doesn't Exist

**Target**: `scipy.datasets._utils._clear_cache`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_clear_cache()` function skips dataset validation when the cache directory doesn't exist, allowing invalid datasets to pass without raising the expected `ValueError`.

## Property-Based Test

```python
import tempfile
import os
import pytest
from hypothesis import given, strategies as st
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map


@given(st.booleans())
def test_clear_cache_validates_dataset_before_checking_cache_exists(cache_exists):
    """
    Property: _clear_cache should validate dataset parameters consistently,
    regardless of whether the cache directory exists or not.

    This test checks that invalid datasets raise ValueError both when
    the cache directory exists and when it doesn't exist.
    """
    def fake_dataset():
        """A dataset function that doesn't exist in the registry"""
        pass
    fake_dataset.__name__ = "nonexistent_dataset"

    with tempfile.TemporaryDirectory() as tmpdir:
        if cache_exists:
            # Use the existing temp directory
            cache_dir = tmpdir
        else:
            # Use a non-existent subdirectory
            cache_dir = os.path.join(tmpdir, "nonexistent_cache_dir")

        print(f"\nTest with cache_exists={cache_exists}")
        print(f"Cache directory: {cache_dir}")
        print(f"Actual cache exists: {os.path.exists(cache_dir)}")

        # Both cases should raise ValueError for invalid dataset
        with pytest.raises(ValueError, match="Dataset method .* doesn't exist"):
            _clear_cache(
                datasets=[fake_dataset],
                cache_dir=cache_dir,
                method_map=method_files_map
            )
        print(f"✓ ValueError correctly raised when cache_exists={cache_exists}")


if __name__ == "__main__":
    # Run the property-based test
    test_clear_cache_validates_dataset_before_checking_cache_exists()
```

<details>

<summary>
**Failing input**: `cache_exists=False`
</summary>
```

Test with cache_exists=False
Cache directory: /tmp/tmpwod8ezx9/nonexistent_cache_dir
Actual cache exists: False
Cache Directory /tmp/tmpwod8ezx9/nonexistent_cache_dir doesn't exist. Nothing to clear.

Test with cache_exists=True
Cache directory: /tmp/tmp0q7eu0vc
Actual cache exists: True
✓ ValueError correctly raised when cache_exists=True

Test with cache_exists=False
Cache directory: /tmp/tmpapf8fmgp/nonexistent_cache_dir
Actual cache exists: False
Cache Directory /tmp/tmpapf8fmgp/nonexistent_cache_dir doesn't exist. Nothing to clear.
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 47, in <module>
    test_clear_cache_validates_dataset_before_checking_cache_exists()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 10, in test_clear_cache_validates_dataset_before_checking_cache_exists
    def test_clear_cache_validates_dataset_before_checking_cache_exists(cache_exists):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 36, in test_clear_cache_validates_dataset_before_checking_cache_exists
    with pytest.raises(ValueError, match="Dataset method .* doesn't exist"):
         ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE <class 'ValueError'>
Falsifying example: test_clear_cache_validates_dataset_before_checking_cache_exists(
    cache_exists=False,
)
```
</details>

## Reproducing the Bug

```python
import tempfile
import os
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map


def fake_dataset():
    """A fake dataset function that is not in the registry"""
    pass

fake_dataset.__name__ = "nonexistent_dataset"

print("Test Case 1: Invalid dataset with non-existent cache directory")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    nonexistent_cache = os.path.join(tmpdir, "nonexistent")

    print(f"Cache directory: {nonexistent_cache}")
    print(f"Cache exists: {os.path.exists(nonexistent_cache)}")
    print("\nCalling _clear_cache with invalid dataset...")

    try:
        _clear_cache(
            datasets=[fake_dataset],
            cache_dir=nonexistent_cache,
            method_map=method_files_map
        )
        print("\n[ERROR] No exception raised! Function returned silently.")
    except ValueError as e:
        print(f"\n[EXPECTED] ValueError raised: {e}")
    except Exception as e:
        print(f"\n[UNEXPECTED] Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test Case 2: Invalid dataset with existing cache directory")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    existing_cache = tmpdir

    print(f"Cache directory: {existing_cache}")
    print(f"Cache exists: {os.path.exists(existing_cache)}")
    print("\nCalling _clear_cache with invalid dataset...")

    try:
        _clear_cache(
            datasets=[fake_dataset],
            cache_dir=existing_cache,
            method_map=method_files_map
        )
        print("\n[ERROR] No exception raised! Function returned silently.")
    except ValueError as e:
        print(f"\n[EXPECTED] ValueError raised: {e}")
    except Exception as e:
        print(f"\n[UNEXPECTED] Exception raised: {type(e).__name__}: {e}")
```

<details>

<summary>
Test shows inconsistent validation behavior between existing and non-existing cache directories
</summary>
```
Test Case 1: Invalid dataset with non-existent cache directory
============================================================
Cache directory: /tmp/tmpxu27y8td/nonexistent
Cache exists: False

Calling _clear_cache with invalid dataset...
Cache Directory /tmp/tmpxu27y8td/nonexistent doesn't exist. Nothing to clear.

[ERROR] No exception raised! Function returned silently.

============================================================
Test Case 2: Invalid dataset with existing cache directory
============================================================
Cache directory: /tmp/tmpayvbvsfb
Cache exists: True

Calling _clear_cache with invalid dataset...

[EXPECTED] ValueError raised: Dataset method nonexistent_dataset doesn't exist. Please check if the passed dataset is a subset of the following dataset methods: ['ascent', 'electrocardiogram', 'face']
```
</details>

## Why This Is A Bug

This violates the principle of consistent error handling. The function contains explicit validation logic (lines 38-42 in `_utils.py`) that checks if dataset names exist in the registry and raises a descriptive `ValueError` when they don't. However, the early return at line 26 when the cache directory doesn't exist bypasses this validation entirely.

This creates an inconsistent API where:
- Invalid datasets correctly raise `ValueError` when the cache directory exists
- Invalid datasets silently pass when the cache directory doesn't exist

The validation is clearly intentional based on the detailed error message that guides users to valid options. This inconsistency violates the fail-fast principle and can lead to confusion when debugging, as the same invalid input produces different behavior based on an unrelated condition (cache directory existence).

## Relevant Context

The bug is located in `/scipy/datasets/_utils.py` at the `_clear_cache` function. The issue stems from the order of operations:

1. Line 24-26: Checks if cache directory exists and returns early if not
2. Line 38-42: Validates that datasets exist in the registry

The validation check in the registry (`method_files_map`) contains only three valid dataset methods: 'ascent', 'electrocardiogram', and 'face'. Any other dataset name should consistently raise a ValueError.

Documentation: The public `clear_cache()` function's docstring doesn't explicitly state that ValueError will be raised for invalid datasets, but the implementation clearly shows this was the intended behavior.

Code location: https://github.com/scipy/scipy/blob/main/scipy/datasets/_utils.py

## Proposed Fix

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -21,10 +21,6 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
                               "conda to install 'pooch'.")
         cache_dir = platformdirs.user_cache_dir("scipy-data")

-    if not os.path.exists(cache_dir):
-        print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
-        return
-
     if datasets is None:
-        print(f"Cleaning the cache directory {cache_dir}!")
-        shutil.rmtree(cache_dir)
+        if os.path.exists(cache_dir):
+            print(f"Cleaning the cache directory {cache_dir}!")
+            shutil.rmtree(cache_dir)
+        else:
+            print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
     else:
         if not isinstance(datasets, list | tuple):
             # single dataset method passed should be converted to list
             datasets = [datasets, ]
         for dataset in datasets:
             assert callable(dataset)
             dataset_name = dataset.__name__  # Name of the dataset method
             if dataset_name not in method_map:
                 raise ValueError(f"Dataset method {dataset_name} doesn't "
                                  "exist. Please check if the passed dataset "
                                  "is a subset of the following dataset "
                                  f"methods: {list(method_map.keys())}")

+        if not os.path.exists(cache_dir):
+            print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
+            return
+
             data_files = method_map[dataset_name]
             data_filepaths = [os.path.join(cache_dir, file)
                               for file in data_files]
```