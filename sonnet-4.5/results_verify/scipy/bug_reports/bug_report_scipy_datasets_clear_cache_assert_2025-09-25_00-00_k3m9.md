# Bug Report: scipy.datasets.clear_cache Uses Assert for Input Validation

**Target**: `scipy.datasets.clear_cache`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clear_cache` function uses an `assert` statement for input validation, which can be bypassed when Python is run with the `-O` optimization flag. This causes confusing error messages when invalid inputs are provided in optimized mode.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import sys

@given(st.text())
def test_clear_cache_validates_non_callable(non_callable_str):
    """Property: clear_cache should reject non-callable inputs regardless of optimization level."""
    import tempfile
    import os
    from scipy.datasets._utils import _clear_cache

    method_map = {"test": ["test.dat"]}
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _clear_cache([non_callable_str], cache_dir=tmpdir, method_map=method_map)
            assert False, "Should have raised ValueError for non-callable"
        except (ValueError, TypeError, AssertionError, AttributeError):
            pass
```

**Failing input**: Any non-callable value when Python is run with `-O` flag

## Reproducing the Bug

```python
import sys
import tempfile
import os

sys.path.insert(0, '/path/to/scipy')
from scipy.datasets._utils import _clear_cache

method_map = {"test": ["test.dat"]}
with tempfile.TemporaryDirectory() as tmpdir:
    test_file = os.path.join(tmpdir, "test.dat")
    with open(test_file, 'w') as f:
        f.write("test")

    non_callable = "not_a_function"

    try:
        _clear_cache([non_callable], cache_dir=tmpdir, method_map=method_map)
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
```

When run normally:
```
Error type: AssertionError
Error message:
```

When run with `python -O`:
```
Error type: AttributeError
Error message: 'str' object has no attribute '__name__'
```

## Why This Is A Bug

1. **Assertion bypass**: Python's `-O` flag disables all assert statements, bypassing the validation
2. **Poor error messages**: When assertions are disabled, users get cryptic `AttributeError: 'str' object has no attribute '__name__'` instead of a clear validation error
3. **PEP 8 violation**: PEP 8 explicitly states "Don't use assertions for data validation" because they can be turned off
4. **API contract violation**: The function should consistently validate inputs regardless of optimization level

## Fix

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -33,7 +33,9 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
             # single dataset method passed should be converted to list
             datasets = [datasets, ]
         for dataset in datasets:
-            assert callable(dataset)
+            if not callable(dataset):
+                raise TypeError(f"Expected callable dataset method, "
+                               f"got {type(dataset).__name__}")
             dataset_name = dataset.__name__  # Name of the dataset method
             if dataset_name not in method_map:
                 raise ValueError(f"Dataset method {dataset_name} doesn't "
```

This ensures proper input validation that cannot be bypassed and provides a clear error message.