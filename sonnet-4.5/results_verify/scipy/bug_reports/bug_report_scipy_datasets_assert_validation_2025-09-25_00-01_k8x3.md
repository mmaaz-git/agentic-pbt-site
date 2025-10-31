# Bug Report: scipy.datasets._clear_cache() Uses Assert for Input Validation

**Target**: `scipy.datasets._utils._clear_cache()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`_clear_cache()` uses `assert callable(dataset)` for input validation, which can be disabled with Python's `-O` flag, leading to unpredictable failures.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pytest
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map

@given(datasets_input=st.one_of(
    st.integers(),
    st.text(),
    st.dictionaries(st.text(), st.integers())
))
def test_clear_cache_invalid_types(datasets_input):
    assume(not callable(datasets_input))

    with pytest.raises((AttributeError, AssertionError, TypeError)):
        _clear_cache(datasets_input, cache_dir="/tmp/test", method_map=method_files_map)
```

**Failing input**: Any non-callable value like `"not_callable"` or `42`

## Reproducing the Bug

```python
from scipy.datasets._utils import _clear_cache

_clear_cache("not_a_callable", cache_dir="/tmp/test")
```

Output:
```
AssertionError
```

With `python -O`:
```
AttributeError: 'str' object has no attribute '__name__'
```

## Why This Is A Bug

Using `assert` for input validation violates Python best practices because:

1. Assertions can be globally disabled with the `-O` (optimize) flag
2. When disabled, the code proceeds to line 37: `dataset_name = dataset.__name__`
3. This causes an unhelpful `AttributeError` instead of a clear validation error
4. The behavior is inconsistent between normal and optimized execution

The Python documentation explicitly states: "Assertions are not a substitute for proper input validation."

## Fix

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -33,7 +33,10 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
             # single dataset method passed should be converted to list
             datasets = [datasets, ]
         for dataset in datasets:
-            assert callable(dataset)
+            if not callable(dataset):
+                raise TypeError(f"Expected callable dataset method, "
+                                f"got {type(dataset).__name__}: {dataset!r}")
             dataset_name = dataset.__name__  # Name of the dataset method
             if dataset_name not in method_map:
                 raise ValueError(f"Dataset method {dataset_name} doesn't "
```