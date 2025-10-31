# Bug Report: pandas.compat.numpy CompatValidator Method Validation Bypass

**Target**: `pandas.compat.numpy.function.CompatValidator.__call__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CompatValidator silently accepts invalid method values when called with empty args and kwargs, bypassing validation that occurs with non-empty inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
from pandas.compat.numpy.function import CompatValidator


@given(method=st.text(min_size=1).filter(lambda x: x not in ["args", "kwargs", "both"]))
def test_compatvalidator_rejects_invalid_methods(method):
    validator = CompatValidator({}, method=method)
    with pytest.raises(ValueError, match="invalid validation method"):
        validator((), {})
```

**Failing input**: `method='0'`

## Reproducing the Bug

```python
from pandas.compat.numpy.function import CompatValidator

validator = CompatValidator({}, method="invalid_method")

validator((), {})

validator((1,), {})
```

Expected: Both calls should raise `ValueError: invalid validation method 'invalid_method'`

Actual: First call returns None silently, second call raises ValueError as expected.

## Why This Is A Bug

The CompatValidator.__call__ method (lines 64-92 in function.py) has an early return when both args and kwargs are empty (lines 72-73), before method validation occurs (lines 83-92). This creates inconsistent behavior where configuration errors are only caught when there's actual data to validate.

While the practical impact is low (all built-in validators use valid methods), this violates the principle of fail-fast validation and could hide configuration bugs during development.

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -69,9 +69,6 @@ class CompatValidator:
         method: str | None = None,
     ) -> None:
-        if not args and not kwargs:
-            return None
-
         fname = self.fname if fname is None else fname
         max_fname_arg_count = (
             self.max_fname_arg_count
@@ -80,6 +77,9 @@ class CompatValidator:
         )
         method = self.method if method is None else method

+        if not args and not kwargs:
+            return None
+
         if method == "args":
             validate_args(fname, args, max_fname_arg_count, self.defaults)
         elif method == "kwargs":
```

Alternatively, validate the method parameter during __init__ to catch invalid methods at construction time.