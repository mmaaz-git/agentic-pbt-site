# Bug Report: xarray.compat.npcompat.isdtype isinstance() argument type error

**Target**: `xarray.compat.npcompat.isdtype`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `isdtype` function crashes with a TypeError when called with a numpy scalar value and a string kind, because it passes a set to `isinstance()` which requires a tuple of types.

## Property-Based Test

While this bug was discovered through static analysis, here's a test that would fail:

```python
from hypothesis import given, strategies as st
import numpy as np
from xarray.compat.npcompat import isdtype

@given(st.integers(min_value=-100, max_value=100))
def test_isdtype_with_numpy_scalars(value):
    scalar = np.int64(value)
    result = isdtype(scalar, "integral")
    assert result is True
```

**Failing input**: Any numpy scalar value with a string kind like `"integral"`, `"numeric"`, etc.

## Reproducing the Bug

```python
import numpy as np
from xarray.compat.npcompat import isdtype

scalar = np.int64(5)
result = isdtype(scalar, "integral")
```

Output:
```
TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
```

## Why This Is A Bug

The `isdtype` function is a compatibility shim for numpy < 2.0, implementing numpy 2.0's `isdtype` function. It has two code paths:

1. For numpy scalars (instances of `np.generic`): uses `isinstance()`
2. For dtype objects: uses `np.issubdtype()`

In the first path (line 73 in npcompat.py), the code does:

```python
translated_kinds = {kind_mapping[k] for k in str_kinds} | type_kinds
if isinstance(dtype, np.generic):
    return isinstance(dtype, translated_kinds)  # BUG: translated_kinds is a set!
```

The `translated_kinds` variable is a set of type objects, but `isinstance()` requires its second argument to be a type or tuple of types, not a set. This causes a TypeError when checking numpy scalar values.

This bug affects any call to `isdtype` where:
- The first argument is a numpy scalar (e.g., `np.int64(5)`, `np.float32(3.14)`)
- The second argument is a string kind (e.g., `"integral"`, `"numeric"`)

## Fix

```diff
--- a/xarray/compat/npcompat.py
+++ b/xarray/compat/npcompat.py
@@ -70,7 +70,7 @@ def isdtype(
     # verified the dtypes already, no need to check again
     translated_kinds = {kind_mapping[k] for k in str_kinds} | type_kinds
     if isinstance(dtype, np.generic):
-        return isinstance(dtype, translated_kinds)
+        return isinstance(dtype, tuple(translated_kinds))
     else:
         return any(np.issubdtype(dtype, k) for k in translated_kinds)
```

The fix converts the set to a tuple before passing it to `isinstance()`. This maintains the same checking logic while using the correct type for the second argument.