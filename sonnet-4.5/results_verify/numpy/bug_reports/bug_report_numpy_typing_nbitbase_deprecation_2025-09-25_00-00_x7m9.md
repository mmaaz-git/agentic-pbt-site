# Bug Report: numpy.typing NBitBase Deprecation Warning Never Triggers

**Target**: `numpy.typing.__getattr__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `NBitBase` deprecation warning defined in `numpy.typing.__getattr__` never triggers because `NBitBase` is imported at module level, bypassing `__getattr__` entirely.

## Property-Based Test

```python
import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_deprecated_attribute_warns(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        getattr(npt, attr_name)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
```

**Failing input**: `"NBitBase"`

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    obj = npt.NBitBase
    print(f"Warnings captured: {len(w)}")
    print(f"Expected: 1, Actual: {len(w)}")
```

## Why This Is A Bug

The `__getattr__` function contains explicit code to show a deprecation warning when `NBitBase` is accessed:

```python
def __getattr__(name: str):
    if name == "NBitBase":
        import warnings
        warnings.warn(
            "`NBitBase` is deprecated...",
            DeprecationWarning,
            stacklevel=2,
        )
        return NBitBase
```

However, since `NBitBase` is imported at module level (`from numpy._typing import ... NBitBase`), Python finds it in the module's `__dict__` during normal attribute lookup, and `__getattr__` is never called. This violates the documented intent to deprecate `NBitBase`.

## Fix

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -159,7 +159,7 @@
 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray
+from numpy._typing import NBitBase as _NBitBase

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -176,7 +176,7 @@
             "bound, instead. (deprecated in NumPy 2.3)",
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase

     if name in __DIR_SET:
         return globals()[name]
```