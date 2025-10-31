# Bug Report: numpy.typing NBitBase Deprecation Warning Never Triggers

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `numpy.typing` module contains a `__getattr__` implementation intended to emit a deprecation warning when accessing `NBitBase`. However, because `NBitBase` is imported at module level and exists in the module's namespace, Python's attribute lookup finds it directly and never invokes `__getattr__`, causing the deprecation warning to never be shown to users.

## Property-Based Test

```python
import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st


def test_nbitbase_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "NBitBase" in str(w[0].message)
```

**Failing input**: `npt.NBitBase` (direct attribute access)

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    obj = npt.NBitBase
    print(f"Warnings caught: {len(w)}")
    print(f"Expected: 1 DeprecationWarning")
```

## Why This Is A Bug

The code in `/numpy/typing/__init__.py` imports `NBitBase` at line 160:
```python
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
```

Then defines `__getattr__` at lines 172-184 to emit a deprecation warning:
```python
def __getattr__(name: str):
    if name == "NBitBase":
        import warnings
        warnings.warn(
            "`NBitBase` is deprecated and will be removed from numpy.typing in the "
            "future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper "
            "bound, instead. (deprecated in NumPy 2.3)",
            DeprecationWarning,
            stacklevel=2,
        )
        return NBitBase
    # ...
```

Python only calls `__getattr__` when normal attribute lookup fails. Since `NBitBase` is in the module's `__dict__`, the standard lookup succeeds and `__getattr__` is never invoked. This violates the documented contract that accessing `NBitBase` should trigger a deprecation warning (deprecated in NumPy 2.3, to be removed 2025-05-01).

## Fix

Remove `NBitBase` from the top-level import so that accessing it triggers `__getattr__`:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,7 +157,8 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray
+from numpy._typing import NBitBase as _NBitBase

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -181,7 +182,7 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase

     if name in __DIR_SET:
         return globals()[name]
```

This ensures `NBitBase` is not in the module's public namespace, forcing attribute access to go through `__getattr__` which will emit the deprecation warning as intended.