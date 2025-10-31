# Bug Report: numpy.typing NBitBase Deprecation Warning Not Emitted

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `NBitBase` attribute is documented as deprecated with an intended deprecation warning, but accessing `numpy.typing.NBitBase` does not emit any warning because the attribute is directly imported into the module namespace, bypassing the `__getattr__` hook that should emit the warning.

## Property-Based Test

```python
import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st
import pytest


def test_nbitbase_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "NBitBase" in str(w[0].message)
```

**Failing input**: Accessing `npt.NBitBase` (any access)

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    print(f"Warnings emitted: {len(w)}")
    assert len(w) == 1, f"Expected 1 DeprecationWarning, got {len(w)}"
```

## Why This Is A Bug

The module's `__getattr__` function (lines 172-184 in `numpy/typing/__init__.py`) explicitly handles the "NBitBase" attribute to emit a deprecation warning:

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
```

However, at line 160, `NBitBase` is imported directly into the module namespace:

```python
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
```

This means `NBitBase` exists in `globals()`, so Python's attribute lookup finds it immediately without ever calling `__getattr__`. The deprecation mechanism is completely bypassed, violating the documented deprecation behavior.

## Fix

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,10 +157,13 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray
+from numpy._typing import NBitBase as _NBitBase_impl

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

+del _NBitBase_impl
+NBitBase = None

 __DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
 __DIR_SET = frozenset(__DIR)
@@ -181,7 +184,7 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        from numpy._typing import NBitBase
+        return NBitBase

     if name in __DIR_SET:
         return globals()[name]
```