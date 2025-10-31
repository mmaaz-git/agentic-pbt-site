# Bug Report: numpy.typing NBitBase Deprecation Warning Never Fires

**Target**: `numpy.typing.__getattr__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `__getattr__` function in `numpy.typing` contains code to issue a deprecation warning when accessing `NBitBase`, but this warning will never fire because `NBitBase` is already in the module's global namespace via a direct import.

## Property-Based Test

```python
import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_deprecated_attributes_trigger_warnings(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)
        assert len(w) >= 1, f"Expected deprecation warning for {attr_name} but got none"
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
```

**Failing input**: `"NBitBase"`

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    assert len(w) == 0

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__('NBitBase')
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
```

## Why This Is A Bug

In Python, `__getattr__` is only called when an attribute is NOT found through normal attribute lookup. The `numpy.typing` module imports `NBitBase` directly:

```python
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]
```

This makes `NBitBase` available in the module's `globals()`, so accessing `npt.NBitBase` finds it immediately without calling `__getattr__`. The deprecation warning code in `__getattr__` is therefore unreachable, defeating the intended deprecation notice to users.

## Fix

Remove `NBitBase` from the direct import and `__all__`, so that accessing it will go through `__getattr__`:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -1,6 +1,6 @@
-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray
+from numpy._typing import NBitBase as _NBitBase_imported

-__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]
+__all__ = ["ArrayLike", "DTypeLike", "NDArray"]


 __DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
@@ -13,7 +13,7 @@ def __dir__() -> list[str]:
 def __getattr__(name: str):
     if name == "NBitBase":
         import warnings

         # Deprecated in NumPy 2.3, 2025-05-01
         warnings.warn(
             "`NBitBase` is deprecated and will be removed from numpy.typing in the "
             "future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper "
             "bound, instead. (deprecated in NumPy 2.3)",
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase_imported

     if name in __DIR_SET:
         return globals()[name]
```