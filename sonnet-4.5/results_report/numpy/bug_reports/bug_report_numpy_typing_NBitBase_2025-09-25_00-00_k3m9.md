# Bug Report: numpy.typing NBitBase Deprecation Warning Not Shown

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The deprecation warning for `NBitBase` is never shown to users because `NBitBase` is imported directly into the module namespace, preventing the `__getattr__` deprecation logic from executing.

## Property-Based Test

```python
import warnings
from hypothesis import given, strategies as st, settings
import pytest
import numpy.typing as npt


def test_nbitbase_access_emits_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "NBitBase" in str(w[0].message)
        assert "deprecated" in str(w[0].message).lower()
```

**Failing input**: Accessing `npt.NBitBase` (any access)

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase

    if len(w) == 0:
        print("BUG: No deprecation warning was emitted")
    else:
        print(f"OK: {len(w)} warning(s) emitted")
```

Output:
```
BUG: No deprecation warning was emitted
```

## Why This Is A Bug

The code in `numpy/typing/__init__.py` contains special handling in `__getattr__` (lines 172-184) to emit a deprecation warning when `NBitBase` is accessed:

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

However, `NBitBase` is also imported directly at line 160:
```python
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
```

Since `NBitBase` exists in the module's `__dict__`, Python's normal attribute lookup finds it before ever calling `__getattr__`. The `__getattr__` method is only called when an attribute is NOT found through normal lookup, making the deprecation warning dead code.

This violates the documented intent (NumPy 2.3 deprecation notice in docstrings and comments) and prevents users from being properly warned about the deprecation.

## Fix

Remove `NBitBase` from the direct import and `__all__` list, letting `__getattr__` handle it exclusively:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,9 +157,9 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray

-__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]
+__all__ = ["ArrayLike", "DTypeLike", "NDArray", "NBitBase"]


 __DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
@@ -172,6 +172,7 @@ def __dir__() -> list[str]:
 def __getattr__(name: str):
     if name == "NBitBase":
         import warnings
+        from numpy._typing import NBitBase

         # Deprecated in NumPy 2.3, 2025-05-01
         warnings.warn(