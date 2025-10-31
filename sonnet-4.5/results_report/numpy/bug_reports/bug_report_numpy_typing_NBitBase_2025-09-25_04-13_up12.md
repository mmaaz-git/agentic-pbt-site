# Bug Report: numpy.typing NBitBase Deprecation Warning Never Triggered

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `NBitBase` class in `numpy.typing` is intended to show a deprecation warning when accessed, but the warning is never triggered because the attribute is found via normal module lookup before `__getattr__` is called.

## Property-Based Test

```python
import warnings
import sys
from hypothesis import given, strategies as st


def fresh_import_numpy_typing():
    if 'numpy.typing' in sys.modules:
        del sys.modules['numpy.typing']
    import numpy.typing
    return numpy.typing


def test_nbits_deprecation_warning():
    npt = fresh_import_numpy_typing()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = npt.NBitBase

        assert len(w) > 0, "Expected DeprecationWarning when accessing NBitBase"
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w), \
            f"Expected DeprecationWarning but got: {[warning.category for warning in w]}"
```

**Failing input**: Accessing `numpy.typing.NBitBase` attribute

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = npt.NBitBase

    if len(w) == 0:
        print("BUG: No deprecation warning triggered")
    else:
        print(f"OK: Warning triggered - {w[0].message}")
```

Output:
```
BUG: No deprecation warning triggered
```

## Why This Is A Bug

The `numpy.typing.__init__.py` file contains a `__getattr__` function that is explicitly designed to trigger a deprecation warning when `NBitBase` is accessed:

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

However, because `NBitBase` is imported at the top of the file (`from numpy._typing import NBitBase`) and added to `__all__`, Python's attribute lookup finds it in the module's `__dict__` before `__getattr__` is ever called. This means users accessing the deprecated `NBitBase` attribute never see the deprecation warning, defeating the purpose of the deprecation notice.

## Fix

Remove `NBitBase` from the direct imports and `__all__`, making it only accessible through `__getattr__`:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -174,9 +174,9 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray

-__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]
+__all__ = ["ArrayLike", "DTypeLike", "NDArray"]


 __DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
@@ -187,6 +187,7 @@ def __dir__() -> list[str]:

 def __getattr__(name: str):
     if name == "NBitBase":
+        from numpy._typing import NBitBase
         import warnings

         # Deprecated in NumPy 2.3, 2025-05-01
@@ -197,6 +198,8 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
+        # Add to __DIR to make it visible in dir()
+        __DIR.append("NBitBase")
         return NBitBase

     if name in __DIR_SET:
```