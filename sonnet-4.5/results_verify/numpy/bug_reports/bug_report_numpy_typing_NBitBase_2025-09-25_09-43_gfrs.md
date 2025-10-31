# Bug Report: numpy.typing NBitBase Deprecation Warning Not Triggered

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `numpy.typing` module has a `__getattr__` method designed to emit a `DeprecationWarning` when `NBitBase` is accessed, but the warning is never triggered because `NBitBase` is directly imported into the module's global namespace.

## Property-Based Test

```python
import warnings
from hypothesis import given, strategies as st


@given(st.just("NBitBase"))
def test_nbitbase_deprecation_warning(attr_name):
    import numpy.typing as npt

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)

        assert len(w) >= 1, f"Expected DeprecationWarning when accessing {attr_name}, but got no warnings"
        assert any(
            issubclass(warning.category, DeprecationWarning) and "NBitBase" in str(warning.message)
            for warning in w
        ), f"Expected DeprecationWarning about NBitBase, but got: {[str(w_.message) for w_ in w]}"
```

**Failing input**: `"NBitBase"`

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase

    print(f"Warnings captured: {len(w)}")
    assert len(w) == 0, "BUG: No deprecation warning was raised!"

print("\nManual __getattr__ call (shows warning IS defined):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__("NBitBase")
    print(f"Warnings captured: {len(w)}")
    if w:
        print(f"Warning: {w[0].message}")
```

## Why This Is A Bug

The `numpy.typing.__init__.py` module defines a `__getattr__` method that is explicitly designed to emit a deprecation warning when `NBitBase` is accessed:

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
    ...
```

However, at the top of the same file, `NBitBase` is directly imported:

```python
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
```

This means `NBitBase` exists in the module's global namespace. When a user accesses `numpy.typing.NBitBase`, Python's attribute lookup finds it in `globals()` and returns it immediately, without ever calling `__getattr__`. As a result, the deprecation warning is never emitted, violating the intended API contract that users should be warned about using this deprecated type.

## Fix

Remove `NBitBase` from the direct import statement and import it inside the `__getattr__` function instead:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -158,7 +158,7 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -172,6 +172,8 @@ def __dir__() -> list[str]:

 def __getattr__(name: str):
     if name == "NBitBase":
+        from numpy._typing import NBitBase
+
         import warnings

         # Deprecated in NumPy 2.3, 2025-05-01
```