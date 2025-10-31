# Bug Report: numpy.typing NBitBase Deprecation Warning Not Triggered

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `NBitBase` attribute is deprecated according to the `__getattr__` implementation in `numpy.typing`, but the deprecation warning is never triggered because `NBitBase` is imported at module level, making the warning code unreachable.

## Property-Based Test

```python
import warnings
import pytest
from hypothesis import given, strategies as st, settings
import numpy.typing as npt


class TestDeprecationWarning:
    def test_nbbitbase_deprecation_warning_triggered(self):
        """
        Property: According to __getattr__ implementation, accessing NBitBase
        should trigger a DeprecationWarning.

        Evidence: Lines 173-184 in __init__.py show special handling for NBitBase
        with a deprecation warning message.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = npt.NBitBase

            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]

            assert len(deprecation_warnings) > 0, \
                "NBitBase access should trigger DeprecationWarning, but none was issued"

            assert any("NBitBase" in str(warning.message) for warning in deprecation_warnings), \
                "DeprecationWarning should mention NBitBase"
```

**Failing behavior**: No deprecation warning is issued when accessing `npt.NBitBase`.

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    print(f"Warnings issued: {len(w)}")
    print(f"Expected: 1 DeprecationWarning")
    print(f"Actual: {len([x for x in w if issubclass(x.category, DeprecationWarning)])} DeprecationWarning(s)")
```

**Output**:
```
Warnings issued: 0
Expected: 1 DeprecationWarning
Actual: 0 DeprecationWarning(s)
```

## Why This Is A Bug

The `__getattr__` function in `numpy/typing/__init__.py` (lines 172-189) contains special handling for `NBitBase` that should issue a deprecation warning:

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

However, `NBitBase` is imported at module level (line 160):
```python
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
```

Since `NBitBase` exists as a module attribute through the import, Python's normal attribute lookup finds it **before** `__getattr__` is called. The `__getattr__` method is only invoked when an attribute is not found through normal lookup, so the deprecation warning code is never executed.

This violates the documented intention to deprecate `NBitBase` and warn users.

## Fix

Remove `NBitBase` from the module-level import and from `__all__` to force attribute access through `__getattr__`:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,9 +157,9 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray, NBitBase as _NBitBase

-__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]
+__all__ = ["ArrayLike", "DTypeLike", "NDArray"]


 __DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
@@ -181,7 +181,7 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase

     if name in __DIR_SET:
         return globals()[name]
```

This ensures that accessing `NBitBase` goes through `__getattr__` and triggers the deprecation warning as intended.