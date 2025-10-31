# Bug Report: numpy.typing NBitBase Deprecation Warning Never Fires

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `NBitBase` class in `numpy.typing` is supposed to emit a `DeprecationWarning` when accessed, but the warning is never emitted because `NBitBase` is directly imported into the module's namespace, bypassing the `__getattr__` hook that contains the warning logic.

## Property-Based Test

```python
import warnings
import numpy.typing as npt

@given(st.just("NBitBase"))
def test_nbitbase_emits_deprecation_warning(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)

        assert len(w) >= 1, "Expected deprecation warning to be emitted"
        assert issubclass(w[0].category, DeprecationWarning)
        assert "NBitBase" in str(w[0].message)
        assert "deprecated" in str(w[0].message).lower()
```

**Failing input**: `"NBitBase"`

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = npt.NBitBase

    print(f"Warnings caught: {len(w)}")
    print(f"Expected: 1 DeprecationWarning")
    print(f"Actual: {len(w)} warnings")

    if len(w) == 0:
        print("BUG: No deprecation warning emitted!")
```

Output:
```
Warnings caught: 0
Expected: 1 DeprecationWarning
Actual: 0 warnings
BUG: No deprecation warning emitted!
```

## Why This Is A Bug

The `numpy/typing/__init__.py` file contains a `__getattr__` function specifically designed to emit a deprecation warning when `NBitBase` is accessed:

```python
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
        return NBitBase
    ...
```

However, earlier in the same file (line 160), `NBitBase` is directly imported:

```python
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
```

This puts `NBitBase` in the module's `__dict__`. Python's attribute lookup mechanism finds attributes in `__dict__` before calling `__getattr__`, so `__getattr__` is never invoked when accessing `npt.NBitBase`, and the deprecation warning is never emitted.

This violates the documented intention to deprecate `NBitBase` as of NumPy 2.3.

## Fix

Remove `NBitBase` from the direct import statement and ensure it's only accessible through `__getattr__`:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,7 +157,10 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray
+
+# Import NBitBase at module level for __getattr__ to reference
+from numpy._typing import NBitBase as _NBitBase

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -177,7 +180,7 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase

     if name in __DIR_SET:
         return globals()[name]
```

This fix:
1. Removes `NBitBase` from the public import, so it's not in `__dict__`
2. Imports it as `_NBitBase` so the `__getattr__` function can still return it
3. Ensures `__getattr__` is called whenever someone accesses `npt.NBitBase`
4. Causes the deprecation warning to be emitted as intended