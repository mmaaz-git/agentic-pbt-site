# Bug Report: numpy.typing NBitBase Deprecation Warning Never Shown

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The deprecation warning for `NBitBase` is never triggered when users access it via `numpy.typing.NBitBase` because the name is imported directly into the module namespace, preventing `__getattr__` from ever being called.

## Property-Based Test

```python
import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st


@given(st.just("NBitBase"))
def test_nbitbase_emits_deprecation_warning(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)

        assert len(w) >= 1, f"Expected deprecation warning for {attr_name}"
        assert any("deprecated" in str(w_item.message).lower() for w_item in w)
```

**Failing input**: `"NBitBase"`

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

print("Accessing npt.NBitBase (should emit deprecation warning but doesn't):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    print(f"Warnings raised: {len(w)}")

print("\nCalling __getattr__ directly (emits warning as intended):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__("NBitBase")
    print(f"Warnings raised: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")
```

Output:
```
Accessing npt.NBitBase (should emit deprecation warning but doesn't):
Warnings raised: 0

Calling __getattr__ directly (emits warning as intended):
Warnings raised: 1
Warning message: `NBitBase` is deprecated and will be removed from numpy.typing in the future...
```

## Why This Is A Bug

The code intends to emit a deprecation warning when users access `NBitBase` (as evidenced by lines 172-184 in `numpy/typing/__init__.py`). However, this warning is never shown because:

1. Line 160 imports `NBitBase` directly: `from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray`
2. This puts `NBitBase` in the module's `__dict__`
3. Python's attribute lookup finds it there before ever calling `__getattr__`
4. Therefore, the deprecation warning code in `__getattr__` is unreachable

Users who are using the deprecated `NBitBase` will never see the deprecation warning and won't know to migrate their code before it's removed in a future NumPy version.

## Fix

Remove `NBitBase` from the direct import on line 160, and instead import it only within `__getattr__`:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,7 +157,7 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -172,6 +172,7 @@ def __dir__() -> list[str]:
 def __getattr__(name: str):
     if name == "NBitBase":
         import warnings
+        from numpy._typing import NBitBase

         # Deprecated in NumPy 2.3, 2025-05-01
         warnings.warn(
```