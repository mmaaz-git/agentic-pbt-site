# Bug Report: numpy.typing NBitBase Deprecation Warning Not Triggered

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The deprecation warning for `NBitBase` is never triggered when users access `numpy.typing.NBitBase`, despite there being code in `__getattr__` intended to emit such a warning.

## Property-Based Test

```python
import warnings

import numpy.typing as npt
from hypothesis import given, strategies as st


@given(st.just('NBitBase'))
def test_deprecated_attributes_should_warn(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        getattr(npt, attr_name)

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, \
            f'Expected DeprecationWarning when accessing {attr_name}, but got {len(deprecation_warnings)} warnings'
```

**Failing input**: `attr_name='NBitBase'`

## Reproducing the Bug

```python
import warnings

import numpy.typing as npt


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    result = npt.NBitBase
    print(f'Warnings captured: {len(w)}')

assert len(w) == 0
```

## Why This Is A Bug

The `numpy.typing.__init__.py` contains a `__getattr__` function with explicit logic to warn about NBitBase deprecation:

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

However, this warning is never triggered because `NBitBase` is imported at the module level:

```python
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
```

When an attribute is already in the module's `__dict__`, Python's attribute lookup finds it immediately and never calls `__getattr__`. Thus, users accessing `numpy.typing.NBitBase` receive no deprecation warning, defeating the purpose of the deprecation notice.

The NBitBase class itself is marked as deprecated (NumPy 2.3, 2025-05-01), and the documentation clearly states it should trigger a deprecation warning, but this never happens in practice.

## Fix

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -171,7 +171,7 @@
 # NOTE: The API section will be appended with additional entries
 # further down in this file

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -182,6 +182,8 @@ __DIR_SET = frozenset(__DIR)
 def __dir__() -> list[str]:
     return __DIR

+from numpy._typing import NBitBase as _NBitBase_impl
+
 def __getattr__(name: str):
     if name == "NBitBase":
         import warnings
@@ -193,7 +195,7 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase_impl

     if name in __DIR_SET:
         return globals()[name]
```