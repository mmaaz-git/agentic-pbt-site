# Bug Report: numpy.typing.__getattr__ NameError

**Target**: `numpy.typing.__getattr__`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `__getattr__` function in numpy.typing crashes with NameError when attempting to return NBitBase after it has been removed from the module's namespace.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings


@given(st.just(None))
@settings(max_examples=1)
def test_nbitbase_getattr_returns_nbitbase_after_deletion(unused):
    """Property: __getattr__ should return NBitBase when it's been deleted from module dict.

    The numpy.typing module has a custom __getattr__ that specifically checks for
    name == "NBitBase" and is supposed to return it with a deprecation warning.
    This should work even if NBitBase has been deleted from the module dict.
    """
    import numpy.typing as npt
    import importlib
    importlib.reload(npt)

    delattr(npt, 'NBitBase')

    obj = npt.NBitBase
    assert obj is not None
```

**Failing input**: Accessing `npt.NBitBase` after deleting it from the module dict

## Reproducing the Bug

```python
import numpy.typing as npt
import importlib

importlib.reload(npt)

delattr(npt, 'NBitBase')

obj = npt.NBitBase
```

Output:
```
NameError: name 'NBitBase' is not defined
```

## Why This Is A Bug

The `__getattr__` function in `numpy/typing/__init__.py` (lines 172-189) has a special case to handle access to `NBitBase`:

```python
def __getattr__(name: str):
    if name == "NBitBase":
        import warnings
        warnings.warn(...)
        return NBitBase  # <-- Bug: NBitBase not in scope

    if name in __DIR_SET:
        return globals()[name]  # <-- This pattern works correctly
```

When NBitBase is not in the module's namespace (e.g., after deletion), the bare reference to `NBitBase` on line 184 causes a NameError. The code should use `globals()['NBitBase']` instead, consistent with the second branch.

## Fix

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -181,7 +181,7 @@ def __getattr__(name: str):
             stacklevel=2,
         )
-        return NBitBase
+        return globals()['NBitBase']

     if name in __DIR_SET:
         return globals()[name]
```