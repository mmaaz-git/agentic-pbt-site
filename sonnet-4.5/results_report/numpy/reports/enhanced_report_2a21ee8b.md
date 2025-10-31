# Bug Report: numpy.typing.NBitBase Deprecation Warning Silent Failure

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `NBitBase` class in `numpy.typing` is documented as deprecated since NumPy 2.3 and has code to emit a deprecation warning, but the warning is never triggered because the class is imported directly into the module namespace, bypassing the `__getattr__` hook responsible for the warning.

## Property-Based Test

```python
import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st
import pytest


def test_nbitbase_deprecation_warning():
    """Test that accessing NBitBase emits a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase
        assert len(w) == 1, f"Expected 1 DeprecationWarning, got {len(w)} warnings"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category}"
        assert "NBitBase" in str(w[0].message), f"Warning message doesn't mention NBitBase: {w[0].message}"


if __name__ == "__main__":
    # Run the test
    try:
        test_nbitbase_deprecation_warning()
        print("Test PASSED: NBitBase deprecation warning was emitted correctly")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"Test ERROR: {e}")
```

<details>

<summary>
**Failing input**: Direct access to `npt.NBitBase` attribute
</summary>
```
Test FAILED: Expected 1 DeprecationWarning, got 0 warnings
```
</details>

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

# Test if accessing NBitBase emits a deprecation warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Access NBitBase
    _ = npt.NBitBase

    # Check for warnings
    print(f"Number of warnings emitted: {len(w)}")

    if len(w) == 0:
        print("ERROR: No deprecation warning was emitted when accessing NBitBase!")
        print("       NBitBase is documented as deprecated but no warning appears.")
    else:
        for warning in w:
            print(f"Warning category: {warning.category.__name__}")
            print(f"Warning message: {warning.message}")

    # Verify the attribute exists and is accessible
    print(f"\nNBitBase type: {type(npt.NBitBase)}")
    print(f"NBitBase module: {npt.NBitBase.__module__ if hasattr(npt.NBitBase, '__module__') else 'N/A'}")
```

<details>

<summary>
ERROR: Missing deprecation warning for NBitBase access
</summary>
```
Number of warnings emitted: 0
ERROR: No deprecation warning was emitted when accessing NBitBase!
       NBitBase is documented as deprecated but no warning appears.

NBitBase type: <class 'type'>
NBitBase module: numpy.typing
```
</details>

## Why This Is A Bug

This violates the documented deprecation contract and the explicit intent in the code. The numpy.typing module contains a `__getattr__` function (lines 172-184) specifically designed to emit a deprecation warning when NBitBase is accessed:

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

However, this warning mechanism is completely bypassed because at line 160, NBitBase is imported directly into the module namespace:

```python
from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
```

When Python's attribute lookup finds `NBitBase` already exists in the module's `globals()`, it returns it immediately without ever calling `__getattr__`. This means users receive no warning about using deprecated functionality, preventing them from migrating their code before NBitBase is removed in a future version.

## Relevant Context

- **NumPy 2.3.0 Release Notes**: NBitBase was officially deprecated in NumPy 2.3.0 (https://numpy.org/doc/stable/release/2.3.0-notes.html)
- **Documentation**: The official NumPy documentation at https://numpy.org/doc/stable/reference/typing.html states "Deprecated since version 2.3"
- **Impact**: All users of NBitBase will continue using deprecated code without any warning, potentially leading to broken code when NBitBase is removed
- **Python Behavior**: Python's `__getattr__` is only called when normal attribute lookup fails. Since NBitBase exists in the module namespace, `__getattr__` is never invoked

## Proposed Fix

Remove NBitBase from the direct import and ensure it's only accessible through the `__getattr__` mechanism:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,7 +157,7 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -178,6 +178,7 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
+        from numpy._typing import NBitBase
         return NBitBase

     if name in __DIR_SET:
```