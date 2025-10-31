# Bug Report: numpy.typing.NBitBase Deprecation Warning Not Shown

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
import numpy.typing as npt


@given(st.none())
@settings(max_examples=1)
def test_nbitbase_access_emits_deprecation(_):
    """Test that accessing npt.NBitBase emits a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase

        assert len(w) == 1, f"Expected 1 deprecation warning, got {len(w)}"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category}"
        assert "NBitBase" in str(w[0].message), f"Warning message doesn't mention NBitBase: {w[0].message}"
        assert "deprecated" in str(w[0].message).lower(), f"Warning message doesn't mention deprecation: {w[0].message}"


if __name__ == "__main__":
    # Run the test
    try:
        test_nbitbase_access_emits_deprecation()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
```

<details>

<summary>
**Failing input**: `None` (any access to `npt.NBitBase`)
</summary>
```
Test failed: Expected 1 deprecation warning, got 0
```
</details>

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase

    if len(w) == 0:
        print("BUG: No deprecation warning was emitted when accessing NBitBase")
    else:
        print(f"OK: {len(w)} warning(s) emitted")
        for warning in w:
            print(f"  - {warning.category.__name__}: {warning.message}")
```

<details>

<summary>
No deprecation warning emitted when accessing NBitBase
</summary>
```
BUG: No deprecation warning was emitted when accessing NBitBase
```
</details>

## Why This Is A Bug

This violates the documented deprecation contract for `NBitBase`. The class is officially deprecated in NumPy 2.3 (as stated in the docstring at `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_typing/_nbit_base.py:20-22`), but users are never warned about this deprecation when they use it.

The code in `numpy/typing/__init__.py` contains specific deprecation handling logic in the `__getattr__` method (lines 172-184) that is designed to emit a `DeprecationWarning` when `NBitBase` is accessed. However, this code is unreachable because:

1. Line 160 imports `NBitBase` directly: `from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray`
2. Line 162 includes it in `__all__`: `__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]`
3. Since `NBitBase` exists in the module's namespace, Python's attribute lookup finds it immediately and never calls `__getattr__`

The `__getattr__` method is only called when an attribute is NOT found through normal lookup, making the deprecation warning dead code that never executes. This means users who rely on `NBitBase` will not be warned that it will be removed in future NumPy versions, preventing them from migrating their code in time.

## Relevant Context

The deprecation timeline and documentation show this is an official deprecation:
- NBitBase was deprecated in NumPy 2.3 (2025-05-01) as noted in comments
- The docstring explicitly states "deprecated:: 2.3"
- Users are advised to use `@typing.overload` or `TypeVar` with scalar-type upper bounds instead
- The deprecation warning text is already written and ready, but never displayed

This issue affects all users of `NBitBase` who need to be notified to update their type annotations before the class is removed in a future NumPy version. The bug prevents the proper API migration path that NumPy intended to provide through deprecation warnings.

## Proposed Fix

Remove `NBitBase` from the direct import statement and let the `__getattr__` method handle it exclusively when accessed:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,9 +157,9 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray

-__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]
+__all__ = ["ArrayLike", "DTypeLike", "NDArray"]


 __DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
@@ -172,6 +172,7 @@ def __dir__() -> list[str]:
 def __getattr__(name: str):
     if name == "NBitBase":
         import warnings
+        from numpy._typing import NBitBase

         # Deprecated in NumPy 2.3, 2025-05-01
         warnings.warn(
```