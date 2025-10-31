# Bug Report: numpy.typing NBitBase Deprecation Warning Never Triggers

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The numpy.typing module contains code to emit a deprecation warning when NBitBase is accessed, but the warning never triggers because Python's attribute lookup finds NBitBase directly in the module namespace, bypassing the __getattr__ method that contains the warning logic.

## Property-Based Test

```python
import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st


def test_nbitbase_deprecation():
    """Test that accessing NBitBase emits a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category}"
        assert "NBitBase" in str(w[0].message), f"'NBitBase' not found in warning message: {w[0].message}"


if __name__ == "__main__":
    # Run the test
    test_nbitbase_deprecation()
    print("Test passed: NBitBase deprecation warning was emitted successfully")
```

<details>

<summary>
**Failing input**: `npt.NBitBase` (direct attribute access)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 18, in <module>
    test_nbitbase_deprecation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 11, in test_nbitbase_deprecation
    assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
           ^^^^^^^^^^^
AssertionError: Expected 1 warning, got 0
```
</details>

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

# Attempting to catch the deprecation warning that should be emitted
# when accessing NBitBase (deprecated in NumPy 2.3)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Access NBitBase - this should trigger a deprecation warning
    obj = npt.NBitBase

    # Check if any warnings were caught
    print(f"Warnings caught: {len(w)}")
    print(f"Expected: 1 DeprecationWarning")

    if len(w) > 0:
        for warning in w:
            print(f"Warning category: {warning.category}")
            print(f"Warning message: {warning.message}")
    else:
        print("No warnings were emitted (BUG: deprecation warning not triggered)")

    # Verify that NBitBase is accessible
    print(f"\nNBitBase object type: {type(obj)}")
    print(f"NBitBase object: {obj}")
```

<details>

<summary>
No deprecation warning emitted when accessing NBitBase
</summary>
```
Warnings caught: 0
Expected: 1 DeprecationWarning
No warnings were emitted (BUG: deprecation warning not triggered)

NBitBase object type: <class 'type'>
NBitBase object: <class 'numpy.typing.NBitBase'>
```
</details>

## Why This Is A Bug

This violates the documented deprecation contract for NBitBase. The code at `/home/npc/miniconda/lib/python3.13/site-packages/numpy/typing/__init__.py` shows clear intent to emit a deprecation warning:

1. **Line 160** imports NBitBase directly: `from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray`
2. **Line 162** includes NBitBase in `__all__`, making it part of the public API
3. **Lines 173-184** define `__getattr__` with specific logic to emit a deprecation warning for NBitBase:
   - Comment on line 176 states: `# Deprecated in NumPy 2.3, 2025-05-01`
   - Warning message states NBitBase is "deprecated in NumPy 2.3" and "will be removed from numpy.typing"
4. However, Python's attribute lookup mechanism finds NBitBase directly in the module's namespace (due to the import on line 160) and never calls `__getattr__`
5. This breaks the deprecation timeline where users should be warned before the planned removal date (2025-05-01)

The bug prevents users from receiving critical migration warnings when using deprecated APIs, potentially causing their code to break without warning when NBitBase is removed in a future NumPy version.

## Relevant Context

The `__getattr__` function in Python modules is only called when normal attribute lookup fails. Since NBitBase is imported at module level and exists in the module's `__dict__`, Python's standard attribute resolution finds it immediately without ever invoking `__getattr__`. This is a common pitfall when implementing module-level deprecation warnings.

NumPy's documentation (numpy.org/doc/stable/reference/typing.html) officially marks NBitBase as deprecated since version 2.3 with migration guidance to use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound instead.

## Proposed Fix

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,7 +157,8 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray
+from numpy._typing import NBitBase as _NBitBase

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -181,7 +182,7 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase

     if name in __DIR_SET:
         return globals()[name]
```