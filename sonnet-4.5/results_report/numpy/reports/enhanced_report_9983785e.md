# Bug Report: numpy.typing NBitBase Deprecation Warning Silently Fails

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `NBitBase` class in `numpy.typing` is documented as deprecated since NumPy 2.3, with a deprecation warning implementation in the code, but the warning never triggers because Python's attribute lookup finds NBitBase in the module namespace before the `__getattr__` deprecation handler is called.

## Property-Based Test

```python
import warnings
import sys
from hypothesis import given, strategies as st


def fresh_import_numpy_typing():
    """Ensure a fresh import of numpy.typing module"""
    if 'numpy.typing' in sys.modules:
        del sys.modules['numpy.typing']
    import numpy.typing
    return numpy.typing


@given(st.just(None))  # Using a dummy strategy since we're not testing with varying inputs
def test_nbits_deprecation_warning(dummy):
    """Test that accessing NBitBase triggers a DeprecationWarning"""
    npt = fresh_import_numpy_typing()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Access the supposedly deprecated NBitBase
        result = npt.NBitBase

        # Verify a deprecation warning was triggered
        assert len(w) > 0, "Expected DeprecationWarning when accessing NBitBase"
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w), \
            f"Expected DeprecationWarning but got: {[warning.category for warning in w]}"

        # Check the warning message content
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0, "No DeprecationWarning found"

        warning_msg = str(deprecation_warnings[0].message)
        assert "NBitBase" in warning_msg, f"Warning message should mention NBitBase: {warning_msg}"
        assert "deprecated" in warning_msg.lower(), f"Warning should mention deprecation: {warning_msg}"


if __name__ == "__main__":
    # Run the test
    test_nbits_deprecation_warning()
```

<details>

<summary>
**Failing input**: `dummy=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 41, in <module>
    test_nbits_deprecation_warning()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 15, in test_nbits_deprecation_warning
    def test_nbits_deprecation_warning(dummy):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 26, in test_nbits_deprecation_warning
    assert len(w) > 0, "Expected DeprecationWarning when accessing NBitBase"
           ^^^^^^^^^^
AssertionError: Expected DeprecationWarning when accessing NBitBase
Falsifying example: test_nbits_deprecation_warning(
    dummy=None,
)
```
</details>

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

# Test if NBitBase deprecation warning is triggered
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Try to access NBitBase
    result = npt.NBitBase

    # Check if any warnings were triggered
    if len(w) == 0:
        print("BUG: No deprecation warning triggered when accessing numpy.typing.NBitBase")
        print(f"Successfully accessed NBitBase: {result}")
    else:
        print(f"OK: Warning triggered")
        for warning in w:
            print(f"  - {warning.category.__name__}: {warning.message}")

# Additional verification - show that NBitBase is directly in module namespace
print(f"\n'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")
print(f"'NBitBase' in npt.__all__: {'NBitBase' in npt.__all__}")

# Show that __getattr__ is never called for NBitBase
print("\nDirect __getattr__ call (should trigger warning):")
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    try:
        result2 = npt.__getattr__("NBitBase")
        if len(w2) > 0:
            print(f"  Warning triggered via __getattr__: {w2[0].message}")
    except AttributeError as e:
        print(f"  AttributeError: {e}")
```

<details>

<summary>
Output showing the bug and root cause
</summary>
```
BUG: No deprecation warning triggered when accessing numpy.typing.NBitBase
Successfully accessed NBitBase: <class 'numpy.typing.NBitBase'>

'NBitBase' in npt.__dict__: True
'NBitBase' in npt.__all__: True

Direct __getattr__ call (should trigger warning):
  Warning triggered via __getattr__: `NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)
```
</details>

## Why This Is A Bug

This violates the documented API contract for several critical reasons:

1. **Documentation explicitly states NBitBase is deprecated**: The NumPy 2.3 release notes and official documentation clearly state that `NBitBase` is deprecated since version 2.3, with instructions to use `@typing.overload` or `TypeVar` instead.

2. **Deprecation warning code exists but is unreachable**: The module contains correct deprecation warning code at lines 173-184 of `/numpy/typing/__init__.py`, but it's never executed because:
   - Line 160 imports NBitBase directly: `from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray`
   - Line 162 adds it to `__all__`: `__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]`
   - Python's attribute lookup finds `NBitBase` in the module's `__dict__` before ever calling `__getattr__`

3. **Silent API breakage risk**: Users have no runtime indication that they're using deprecated functionality. When NBitBase is eventually removed, their code will break without warning.

4. **Inconsistent with NumPy's deprecation policy**: NumPy's standard practice is to provide runtime deprecation warnings before removing features, allowing users time to migrate their code.

## Relevant Context

The bug demonstrates a fundamental misunderstanding of Python's attribute resolution order. The `__getattr__` method is only called as a fallback when normal attribute lookup fails. Since NBitBase is imported directly into the module namespace, normal lookup succeeds and `__getattr__` is never invoked.

The output clearly shows:
- `'NBitBase' in npt.__dict__: True` - NBitBase exists in the module namespace
- Directly calling `npt.__getattr__("NBitBase")` DOES trigger the warning, proving the deprecation code works
- Normal attribute access (`npt.NBitBase`) bypasses `__getattr__` entirely

This affects all NumPy 2.3+ users who are using NBitBase in their type annotations. They receive no indication that they should migrate to the recommended alternatives.

## Proposed Fix

Remove NBitBase from direct imports and `__all__`, ensuring it's only accessible through `__getattr__`:

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
@@ -171,6 +171,7 @@ def __dir__() -> list[str]:

 def __getattr__(name: str):
     if name == "NBitBase":
+        from numpy._typing import NBitBase
         import warnings

         # Deprecated in NumPy 2.3, 2025-05-01
```