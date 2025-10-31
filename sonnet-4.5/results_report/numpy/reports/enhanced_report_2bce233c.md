# Bug Report: numpy.typing NBitBase Deprecation Warning Never Triggers

**Target**: `numpy.typing.__getattr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The deprecation warning for `NBitBase` in numpy.typing is never triggered because `NBitBase` is imported directly into the module namespace, bypassing the `__getattr__` method that contains the warning logic.

## Property-Based Test

```python
import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_deprecated_attribute_warns(attr_name):
    """Test that accessing NBitBase triggers a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        getattr(npt, attr_name)
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category}"

# Run the test
if __name__ == "__main__":
    test_deprecated_attribute_warns()
```

<details>

<summary>
**Failing input**: `'NBitBase'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 16, in <module>
    test_deprecated_attribute_warns()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_deprecated_attribute_warns
    def test_deprecated_attribute_warns(attr_name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 11, in test_deprecated_attribute_warns
    assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
           ^^^^^^^^^^^
AssertionError: Expected 1 warning, got 0
Falsifying example: test_deprecated_attribute_warns(
    attr_name='NBitBase',
)
```
</details>

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

# Test 1: Normal attribute access (expected to trigger warning but doesn't)
print("Test 1: Normal attribute access")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    obj = npt.NBitBase
    print(f"Warnings captured: {len(w)}")
    print(f"Expected: 1, Actual: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")
    else:
        print("No warning triggered!")

print("\n" + "="*50 + "\n")

# Test 2: Verify NBitBase exists and is accessible
print("Test 2: Verify NBitBase exists")
print(f"NBitBase type: {type(npt.NBitBase)}")
print(f"NBitBase is in module __dict__: {'NBitBase' in npt.__dict__}")
print(f"NBitBase is in __all__: {'NBitBase' in npt.__all__}")

print("\n" + "="*50 + "\n")

# Test 3: Direct call to __getattr__ (this SHOULD work)
print("Test 3: Direct __getattr__ call")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    obj = npt.__getattr__("NBitBase")
    print(f"Warnings captured: {len(w)}")
    print(f"Expected: 1, Actual: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")
        print(f"Warning category: {w[0].category}")
```

<details>

<summary>
Normal access fails to trigger warning, but direct __getattr__ call works
</summary>
```
Test 1: Normal attribute access
Warnings captured: 0
Expected: 1, Actual: 0
No warning triggered!

==================================================

Test 2: Verify NBitBase exists
NBitBase type: <class 'type'>
NBitBase is in module __dict__: True
NBitBase is in __all__: True

==================================================

Test 3: Direct __getattr__ call
Warnings captured: 1
Expected: 1, Actual: 1
Warning message: `NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)
Warning category: <class 'DeprecationWarning'>
```
</details>

## Why This Is A Bug

This violates the documented deprecation contract. The `__getattr__` method in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/typing/__init__.py` (lines 173-184) contains explicit code to emit a `DeprecationWarning` when `NBitBase` is accessed:

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
```

However, because `NBitBase` is imported directly at the module level (line 160: `from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray`), Python's attribute lookup mechanism finds it in the module's `__dict__` before ever calling `__getattr__`. The `__getattr__` method is only called as a fallback when an attribute cannot be found through normal lookup.

This means users upgrading to NumPy 2.3+ are not receiving the intended deprecation warnings about their usage of `NBitBase`, potentially leading to broken code when `NBitBase` is eventually removed.

## Relevant Context

- **NumPy Version**: 2.3.0
- **Python's Attribute Lookup Order**: When accessing an attribute like `npt.NBitBase`, Python first checks the module's `__dict__`. Only if the attribute is not found does it call `__getattr__` as a fallback.
- **Documentation**: The NumPy 2.3.0 release notes officially announce the deprecation of `NBitBase`, recommending users switch to `@typing.overload` or `TypeVar` with a scalar-type upper bound.
- **Impact**: Without the warning, users won't be notified at runtime about deprecated API usage, which is especially problematic for large codebases where deprecation warnings help track necessary migrations.

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

@@ -180,7 +181,7 @@ def __getattr__(name: str):
             "bound, instead. (deprecated in NumPy 2.3)",
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase

     if name in __DIR_SET:
         return globals()[name]
```