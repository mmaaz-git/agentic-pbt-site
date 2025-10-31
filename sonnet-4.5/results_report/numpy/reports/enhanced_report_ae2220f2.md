# Bug Report: numpy.typing NBitBase Deprecation Warning Not Triggered

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `numpy.typing` module's `__getattr__` method is designed to emit a `DeprecationWarning` when `NBitBase` is accessed, but the warning never triggers because `NBitBase` is directly imported into the module's global namespace, bypassing the `__getattr__` mechanism entirely.

## Property-Based Test

```python
import warnings
from hypothesis import given, strategies as st


@given(st.just("NBitBase"))
def test_nbitbase_deprecation_warning(attr_name):
    import numpy.typing as npt

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)

        assert len(w) >= 1, f"Expected DeprecationWarning when accessing {attr_name}, but got no warnings"
        assert any(
            issubclass(warning.category, DeprecationWarning) and "NBitBase" in str(warning.message)
            for warning in w
        ), f"Expected DeprecationWarning about NBitBase, but got: {[str(w_.message) for w_ in w]}"


if __name__ == "__main__":
    test_nbitbase_deprecation_warning()
```

<details>

<summary>
**Failing input**: `'NBitBase'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 21, in <module>
    test_nbitbase_deprecation_warning()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 6, in test_nbitbase_deprecation_warning
    def test_nbitbase_deprecation_warning(attr_name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 13, in test_nbitbase_deprecation_warning
    assert len(w) >= 1, f"Expected DeprecationWarning when accessing {attr_name}, but got no warnings"
           ^^^^^^^^^^^
AssertionError: Expected DeprecationWarning when accessing NBitBase, but got no warnings
Falsifying example: test_nbitbase_deprecation_warning(
    attr_name='NBitBase',
)
```
</details>

## Reproducing the Bug

```python
import warnings
import numpy.typing as npt

print("Testing NBitBase deprecation warning...\n")

# Test 1: Direct access to NBitBase (should show warning but doesn't)
print("Test 1: Direct access to npt.NBitBase")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase

    print(f"Warnings captured: {len(w)}")
    if len(w) == 0:
        print("BUG CONFIRMED: No deprecation warning was raised when accessing NBitBase!")
    else:
        for warning in w:
            print(f"Warning: {warning.message}")

print("\n" + "="*50 + "\n")

# Test 2: Verify the warning exists in __getattr__ (manual call)
print("Test 2: Manual __getattr__ call (to prove the warning IS defined)")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__("NBitBase")

    print(f"Warnings captured: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")

print("\n" + "="*50 + "\n")

# Test 3: Show NBitBase is in module namespace
print("Test 3: Checking module namespace")
print(f"'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")
print(f"'NBitBase' in dir(npt): {'NBitBase' in dir(npt)}")

print("\n" + "="*50 + "\n")

# Test 4: Demonstrate the attribute lookup order issue
print("Test 4: Understanding Python's attribute lookup order")
print("When accessing npt.NBitBase, Python:")
print("1. First checks npt.__dict__ (module globals)")
print("2. Only calls __getattr__ if not found in __dict__")
print(f"\nSince NBitBase IS in __dict__, __getattr__ is never called!")
print(f"NBitBase object from direct access: {npt.NBitBase}")
print(f"NBitBase object from __getattr__: {npt.__getattr__('NBitBase')}")
print(f"They are the same object: {npt.NBitBase is npt.__getattr__('NBitBase')}")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/6/repo.py:47: DeprecationWarning: `NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)
  print(f"NBitBase object from __getattr__: {npt.__getattr__('NBitBase')}")
/home/npc/pbt/agentic-pbt/worker_/6/repo.py:48: DeprecationWarning: `NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)
  print(f"They are the same object: {npt.NBitBase is npt.__getattr__('NBitBase')}")
Testing NBitBase deprecation warning...

Test 1: Direct access to npt.NBitBase
Warnings captured: 0
BUG CONFIRMED: No deprecation warning was raised when accessing NBitBase!

==================================================

Test 2: Manual __getattr__ call (to prove the warning IS defined)
Warnings captured: 1
Warning message: `NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)

==================================================

Test 3: Checking module namespace
'NBitBase' in npt.__dict__: True
'NBitBase' in dir(npt): True

==================================================

Test 4: Understanding Python's attribute lookup order
When accessing npt.NBitBase, Python:
1. First checks npt.__dict__ (module globals)
2. Only calls __getattr__ if not found in __dict__

Since NBitBase IS in __dict__, __getattr__ is never called!
NBitBase object from direct access: <class 'numpy.typing.NBitBase'>
NBitBase object from __getattr__: <class 'numpy.typing.NBitBase'>
They are the same object: True
```
</details>

## Why This Is A Bug

This violates the expected deprecation contract in multiple ways:

1. **NumPy 2.3 officially deprecated NBitBase**: The [official documentation](https://numpy.org/doc/stable/reference/typing.html#numpy.typing.NBitBase) states "Deprecated since version 2.3" and the release notes confirm NBitBase should emit warnings when accessed.

2. **The warning mechanism exists but is bypassed**: The code at `/numpy/typing/__init__.py:173-184` contains explicit logic to emit a `DeprecationWarning` when `NBitBase` is accessed through `__getattr__`, but this code is never executed.

3. **Python's attribute resolution order is violated**: When `npt.NBitBase` is accessed, Python follows this order:
   - First checks the module's `__dict__` (globals)
   - Only calls `__getattr__` if the attribute is not found

   Since line 160 imports `NBitBase` directly (`from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray`), it exists in the module's globals, so `__getattr__` is never invoked.

4. **Users are not warned about deprecated API**: Without the warning, users continue using `NBitBase` unaware it will be removed, preventing them from updating their code proactively. This breaks the deprecation contract that gives users time to migrate.

## Relevant Context

The bug occurs specifically in `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/typing/__init__.py`:

- **Line 160**: `from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray` - This imports NBitBase into module globals
- **Line 162**: `__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]` - NBitBase is part of the public API
- **Lines 173-184**: The `__getattr__` function contains the deprecation warning logic that never executes

The deprecation warning message indicates users should "Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead" as the recommended migration path.

This is particularly problematic because:
- `DeprecationWarning` is ignored by default unless code is run from `__main__`
- Many users rely on these warnings during development to update their code
- The NumPy 2.3 release notes specifically mention this deprecation

## Proposed Fix

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,7 +157,7 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -171,6 +171,8 @@ def __dir__() -> list[str]:

 def __getattr__(name: str):
     if name == "NBitBase":
+        from numpy._typing import NBitBase
+
         import warnings

         # Deprecated in NumPy 2.3, 2025-05-01
```