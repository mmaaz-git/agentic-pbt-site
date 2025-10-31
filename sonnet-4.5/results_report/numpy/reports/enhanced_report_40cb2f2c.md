# Bug Report: numpy.typing NBitBase Deprecation Warning Never Shown

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The deprecation warning for `NBitBase` is never triggered when users access it via `numpy.typing.NBitBase` because the name is imported directly into the module namespace, bypassing the `__getattr__` method that contains the warning logic.

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


# Run the test
test_nbitbase_emits_deprecation_warning()
```

<details>

<summary>
**Failing input**: `"NBitBase"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 17, in <module>
    test_nbitbase_emits_deprecation_warning()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 7, in test_nbitbase_emits_deprecation_warning
    def test_nbitbase_emits_deprecation_warning(attr_name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 12, in test_nbitbase_emits_deprecation_warning
    assert len(w) >= 1, f"Expected deprecation warning for {attr_name}"
           ^^^^^^^^^^^
AssertionError: Expected deprecation warning for NBitBase
Falsifying example: test_nbitbase_emits_deprecation_warning(
    attr_name='NBitBase',
)
```
</details>

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

print("\nChecking if NBitBase is in module __dict__:")
print(f"'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")
```

<details>

<summary>
Output showing no warning for normal access but warning for direct __getattr__ call
</summary>
```
Accessing npt.NBitBase (should emit deprecation warning but doesn't):
Warnings raised: 0

Calling __getattr__ directly (emits warning as intended):
Warnings raised: 1
Warning message: `NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)

Checking if NBitBase is in module __dict__:
'NBitBase' in npt.__dict__: True
```
</details>

## Why This Is A Bug

This violates NumPy's deprecation contract with its users. The code clearly intends to emit a deprecation warning when users access `NBitBase`, as evidenced by:

1. **Documentation explicitly marks it as deprecated**: The NBitBase class docstring states `.. deprecated:: 2.3` (line 20 in `/numpy/_typing/_nbit_base.py`)

2. **Warning mechanism exists but is unreachable**: The `__getattr__` method in `/numpy/typing/__init__.py` (lines 172-184) contains explicit code to check for "NBitBase" and issue a DeprecationWarning with detailed migration instructions

3. **Python's attribute resolution bypasses the warning**: Line 160 imports `NBitBase` directly (`from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray`), placing it in the module's `__dict__`. When users access `npt.NBitBase`, Python finds it in the module dictionary and never calls `__getattr__`, making the deprecation warning unreachable

4. **Users lose critical migration information**: Without the warning, users won't know to migrate to the recommended alternatives (`@typing.overload` or `TypeVar` with scalar-type bounds) before NBitBase is removed in a future NumPy version

5. **Breaks standard deprecation workflow**: NumPy follows a strict deprecation policy where features are warned about before removal. This bug violates that policy for all users of NBitBase.

## Relevant Context

The NBitBase class was deprecated in NumPy 2.3 (2025-05-01) because changes in NumPy 2.2.0 to float64 and complex128 made static type-checkers reject code using NBitBase as a generic upper bound. The recommended migration path is to use `@typing.overload` decorators or `TypeVar` with scalar-type bounds instead.

The deprecation is documented in:
- NBitBase class docstring: `/numpy/_typing/_nbit_base.py:20`
- NumPy 2.3 release notes
- Online NumPy documentation at numpy.org

The `__getattr__` implementation exists specifically to handle deprecated names and show warnings, but the direct import prevents it from ever being invoked for NBitBase.

## Proposed Fix

Remove `NBitBase` from the direct import on line 160 and let `__getattr__` handle it:

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,7 +157,10 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray
+
+# NBitBase is deliberately not imported directly to ensure
+# the deprecation warning in __getattr__ is triggered

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -172,6 +175,7 @@ def __dir__() -> list[str]:
 def __getattr__(name: str):
     if name == "NBitBase":
         import warnings
+        from numpy._typing import NBitBase

         # Deprecated in NumPy 2.3, 2025-05-01
         warnings.warn(
```