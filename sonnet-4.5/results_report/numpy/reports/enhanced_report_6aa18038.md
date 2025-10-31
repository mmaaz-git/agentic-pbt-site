# Bug Report: numpy.typing NBitBase Deprecation Warning Never Fires

**Target**: `numpy.typing.NBitBase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `NBitBase` class in `numpy.typing` is documented as deprecated in NumPy 2.3 and has code to emit a `DeprecationWarning` when accessed, but the warning never fires because `NBitBase` is directly imported into the module namespace, bypassing the `__getattr__` hook entirely.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for NBitBase deprecation warning"""

import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_nbitbase_emits_deprecation_warning(attr_name):
    """Test that accessing NBitBase emits a DeprecationWarning"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)

        assert len(w) >= 1, "Expected deprecation warning to be emitted"
        assert issubclass(w[0].category, DeprecationWarning)
        assert "NBitBase" in str(w[0].message)
        assert "deprecated" in str(w[0].message).lower()

if __name__ == "__main__":
    # Run the test with Hypothesis
    test_nbitbase_emits_deprecation_warning()
```

<details>

<summary>
**Failing input**: `"NBitBase"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 22, in <module>
    test_nbitbase_emits_deprecation_warning()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 9, in test_nbitbase_emits_deprecation_warning
    def test_nbitbase_emits_deprecation_warning(attr_name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 15, in test_nbitbase_emits_deprecation_warning
    assert len(w) >= 1, "Expected deprecation warning to be emitted"
           ^^^^^^^^^^^
AssertionError: Expected deprecation warning to be emitted
Falsifying example: test_nbitbase_emits_deprecation_warning(
    attr_name='NBitBase',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of NBitBase deprecation warning bug"""

import warnings
import numpy.typing as npt

print("Testing NBitBase deprecation warning in numpy.typing")
print("=" * 60)

# Test 1: Direct access to NBitBase
print("\nTest 1: Direct access to NBitBase")
print("-" * 40)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = npt.NBitBase

    print(f"Accessed: npt.NBitBase")
    print(f"Result type: {type(result)}")
    print(f"Warnings caught: {len(w)}")

    if len(w) > 0:
        for warning in w:
            print(f"  Warning: {warning.category.__name__}: {warning.message}")
    else:
        print("  BUG: No deprecation warning emitted!")

# Test 2: Access via getattr
print("\nTest 2: Access via getattr()")
print("-" * 40)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = getattr(npt, "NBitBase")

    print(f"Accessed: getattr(npt, 'NBitBase')")
    print(f"Result type: {type(result)}")
    print(f"Warnings caught: {len(w)}")

    if len(w) > 0:
        for warning in w:
            print(f"  Warning: {warning.category.__name__}: {warning.message}")
    else:
        print("  BUG: No deprecation warning emitted!")

# Test 3: Check module internals
print("\nTest 3: Module internals")
print("-" * 40)
print(f"'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")
print(f"'NBitBase' in npt.__all__: {'NBitBase' in npt.__all__}")

# Test 4: Expected behavior summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("Expected: DeprecationWarning when accessing NBitBase")
print("Actual: No warning emitted")
print("\nRoot cause: NBitBase is imported directly into module namespace")
print("at line 160 of numpy/typing/__init__.py, bypassing the __getattr__")
print("hook (lines 173-184) that contains the deprecation warning.")
```

<details>

<summary>
Output showing the bug
</summary>
```
Testing NBitBase deprecation warning in numpy.typing
============================================================

Test 1: Direct access to NBitBase
----------------------------------------
Accessed: npt.NBitBase
Result type: <class 'type'>
Warnings caught: 0
  BUG: No deprecation warning emitted!

Test 2: Access via getattr()
----------------------------------------
Accessed: getattr(npt, 'NBitBase')
Result type: <class 'type'>
Warnings caught: 0
  BUG: No deprecation warning emitted!

Test 3: Module internals
----------------------------------------
'NBitBase' in npt.__dict__: True
'NBitBase' in npt.__all__: True

============================================================
SUMMARY:
Expected: DeprecationWarning when accessing NBitBase
Actual: No warning emitted

Root cause: NBitBase is imported directly into module namespace
at line 160 of numpy/typing/__init__.py, bypassing the __getattr__
hook (lines 173-184) that contains the deprecation warning.
```
</details>

## Why This Is A Bug

This violates the documented deprecation contract. The NumPy 2.3.0 release notes explicitly state that `numpy.typing.NBitBase` is deprecated. The source code at `/numpy/typing/__init__.py:173-184` contains a carefully implemented `__getattr__` hook specifically to emit a `DeprecationWarning` with the message: "`NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)".

However, this warning mechanism is completely bypassed due to a Python attribute resolution order issue. Line 160 imports `NBitBase` directly: `from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray`. This places `NBitBase` directly in the module's `__dict__`. When Python resolves `npt.NBitBase`, it finds the attribute in `__dict__` and returns it immediately, never calling `__getattr__`. The deprecation warning code is effectively dead code that can never execute.

This means users upgrading to NumPy 2.3 will not receive any runtime warnings that `NBitBase` is deprecated, preventing them from updating their code before it breaks in a future release when `NBitBase` is removed entirely.

## Relevant Context

The deprecation mechanism in NumPy follows standard Python practices where deprecated features emit `DeprecationWarning` at runtime to give users advance notice before removal. The comment in the source code indicates "Deprecated in NumPy 2.3, 2025-05-01", showing this was a planned deprecation.

The `__getattr__` pattern is commonly used for module-level deprecations in Python, but it only works when the deprecated attribute is not already present in the module's namespace. The NumPy documentation at line 95 of the same file even references `NBitBase` as part of the typing system, indicating it's a well-established part of the API that users may be relying upon.

Source code location: `/numpy/typing/__init__.py`
- Line 160: Direct import that causes the bug
- Lines 173-184: Deprecation warning code that never executes

## Proposed Fix

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -157,7 +157,10 @@

 # pyright: reportDeprecated=false

-from numpy._typing import ArrayLike, DTypeLike, NBitBase, NDArray
+from numpy._typing import ArrayLike, DTypeLike, NDArray
+
+# Import NBitBase privately for __getattr__ to use
+from numpy._typing import NBitBase as _NBitBase

 __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

@@ -180,7 +183,7 @@ def __getattr__(name: str):
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return _NBitBase

     if name in __DIR_SET:
         return globals()[name]
```